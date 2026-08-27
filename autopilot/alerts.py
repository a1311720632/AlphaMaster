"""关键告警（B4/ADR-0007）：飞书通知 + 节流。

设计要点：
  - 引擎内直调：autopilot 子进程懒 import web.feishu_notify（纯 HTTP，读 settings），
    **不依赖 web 进程活着**——web 死了告警也必须能发。
  - 节流：同一 key 的状态"未触发→触发"发一次、"触发→恢复"发一次；critical=True
    无视节流必发（熔断类事件天然低频，不存在刷屏；节流防的是连环重复）。
  - 发送失败只 log：告警失败绝不阻断 halt/交易路径。
"""
from __future__ import annotations

from typing import Callable


class Alerter:
    """飞书告警发送器（进程内单实例，随 engine 生命周期）。"""

    def __init__(self, log: Callable[[str], None] = print) -> None:
        self._log = log
        self._active: dict[str, bool] = {}  # key → 当前是否处于触发态

    def _send_raw(self, text: str, attempts: int = 3) -> bool:
        """底层发送（含重试）。webhook 未配置/告警关闭 → log 并返回 False。"""
        if not self.configured:
            self._log("[autopilot] [alert] 飞书通知未启用或 webhook 未配置，事件仅入账本")
            return False
        try:
            from web.feishu_notify import send_text
        except Exception as exc:  # noqa: BLE001 - web 包不可用（理论不该发生）
            self._log(f"[autopilot] [alert] web.feishu_notify 导入失败: {exc}")
            return False
        for i in range(max(1, attempts)):
            try:
                ok, msg = send_text(text)
            except Exception as exc:  # noqa: BLE001 - 发送自身异常不外抛
                ok, msg = False, str(exc)
            if ok:
                return True
            if i + 1 < attempts:
                continue
        self._log(f"[autopilot] [alert] 飞书发送失败: {msg}")
        return False

    def send(self, key: str, text: str, *, critical: bool = False) -> None:
        """事件告警。key = 事件类目（节流粒度）；critical=True 无视节流必发。

        飞书总开关（feishu_autopilot_enabled，默认开）关闭时事件仍进冷账本，只是不推送。
        """
        was = self._active.get(key, False)
        if critical or not was:
            self._send_raw(text)
        self._active[key] = True

    def resolve(self, key: str, text: str = "") -> None:
        """状态恢复通知（触发态 → 正常态才发）。"""
        was = self._active.get(key, False)
        if was:
            self._send_raw(text or f"[autopilot] {key} 已恢复")
        self._active[key] = False

    @property
    def configured(self) -> bool:
        """自动驾驶飞书通知是否生效：开关打开 且 webhook 已配置。"""
        try:
            from web.settings import load_settings

            s = load_settings()
            enabled = bool(s.get("feishu_autopilot_enabled", True))
            return enabled and bool((s.get("feishu_webhook_url") or "").strip())
        except Exception:  # noqa: BLE001
            return False
