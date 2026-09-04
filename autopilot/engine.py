"""自动驾驶主循环（模式无关）。

信号计算与回测**逐字同路径**：bars → compute_features → StackVM.execute →
compute_target_positions_stateless（model_core/backtest.py:424）。取 [:, -1] 为当前
目标仓位（含 MIN_TRADE_EXPOSURE 地板）。

规模（ADR-0002）：target_notional = target_pos × equity，≤1x。
对账自愈（ADR-0006）：delta = target_notional − actual，以交易所实际持仓为准。
运营熔断（ADR-0005）：回撤 + 断网；告警监控只记录不动仓位。
"""
from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless
from web.data_sources.base import Bar, DataSource, bars_to_raw_dict

from autopilot.alerts import Alerter
from autopilot.backends import ExecutionBackend, OrderResult
from autopilot.breakers import ConnectivityBreaker, DrawdownBreaker, Monitors
from autopilot.ledger import Ledger, ledger_path
from autopilot.state import AutopilotState, BarRecord, archive_if_mismatch
from autopilot.strategy_loader import StrategySpec

# 词表驱动、构造后无状态；进程内复用（与 strategy_manager/live_signal.py 同样的单例模式）
_VM = StackVM()

# 周期 → 轮询节拍（秒）与 bar 秒数，镜像 web/realtime_manager 的 _CADENCE / _TF_SECONDS
_CADENCE = {
    "1m": 15, "5m": 30, "15m": 45, "30m": 60,
    "1h": 60, "4h": 120, "1d": 300, "1w": 600, "1M": 600,
}
_TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000,
}
# 低于此 bar 数视为 warmup 不足，跳过。
# 特征归一化用滚动窗 MT5FeatureEngineer._NORM_WINDOW=200，不足则特征全零→信号恒为 0
# （静默空仓）。故 warmup 下限取 200；默认 lookback=300 给足余量。
_MIN_BARS = 200


def _ensure_closed_bars(bars: list[Bar], now: float | None = None) -> list[Bar]:
    """剔除时间戳仍在未来的 bar（双保险，防数据源时钟偏快）。

    与 web/realtime_manager._ensure_closed_bars 同义；此处内联以避免把整个
    realtime_manager（含 settings/factory 等重依赖）拉进 autopilot 核心。
    """
    if not bars:
        return bars
    now_i = int(now if now is not None else time.time())
    out = list(bars)
    while out and int(out[-1].ts) > now_i:
        out.pop()
    return out


def compute_target_from_bars(
    formula: list[int], bars: list[Bar]
) -> tuple[float | None, float]:
    """信号计算（与回测同一条路径）。返回 (target_position[:, -1], last_close)。

    target_position 已含 MIN_TRADE_EXPOSURE 地板（由 compute_target_positions_stateless
    内部生效）。这是 ADR-0001 奇偶不变量的可执行形式——autopilot 不加任何额外变换
    （不做 sign 塌缩、不做额外缩放/截断）。返回 None 表示公式无有效输出。

    该函数被 tests/unit/test_autopilot_parity.py 直接钉住，任何额外变换都会让奇偶
    测试变红。
    """
    raw = bars_to_raw_dict(bars)                            # [1, T]
    feats = MT5FeatureEngineer.compute_features(raw)        # [1, F, T]
    factor = _VM.execute([int(t) for t in formula], feats)  # [1, T] or None
    last_close = float(bars[-1].close)
    if factor is None or factor.ndim != 2 or factor.shape[1] == 0:
        return None, last_close
    pos = compute_target_positions_stateless(factor)        # [1, T]（tanh + 地板）
    val = float(pos[0, -1].item())
    if not math.isfinite(val):
        return None, last_close
    return val, last_close


def _classify_action(before: float, after: float, eps: float = 1e-9) -> str:
    """根据调仓前后持仓（带方向名义）分类动作，供成交记录校验开平仓逻辑。

    开仓/平仓优先（一边≈0），再判反手（穿零），再加仓/减仓（同向）。
    """
    b, a = abs(before), abs(after)
    if b < eps and a >= eps:
        return "开仓"
    if a < eps and b >= eps:
        return "平仓"
    if before * after < 0:  # 穿零反手
        return "反手"
    return "加仓" if a > b else "减仓"


def _realized_pnl(
    before: float, filled: float, fill_price: float, entry_before: float
) -> float:
    """这次 fill 中【平仓部分】的实现盈亏（与 mark_to_market 同口径，收益率×名义）。

    开仓/加仓（同向，无平仓）→ 0；减仓/平仓/反手（与 before 反向的部分）→
    平仓量 × sign(before) × (fill/entry_before − 1)。
    entry_before 必须用调仓前均价（反手后 entry 被重置为 fill，不能用调仓后值）。
    """
    if entry_before <= 0 or fill_price <= 0 or before == 0:
        return 0.0
    if (filled > 0) == (before > 0):  # 同向（开仓/加仓），无平仓部分
        return 0.0
    closed = min(abs(filled), abs(before))  # 这次平掉的部分名义
    return math.copysign(closed, before) * (fill_price / entry_before - 1.0)


def _fmt_ts(ts: int) -> str:
    """Unix 秒 → 人读时间（UTC 标注，与冷账本/日报的 UTC 口径一致）。

    ≤0 无 bar 语义（空账本首启），原样返回避免打出 1970 年的假时刻。
    """
    if int(ts) <= 0:
        return str(ts)
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M") + " UTC"


class AutopilotEngine:
    """模式无关的调仓主循环。装配好后调用 run_forever()。"""

    def __init__(
        self,
        *,
        strategy: StrategySpec,
        datasource: DataSource,
        backend: ExecutionBackend,
        lookback_bars: int,
        breaker_max_drawdown_pct: float,
        breaker_drawdown_enabled: bool = True,
        breaker_max_bars_stale: int = 3,
        min_notional_delta: float,
        state_path: str | Path,
        stop_signal_paths: list[str | Path],
        cadence_s: int | None = None,
        bar_seconds: int | None = None,
        max_bars: int | None = None,
        ledger_dir: str | Path | None = None,
        ledger_file: str | Path | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        alerter: Alerter | None = None,
        heartbeat_url: str = "",
        heartbeat_max_silent_s: int = 900,
        log: Callable[[str], None] = print,
    ) -> None:
        self.strategy = strategy
        self.datasource = datasource
        self.backend = backend
        self.lookback = max(int(lookback_bars), _MIN_BARS)
        self.min_delta = float(min_notional_delta)
        self.state_path = Path(state_path)
        self.stop_paths = [Path(p) for p in stop_signal_paths]
        self.log = log

        tf = strategy.timeframe or "1h"
        self.cadence_s = int(cadence_s if cadence_s is not None else _CADENCE.get(tf, 60))
        self.bar_seconds = int(bar_seconds if bar_seconds else _TF_SECONDS.get(tf, 3600))

        self.breaker_drawdown_enabled = bool(breaker_drawdown_enabled)
        self.drawdown = DrawdownBreaker(
            breaker_max_drawdown_pct, enabled=self.breaker_drawdown_enabled
        )
        self.connectivity = ConnectivityBreaker(breaker_max_bars_stale)
        self.monitors = Monitors()

        # 三元组不匹配先归档（C2/ADR-0007）：换模式/品种/周期 = 换一笔钱，旧账本留档
        bak = archive_if_mismatch(
            self.state_path, strategy.symbol, strategy.timeframe, backend.mode
        )
        if bak:
            self.log(f"[autopilot] 旧账本三元组不匹配，已归档: {bak}")

        # 恢复或新建状态
        loaded = AutopilotState.load(self.state_path)
        if loaded and loaded.symbol == strategy.symbol and loaded.timeframe == strategy.timeframe:
            self.state = loaded
            # B1/ADR-0007：回填 peak 使回撤基线跨重启连续——重启进程 ≠ 重置熔断额度
            if loaded.peak_equity > 0:
                self.drawdown = DrawdownBreaker(
                    breaker_max_drawdown_pct, loaded.peak_equity,
                    enabled=self.breaker_drawdown_enabled,
                )
            self.log(
                f"[autopilot] 恢复状态: last_ts={_fmt_ts(loaded.last_ts)} peak={loaded.peak_equity:.6f}"
            )
        else:
            self.state = AutopilotState(
                symbol=strategy.symbol, timeframe=strategy.timeframe, mode=backend.mode
            )
        # 总收益基线 start_equity 在首根 tick 拿到 equity 时 lazy 设定（见 _tick），
        # 避免在此额外调用 fetch_equity 破坏 backend 的调用计数语义。

        self._last_ts = self.state.last_ts
        self._prev_close = float("nan")
        self._halt_reason = ""
        # 恢复 state 时把末根持仓/权益/entry/收盘价喂回 backend（paper SimBackend），
        # 并恢复 _prev_close 让首根新 bar 的 mark_to_market 计入 carry、权益不断层。
        # 否则 SimBackend 新建重置 position=0/equity=start_equity，restart 后持仓/权益丢失。
        if self.state.history:
            last = self.state.history[-1]
            self.backend.restore(
                float(last["actual_notional"]),
                float(last["equity"]),
                float(last["entry_price"]),
                float(last["close"]),
            )
            self._prev_close = float(last["close"])
        # 非 None 时处理完这么多根新 bar 后退出（冒烟/冒烟测试用；None=永久运行）
        self._max_bars = max_bars

        # 冷账本（E1/ADR-0007）：bar/trade/event 三类行 append-only，审计视图读它。
        # ledger_file 显式指定时用之（D2 离线补算隔离：自定义 state-file 必须配独立账本，
        # 否则补算的 bar/trade 会污染生产审计与 readiness 门槛计数）
        lf = Path(ledger_file) if ledger_file else ledger_path(
            strategy.symbol, backend.mode, ledger_dir
        )
        self.ledger = Ledger(lf, log=self.log)

        # 冷账本（E1/ADR-0007）：bar/trade/event 三类行 append-only，审计视图读它。
        # ledger_file 显式指定时用之（D2 离线补算隔离：自定义 state-file 必须配独立账本，
        # 否则补算的 bar/trade 会污染生产审计与 readiness 门槛计数）
        lf = Path(ledger_file) if ledger_file else ledger_path(
            strategy.symbol, backend.mode, ledger_dir
        )
        self.ledger = Ledger(lf, log=self.log)
        self._event(
            "run_start",
            f"mode={backend.mode} symbol={strategy.symbol} tf={strategy.timeframe} "
            f"formula_len={len(strategy.formula)}",
        )
        # 可注入 sleep（下单重试间隔；测试传 no-op 免真等 2s）
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        # 告警（B4/ADR-0007）：None = 无告警（paper 冒烟不配飞书也能跑）
        self.alerter = alerter
        # 心跳（B5/ADR-0007）：外部 watchdog ping；空 URL = 禁用
        self._hb_url = (heartbeat_url or "").strip()
        self._hb_max_silent = int(heartbeat_max_silent_s)
        self._hb_last_ok = 0.0
        # 昨日日报的 UTC 日标记（B4：摘要兼心跳）。壁钟驱动，日切+5min 发送
        self._last_day = ""

        # 空账本首启保护（b 决策，testnet/live）：当前这根进行中的 bar 只观察不下单，
        # 等【下一根】收盘 bar 才允许首次调仓。防止“点完启动冷不丁吃一单”——尤其
        # 一键清除→立刻重启的序列会把同一根 bar 当两次新 bar 各打一枪。
        # 已有账本的恢复启动不走此闸（其 last_ts 锁定同根 bar，天然等下根）。
        self._defer_first_trade = (
            not self.state.history and backend.mode in ("testnet", "live")
        )
        if self._defer_first_trade:
            self.log("[autopilot] 空账本首启：本根 bar 仅观察，下一根收盘后开始调仓")
            self._event("boot_defer", f"mode={backend.mode} 观察一根后进场")

    # ── 主循环 ─────────────────────────────────────────────────────────
    def run_forever(self) -> str:
        """主循环。返回 halt 原因（熔断 / STOP_SIGNAL / 异常）。"""
        self.log(
            f"[autopilot] 启动 mode={self.backend.mode} symbol={self.strategy.symbol} "
            f"tf={self.strategy.timeframe} formula_len={len(self.strategy.formula)} "
            f"lookback={self.lookback} cadence={self.cadence_s}s "
            f"dd_breaker={'on' if self.breaker_drawdown_enabled else 'off'}"
        )
        while True:
            if self._stop_signalled():
                self._halt("STOP_SIGNAL 文件检出，优雅退出")
                break
            if self._halt_reason:
                break
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - 单 tick 异常不杀进程
                self.log(f"[autopilot] tick 异常: {exc}")
            try:
                self._maybe_daily_digest()  # 壁钟驱动：UTC 日切+5min 发昨日日报
            except Exception as exc:  # noqa: BLE001 - 日报异常同样不杀交易循环
                self.log(f"[autopilot] 日报异常: {exc}")
            if self._halt_reason:
                break
            if self._max_bars is not None and len(self.state.history) >= self._max_bars:
                self._halt_reason = f"max_bars 达成（{self._max_bars} 根新 bar）"
                break
            time.sleep(self.cadence_s)
        self.log(f"[autopilot] 结束 reason={self._halt_reason or 'stopped'}")
        self._event("run_end", self._halt_reason or "stopped")
        self.ledger.close()
        return self._halt_reason or "stopped"

    # ── 单次轮询 ───────────────────────────────────────────────────────
    def _tick(self) -> None:
        # 1. 拉取已收盘 bar
        try:
            bars = self.datasource.fetch_bars(
                self.strategy.symbol,
                self.strategy.timeframe,
                self.lookback,
                drop_forming=True,
            )
        except Exception as exc:  # noqa: BLE001 - 任何拉取失败都计入断网熔断
            self.log(f"[autopilot] 行情拉取失败: {exc}")
            if self.connectivity.fail():
                # 备源全挂（B2：链全失败才走到这）→ 🔴 级告警再停
                self._alert_critical(
                    "断连熔断（备源全挂）",
                    f"{self.connectivity.reason}。持仓保持现状（未平仓），请人工检查网络/交易所。",
                )
                self._event("breaker_connectivity", str(exc))
                self._halt(self.connectivity.reason)
            return

        bars = _ensure_closed_bars(bars)
        if len(bars) < _MIN_BARS:
            self.log(f"[autopilot] bar 不足 {len(bars)}/{_MIN_BARS}，等待 warmup")
            self._maybe_heartbeat()
            return
        self.connectivity.reset()
        # 备源链切换事件消费（B2/ADR-0007）：拉取式——读后置 None，避免回调侵入
        sw = getattr(self.datasource, "last_switch", None)
        if sw:
            self.datasource.last_switch = None  # type: ignore[attr-defined]
            frm, to, _ = sw
            self._event("source_switch", f"{frm} → {to}")
            self._notify(
                "source_switch",
                f"[autopilot][{self.strategy.symbol}] 行情源切换: {frm} → {to}",
            )
        # 无新 bar 的 cadence 轮也喂心跳（低频周期/休市防饿死，B5）
        self._maybe_heartbeat()

        cur_ts = int(bars[-1].ts)
        cur_close = float(bars[-1].close)

        # 2. 无新收盘 bar
        if cur_ts == self._last_ts:
            return
        is_first = self._last_ts == 0
        # 首启延迟闸：本根用局部快照判断（观察根不交易），闸的解除自下一根生效
        deferred_now = self._defer_first_trade
        self._defer_first_trade = False
        self._last_ts = cur_ts

        # 3. 信号（与回测同路径）
        target_pos, _ = compute_target_from_bars(self.strategy.formula, bars)
        if target_pos is None:
            self.monitors.alerts.append("公式无有效输出，本期保持仓位")
            self._save()
            return

        # 4. 盘前 mark-to-market（上一期持仓 × close-to-close 收益；交易所后端 no-op）
        #    fresh start（is_first）或 _prev_close 未就绪时用 (cur,cur)：pnl=0 不动权益，
        #    但初始化 _last_close 作首单 fill 价；恢复 state 后 _prev_close 已从末根 close
        #    回填（见 __init__），走 (prev,cur) 计入 carry、权益连续。
        if is_first or self._prev_close != self._prev_close:  # NaN 检查（prev 未就绪）
            self.backend.mark_to_market(cur_close, cur_close)
        else:
            self.backend.mark_to_market(self._prev_close, cur_close)
        self._prev_close = cur_close

        # 5. 规模（ADR-0002）+ 回撤熔断（ADR-0005）
        equity = self.backend.fetch_equity()
        if self.state.start_equity <= 0:
            # 首次拿到 equity 时定总收益基线（paper=起点权益，live=真实余额快照，决策 #3/#6）
            self.state.start_equity = float(equity)
        peak, dd, tripped = self.drawdown.update(equity)
        if tripped:
            _, entry_before, _ = self.backend.fetch_position_detail(self.strategy.symbol)
            res = self.backend.flatten_all(self.strategy.symbol)
            actual, entry, unreal = self.backend.fetch_position_detail(self.strategy.symbol)
            self._record_trade(
                res, cur_ts,
                before=actual - res.filled_notional, after=actual,
                entry=entry, entry_before=entry_before, reason="breaker_flatten",
            )
            if not res.ok:
                # 全平失败（最凶险）：halt 前强制告警——仓位在场内且马上无人照看
                self._alert_critical(
                    "回撤熔断但全平失败",
                    f"回撤 {dd * 100:.2f}% 触发熔断，但 flatten_all 失败（{res.message}）。"
                    f"仓位仍在场内: {actual:+.4f} USDT，请人工立即处理。",
                )
            self._event("breaker_drawdown", f"flatten ok={res.ok} detail={res.message}")
            self._halt(self.drawdown.reason)
            self._record(
                cur_ts, cur_close, target_pos, target_pos * equity, actual, equity, peak, dd,
                entry_price=entry, unrealized_pnl=unreal,
            )
            return

        # 6. 对账自愈（ADR-0006）：delta 以交易所实际持仓为准
        #    首启延迟闸（b 决策）激活时本根只观察：不下单、不记漂移告警，
        #    真实持仓若在也交给下根 bar 对账。
        target_notional = target_pos * equity
        actual, entry_before, unreal = self.backend.fetch_position_detail(self.strategy.symbol)
        delta = target_notional - actual
        fill_ok = True
        entry = entry_before  # 不调仓时 entry 不变
        if deferred_now:
            pass
        elif abs(delta) > 0 and abs(delta) >= self.min_delta:
            # 执行熔断（B3/ADR-0007）：bar 内 3 连败即停机（不靠下根 bar 自愈——
            # 能穿透 3 次重试的失败基本是持久的：凭据吊销/保证金不足/账户限制）
            res, last_err = self._place_delta_with_retry(delta)
            fill_ok = res.ok
            if not res.ok:
                self._alert_critical(
                    "执行熔断",
                    f"下单 3 连败已停机。目标 {target_notional:+.4f} / 实际 {actual:+.4f} USDT，"
                    f"delta {delta:+.4f}。最近错误: {last_err}。仓位保持现状，请人工处理。",
                )
                self._event("breaker_execution", f"delta={delta:+.4f} err={last_err}")
                actual, entry, unreal = self.backend.fetch_position_detail(self.strategy.symbol)
                self._halt(f"执行熔断: {last_err}")
                self._record(
                    cur_ts, cur_close, target_pos, target_notional, actual, equity, peak, dd,
                    entry_price=entry, unrealized_pnl=unreal,
                )
                return
            actual, entry, unreal = self.backend.fetch_position_detail(self.strategy.symbol)
            self._record_trade(
                res, cur_ts,
                before=actual - res.filled_notional, after=actual,
                entry=entry, entry_before=entry_before,
            )
            self.monitors.observe(actual, target_notional, fill_ok)
        else:
            self.monitors.observe(actual, target_notional, fill_ok)

        self._record(
            cur_ts, cur_close, target_pos, target_notional, actual, equity, peak, dd,
            entry_price=entry, unrealized_pnl=unreal,
        )

    # ── 辅助 ───────────────────────────────────────────────────────────
    def _place_delta_with_retry(
        self, delta: float, attempts: int = 3, base_interval_s: float = 30.0
    ) -> tuple[OrderResult, str]:
        """bar 内下单重试（B3/ADR-0007）。全败返回 (失败 OrderResult, 聚合错误)。

        ok=True（含"delta 低于最小手"这类成功空单）直接返回；连打 3 发打的是同一扇门，
        指数退避 base×2^i（默认 30s→60s）——demo 的 50013 "Systems are busy" 这类
        瞬时过载要几十秒才缓过来，2s 连打必穿透（2026-09-04 XRPUSDT 实战）；持久性
        失败多等一分钟无风险（不下单本身就是安全方向）。可注入 sleep_fn 供测试。
        """
        last_err = ""
        for i in range(max(1, attempts)):
            res = self.backend.place_delta_order(self.strategy.symbol, delta)
            if res.ok:
                return res, ""
            last_err = res.message or f"attempt {i + 1}"
            self.log(f"[autopilot] 下单失败({i + 1}/{attempts}): {last_err}")
            if i + 1 < attempts:
                wait = base_interval_s * (2 ** i)
                self.log(f"[autopilot] {wait:.0f}s 后重试")
                self._sleep(wait)
        return res, last_err

    def _alert_critical(self, title: str, detail: str) -> None:
        """关键告警（B4/ADR-0007）：飞书必发（critical 无节流）+ 冷账本留痕。"""
        self.log(f"[autopilot] [CRITICAL] {title}: {detail}")
        self._event("alert_critical", f"{title}: {detail}")
        if self.alerter is not None:
            self.alerter.send(title, f"[autopilot][{self.strategy.symbol}] {title}\n{detail}",
                              critical=True)

    def _notify(self, key: str, text: str) -> None:
        """知会类告警（🟡 级，节流）：源切换/halt 兜底等。"""
        self.log(f"[autopilot] [notify] {text}")
        if self.alerter is not None:
            self.alerter.send(key, text)

    def _record_trade(
        self, res: OrderResult, ts: int, *,
        before: float, after: float, entry: float, entry_before: float,
        reason: str = "delta",
    ) -> None:
        """成交(fill)入账 + 动作分类 + 方向 + 实现盈亏，供前端校验开平仓逻辑。

        before/after = 调仓前/后持仓；entry = 调仓后开仓均价（展示用）；entry_before =
        调仓前均价（算实现盈亏——反手后 entry 被重置为 fill，不能用调仓后值）。
        """
        if not res or not res.ok or abs(res.filled_notional) <= 1e-9:
            return
        realized = _realized_pnl(before, res.filled_notional, res.price, entry_before)
        direction = "多" if after > 0 else "空" if after < 0 else "平"
        row = {
            "ts": ts,
            "action": _classify_action(before, after),
            "direction": direction,
            "side": "buy" if res.filled_notional > 0 else "sell",
            "filled_notional": float(res.filled_notional),
            "price": float(res.price),
            "fee": float(res.fee),
            "pos_before": float(before),
            "pos_after": float(after),
            "entry": float(entry),
            "realized": float(realized),
            "reason": reason,
        }
        self.state.record_trade(row)
        self.ledger.append("trade", row)

    def _record(
        self,
        ts: int,
        close: float,
        target_pos: float,
        target_notional: float,
        actual_notional: float,
        equity: float,
        peak: float,
        dd: float,
        entry_price: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> None:
        rec = BarRecord(
            ts=ts,
            close=close,
            target_pos=target_pos,
            target_notional=target_notional,
            actual_notional=actual_notional,
            equity=equity,
            peak_equity=peak,
            drawdown_pct=dd,
            alerts=self.monitors.drain(),
            entry_price=entry_price,
            unrealized_pnl=unrealized_pnl,
        )
        self.state.record(rec)
        self.ledger.append("bar", asdict(rec))
        self._save()
        self.log(
            f"[autopilot] bar_ts={_fmt_ts(ts)} pos={target_pos:+.4f} "
            f"target_notional={target_notional:+.4f} actual={actual_notional:+.4f} "
            f"equity={equity:.6f} dd={dd * 100:.2f}%"
            + (f" alerts={rec.alerts}" if rec.alerts else "")
        )

    def _save(self) -> None:
        self.state.breaker_tripped = bool(self._halt_reason)
        self.state.breaker_reason = self._halt_reason
        try:
            self.state.save(self.state_path)
        except OSError as exc:  # noqa: BLE001
            self.log(f"[autopilot] 状态保存失败: {exc}")

    def _event(self, name: str, detail: str = "") -> None:
        """运营事件入冷账本（源切换/熔断/心跳等统一入口，E1/ADR-0007）。"""
        self.ledger.append("event", {"name": name, "detail": detail})

    # ── 心跳 + 每日摘要（B4/B5/ADR-0007）─────────────────────────────
    def _maybe_heartbeat(self) -> None:
        """外部 watchdog ping（fire-and-forget）。距上次成功超 max_silent 即补发。

        5s 超时独立 urlopen，绝不阻塞主路径；失败静默（不更新 _hb_last_ok，
        下次必然重试——不会因一次失败把心跳永久静音）。runbook：watchdog 侧
        grace ≥ 3×bar_seconds。
        """
        if not self._hb_url:
            return
        if self._hb_last_ok and (time.monotonic() - self._hb_last_ok) < self._hb_max_silent:
            return
        try:
            with urllib.request.urlopen(self._hb_url, timeout=5):
                pass
            self._hb_last_ok = time.monotonic()
            self.state.heartbeat_last_ok_ts = int(time.time())
        except Exception:  # noqa: BLE001 - ping 失败不影响交易路径
            pass

    def _maybe_daily_digest(self, now_s: float | None = None) -> None:
        """UTC 日切 +5min → 发昨日日报（B4：摘要兼系统心跳）。数据不足静默跳过。

        壁钟驱动，由 run_forever 每轮调用（cadence 默认 60s），不再挂 bar 收盘——
        1h bar 下按 bar 判日切要等到 01:00 UTC 才触发，现在 00:05 UTC 准点发。
        critical 发送绕过 Alerter 节流：日报天然日频无刷屏，而节流（同 key 置位后
        无人 resolve）会让"摘要停发=系统死的次级信号"（ADR-0007 决策5）变成常态
        静默、只发出第一份。当日% 口径 =（昨收 − 前收）/ 前收，收盘对收盘，
        与回测 target_ret 的 close-to-close 同口径；前一日无数据（如上线首日）
        退回昨日首根权益作基准。
        """
        now_dt = datetime.fromtimestamp(
            int(now_s if now_s is not None else time.time()), tz=timezone.utc
        )
        if now_dt.hour * 3600 + now_dt.minute * 60 + now_dt.second < 300:
            return  # 日切后 5min 内不发（00:00–00:05 UTC）
        day = now_dt.strftime("%Y-%m-%d")
        prev = self._last_day
        self._last_day = day
        if not prev or prev == day:
            return

        def rows_of(d: str) -> list[dict]:
            return [
                h for h in self.state.history
                if datetime.fromtimestamp(int(h["ts"]), tz=timezone.utc).strftime("%Y-%m-%d") == d
            ]

        # 从 hot state history 聚合昨日与前日（cap 1000 ≈ 41 天，1h bar 下两天全在）
        day_rows = rows_of(prev)
        if not day_rows:
            return
        last_eq = float(day_rows[-1].get("equity") or 0.0)
        prev2 = (datetime.strptime(prev, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                 - timedelta(days=1)).strftime("%Y-%m-%d")
        base_rows = rows_of(prev2)
        base = (float(base_rows[-1].get("equity") or 0.0) if base_rows
                else float(day_rows[0].get("equity") or 0.0))
        if last_eq <= 0 or base <= 0:
            return
        day_ret = (last_eq - base) / base
        trades_cnt = sum(
            1 for t in self.state.trades
            if datetime.fromtimestamp(int(t["ts"]), tz=timezone.utc).strftime("%Y-%m-%d") == prev
        )

        mood = "小赚一笔 ✌️" if day_ret > 0 else ("小亏，一杯奶茶的钱 🥤" if day_ret < 0 else "原地踏步，深藏功与名")
        trades_txt = f"昨日成交 {trades_cnt} 笔，bot 搬砖不停歇 💪" if trades_cnt else "昨日 0 单成交，安静持有躺平中 🐢"
        text = (
            f"[autopilot][{self.strategy.symbol}] 📊 {prev} 日报\n"
            f"💰 权益 {last_eq:.2f} USDT，当日 {day_ret * 100:+.2f}%，{mood}\n"
            f"📦 当前仓位：{self._position_digest_text()}\n"
            f"🔁 {trades_txt}"
        )
        self.log(f"[autopilot] [digest] {text}")
        if self.alerter is not None:
            self.alerter.send("daily_digest", text, critical=True)
        self._save()

    def _position_digest_text(self) -> str:
        """日报持仓描述：后端实时明细为准，拉取失败退回末根 bar 快照。"""
        actual: float | None = None
        entry = unreal = 0.0
        try:
            actual, entry, unreal = self.backend.fetch_position_detail(self.strategy.symbol)
            actual = float(actual)
        except Exception:  # noqa: BLE001 - 持仓查询失败不拦日报
            if self.state.history:
                last = self.state.history[-1]
                actual = float(last.get("actual_notional") or 0.0)
                entry = float(last.get("entry_price") or 0.0)
                unreal = float(last.get("unrealized_pnl") or 0.0)
        if actual is None:
            return "查询失败，看板自提 🔍"
        if abs(actual) < 1e-9:
            return "空仓休息 🍹"
        detail = f"{'多头' if actual > 0 else '空头'} {abs(actual):.2f} USDT"
        if float(entry) > 0:
            detail += f"，开仓价 {float(entry):.2f}"
        if float(unreal):
            detail += f"，浮盈亏 {float(unreal):+.2f} USDT"
        return detail

    def _halt(self, reason: str) -> None:
        if not self._halt_reason:
            self._halt_reason = reason
            self.log(f"[autopilot] HALT: {reason}")
            self._event("halt", reason)
            self._save()

    def _stop_signalled(self) -> bool:
        return any(p.exists() for p in self.stop_paths)
