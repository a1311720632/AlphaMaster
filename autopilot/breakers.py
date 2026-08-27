"""运营熔断与告警监控（ADR-0005）。

两类**硬熔断**（触发即 halt，破坏奇偶性但为运营安全有意为之）：
  - DrawdownBreaker:    峰值权益回撤 ≤ 阈值 → trip（全平 + 停）
  - ConnectivityBreaker: 连续 N 次 行情/持仓 拉取失败 → trip（停，不再基于陈旧数据调仓）

Monitors（告警监控）：只记录/通知，**不动仓位，不破坏奇偶性**——账实漂移、未成交、
孤儿单。与运营熔断的关键区别：告警监控不改仓位，运营熔断会。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BreakerStatus:
    drawdown_tripped: bool = False
    connectivity_tripped: bool = False
    peak_equity: float = 0.0
    cur_equity: float = 0.0
    drawdown_pct: float = 0.0
    reason: str = ""

    @property
    def tripped(self) -> bool:
        return self.drawdown_tripped or self.connectivity_tripped


class DrawdownBreaker:
    """峰值权益回撤熔断。trip 后恒 tripped（不可自愈，需人工重置）。

    initial_peak（B1/ADR-0007）：进程重启时回填上次运行留下的峰值权益，使回撤基线
    跨重启连续——否则每次重启白送一次新的 −10% 额度（peak 从 -inf 重新起算）。
    """

    def __init__(self, max_drawdown_pct: float, initial_peak: float | None = None) -> None:
        # max_drawdown_pct 为负数，如 -0.10
        self._threshold = float(max_drawdown_pct)
        self._peak = float(initial_peak) if (initial_peak is not None and initial_peak > 0) else float("-inf")
        self._tripped = False
        self._reason = ""

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def reason(self) -> str:
        return self._reason

    def update(self, equity: float) -> tuple[float, float, bool]:
        """更新峰值权益。返回 (peak, drawdown_pct, tripped)。"""
        if equity > self._peak:
            self._peak = equity
        dd = 0.0 if self._peak <= 0 else (equity - self._peak) / self._peak
        if not self._tripped and dd <= self._threshold:
            self._tripped = True
            self._reason = (
                f"回撤熔断: 回撤 {dd * 100:.2f}% ≤ 阈值 {self._threshold * 100:.2f}%"
            )
        return self._peak, dd, self._tripped


class ConnectivityBreaker:
    """连续 N 次拉取失败熔断。成功一次即重置计数。trip 后恒 tripped。"""

    def __init__(self, max_failures: int) -> None:
        self._limit = max(1, int(max_failures))
        self._fails = 0
        self._tripped = False
        self._reason = ""

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def reason(self) -> str:
        return self._reason

    def reset(self) -> None:
        """成功拉取一次后重置失败计数（不解除已 trip 的熔断）。"""
        self._fails = 0

    def fail(self) -> bool:
        """记录一次失败，返回是否（刚刚）触发熔断。"""
        self._fails += 1
        if not self._tripped and self._fails >= self._limit:
            self._tripped = True
            self._reason = f"断网熔断: 连续 {self._fails} 次拉取失败"
        return self._tripped


@dataclass
class Monitors:
    """告警监控：只记录/通知，不动仓位（ADR-0005）。"""

    alerts: list[str] = field(default_factory=list)

    def observe(
        self, actual_notional: float, target_notional: float, fill_ok: bool
    ) -> None:
        if not fill_ok:
            self.alerts.append(
                f"未成交漂移: 目标名义 {target_notional:.4f} 未能落地"
            )
        drift = target_notional - actual_notional
        # 只在目标非零但有显著漂移时告警（目标≈0 且实际≈0 不算异常）
        if abs(target_notional) > 1e-9 and abs(drift) > 1e-6:
            self.alerts.append(
                f"账实漂移: 目标 {target_notional:.4f} / 实际 {actual_notional:.4f}"
            )

    def orphan(self, symbol: str) -> None:
        self.alerts.append(f"孤儿单: 交易所有 {symbol} 持仓但本地无对应策略")

    def drain(self) -> list[str]:
        out = list(self.alerts)
        self.alerts = []
        return out
