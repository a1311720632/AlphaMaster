"""自动驾驶运行态持久化（与 MT5 portfolio_state.json 分离）。

记录：峰值权益、熔断状态、last_ts、逐 bar 账本（目标/实际仓位·名义·权益·回撤·告警）。
用于 web 第四步的状态展示与 crash 后恢复（恢复时强制对账，ADR-0006）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_MAX_HISTORY = 1000  # 内存/磁盘里保留的最近 bar 数


@dataclass
class BarRecord:
    ts: int
    close: float
    target_pos: float           # tanh 仓位 [-1,1]（含 MIN_TRADE_EXPOSURE 地板）
    target_notional: float      # 带方向名义 = target_pos × equity（ADR-0002）
    actual_notional: float      # 交易所/模拟实际名义（带方向）
    equity: float
    peak_equity: float
    drawdown_pct: float
    alerts: list[str] = field(default_factory=list)
    entry_price: float = 0.0        # 当前持仓开仓均价（展示用；paper 由 SimBackend 跟踪）
    unrealized_pnl: float = 0.0     # 截至本 bar 收盘的未实现盈亏（带方向名义口径）


@dataclass
class AutopilotState:
    symbol: str
    timeframe: str
    mode: str
    peak_equity: float = 0.0
    last_ts: int = 0
    breaker_tripped: bool = False
    breaker_reason: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    start_equity: float = 0.0      # 总收益基线：paper=起点权益，live=首次余额快照
    trades: list[dict[str, Any]] = field(default_factory=list)  # 成交(fill)流水

    def record(self, rec: BarRecord) -> None:
        self.history.append(asdict(rec))
        self.last_ts = rec.ts
        self.peak_equity = rec.peak_equity
        if len(self.history) > _MAX_HISTORY:
            # 保留最近 _MAX_HISTORY 条，避免无限增长
            self.history = self.history[-_MAX_HISTORY:]

    def record_trade(self, t: dict[str, Any]) -> None:
        """追加一条成交(fill)记录，与 history 同样 cap 在 _MAX_HISTORY。"""
        self.trades.append(t)
        if len(self.trades) > _MAX_HISTORY:
            self.trades = self.trades[-_MAX_HISTORY:]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AutopilotState":
        return cls(
            symbol=str(d.get("symbol", "")),
            timeframe=str(d.get("timeframe", "")),
            mode=str(d.get("mode", "")),
            peak_equity=float(d.get("peak_equity", 0.0)),
            last_ts=int(d.get("last_ts", 0)),
            breaker_tripped=bool(d.get("breaker_tripped", False)),
            breaker_reason=str(d.get("breaker_reason", "")),
            history=list(d.get("history", [])),
            start_equity=float(d.get("start_equity", 0.0)),
            trades=list(d.get("trades", [])),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "AutopilotState | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None
