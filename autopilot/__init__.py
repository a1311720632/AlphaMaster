"""自动驾驶（第四步）——按策略自动调仓的实盘执行层。

设计见 docs/adr/0001…0006 与 CONTEXT.md。核心约束：信号/调仓逻辑与回测字面一致
（连续 tanh 仓位，无单笔 SL/TP），三模式（paper/testnet/live）共享同一信号核心，
仅执行后端不同。
"""
from autopilot.strategy_loader import (
    StrategySpec,
    StrategyLoadError,
    load_strategy,
    normalize_timeframe,
)

__all__ = ["StrategySpec", "StrategyLoadError", "load_strategy", "normalize_timeframe"]
