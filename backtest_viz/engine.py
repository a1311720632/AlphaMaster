"""
backtest_viz/engine.py — 逐 bar 可视化回测引擎

与训练用 backtest.py 共享相同的信号逻辑（tanh 连续仓位），
但额外记录每笔交易的开平仓细节，供图表标注使用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from model_core.vm import StackVM
from strategy_manager.signal import target_to_direction

_H1_PERIODS_PER_YEAR = 6240


@dataclass
class Trade:
    """一笔完整交易记录（开仓 → 平仓/反手）"""
    symbol:      str
    direction:   int          # +1 多 / -1 空
    entry_bar:   int          # 开仓 bar 索引（相对于整个序列）
    entry_time:  int          # Unix 秒
    entry_price: float        # 开仓价（用 open 价格）
    exit_bar:    Optional[int]   = None
    exit_time:   Optional[int]   = None
    exit_price:  Optional[float] = None
    pnl:         float           = 0.0   # 本笔税后 PnL（log return - cost）
    cum_pnl:     float           = 0.0   # 截至本笔结束的累计 PnL


@dataclass
class SymbolResult:
    """单个品种的完整回测结果"""
    symbol:       str
    times:        np.ndarray     # Unix 秒，shape [T]
    open:         np.ndarray     # [T]
    high:         np.ndarray     # [T]
    low:          np.ndarray     # [T]
    close:        np.ndarray     # [T]
    volume:       np.ndarray     # [T]
    factor:       np.ndarray     # StackVM 输出，[T]
    signal:       np.ndarray     # tanh(factor)，[T]
    position:     np.ndarray     # 连续仓位 ∈ [-1,+1]，[T]
    pnl:          np.ndarray     # 逐 bar PnL，[T]
    cum_pnl:      np.ndarray     # 累计 PnL，[T]
    buy_hold:     np.ndarray     # 买入持有累计对数收益（恒 +1 仓位基准），[T]
    drawdown:     np.ndarray     # 水下回撤序列 equity/running_max - 1（≤0），[T]
    trades:       list[Trade]    = field(default_factory=list)
    sortino:      float          = 0.0
    total_return: float          = 0.0
    n_trades:     int            = 0
    win_rate:     float          = 0.0
    max_drawdown: float          = 0.0
    avg_hold_bars:float          = 0.0
    profit_loss_ratio: float | None = None  # 盈亏比 = 平均盈利 / 平均亏损
    # ── A 波扩展指标（风险 / 基准 / 暴露 / 成本 / 尾部）──────────────
    ann_return:      float = 0.0   # 年化对数收益 = mean(pnl) * periods_per_year
    calmar:          float = 0.0   # 年化收益 / 最大回撤
    buy_hold_total:  float = 0.0   # 买入持有累计对数收益（基准）
    long_pct:        float = 0.0   # 多头时间占比 ∈ [0,1]
    short_pct:       float = 0.0   # 空头时间占比 ∈ [0,1]
    flat_pct:        float = 0.0   # 空仓时间占比 ∈ [0,1]
    avg_position:    float = 0.0   # 平均 |仓位|
    cost_ratio:      float = 0.0   # 总成本 / |毛收益|
    avg_turnover:    float = 0.0   # 平均每 bar 换手
    var95:           float = 0.0   # 95% VaR（正数 = 损失幅度）
    cvar95:          float = 0.0   # 95% CVaR / 预期短缺
    worst_bar:       float = 0.0   # 最差单根 bar 收益


class BacktestEngine:
    """逐 bar 可视化回测引擎。

    用法：
        engine = BacktestEngine(formula=[6,15,8,...])
        results = engine.run(raw_dict, times, symbols)
    """

    def __init__(
        self,
        formula:         list[int],
        cost_rate:       float = 0.0001,
        periods_per_year:int   = _H1_PERIODS_PER_YEAR,
    ):
        self.formula          = formula
        self.cost_rate        = cost_rate
        self.periods_per_year = periods_per_year
        self.vm               = StackVM()

    # ─────────────────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        raw_dict: dict,          # {open/high/low/close/volume/time: Tensor[N,T]}
        feat_tensor: torch.Tensor,  # [N, F, T]
        symbols: list[str],
    ) -> list[SymbolResult]:
        """执行所有品种的回测，返回每个品种的 SymbolResult。"""

        factors_all = self.vm.execute(self.formula, feat_tensor)  # [N, T]
        if factors_all is None:
            raise RuntimeError(
                f"StackVM 无法执行公式 {self.formula}。"
                "请检查公式 token 是否合法。"
            )

        results = []
        N = len(symbols)
        for n in range(N):
            sym = symbols[n]
            sym_result = self._backtest_symbol(
                symbol     = sym,
                raw_dict   = {k: v[n] for k, v in raw_dict.items()},   # [T] 各字段
                factor_1d  = factors_all[n],                            # [T]
            )
            results.append(sym_result)

        return results

    # ─────────────────────────────────────────────────────────────────────
    # 单品种回测
    # ─────────────────────────────────────────────────────────────────────

    def _backtest_symbol(
        self,
        symbol:   str,
        raw_dict: dict,         # 每个值是 [T] 的 Tensor
        factor_1d: torch.Tensor,  # [T]
    ) -> SymbolResult:

        T = factor_1d.shape[0]

        # numpy 转换（便于后续图表处理）
        factor_np   = factor_1d.detach().float().numpy()
        signal_np   = np.tanh(factor_np)
        # 连续仓位：与训练 / 实盘严格 parity——走 strategy_manager.signal 的同一个函数，
        # 内部做 tanh + MIN_TRADE_EXPOSURE 地板（|pos|<0.05 → 空仓）。
        # 修复原先「只 tanh、不地板」的偏差：弱信号此前会被当成微小仓位计入回测，
        # 与训练 backtest.py / 实盘 runner 的信号逻辑不一致。
        from strategy_manager.signal import compute_target_positions_stateless as _to_positions
        position_np = _to_positions(factor_1d).detach().float().numpy()

        open_np   = raw_dict["open"].float().numpy()
        high_np   = raw_dict["high"].float().numpy()
        low_np    = raw_dict["low"].float().numpy()
        close_np  = raw_dict["close"].float().numpy()
        volume_np = raw_dict["volume"].float().numpy()

        if "time" in raw_dict:
            times_np = raw_dict["time"].long().numpy()
        else:
            times_np = np.arange(T, dtype=np.int64)

        # ── 计算 PnL 序列（与 backtest.py 完全一致）─────────────────
        # target_ret[t] = log(open[t+2] / open[t+1])
        target_ret = np.zeros(T, dtype=np.float32)
        if T >= 3:
            target_ret[: T - 2] = np.log(
                (open_np[2:] + 1e-12) / (open_np[1:-1] + 1e-12)
            )

        prev_pos = np.zeros(T, dtype=np.float32)
        prev_pos[1:] = position_np[:-1]
        turnover = np.abs(position_np - prev_pos)

        pnl_np    = position_np * target_ret - turnover * self.cost_rate
        cum_pnl   = np.cumsum(pnl_np)

        # ── 提取交易记录 ──────────────────────────────────────────────
        trades = self._extract_trades(
            symbol, position_np, open_np, times_np, pnl_np
        )

        # ── 统计指标 ─────────────────────────────────────────────────
        sortino       = self._calc_sortino(pnl_np)
        total_return  = float(cum_pnl[-1]) if len(cum_pnl) else 0.0
        n_trades      = len(trades)
        win_rate      = (
            sum(1 for t in trades if t.pnl > 0) / n_trades
            if n_trades else 0.0
        )
        avg_hold      = (
            sum(
                (t.exit_bar - t.entry_bar)
                for t in trades if t.exit_bar is not None
            ) / n_trades
            if n_trades else 0.0
        )
        pl_ratio      = self._calc_profit_loss_ratio(trades)

        # ── A 波扩展指标 ─────────────────────────────────────────────
        # 最大回撤 + 水下曲线（基于复利净值 equity = exp(cum_pnl)，修复原 max_drawdown=0.0 stub）
        if len(cum_pnl):
            equity      = np.exp(cum_pnl)
            running_max = np.maximum.accumulate(equity)
            safe_peak   = np.where(running_max <= 0, 1e-12, running_max)
            dd          = (running_max - equity) / safe_peak     # 回撤幅度 ∈ [0,1]
            max_drawdown = float(np.clip(np.max(dd), 0.0, 1.0))
            drawdown_np = (equity / safe_peak) - 1.0             # 水下曲线 ≤ 0
        else:
            max_drawdown = 0.0
            drawdown_np = np.zeros_like(cum_pnl)

        ann_ret = float(pnl_np.mean() * self.periods_per_year) if len(pnl_np) else 0.0
        calmar  = float(ann_ret / max_drawdown) if max_drawdown > 1e-9 else 0.0

        # 买入持有基准：恒 +1 仓位、无换手成本
        bh_cum         = np.cumsum(target_ret) if len(target_ret) else np.zeros_like(cum_pnl)
        buy_hold_total = float(bh_cum[-1]) if len(bh_cum) else 0.0

        # 多空 / 空仓时间占比 + 平均仓位
        _eps = 1e-6
        _n   = len(position_np)
        long_pct  = float(np.mean(position_np >  _eps)) if _n else 0.0
        short_pct = float(np.mean(position_np < -_eps)) if _n else 0.0
        flat_pct  = float(np.mean(np.abs(position_np) <= _eps)) if _n else 0.0
        avg_pos   = float(np.mean(np.abs(position_np))) if _n else 0.0

        # 成本占比 + 平均换手
        gross_ret  = float(np.sum(position_np * target_ret)) if _n else 0.0
        total_cost = float(np.sum(turnover * self.cost_rate)) if len(turnover) else 0.0
        cost_ratio = float(total_cost / abs(gross_ret)) if abs(gross_ret) > 1e-9 else 0.0
        avg_turn   = float(np.mean(turnover)) if len(turnover) else 0.0

        # 尾部风险（逐 bar pnl 分布）
        if len(pnl_np):
            q5        = float(np.percentile(pnl_np, 5))
            var95     = -q5
            tail      = pnl_np[pnl_np <= q5]
            cvar95    = float(-tail.mean()) if len(tail) else var95
            worst_bar = float(np.min(pnl_np))
        else:
            var95 = cvar95 = worst_bar = 0.0

        return SymbolResult(
            symbol       = symbol,
            times        = times_np,
            open         = open_np,
            high         = high_np,
            low          = low_np,
            close        = close_np,
            volume       = volume_np,
            factor       = factor_np,
            signal       = signal_np,
            position     = position_np,
            pnl          = pnl_np,
            cum_pnl      = cum_pnl,
            buy_hold     = bh_cum,
            drawdown     = drawdown_np,
            trades       = trades,
            sortino      = sortino,
            total_return = total_return,
            n_trades     = n_trades,
            win_rate     = win_rate,
            max_drawdown = max_drawdown,
            avg_hold_bars= avg_hold,
            profit_loss_ratio = pl_ratio,
            ann_return      = ann_ret,
            calmar          = calmar,
            buy_hold_total  = buy_hold_total,
            long_pct        = long_pct,
            short_pct       = short_pct,
            flat_pct        = flat_pct,
            avg_position    = avg_pos,
            cost_ratio      = cost_ratio,
            avg_turnover    = avg_turn,
            var95           = var95,
            cvar95          = cvar95,
            worst_bar       = worst_bar,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 交易记录提取
    # ─────────────────────────────────────────────────────────────────────

    def _extract_trades(
        self,
        symbol:      str,
        position:    np.ndarray,   # [T] 连续仓位
        open_prices: np.ndarray,   # [T]
        times:       np.ndarray,   # [T]
        pnl:         np.ndarray,   # [T]
    ) -> list[Trade]:
        """从仓位序列中提取完整交易列表（含开平仓 bar、价格、PnL）。

        执行价对齐逻辑（与 target_ret 计算保持一致）：
          target_ret[t] = log(open[t+2] / open[t+1])
          position[t] 产生的收益对应 open[t+1] → open[t+2]
          因此：信号在 entry_bar 产生 → 实际成交价 = open[entry_bar + 1]
                信号在 exit_bar 翻转 → 实际成交价 = open[exit_bar + 1]

        PnL 计算：把持仓期间的逐 bar pnl 累加作为本笔盈亏。
        """
        T = len(position)
        trades:       list[Trade] = []
        cum_pnl_total = 0.0

        current_dir: int = 0
        entry_bar:   int = 0

        def _exec_price(bar: int) -> float:
            """信号在 bar 产生，执行价为下一根 open（若越界则取最后一根）。"""
            idx = min(bar + 1, T - 1)
            return float(open_prices[idx])

        def _exec_time(bar: int) -> int:
            idx = min(bar + 1, T - 1)
            return int(times[idx])

        for t in range(T):
            new_dir = target_to_direction(float(position[t]))

            if new_dir != current_dir:
                # 平掉旧仓
                if current_dir != 0:
                    trade_pnl = float(pnl[entry_bar:t].sum())
                    cum_pnl_total += trade_pnl
                    trade = Trade(
                        symbol      = symbol,
                        direction   = current_dir,
                        entry_bar   = entry_bar,
                        entry_time  = _exec_time(entry_bar),
                        entry_price = _exec_price(entry_bar),
                        exit_bar    = t,
                        exit_time   = _exec_time(t),
                        exit_price  = _exec_price(t),
                        pnl         = trade_pnl,
                        cum_pnl     = cum_pnl_total,
                    )
                    trades.append(trade)

                current_dir = new_dir
                entry_bar   = t

        # 序列末尾强平
        if current_dir != 0:
            trade_pnl = float(pnl[entry_bar:].sum())
            cum_pnl_total += trade_pnl
            trades.append(Trade(
                symbol      = symbol,
                direction   = current_dir,
                entry_bar   = entry_bar,
                entry_time  = _exec_time(entry_bar),
                entry_price = _exec_price(entry_bar),
                exit_bar    = T - 1,
                exit_time   = _exec_time(T - 1),
                exit_price  = _exec_price(T - 1),
                pnl         = trade_pnl,
                cum_pnl     = cum_pnl_total,
            ))

        return trades

    # ─────────────────────────────────────────────────────────────────────
    # 统计辅助
    # ─────────────────────────────────────────────────────────────────────

    def _calc_sortino(self, pnl: np.ndarray) -> float:
        mean_pnl = float(np.mean(pnl))
        downside = pnl[pnl < 0]
        if len(downside) == 0:
            return 0.0
        ds_std = float(np.std(downside, ddof=0))
        floor  = max(abs(mean_pnl), 1e-8)
        ds_std = max(ds_std, floor)
        sortino = mean_pnl / ds_std * math.sqrt(self.periods_per_year)
        return float(np.clip(sortino, -20.0, 20.0))

    @staticmethod
    def _calc_profit_loss_ratio(trades: list[Trade]) -> float | None:
        """盈亏比 = 平均盈利 / 平均亏损绝对值。无盈利或无亏损时返回 None。"""
        wins = [t.pnl for t in trades if t.pnl is not None and t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl is not None and t.pnl < 0]
        if not wins or not losses:
            return None
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        if avg_loss <= 0:
            return None
        return float(avg_win / avg_loss)
