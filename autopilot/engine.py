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
from pathlib import Path
from typing import Callable

from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless
from web.data_sources.base import Bar, DataSource, bars_to_raw_dict

from autopilot.backends import ExecutionBackend, OrderResult
from autopilot.breakers import ConnectivityBreaker, DrawdownBreaker, Monitors
from autopilot.state import AutopilotState, BarRecord
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
        breaker_max_bars_stale: int,
        min_notional_delta: float,
        state_path: str | Path,
        stop_signal_paths: list[str | Path],
        cadence_s: int | None = None,
        bar_seconds: int | None = None,
        max_bars: int | None = None,
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

        self.drawdown = DrawdownBreaker(breaker_max_drawdown_pct)
        self.connectivity = ConnectivityBreaker(breaker_max_bars_stale)
        self.monitors = Monitors()

        # 恢复或新建状态
        loaded = AutopilotState.load(self.state_path)
        if loaded and loaded.symbol == strategy.symbol and loaded.timeframe == strategy.timeframe:
            self.state = loaded
            self.log(
                f"[autopilot] 恢复状态: last_ts={loaded.last_ts} peak={loaded.peak_equity:.6f}"
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

    # ── 主循环 ─────────────────────────────────────────────────────────
    def run_forever(self) -> str:
        """主循环。返回 halt 原因（熔断 / STOP_SIGNAL / 异常）。"""
        self.log(
            f"[autopilot] 启动 mode={self.backend.mode} symbol={self.strategy.symbol} "
            f"tf={self.strategy.timeframe} formula_len={len(self.strategy.formula)} "
            f"lookback={self.lookback} cadence={self.cadence_s}s"
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
            if self._halt_reason:
                break
            if self._max_bars is not None and len(self.state.history) >= self._max_bars:
                self._halt_reason = f"max_bars 达成（{self._max_bars} 根新 bar）"
                break
            time.sleep(self.cadence_s)
        self.log(f"[autopilot] 结束 reason={self._halt_reason or 'stopped'}")
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
                self._halt(self.connectivity.reason)
            return

        bars = _ensure_closed_bars(bars)
        if len(bars) < _MIN_BARS:
            self.log(f"[autopilot] bar 不足 {len(bars)}/{_MIN_BARS}，等待 warmup")
            return
        self.connectivity.reset()

        cur_ts = int(bars[-1].ts)
        cur_close = float(bars[-1].close)

        # 2. 无新收盘 bar
        if cur_ts == self._last_ts:
            return
        is_first = self._last_ts == 0
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
            self._halt(self.drawdown.reason)
            self._record(
                cur_ts, cur_close, target_pos, target_pos * equity, actual, equity, peak, dd,
                entry_price=entry, unrealized_pnl=unreal,
            )
            return

        # 6. 对账自愈（ADR-0006）：delta 以交易所实际持仓为准
        #    首根新 bar 即调仓（fresh start 进场 / restart 恢复后立即对账回 target）。
        #    warmup 由 _MIN_BARS 兜底、forming bar 由 _ensure_closed_bars 兜底，无需延迟首调仓。
        target_notional = target_pos * equity
        actual, entry_before, unreal = self.backend.fetch_position_detail(self.strategy.symbol)
        delta = target_notional - actual
        fill_ok = True
        entry = entry_before  # 不调仓时 entry 不变
        if abs(delta) > 0 and abs(delta) >= self.min_delta:
            res: OrderResult = self.backend.place_delta_order(self.strategy.symbol, delta)
            fill_ok = res.ok
            actual, entry, unreal = self.backend.fetch_position_detail(self.strategy.symbol)
            self._record_trade(
                res, cur_ts,
                before=actual - res.filled_notional, after=actual,
                entry=entry, entry_before=entry_before,
            )
        self.monitors.observe(actual, target_notional, fill_ok)

        self._record(
            cur_ts, cur_close, target_pos, target_notional, actual, equity, peak, dd,
            entry_price=entry, unrealized_pnl=unreal,
        )

    # ── 辅助 ───────────────────────────────────────────────────────────
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
        self.state.record_trade({
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
        })

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
        self._save()
        self.log(
            f"[autopilot] ts={ts} pos={target_pos:+.4f} "
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

    def _halt(self, reason: str) -> None:
        if not self._halt_reason:
            self._halt_reason = reason
            self.log(f"[autopilot] HALT: {reason}")
            self._save()

    def _stop_signalled(self) -> bool:
        return any(p.exists() for p in self.stop_paths)
