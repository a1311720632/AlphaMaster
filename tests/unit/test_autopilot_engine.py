"""AutopilotEngine 离线端到端（paper 全链路，无网络）。

用 FakeSource 喂合成 bar，跑通“拉取→信号→规模→对账→记录→存档”主循环，
作为 paper 模式的确定性冒烟测试；并验证两类运营熔断（ADR-0005）真正停机。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from web.data_sources.base import Bar, DataSource, DataSourceError

from autopilot.backends import ExecutionBackend, OrderResult
from autopilot.engine import AutopilotEngine
from autopilot.strategy_loader import StrategySpec


# ── 测试夹具：合成 bar 池 ────────────────────────────────────────────────────
def _pool(n: int = 310, seed: int = 3) -> list[Bar]:
    bars: list[Bar] = []
    state = seed + 1
    price = 100.0
    ts0 = 1_700_000_000
    for i in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        rnd = (state / 0x7FFFFFFF) - 0.5
        op = price
        price = max(1.0, price * (1.0 + rnd * 0.02))
        bars.append(
            Bar(
                ts=ts0 + i * 3600,
                open=op,
                high=max(op, price) * 1.005,
                low=min(op, price) * 0.995,
                close=price,
                volume=1000.0,
            )
        )
    return bars


class FakeSource(DataSource):
    """每次 fetch 返回以 idx 结尾的窗口，随后 idx 前进 1（模拟每轮一根新 bar）。"""

    kind = "fake"

    def __init__(self, pool: list[Bar]) -> None:
        self.pool = pool
        self.idx = 300  # 起始窗口终点（已 warmup ≥ NORM_WINDOW=200）

    def available(self):
        return (True, "fake")

    def supported_timeframes(self):
        return ["1h"]

    def preset_symbols(self):
        return ["TEST"]

    def fetch_bars(self, symbol, timeframe, n, drop_forming=True):
        end = min(self.idx, len(self.pool))
        window = list(self.pool[max(0, end - n) : end])
        self.idx = min(self.idx + 1, len(self.pool))
        return window


class FailingSource(DataSource):
    kind = "fail"

    def available(self):
        return (True, "fail")

    def supported_timeframes(self):
        return ["1h"]

    def preset_symbols(self):
        return ["TEST"]

    def fetch_bars(self, symbol, timeframe, n, drop_forming=True):
        raise DataSourceError("模拟断网")


class PlungeBackend(ExecutionBackend):
    """权益随调用次序骤降，用于触发回撤熔断。"""

    mode = "test"

    def __init__(self):
        self._calls = 0
        self.flattened = False

    def fetch_equity(self):
        self._calls += 1
        # 第 1 次 100（建峰），第 2 次起 84（dd=-16% → trip）
        return 100.0 if self._calls == 1 else 84.0

    def fetch_position_notional(self, symbol):
        return 0.0

    def place_delta_order(self, symbol, delta_notional):
        return OrderResult(ok=True, filled_notional=delta_notional)

    def flatten_all(self, symbol):
        self.flattened = True
        return OrderResult(ok=True, message="flattened")


def _spec() -> StrategySpec:
    return StrategySpec(
        formula=[0], symbol="TEST", timeframe="1h", score=1.0,
        vocab_version="test", path="",
    )


def _make_engine(tmp_path: Path, source, backend, **kw) -> AutopilotEngine:
    return AutopilotEngine(
        strategy=_spec(),
        datasource=source,
        backend=backend,
        lookback_bars=300,
        breaker_max_drawdown_pct=-0.10,
        breaker_max_bars_stale=kw.get("max_stale", 3),
        min_notional_delta=0.0,
        state_path=tmp_path / "autopilot_state.json",
        stop_signal_paths=[tmp_path / "AUTOPILOT_STOP_SIGNAL", tmp_path / "STOP_SIGNAL"],
        cadence_s=0,
        max_bars=kw.get("max_bars"),
        log=lambda m: None,
    )


# ── 端到端：paper 主循环 ────────────────────────────────────────────────────
def test_engine_paper_loop_records_ledger(tmp_path):
    from autopilot.backends import SimBackend

    be = SimBackend(start_equity=1.0, cost_rate=0.0003)
    eng = _make_engine(tmp_path, FakeSource(_pool()), be, max_bars=3)
    reason = eng.run_forever()

    assert reason.startswith("max_bars")
    assert len(eng.state.history) == 3
    # 每条记录字段合法
    for rec in eng.state.history:
        assert -1.0 <= rec["target_pos"] <= 1.0
        assert rec["equity"] > 0
        assert rec["ts"] > 0
    # 状态文件已落盘
    assert (tmp_path / "autopilot_state.json").exists()


def test_engine_connectivity_halt(tmp_path):
    from autopilot.backends import SimBackend

    be = SimBackend(start_equity=1.0, cost_rate=0.0003)
    eng = _make_engine(tmp_path, FailingSource(), be, max_stale=2)
    reason = eng.run_forever()
    assert "断网" in reason
    assert eng.state.breaker_tripped


def test_engine_drawdown_halt_flattens(tmp_path):
    src = FakeSource(_pool())
    be = PlungeBackend()
    eng = _make_engine(tmp_path, src, be, max_bars=10)
    reason = eng.run_forever()
    assert "回撤" in reason
    assert be.flattened  # 熔断触发时调用了 flatten_all
    assert eng.state.breaker_tripped
