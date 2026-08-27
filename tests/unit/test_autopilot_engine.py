"""AutopilotEngine 离线端到端（paper 全链路，无网络）。

用 FakeSource 喂合成 bar，跑通“拉取→信号→规模→对账→记录→存档”主循环，
作为 paper 模式的确定性冒烟测试；并验证两类运营熔断（ADR-0005）真正停机。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from web.data_sources.base import Bar, DataSource, DataSourceError

from autopilot.backends import ExecutionBackend, OrderResult, SimBackend
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
        sleep_fn=kw.get("sleep_fn"),
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


def test_simbackend_entry_weighted_average():
    """SimBackend 移动加权开仓价：开仓→同向加仓加权→减仓不穿零不变→穿零重置。"""
    from autopilot.backends import SimBackend

    b = SimBackend(start_equity=10000.0, cost_rate=0.0)
    b.mark_to_market(0, 100.0)              # 初始化 _last_close=100
    b.place_delta_order("BTC", 1000.0)      # 开多 entry=100
    assert b._entry_price == pytest.approx(100.0)
    b.mark_to_market(100.0, 120.0)
    b.place_delta_order("BTC", 800.0)       # 同向加仓 → 加权
    assert b._entry_price == pytest.approx((1000 * 100 + 800 * 120) / 1800)
    b.mark_to_market(120.0, 110.0)
    b.place_delta_order("BTC", -1000.0)     # 减仓不穿零 → entry 不变
    assert b._entry_price == pytest.approx((1000 * 100 + 800 * 120) / 1800)
    b.mark_to_market(110.0, 130.0)
    b.place_delta_order("BTC", -1500.0)     # 穿零 → entry 重置为 fill
    assert b._entry_price == pytest.approx(130.0)
    assert b._position_notional == pytest.approx(-700.0)


def test_engine_records_entry_unrealized_and_trades(tmp_path):
    """paper 端到端：history 含 entry_price/unrealized_pnl；start_equity=起点；trades 落地。"""
    from autopilot.backends import SimBackend

    be = SimBackend(start_equity=1.0, cost_rate=0.0003)
    eng = _make_engine(tmp_path, FakeSource(_pool()), be, max_bars=3)
    eng.run_forever()

    assert eng.state.start_equity == pytest.approx(1.0)  # paper 起点权益
    for rec in eng.state.history:
        assert rec["entry_price"] >= 0
        assert rec["unrealized_pnl"] == pytest.approx(rec["unrealized_pnl"])  # finite
    # min_delta=0 → 每根调仓 bar 都有 fill（除非 target 恒为 0）；trades 是 list
    assert isinstance(eng.state.trades, list)
    for t in eng.state.trades:
        assert t["side"] in ("buy", "sell")
        assert t["action"] in ("开仓", "加仓", "减仓", "平仓", "反手")
        for k in ("filled_notional", "price", "fee", "pos_before", "pos_after", "entry"):
            assert k in t, f"trade 缺字段 {k}: {t}"


def test_engine_first_tick_initializes_last_close(tmp_path):
    """首根 tick：mark_to_market(cur,cur) 初始化 _last_close（防 entry 卡 0 regression）。

    去 first_tick 闸后首根即调仓进场（不再延迟到下根）。
    """
    from autopilot.backends import SimBackend

    be = SimBackend(start_equity=1.0, cost_rate=0.0)
    eng = _make_engine(tmp_path, FakeSource(_pool()), be, max_bars=1)
    eng.run_forever()
    # 首根 mark(cur,cur) 被调 → _last_close 就绪（防 entry 卡 0）
    assert be._last_close > 0
    # 去闸后首根即调仓：若 target≠0 则有成交进场
    if eng.state.history:
        target = eng.state.history[-1]["target_pos"]
        if abs(target) > 1e-9:
            assert len(eng.state.trades) > 0


def test_engine_resume_initializes_last_close(tmp_path):
    """恢复 state 后：backend.restore 回填 _last_close/_prev_close，权益 carry 连续。

    regression: 旧代码恢复时 _prev_close=NaN 且 backend 未恢复，首根 mark 分支异常 →
    _last_close 卡 0、entry 卡 0、未实现盈亏失真。现 __init__ 从末根 restore 回填。
    """
    from autopilot.backends import SimBackend

    # 第一次跑产生 state（有 1 根 history）
    be1 = SimBackend(start_equity=1.0, cost_rate=0.0)
    eng1 = _make_engine(tmp_path, FakeSource(_pool()), be1, max_bars=1)
    eng1.run_forever()
    assert (tmp_path / "autopilot_state.json").exists()
    # 恢复：新 backend 经 __init__ restore 回填；max_bars=3 让再跑 2 根新 bar（首根即调仓）
    be2 = SimBackend(start_equity=1.0, cost_rate=0.0)
    eng2 = _make_engine(tmp_path, FakeSource(_pool()), be2, max_bars=3)
    eng2.run_forever()
    # restore 已回填 _last_close>0；fill 价/entry 也非 0
    assert be2._last_close > 0
    for t in eng2.state.trades:
        assert t["price"] > 0, f"恢复首根 fill 价为 0: {t}"


def test_engine_restore_recovers_backend_state(tmp_path):
    """restart 后新 SimBackend 从 state 末根恢复持仓/权益/entry/close（不再重置丢失）。"""
    from autopilot.backends import SimBackend

    # 第一次跑产生持仓
    be1 = SimBackend(start_equity=10000.0, cost_rate=0.0003)
    eng1 = _make_engine(tmp_path, FakeSource(_pool()), be1, max_bars=3)
    eng1.run_forever()
    last = eng1.state.history[-1]
    # 第二次：新 SimBackend（内存重置）+ 恢复 state → __init__ restore 应回填 4 个字段
    be2 = SimBackend(start_equity=10000.0, cost_rate=0.0003)
    eng2 = _make_engine(tmp_path, FakeSource(_pool()), be2, max_bars=4)
    assert be2._position_notional == pytest.approx(last["actual_notional"])
    assert be2._equity == pytest.approx(last["equity"])
    assert be2._entry_price == pytest.approx(last["entry_price"])
    assert be2._last_close == pytest.approx(last["close"])


def test_engine_restore_equity_no_reset(tmp_path):
    """restart 后权益接续末根、不重置回 start_equity（权益曲线不断层）。"""
    from autopilot.backends import SimBackend

    be1 = SimBackend(start_equity=10000.0, cost_rate=0.0003)
    eng1 = _make_engine(tmp_path, FakeSource(_pool()), be1, max_bars=5)
    eng1.run_forever()
    last_eq = eng1.state.history[-1]["equity"]
    # 第二次恢复：equity 应接续末根，不回 start_equity
    be2 = SimBackend(start_equity=10000.0, cost_rate=0.0003)
    eng2 = _make_engine(tmp_path, FakeSource(_pool()), be2, max_bars=6)
    assert be2._equity == pytest.approx(last_eq)
    # 第一次确实产生了盈亏（equity 偏离 start_equity）→ 恢复后不得断层回 10000
    if abs(last_eq - 10000.0) > 1e-9:
        assert be2._equity != pytest.approx(10000.0)


def test_engine_resume_keeps_drawdown_baseline(tmp_path):
    """B1：回撤基线跨重启——恢复 state 时回填 peak，重启不重置回撤额度。

    场景：peak=12000、当前权益 10800（dd=-10%）→ 重启 → DrawdownBreaker 应回填
    12000，首根 tick 权益 10800 仍触发熔断（现状：peak 从 -inf 起算 → 10800 成新峰 → 不触发）。
    """
    from autopilot.backends import SimBackend
    from autopilot.state import AutopilotState, BarRecord

    # 预置一份 peak=12000 / 末根 equity=10800 的 state
    st = AutopilotState(symbol="TEST", timeframe="1h", mode="paper",
                        peak_equity=12000.0, last_ts=1_700_000_000 + 299 * 3600)
    st.record(BarRecord(ts=1_700_000_000 + 299 * 3600, close=100.0, target_pos=0.0,
                        target_notional=0.0, actual_notional=0.0, equity=10800.0,
                        peak_equity=12000.0, drawdown_pct=-0.10))
    state_file = tmp_path / "autopilot_state.json"
    st.save(state_file)

    # 恒定权益 10800 的 backend（峰 12000 → dd=-10% ≤ -0.10 触发；若 peak 丢失则 dd=0）
    class FlatBackend(SimBackend):
        def fetch_equity(self):
            return 10800.0

    be = FlatBackend(start_equity=10800.0, cost_rate=0.0)
    eng = _make_engine(tmp_path, FakeSource(_pool()), be, max_bars=5)
    reason = eng.run_forever()
    assert "回撤" in reason
    assert eng.state.breaker_tripped


def test_engine_archives_state_on_mode_mismatch(tmp_path):
    """C2：三元组不匹配 → 旧 state 归档 .bak_{mode}_{date}，新 state 空白起算。"""
    from autopilot.backends import SimBackend

    be1 = SimBackend(start_equity=1.0, cost_rate=0.0003)
    eng1 = _make_engine(tmp_path, FakeSource(_pool()), be1, max_bars=2)
    eng1.run_forever()
    state_file = tmp_path / "autopilot_state.json"
    assert state_file.exists()

    # 换 mode（paper → 模拟 testnet 语义：mode 由 backend 决定）重启 → 归档
    class TestnetBackend(SimBackend):
        mode = "testnet"

    be2 = TestnetBackend(start_equity=5000.0, cost_rate=0.0003)
    eng2 = _make_engine(tmp_path, FakeSource(_pool()), be2, max_bars=2)
    baks = list(tmp_path.glob("autopilot_state.bak_paper_*"))
    assert len(baks) == 1, f"应恰好归档一份 paper state: {baks}"
    # 新 state 三元组已是 testnet 且 history 重新起算（max_bars=2 内含熔断前记录≥1）
    assert eng2.state.mode == "testnet"
    assert eng2.state.start_equity <= 0  # 尚未 lazy 设定基线（新账本空白）


def test_engine_no_archive_when_matching(tmp_path):
    """C2：三元组匹配（同 symbol/tf/mode）→ 不归档、正常恢复。"""
    from autopilot.backends import SimBackend

    be1 = SimBackend(start_equity=1.0, cost_rate=0.0003)
    eng1 = _make_engine(tmp_path, FakeSource(_pool()), be1, max_bars=2)
    eng1.run_forever()
    n_hist = len(eng1.state.history)

    be2 = SimBackend(start_equity=1.0, cost_rate=0.0003)
    eng2 = _make_engine(tmp_path, FakeSource(_pool()), be2, max_bars=3)
    assert not list(tmp_path.glob("autopilot_state.bak_*")), "匹配时不应产生归档"
    assert len(eng2.state.history) >= n_hist  # 同一账本延续


class FailingOrderBackend(ExecutionBackend):
    """place_delta_order 恒失败（执行熔断测试替身）。"""

    mode = "paper"

    def __init__(self, msg: str = "simulated reject"):
        self.order_calls = 0
        self._msg = msg

    def fetch_equity(self):
        return 10000.0

    def fetch_position_notional(self, symbol):
        return 0.0

    def place_delta_order(self, symbol, delta_notional):
        self.order_calls += 1
        return OrderResult(ok=False, filled_notional=0.0, message=self._msg)

    def flatten_all(self, symbol):
        return OrderResult(ok=True, message="flattened")


def test_engine_execution_breaker_halts(tmp_path):
    """B3：下单 3 连败 → 执行熔断 halt（不靠下根 bar 自愈），reason 前缀钉死。

    sleep_fn 注入 no-op 免真等 2s；熔断时 state 落盘、halt 前留最后一根账。
    """
    be = FailingOrderBackend()
    eng = _make_engine(tmp_path, FakeSource(_pool()), be, max_bars=5, sleep_fn=lambda s: None)
    reason = eng.run_forever()
    assert reason.startswith("执行熔断")
    assert "simulated reject" in reason
    assert be.order_calls == 3  # 恰好 3 连重试（bar 内），halt 后不再有下一根 bar 的第 4 次
    assert eng.state.breaker_tripped
    assert len(eng.state.history) >= 1  # halt 前留了最后一根账


def test_engine_order_retry_succeeds_no_halt(tmp_path):
    """B3：第 2 次重试成功 → 不熔断，正常调仓继续跑。"""

    class FlakyOrderBackend(FailingOrderBackend):
        def place_delta_order(self, symbol, delta_notional):
            self.order_calls += 1
            if self.order_calls == 1:
                return OrderResult(ok=False, filled_notional=0.0, message="transient")
            return OrderResult(ok=True, filled_notional=delta_notional, price=100.0,
                               message="recovered")

    be = FlakyOrderBackend()
    eng = _make_engine(tmp_path, FakeSource(_pool()), be, max_bars=3, sleep_fn=lambda s: None)
    reason = eng.run_forever()
    assert reason.startswith("max_bars")  # 未熔断，正常跑完
    assert be.order_calls >= 2


def test_classify_action():
    """动作分类：开仓/加仓/减仓/平仓/反手（成交记录校验开平仓逻辑的基础）。"""
    from autopilot.engine import _classify_action

    assert _classify_action(0.0, 1000.0) == "开仓"       # 空仓 → 多
    assert _classify_action(0.0, -1000.0) == "开仓"      # 空仓 → 空
    assert _classify_action(1000.0, 0.0) == "平仓"       # 多 → 空仓
    assert _classify_action(-1000.0, 0.0) == "平仓"      # 空 → 空仓
    assert _classify_action(1000.0, 1800.0) == "加仓"    # 同向增大
    assert _classify_action(-1000.0, -1800.0) == "加仓"  # 空同向增大
    assert _classify_action(1000.0, 500.0) == "减仓"     # 同向减小
    assert _classify_action(-1000.0, -500.0) == "减仓"   # 空同向减小
    assert _classify_action(1000.0, -500.0) == "反手"    # 穿零
    assert _classify_action(-1000.0, 500.0) == "反手"    # 穿零


def test_realized_pnl():
    """实现盈亏：开仓/加仓=0；减仓/平仓/反手=平仓量×sign(before)×(fill/entry_before−1)。"""
    from autopilot.engine import _realized_pnl

    # 开仓（before=0）→ 0
    assert _realized_pnl(0.0, 1000.0, 120.0, 0.0) == 0.0
    # 加仓（同向）→ 0
    assert _realized_pnl(1000.0, 500.0, 120.0, 100.0) == 0.0
    # 多头减仓：fill>entry 盈利
    assert _realized_pnl(1000.0, -400.0, 120.0, 100.0) == pytest.approx(80.0)
    # 空头减仓：fill<entry 盈利（低价买回）
    assert _realized_pnl(-1000.0, 400.0, 80.0, 100.0) == pytest.approx(80.0)
    # 多头全平
    assert _realized_pnl(1000.0, -1000.0, 110.0, 100.0) == pytest.approx(100.0)
    # 反手（多→空）：平掉整个 before，用调仓前均价
    assert _realized_pnl(1000.0, -1500.0, 110.0, 100.0) == pytest.approx(100.0)
    # entry 缺失 → 0（防 ZeroDivision）
    assert _realized_pnl(1000.0, -400.0, 120.0, 0.0) == 0.0
