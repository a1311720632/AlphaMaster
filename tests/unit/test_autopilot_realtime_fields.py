"""_autopilot_state_dict 实时字段：两基准公式 + 衔接不变量 + 降级。

钉死的关键不变量（plan 易错点 #1）：
- 持仓浮盈用 entry 基准（自开仓）：actual×(P/entry−1)
- 实时权益用 close 基准（自上根收盘）：equity + actual×(P/close−1)
- 衔接：P==close 时 unrealized_pnl_live==0、realtime_equity==equity
"""
from __future__ import annotations

import pytest

import web.app as app
from autopilot.state import AutopilotState, BarRecord
import config


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    path = tmp_path / "autopilot_state.json"
    monkeypatch.setattr(config.Config, "AUTOPILOT_STATE_FILE", str(path))
    return path


def _save(path, actual, entry, equity=10000.0, start=10000.0, close=100.0):
    st = AutopilotState(symbol="BTCUSDT", timeframe="1h", mode="paper", start_equity=start)
    st.record(
        BarRecord(
            ts=1, close=close, target_pos=0.1, target_notional=actual,
            actual_notional=actual, equity=equity, peak_equity=equity,
            drawdown_pct=0.0, entry_price=entry, unrealized_pnl=0.0,
        )
    )
    st.save(path)


def test_seam_invariant_last_equals_close(tmp_state, monkeypatch):
    """P==close：浮盈=0、实时权益=bar 权益、收益=0（两基准没串台）。"""
    _save(tmp_state, actual=1000.0, entry=100.0)
    monkeypatch.setattr(app, "_get_live_price", lambda s: 100.0)
    d = app._autopilot_state_dict()
    assert d["unrealized_pnl_live"] == pytest.approx(0.0)
    assert d["realtime_equity"] == pytest.approx(10000.0)
    assert d["realtime_total_return"] == pytest.approx(0.0)


def test_realtime_formulas_long(tmp_state, monkeypatch):
    """多仓 P=110：浮盈 100、权益 10100、收益 1%（不是 10%！）。"""
    _save(tmp_state, actual=1000.0, entry=100.0)
    monkeypatch.setattr(app, "_get_live_price", lambda s: 110.0)
    d = app._autopilot_state_dict()
    assert d["unrealized_pnl_live"] == pytest.approx(100.0)   # 1000×(110/100−1)
    assert d["realtime_equity"] == pytest.approx(10100.0)     # 10000 + 100
    assert d["realtime_total_return"] == pytest.approx(0.01)  # 100/10000


def test_realtime_formulas_short(tmp_state, monkeypatch):
    """空仓 actual<0：价格涨 → 浮亏、权益降。"""
    _save(tmp_state, actual=-1000.0, entry=100.0)
    monkeypatch.setattr(app, "_get_live_price", lambda s: 110.0)
    d = app._autopilot_state_dict()
    assert d["unrealized_pnl_live"] == pytest.approx(-100.0)
    assert d["realtime_equity"] == pytest.approx(9900.0)


def test_flat_position(tmp_state, monkeypatch):
    _save(tmp_state, actual=0.0, entry=0.0)
    monkeypatch.setattr(app, "_get_live_price", lambda s: 110.0)
    d = app._autopilot_state_dict()
    assert d["unrealized_pnl_live"] == 0.0
    assert d["realtime_equity"] == pytest.approx(10000.0)


def test_ticker_none_degrades(tmp_state, monkeypatch):
    """ticker 取不到 → 三实时字段 None，status 不 500。"""
    _save(tmp_state, actual=1000.0, entry=100.0)
    monkeypatch.setattr(app, "_get_live_price", lambda s: None)
    d = app._autopilot_state_dict()
    assert d["unrealized_pnl_live"] is None
    assert d["realtime_equity"] is None
    assert d["realtime_total_return"] is None


def test_get_live_price_caches(monkeypatch):
    """成功才写缓存；命中缓存不再拉取。"""
    import web.app as a
    a._ticker_cache.clear()
    calls = [0]

    class _FakeOKX:
        def fetch_ticker(self, symbol):
            calls[0] += 1
            return 99.0

    monkeypatch.setattr(a, "get_source", lambda kind: _FakeOKX())
    p1 = a._get_live_price("BTCUSDT")
    p2 = a._get_live_price("BTCUSDT")  # 命中缓存，不再调用 fetch_ticker
    assert p1 == pytest.approx(99.0) and p2 == pytest.approx(99.0)
    assert calls[0] == 1  # 只拉一次
