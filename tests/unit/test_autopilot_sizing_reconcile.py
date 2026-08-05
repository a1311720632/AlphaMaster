"""SimBackend 规模/成本/对账单元测试（ADR-0002 名义价值、ADR-0006 自愈）。

- notional = target_pos × equity，|target_pos|<1（tanh）→ 结构性 ≤1x
- mark_to_market：多头在涨价盈利、空头在跌价盈利
- place_delta_order：成本 = |delta|×cost_rate 从权益扣
- flatten_all：把持仓清零
- delta = target − actual；部分成交留下残差，下根 bar 自动修正（自愈）
"""
from __future__ import annotations

import pytest

from autopilot.backends import SimBackend


def test_sizing_notional_is_equity_fraction_and_le1x():
    # ADR-0002：notional = target_pos × equity；|pos|<1 → |notional| < equity（≤1x）
    be = SimBackend(start_equity=1000.0, cost_rate=0.0003)
    for pos in (0.0, 0.3, 0.5, 0.9, 0.999, -0.4, -0.999):
        target_notional = pos * be.fetch_equity()
        assert abs(target_notional) <= be.fetch_equity() + 1e-9


def test_mark_to_market_long_gains_on_rise():
    be = SimBackend(start_equity=1000.0, cost_rate=0.0)
    be.place_delta_order("X", 500.0)              # 持 500 名义多头
    eq0 = be.fetch_equity()
    pnl = be.mark_to_market(100.0, 101.0)         # 价格 +1%
    assert pnl == pytest.approx(500.0 * 0.01)     # 500 × 1% = 5
    assert be.fetch_equity() == pytest.approx(eq0 + 5.0)


def test_mark_to_market_short_gains_on_fall():
    be = SimBackend(start_equity=1000.0, cost_rate=0.0)
    be.place_delta_order("X", -500.0)             # 持 500 名义空头
    pnl = be.mark_to_market(100.0, 98.0)          # 价格 -2%
    assert pnl == pytest.approx(-500.0 * -0.02)   # 空头跌价盈利 = 10
    assert be.fetch_equity() == pytest.approx(1000.0 + 10.0)


def test_place_delta_charges_cost():
    be = SimBackend(start_equity=1000.0, cost_rate=0.0003)
    be.place_delta_order("X", 500.0)              # 成本 500×0.0003 = 0.15
    assert be.fetch_equity() == pytest.approx(1000.0 - 0.15)
    assert be.fetch_position_notional("X") == pytest.approx(500.0)


def test_flatten_zeroes_position():
    be = SimBackend(start_equity=1000.0, cost_rate=0.0)
    be.place_delta_order("X", 400.0)
    assert be.fetch_position_notional("X") == pytest.approx(400.0)
    be.flatten_all("X")
    assert be.fetch_position_notional("X") == pytest.approx(0.0)


def test_delta_self_heal_after_partial_drift():
    # ADR-0006：delta = target − actual；漂移（如部分成交/手动干预）由下根 bar 的 delta 自愈
    be = SimBackend(start_equity=1000.0, cost_rate=0.0)
    target = 500.0
    # 第一根：actual=0 → delta=500，全额成交（paper）
    actual = be.fetch_position_notional("X")
    assert actual == 0.0
    be.place_delta_order("X", target - actual)
    assert be.fetch_position_notional("X") == pytest.approx(500.0)

    # 模拟交易所侧出现漂移：实际只成交了 480（部分成交）。对账以交易所为准 → delta=20
    be._position_notional = 480.0  # 直接模拟交易所报告的实际持仓
    delta = target - be.fetch_position_notional("X")
    assert delta == pytest.approx(20.0)
    be.place_delta_order("X", delta)
    assert be.fetch_position_notional("X") == pytest.approx(500.0)  # 自愈回到目标
