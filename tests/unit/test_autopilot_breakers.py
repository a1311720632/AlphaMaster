"""运营熔断单元测试（ADR-0005）。

DrawdownBreaker：峰值权益回撤 ≤ 阈值 → trip，且 trip 后恒 tripped（不可自愈）。
ConnectivityBreaker：连续 N 次失败 → trip；成功一次重置计数；trip 后恒 tripped。
"""
from __future__ import annotations

import pytest

from autopilot.breakers import ConnectivityBreaker, DrawdownBreaker


# ── DrawdownBreaker ──────────────────────────────────────────────────────────
def test_drawdown_tracks_peak_no_trip_on_gain():
    b = DrawdownBreaker(-0.10)
    peak, dd, tripped = b.update(100.0)
    assert peak == 100.0 and dd == 0.0 and not tripped
    peak, dd, tripped = b.update(105.0)
    assert peak == 105.0 and dd == 0.0 and not tripped


def test_drawdown_trips_at_threshold_and_is_sticky():
    b = DrawdownBreaker(-0.10)
    b.update(100.0)                       # peak=100
    peak, dd, tripped = b.update(89.0)    # dd = -11% ≤ -10% → trip
    assert tripped
    assert dd == pytest.approx(-0.11)
    assert peak == 100.0
    # sticky：即便权益反弹，仍 tripped
    _, _, tripped2 = b.update(120.0)
    assert tripped2


def test_drawdown_boundary_equal_trips():
    b = DrawdownBreaker(-0.10)
    b.update(100.0)
    _, _, tripped = b.update(90.0)        # dd 恰好 -10% ≤ -10% → trip
    assert tripped


def test_drawdown_just_above_threshold_no_trip():
    b = DrawdownBreaker(-0.10)
    b.update(100.0)
    _, _, tripped = b.update(91.0)        # dd ≈ -9% > -10% → 不 trip
    assert not tripped


# ── DrawdownBreaker.enabled（web 可视化开关，默认 True 保持既有语义）──────────
def test_drawdown_disabled_never_trips_but_tracks():
    """关闭后深跌不 trip，但 peak/dd 照常跟踪（前端曲线/日报/审计依赖）。"""
    b = DrawdownBreaker(-0.10, enabled=False)
    assert b.enabled is False
    peak, dd, tripped = b.update(100.0)
    assert (peak, tripped) == (100.0, False) and dd == 0.0
    peak, dd, tripped = b.update(80.0)    # dd = -20% 远超阈值 → 仍不 trip
    assert tripped is False
    assert peak == 100.0
    assert dd == pytest.approx(-0.20)
    peak, _, tripped = b.update(200.0)    # 新高照常记录，后续回落仍不 trip
    assert peak == 200.0 and not tripped
    _, _, tripped = b.update(150.0)
    assert not tripped


def test_drawdown_default_enabled():
    """不传 enabled 的既有构造（引擎/测试）语义不变：默认开启、可 trip。"""
    b = DrawdownBreaker(-0.10)
    assert b.enabled is True
    b.update(100.0)
    assert b.update(89.0)[2]


# ── ConnectivityBreaker ──────────────────────────────────────────────────────
def test_connectivity_trips_after_n_failures():
    b = ConnectivityBreaker(3)
    assert not b.fail()
    assert not b.fail()
    assert b.fail()                       # 第 3 次 → trip


def test_connectivity_reset_on_success():
    b = ConnectivityBreaker(3)
    b.fail()
    b.fail()
    b.reset()                             # 一次成功重置计数
    assert not b.fail()
    assert not b.fail()
    assert b.fail()                       # 又需 3 次才 trip


def test_connectivity_is_sticky():
    b = ConnectivityBreaker(2)
    b.fail()
    assert b.fail()
    b.reset()                             # trip 后 reset 不解除
    assert b.tripped
