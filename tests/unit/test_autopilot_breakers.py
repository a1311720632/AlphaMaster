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
