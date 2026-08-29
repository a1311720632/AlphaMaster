"""回撤熔断 web 设置：默认关闭、clamp 合法域、roundtrip、resolver 优先级。

conftest 的 autouse fixture 已把 SETTINGS_PATH 隔离进 tmp_path，可直接读写。
"""
from __future__ import annotations

import pytest

from web.settings import (
    _as_drawdown_pct,
    drawdown_breaker_config,
    load_settings,
    save_settings,
)


# ── 默认值（用户决策：默认关闭）─────────────────────────────────────────────
def test_default_disabled_when_no_file():
    s = load_settings()
    assert s["autopilot_breaker_drawdown_enabled"] is False
    assert s["autopilot_breaker_drawdown_pct"] == pytest.approx(0.10)
    assert drawdown_breaker_config({}) == (False, pytest.approx(0.10))


# ── clamp 合法域 (0, 0.95]：越界回落默认，绝不 clamp 进危险区 ────────────────
@pytest.mark.parametrize("bad", [0, -0.1, 0.96, 2.0, "abc", None, float("nan"), float("inf")])
def test_pct_invalid_falls_back(bad):
    assert _as_drawdown_pct(bad, 0.10) == pytest.approx(0.10)


@pytest.mark.parametrize("good,expected", [(0.05, 0.05), (0.95, 0.95), ("0.2", 0.2)])
def test_pct_valid_passes(good, expected):
    assert _as_drawdown_pct(good, 0.10) == pytest.approx(expected)


# ── roundtrip：存什么读回什么；写其他键不丢熔断键 ───────────────────────────
def test_roundtrip_and_no_loss_on_partial_save():
    save_settings({"autopilot_breaker_drawdown_enabled": True, "autopilot_breaker_drawdown_pct": 0.05})
    s = load_settings()
    assert s["autopilot_breaker_drawdown_enabled"] is True
    assert s["autopilot_breaker_drawdown_pct"] == pytest.approx(0.05)

    save_settings({"debug_mode": True})  # 只写别的键
    s = load_settings()
    assert s["autopilot_breaker_drawdown_enabled"] is True
    assert s["autopilot_breaker_drawdown_pct"] == pytest.approx(0.05)
    assert s["debug_mode"] is True

    save_settings({"autopilot_breaker_drawdown_enabled": False})
    assert load_settings()["autopilot_breaker_drawdown_enabled"] is False


# ── resolver：env 不可达阈值（-2.0 off 手法）映射为关闭 ─────────────────────
def test_resolver_unreachable_env_threshold_maps_to_off(monkeypatch):
    from config import Config

    import web.settings as settings_mod

    monkeypatch.setattr(
        settings_mod, "load_raw_settings", lambda: {"autopilot_breaker_drawdown_enabled": True}
    )
    monkeypatch.setattr(Config, "AUTOPILOT_BREAKER_MAX_DRAWDOWN_PCT", -2.0)
    enabled, _pct = drawdown_breaker_config()
    assert enabled is False
