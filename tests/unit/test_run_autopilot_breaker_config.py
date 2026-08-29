"""run_autopilot 回撤熔断 resolver 优先级：web_settings 显式键 > Config env。"""
from __future__ import annotations

import run_autopilot
from config import Config


def test_env_magnitude_fallback_when_no_explicit_key(monkeypatch):
    """原始 JSON 无键 → env 幅度兜底；开关默认关闭（用户决策）。"""
    monkeypatch.setattr("web.settings.load_raw_settings", lambda: {})
    monkeypatch.setattr(Config, "AUTOPILOT_BREAKER_MAX_DRAWDOWN_PCT", -0.25)
    assert run_autopilot._resolve_drawdown_breaker() == (False, 0.25)


def test_env_unreachable_threshold_maps_to_off(monkeypatch):
    """env 设 -2.0（config.py 注释记载的 off 手法）→ 诚实报告关闭。"""
    monkeypatch.setattr("web.settings.load_raw_settings", lambda: {})
    monkeypatch.setattr(Config, "AUTOPILOT_BREAKER_MAX_DRAWDOWN_PCT", -2.0)
    enabled, _pct = run_autopilot._resolve_drawdown_breaker()
    assert enabled is False


def test_web_explicit_keys_win_over_env(monkeypatch):
    monkeypatch.setattr("web.settings.load_raw_settings", lambda: {
        "autopilot_breaker_drawdown_enabled": True,
        "autopilot_breaker_drawdown_pct": 0.05,
    })
    monkeypatch.setattr(Config, "AUTOPILOT_BREAKER_MAX_DRAWDOWN_PCT", -0.25)
    assert run_autopilot._resolve_drawdown_breaker() == (True, 0.05)


def test_web_invalid_pct_falls_back_to_env_magnitude(monkeypatch):
    monkeypatch.setattr("web.settings.load_raw_settings", lambda: {
        "autopilot_breaker_drawdown_enabled": True,
        "autopilot_breaker_drawdown_pct": 0,  # 非法（首根 bar 即 trip 的危险值）
    })
    monkeypatch.setattr(Config, "AUTOPILOT_BREAKER_MAX_DRAWDOWN_PCT", -0.25)
    assert run_autopilot._resolve_drawdown_breaker() == (True, 0.25)
