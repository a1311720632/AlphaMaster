"""β 自动续命的逻辑测试（不拉子进程，不碰奇偶命门）。

覆盖：
- _classify_exit 决策表（stay_down vs relaunch）
- autopilot_intended_running 标志的 settings 往返
- relaunch_if_intended 的幂等、flag-False 短路、advisory stay-down
"""
from __future__ import annotations

import pytest

from web.autopilot_manager import autopilot_manager
from web.settings import load_settings, save_settings


@pytest.fixture(autouse=True)
def _reset_manager(monkeypatch):
    """每个用例前重置单例的 β 状态，避免互相污染。"""
    autopilot_manager._boot_relaunch_done = False
    autopilot_manager._last_exit_reason = ""
    autopilot_manager._stopped_by_user = False
    autopilot_manager._relaunch_attempt = 0
    autopilot_manager._last_handled_finished_at = None
    # 确保无残留 job/proc 句柄
    autopilot_manager._proc = None
    autopilot_manager._job = None
    # 隔离真实 autopilot_state.json（仓库根可能有本地跑残留）→ reason 恒空，
    # 让 advisory stay-down 判断不被外部文件污染
    monkeypatch.setattr(autopilot_manager, "_read_breaker_reason", lambda: "")
    yield


# ── _classify_exit 决策表 ────────────────────────────────────────────────
@pytest.mark.parametrize("stopped_by_user,code,reason,expected", [
    (True, 1, "", "stay_down"),                       # 用户 stop
    (True, -9, "", "stay_down"),                      # 用户 stop（即使被信号杀）
    (False, 1, "回撤熔断: 回撤 12.34%", "stay_down"),  # 策略爆了
    (False, 0, "STOP_SIGNAL 文件检出，优雅退出", "stay_down"),
    (False, 0, "", "stay_down"),                      # 干净退出
    (False, 0, "max_bars 达成（2 根新 bar）", "stay_down"),
    (False, 1, "断网熔断: 连续 3 次拉取失败", "relaunch"),  # 瞬时，自愈
    (False, 1, "", "relaunch"),                       # 未知崩溃
    (False, 137, "", "relaunch"),                     # 信号死亡（code>0 视为崩溃）
    (False, -11, "", "relaunch"),                     # SIGSEGV 等（code<0 非 user-stop）
])
def test_classify_exit(stopped_by_user, code, reason, expected):
    assert autopilot_manager._classify_exit(stopped_by_user, code, reason) == expected


# ── 标志位 settings 往返 ──────────────────────────────────────────────────
def test_intended_running_flag_roundtrip():
    save_settings({"autopilot_intended_running": True})
    assert load_settings()["autopilot_intended_running"] is True
    save_settings({"autopilot_intended_running": False})
    assert load_settings()["autopilot_intended_running"] is False


def test_intended_running_default_false():
    """新设置文件默认 False（不会在没启动过的情况下误重拉）。"""
    save_settings({})  # 触发一次写入
    assert load_settings()["autopilot_intended_running"] is False


# ── relaunch_if_intended ──────────────────────────────────────────────────
def test_relaunch_short_circuits_when_flag_false(monkeypatch):
    """flag=False → 绝不调 start。"""
    calls = []
    monkeypatch.setattr(autopilot_manager, "start", lambda **kw: calls.append(kw))
    save_settings({"autopilot_intended_running": False,
                   "autopilot_last_strategy": "strategies/best_BTCUSDT.json"})
    autopilot_manager.relaunch_if_intended()
    assert calls == []


def test_relaunch_relaunches_when_flag_true(monkeypatch):
    """flag=True + 有效 strategy → 调一次 start，参数来自持久化设置。"""
    calls = []
    monkeypatch.setattr(autopilot_manager, "start", lambda **kw: calls.append(kw))
    save_settings({
        "autopilot_intended_running": True,
        "autopilot_last_strategy": "strategies/best_BTCUSDT.json",
        "autopilot_mode": "paper",
        "autopilot_symbol": "BTCUSDT",
        "autopilot_timeframe": "1h",
    })
    autopilot_manager.relaunch_if_intended()
    assert len(calls) == 1
    assert calls[0]["strategy_file"] == "strategies/best_BTCUSDT.json"
    assert calls[0]["mode"] == "paper"
    assert calls[0]["symbol"] == "BTCUSDT"


def test_relaunch_idempotent(monkeypatch):
    """连调两次只 spawn 一次（_boot_relaunch_done 守卫）。"""
    calls = []
    monkeypatch.setattr(autopilot_manager, "start", lambda **kw: calls.append(kw))
    save_settings({"autopilot_intended_running": True,
                   "autopilot_last_strategy": "strategies/best_BTCUSDT.json"})
    autopilot_manager.relaunch_if_intended()
    autopilot_manager.relaunch_if_intended()
    assert len(calls) == 1


def test_relaunch_advisory_stay_down_on_drawdown(monkeypatch):
    """上一轮 halt 是回撤熔断 → 不重拉，且清掉意图标志。"""
    calls = []
    monkeypatch.setattr(autopilot_manager, "start", lambda **kw: calls.append(kw))
    autopilot_manager._last_exit_reason = "回撤熔断: 回撤 15.00% ≤ 阈值 -200.00%"
    save_settings({"autopilot_intended_running": True,
                   "autopilot_last_strategy": "strategies/best_BTCUSDT.json"})
    autopilot_manager.relaunch_if_intended()
    assert calls == []
    assert load_settings()["autopilot_intended_running"] is False


def test_relaunch_clears_flag_when_strategy_missing(monkeypatch):
    """flag=True 但无 strategy_file → 直接 return，不动 flag（无可拉对象）。"""
    calls = []
    monkeypatch.setattr(autopilot_manager, "start", lambda **kw: calls.append(kw))
    save_settings({"autopilot_intended_running": True, "autopilot_last_strategy": ""})
    autopilot_manager.relaunch_if_intended()
    assert calls == []
