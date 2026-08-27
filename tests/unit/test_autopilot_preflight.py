"""readiness 汇总 + preflight 检查测试（C1/D1/ADR-0007）。"""
from __future__ import annotations

import json
from pathlib import Path

from autopilot.ledger import Ledger
from autopilot.readiness import LIVE_MIN_DAYS, LIVE_MIN_TRADES, live_readiness, summarize_testnet_run


def _seed_ledger(path: Path, *, days: float, trades: list[dict] | None = None,
                 bar_alerts: list[str] | None = None, events: list[dict] | None = None,
                 gap_bars: int = 0) -> None:
    """构造一份 testnet 冷账本：days 天逐小时 bar + 可选 trades/events/alerts/断档。

    gap_bars>0 时在序列中段挖一个 gap_bars×3600s 的洞（模拟 relaunch 空窗）。
    """
    lg = Ledger(path, log=lambda m: None)
    n_bars = int(days * 24)
    ts0 = 1_700_000_000
    half = n_bars // 2
    for i in range(n_bars + 1):
        extra = gap_bars * 3600 if i > half else 0
        alerts = bar_alerts if (bar_alerts and i == half) else []
        lg.append("bar", {"ts": ts0 + i * 3600 + extra, "close": 100.0,
                          "equity": 10000.0, "alerts": alerts})
    for t in trades or []:
        lg.append("trade", t)
    for e in events or []:
        lg.append("event", e)
    lg.close()


def test_summarize_empty_ledger(tmp_path):
    s = summarize_testnet_run(tmp_path / "nope.jsonl")
    assert s["days"] == 0.0 and s["trades"] == 0 and not s["has_open_and_close"]


def test_summarize_full_run(tmp_path):
    p = tmp_path / "l.jsonl"
    _seed_ledger(
        p, days=20,
        trades=[
            {"ts": 1, "action": "开仓", "side": "buy"},
            {"ts": 2, "action": "加仓", "side": "buy"},
            {"ts": 3, "action": "平仓", "side": "sell"},
        ] * 4,  # 12 笔，含开+平
        events=[{"name": "source_switch", "detail": "okx → bybit"}],  # 非关键事件不计
    )
    s = summarize_testnet_run(p)
    assert s["days"] >= 19.9
    assert s["trades"] == 12
    assert s["has_open_and_close"]
    assert s["drift_alerts"] == 0
    assert s["critical_halts"] == 0
    ready, checks = live_readiness(s)
    assert ready, [c for c in checks if c["status"] != "pass"]
    assert len(checks) == 6


def test_summarize_counts_critical_and_drift(tmp_path):
    p = tmp_path / "l.jsonl"
    _seed_ledger(
        p, days=1,
        trades=[{"ts": 1, "action": "开仓"}],
        bar_alerts=["账实漂移: 目标 1.0 / 实际 0.5"],
        events=[
            {"name": "breaker_drawdown", "detail": "..."},
            {"name": "breaker_connectivity", "detail": "..."},  # 断连不计（自愈）
        ],
    )
    s = summarize_testnet_run(p)
    assert s["drift_alerts"] == 1
    assert s["critical_halts"] == 1
    ready, checks = live_readiness(s)
    assert not ready
    assert {c["id"] for c in checks if c["status"] == "fail"} >= {"drift", "halts", "days", "trades"}


def test_gap_detected(tmp_path):
    """断档（relaunch 空窗）计入 max_gap_bars。"""
    p = tmp_path / "l.jsonl"
    _seed_ledger(p, days=1, gap_bars=48)  # 中间空 48 小时 → gap=48 根
    s = summarize_testnet_run(p)
    assert s["max_gap_bars"] >= 40
    ready, checks = live_readiness(s)
    assert not ready


# ── preflight（web 层）──────────────────────────────────────────────────
def test_preflight_passes_without_credentials_when_ccxt_missing():
    """ccxt 未装 + 凭据缺失 → ok=False 且两检查项 fail（其余项不炸）。"""
    import fastapi.testclient

    from web.app import app

    client = fastapi.testclient.TestClient(app)
    resp = client.post("/api/autopilot/preflight", json={
        "strategy_file": "strategies/best_BTCUSDT.json",
        "mode": "testnet", "symbol": "BTCUSDT", "timeframe": "1h",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    ids = {c["id"]: c["status"] for c in body["checks"]}
    assert ids.get("ccxt_installed") in ("pass", "fail")   # 本机装没装 ccxt 都合法
    assert ids["credentials"] == "fail"                     # CI 必然没凭据
    assert ids["state_match"] in ("pass", "warn")


def test_preflight_rejects_paper_mode():
    import fastapi.testclient

    from web.app import app

    client = fastapi.testclient.TestClient(app)
    resp = client.post("/api/autopilot/preflight", json={
        "strategy_file": "x", "mode": "paper",
    })
    assert resp.status_code == 400
