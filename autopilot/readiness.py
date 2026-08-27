"""testnet 长跑达标汇总（D1/ADR-0007）：上实盘是一个状态，不是仪式。

扫 testnet 冷账本 → 机检清单 → live 确认弹窗逐项绿勾。

口径说明（写死在这里避免将来误判）：
  - days = 首末 bar 跨度天数（容忍 relaunch 断档；断档大小计 max_gap_bars）
  - critical_halts 只计回撤/执行熔断——断连熔断走 watcher relaunch 自动恢复
    （B2 决策保留），不算 testnet 失败
  - drift_alerts = bar 行 alerts 里"账实漂移"计数
"""
from __future__ import annotations

from pathlib import Path

from autopilot.ledger import Ledger

# live 门槛（D1：testnet 长跑 2-4 周，取下限 14 天）
LIVE_MIN_DAYS = 14.0
LIVE_MIN_TRADES = 8
LIVE_MAX_GAP_BARS = 6  # relaunch 断档容忍上限（根）


def summarize_testnet_run(ledger_path: str | Path) -> dict:
    """扫 ledger → {"days","trades","has_open_and_close","drift_alerts",
    "critical_halts","max_gap_bars","first_ts","last_ts"}。文件缺失 → 全零。"""
    rows = Ledger(ledger_path).read_all()
    bars = [r for r in rows if r.get("type") == "bar"]
    trades = [r for r in rows if r.get("type") == "trade"]
    events = [r for r in rows if r.get("type") == "event"]

    out = {
        "days": 0.0, "trades": 0, "has_open_and_close": False,
        "drift_alerts": 0, "critical_halts": 0, "max_gap_bars": 0,
        "first_ts": 0, "last_ts": 0,
    }
    if not bars:
        return out
    ts_list = [int(b.get("ts") or 0) for b in bars]
    ts_list.sort()
    out["first_ts"], out["last_ts"] = ts_list[0], ts_list[-1]
    span = ts_list[-1] - ts_list[0]
    # bar 间隔取相邻差的中位数（抗断档/补数毛刺）
    diffs = [b - a for a, b in zip(ts_list, ts_list[1:]) if b > a]
    med = sorted(diffs)[len(diffs) // 2] if diffs else 3600
    out["days"] = span / 86400.0
    out["max_gap_bars"] = (max(diffs) // med) if diffs and med > 0 else 0
    out["trades"] = len(trades)
    actions = {str(t.get("action")) for t in trades}
    out["has_open_and_close"] = "开仓" in actions and "平仓" in actions
    out["drift_alerts"] = sum(
        1 for b in bars
        if any("账实漂移" in str(a) for a in (b.get("alerts") or []))
    )
    out["critical_halts"] = sum(
        1 for e in events
        if str(e.get("name")) in ("breaker_drawdown", "breaker_execution")
    )
    return out


def live_readiness(summary: dict) -> tuple[bool, list[dict]]:
    """门槛逐项判定 → (全部通过, 检查项列表)。每项 {id,label,status,detail}。"""
    checks = [
        {
            "id": "days", "label": f"连续运行 ≥ {LIVE_MIN_DAYS:.0f} 天",
            "status": "pass" if summary["days"] >= LIVE_MIN_DAYS else "fail",
            "detail": f"当前 {summary['days']:.1f} 天",
        },
        {
            "id": "trades", "label": f"真实订单 ≥ {LIVE_MIN_TRADES} 笔",
            "status": "pass" if summary["trades"] >= LIVE_MIN_TRADES else "fail",
            "detail": f"当前 {summary['trades']} 笔",
        },
        {
            "id": "open_close", "label": "覆盖开仓与平仓",
            "status": "pass" if summary["has_open_and_close"] else "fail",
            "detail": "开+平均发生过" if summary["has_open_and_close"] else "缺少开仓或平仓动作",
        },
        {
            "id": "drift", "label": "账实漂移告警 = 0",
            "status": "pass" if summary["drift_alerts"] == 0 else "fail",
            "detail": f"当前 {summary['drift_alerts']} 次",
        },
        {
            "id": "halts", "label": "无关键熔断（回撤/执行）",
            "status": "pass" if summary["critical_halts"] == 0 else "fail",
            "detail": f"当前 {summary['critical_halts']} 次（断连自愈不计）",
        },
        {
            "id": "gap", "label": f"最大断档 ≤ {LIVE_MAX_GAP_BARS} 根",
            "status": "pass" if summary["max_gap_bars"] <= LIVE_MAX_GAP_BARS else "fail",
            "detail": f"当前 {summary['max_gap_bars']} 根",
        },
    ]
    return all(c["status"] == "pass" for c in checks), checks
