"""一键清除的冷账本归档（2026-09-04）。

回归背景：reset 原来只删 autopilot_state.json——下次启动新 state 三元组又指向
同一 ledger 文件，旧成交/曲线全部回来，"一键清除"落空。锁定：state 删除 +
ledger 主文件归档 .cleared_*（不直删，append-only 审计语义）。
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import web.autopilot_manager as am
from web.settings import save_settings


@pytest.fixture(autouse=True)
def _isolate_project_root(monkeypatch, tmp_path):
    """reset/_archive_ledger 都以模块级 PROJECT_ROOT 拼路径——重定向到 tmp，
    绝不碰仓库根真实 state/ledger；settings 已由 conftest autouse 隔离。"""
    monkeypatch.setattr(am, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(am, "_PROJECT_ROOT_RESOLVED", tmp_path.resolve())
    am.autopilot_manager._proc = None
    am.autopilot_manager._job = None
    yield


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def test_reset_archives_ledger(tmp_path):
    """state 删除 + ledger 归档 .cleared_YYYYMMDD，返回值如实汇报。"""
    save_settings({"autopilot_symbol": "XRPUSDT", "autopilot_mode": "testnet"})
    (tmp_path / "autopilot_state.json").write_text("{}", encoding="utf-8")
    ledger = tmp_path / "autopilot_ledger_XRPUSDT_testnet.jsonl"
    ledger.write_text('{"type": "bar"}\n', encoding="utf-8")

    result = am.autopilot_manager.reset()

    assert result["deleted"] is True
    assert result["ledger_archived"] == f"autopilot_ledger_XRPUSDT_testnet.jsonl.cleared_{_today()}"
    assert not ledger.exists()                    # 主文件已让位
    assert (tmp_path / result["ledger_archived"]).is_file()  # 内容归档保留
    assert not (tmp_path / "autopilot_state.json").exists()


def test_reset_archive_same_day_gets_sequence(tmp_path):
    """同日重复清除：.cleared_YYYYMMDD 已存在 → 追加 _2 防覆盖。"""
    save_settings({"autopilot_symbol": "XRPUSDT", "autopilot_mode": "testnet"})
    ledger = tmp_path / "autopilot_ledger_XRPUSDT_testnet.jsonl"
    ledger.write_text("new\n", encoding="utf-8")
    (tmp_path / f"autopilot_ledger_XRPUSDT_testnet.jsonl.cleared_{_today()}").write_text(
        "old\n", encoding="utf-8")

    result = am.autopilot_manager.reset()

    assert result["ledger_archived"].endswith(f".cleared_{_today()}_2")
    assert (tmp_path / result["ledger_archived"]).read_text(encoding="utf-8") == "new\n"


def test_reset_without_ledger_reports_empty(tmp_path):
    """无 ledger 文件 / 三元组缺失：ledger_archived 空串，不算失败。"""
    # 有三元组但账本文件不存在（从未跑过）
    assert am.autopilot_manager._archive_ledger(
        {"autopilot_symbol": "XRPUSDT", "autopilot_mode": "testnet"}) == ""

    # symbol 缺失：即使账本文件在也绝不动它
    ledger = tmp_path / "autopilot_ledger_XRPUSDT_testnet.jsonl"
    ledger.write_text("x\n", encoding="utf-8")
    assert am.autopilot_manager._archive_ledger(
        {"autopilot_symbol": "", "autopilot_mode": "testnet"}) == ""
    assert ledger.exists()

    # mode 非法同样不动
    assert am.autopilot_manager._archive_ledger(
        {"autopilot_symbol": "XRPUSDT", "autopilot_mode": "demo"}) == ""
    assert ledger.exists()
