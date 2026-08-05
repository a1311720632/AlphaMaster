"""autopilot_manager 输入净化测试（安全评审 #1/#2）。

- _safe_log_symbol：所有路径分隔符/特殊字符被剥离，日志路径留在 logs/ 内。
- _validate_strategy_path：拒绝不存在/非 .json/项目根外的策略路径。
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

import web.autopilot_manager as m


def test_safe_log_symbol_strips_traversal():
    for evil in ("../../etc/passwd", "..\\..\\evil", "../logs/x", "BTC/USDT:USDT", "a;b|c", "..."):
        clean = m._safe_log_symbol(evil)
        # 仅剩单词字符与连字符
        assert re.match(r"^[\w\-]+$", clean), f"{evil!r} -> {clean!r}"
        assert len(clean) <= 50


def test_safe_log_symbol_log_path_stays_within_logdir(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "LOG_DIR", tmp_path)
    monkeypatch.setattr(m, "_LOG_DIR_RESOLVED", tmp_path.resolve())
    clean = m._safe_log_symbol("../../evil")
    log_path = (tmp_path / f"autopilot_{clean}_t.log").resolve()
    # 关键：穿越输入不会让日志落到 logs/ 之外
    assert log_path.parent == tmp_path.resolve()
    assert log_path.is_absolute()


def test_validate_strategy_path_accepts_in_root(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "_PROJECT_ROOT_RESOLVED", tmp_path.resolve())
    good = tmp_path / "best_X.json"
    good.write_text("{}", encoding="utf-8")
    assert m._validate_strategy_path(str(good)) == str(good.resolve())


def test_validate_strategy_path_rejects_outside_root(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "_PROJECT_ROOT_RESOLVED", tmp_path.resolve())
    outside = Path(tempfile.mkdtemp()) / "best_Y.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        m._validate_strategy_path(str(outside))


def test_validate_strategy_path_rejects_non_json_and_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "_PROJECT_ROOT_RESOLVED", tmp_path.resolve())
    notjson = tmp_path / "x.txt"
    notjson.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        m._validate_strategy_path(str(notjson))            # 非 .json
    with pytest.raises(ValueError):
        m._validate_strategy_path(str(tmp_path / "nope.json"))  # 不存在
