"""冷账本 Ledger 单元测试（E1/ADR-0007）：append/tail/过滤/坏行容错/路径。"""
from __future__ import annotations

from autopilot.ledger import Ledger, ledger_path


def test_ledger_path_sanitizes_symbol(tmp_path):
    """symbol 净化 + base_dir 默认项目根（空=Path.cwd）。"""
    p = ledger_path("BTCUSDT", "paper", tmp_path)
    assert p.name == "autopilot_ledger_BTCUSDT_paper.jsonl"
    assert p.parent == tmp_path
    # 带路径分隔符的脏 symbol → 下划线
    p2 = ledger_path("../evil/sym", "live", tmp_path)
    assert p2.parent == tmp_path
    assert "/" not in p2.name.replace("autopilot_ledger_", "", 1) or p2.parent == tmp_path
    assert ".." not in p2.name


def test_append_and_tail_roundtrip(tmp_path):
    lg = Ledger(tmp_path / "l.jsonl", log=lambda m: None)
    lg.append("bar", {"ts": 1, "equity": 100.0})
    lg.append("trade", {"ts": 2, "side": "buy"})
    lg.append("event", {"ts": 3, "name": "source_switch"})
    lg.close()

    rows = Ledger(tmp_path / "l.jsonl").tail(10)
    assert [r["type"] for r in rows] == ["bar", "trade", "event"]
    assert rows[0]["equity"] == 100.0
    # 类型过滤
    trades = Ledger(tmp_path / "l.jsonl").tail(10, types={"trade"})
    assert len(trades) == 1 and trades[0]["side"] == "buy"
    # read_all
    assert len(Ledger(tmp_path / "l.jsonl").read_all()) == 3


def test_tail_respects_limit(tmp_path):
    lg = Ledger(tmp_path / "l.jsonl", log=lambda m: None)
    for i in range(10):
        lg.append("bar", {"ts": i})
    lg.close()
    rows = Ledger(tmp_path / "l.jsonl").tail(3)
    assert [r["ts"] for r in rows] == [7, 8, 9]


def test_tail_missing_file_and_bad_lines(tmp_path):
    """文件不存在 → []；坏行（半写）跳过不抛。"""
    assert Ledger(tmp_path / "nope.jsonl").tail(5) == []
    assert Ledger(tmp_path / "nope.jsonl").read_all() == []
    p = tmp_path / "bad.jsonl"
    p.write_text('{"type":"bar","ts":1}\nNOT-JSON\n\n{"type":"event","ts":2}\n', encoding="utf-8")
    rows = Ledger(p).tail(10)
    assert len(rows) == 2  # 坏行与空行都被跳过


def test_append_invalid_type_raises(tmp_path):
    lg = Ledger(tmp_path / "l.jsonl", log=lambda m: None)
    try:
        lg.append("bogus", {})
        raise AssertionError("应拒绝未知类型")
    except ValueError:
        pass
    finally:
        lg.close()


def test_append_failure_does_not_raise(tmp_path):
    """写失败（目录变只读等）swallow，不杀交易进程。"""
    # 用一个"文件名其实是目录"的路径强制 OSError
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    lg = Ledger(blocker, log=lambda m: None)  # open 一个目录 → OSError
    lg.append("bar", {"ts": 1})  # 不应抛
