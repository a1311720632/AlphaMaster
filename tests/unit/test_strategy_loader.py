"""strategy_loader 周期归一化测试。

策略文件按 MT5 风格记录训练周期（H1/M15/D1…，来自 Config.get_timeframe），实盘行情源
（OKXSource）与 autopilot engine 的 _CADENCE/_TF_SECONDS 用项目规范周期（1h/15m/1d…）。
load_strategy 必须在落地 StrategySpec 前归一化，否则报「OKX 不支持周期 H1」。
"""
from __future__ import annotations

import json

from model_core.vocab import FORMULA_VOCAB
from autopilot.strategy_loader import load_strategy, normalize_timeframe


# ── normalize_timeframe：纯函数行为（不依赖词表）──────────────────────────────
def test_mt5_style_mapped_to_canon():
    assert normalize_timeframe("H1") == "1h"
    assert normalize_timeframe("M15") == "15m"
    assert normalize_timeframe("M5") == "5m"
    assert normalize_timeframe("D1") == "1d"
    assert normalize_timeframe("MN1") == "1M"
    assert normalize_timeframe("W1") == "1w"


def test_canonical_timeframe_idempotent():
    # 已是规范周期 → 原样返回（幂等，不会二次变换）
    for canon in ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"):
        assert normalize_timeframe(canon) == canon


def test_case_and_whitespace_tolerant():
    assert normalize_timeframe("h1") == "1h"
    assert normalize_timeframe("  H1 ") == "1h"
    assert normalize_timeframe("m15") == "15m"


def test_unknown_timeframe_passthrough():
    # 未知周期原样返回，交给数据源报「不支持周期」
    assert normalize_timeframe("2h") == "2h"
    assert normalize_timeframe("") == ""


# ── load_strategy：策略文件内 H1/M15 落地为规范周期 ──────────────────────────
def _write_strategy(tmp_path, timeframe: str) -> str:
    p = tmp_path / f"best_TEST_{timeframe}.json"
    p.write_text(
        json.dumps(
            {
                "symbol": "TEST",
                "timeframe": timeframe,
                "formula": [0],  # token 0 = 第一个激活特征，合法
                "vocab_version": FORMULA_VOCAB.version,
                "best_score": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return str(p)


def test_load_strategy_normalizes_mt5_h1(tmp_path):
    spec = load_strategy(_write_strategy(tmp_path, "H1"))
    assert spec.timeframe == "1h"


def test_load_strategy_normalizes_mt5_m15(tmp_path):
    spec = load_strategy(_write_strategy(tmp_path, "M15"))
    assert spec.timeframe == "15m"


def test_load_strategy_keeps_canonical(tmp_path):
    spec = load_strategy(_write_strategy(tmp_path, "1h"))
    assert spec.timeframe == "1h"
