"""策略公式 AI 解读：prompt 组装、磁盘缓存失效、explain-strategy 端点。"""
from __future__ import annotations

from pathlib import Path

import pytest

import web.ai_analyze as ai_mod
from web.ai_analyze import (
    _formula_hash,
    build_explain_user_message,
    load_explanation,
    save_explanation,
)


# ── prompt 组装：分型 + 元信息 ───────────────────────────────────────────────
def test_user_message_types_tokens_and_meta():
    # 词表：offset=65（65 个特征），48=特征 DMI_ADX_14，90=算子 TS_MIN_10
    msg = build_explain_user_message({
        "symbol": "XRPUSDT",
        "timeframe": "H1",
        "best_score": 2.63,
        "formula": [48, 90],
        "formula_decoded": "DMI_ADX_14 → TS_MIN_10",
    })
    assert "XRPUSDT" in msg and "H1" in msg and "2.63" in msg
    assert "DMI_ADX_14 → TS_MIN_10" in msg
    assert "1. 特征·DMI_ADX_14" in msg
    assert "2. 算子·TS_MIN_10" in msg


def test_user_message_tolerates_out_of_range_tokens():
    msg = build_explain_user_message({"formula": [9999], "symbol": "T"})
    assert "?9999" in msg  # 越界 token 不炸，标问号


# ── 磁盘缓存：命中 / 公式或词表变化失效 ─────────────────────────────────────
@pytest.fixture
def cache_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "strategy_explanations.json"
    monkeypatch.setattr(ai_mod, "_EXPLAIN_CACHE_PATH", p)
    return p


def test_cache_roundtrip_and_invalidation(cache_path: Path):
    h1 = _formula_hash("v1", [1, 2])
    save_explanation("best_A.json", h1, {"explanation": "解读", "provider": "deepseek"})
    assert load_explanation("best_A.json", h1)["explanation"] == "解读"
    # 公式/词表一变（hash 不同）→ 失效
    assert load_explanation("best_A.json", _formula_hash("v1", [1, 2, 3])) is None
    assert load_explanation("best_A.json", _formula_hash("v2", [1, 2])) is None
    assert load_explanation("best_B.json", h1) is None


def test_cache_missing_or_corrupt_file_returns_none(cache_path: Path, monkeypatch):
    assert load_explanation("best_A.json", "h") is None
    cache_path.write_text("not json{", encoding="utf-8")
    assert load_explanation("best_A.json", "h") is None


# ── 端点（TestClient）───────────────────────────────────────────────────────
def _client():
    import fastapi.testclient

    from web.app import app

    return fastapi.testclient.TestClient(app)


def test_endpoint_rejects_path_traversal():
    resp = _client().post("/api/ai/explain-strategy", json={"strategy_file": "../../web_settings.json"})
    assert resp.status_code == 400


def test_endpoint_rejects_missing_formula(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from web.progress import STRATEGIES_DIR

    monkeypatch.setattr("web.progress.STRATEGIES_DIR", tmp_path)
    (tmp_path / "best_EMPTY.json").write_text('{"symbol": "T"}', encoding="utf-8")
    resp = _client().post("/api/ai/explain-strategy", json={"strategy_file": "best_EMPTY.json"})
    assert resp.status_code == 400


def test_endpoint_serves_from_cache_without_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """缓存命中 → cached=True 且绝不触碰 LLM（monkeypatch explain_strategy 为炸弹）。"""
    from web.progress import STRATEGIES_DIR

    monkeypatch.setattr("web.progress.STRATEGIES_DIR", tmp_path)
    (tmp_path / "best_A.json").write_text(
        '{"symbol": "T", "formula": [48, 90], "vocab_version": "v1"}', encoding="utf-8"
    )
    monkeypatch.setattr(ai_mod, "_EXPLAIN_CACHE_PATH", tmp_path / "cache.json")
    save_explanation("best_A.json", _formula_hash("v1", [48, 90]), {"explanation": "缓存版解读"})

    resp = _client().post("/api/ai/explain-strategy", json={"strategy_file": "best_A.json"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["cached"] is True
    assert body["explanation"] == "缓存版解读"


def test_endpoint_generates_and_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """未命中 → 调 explain_strategy（打桩）→ cached=False 且落盘，二次请求走缓存。"""
    from web.progress import STRATEGIES_DIR

    monkeypatch.setattr("web.progress.STRATEGIES_DIR", tmp_path)
    (tmp_path / "best_B.json").write_text(
        '{"symbol": "T", "formula": [1], "vocab_version": "v1"}', encoding="utf-8"
    )
    monkeypatch.setattr(ai_mod, "_EXPLAIN_CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(ai_mod, "explain_strategy", lambda **kw: {
        "explanation": "新生成解读", "provider": "deepseek", "model": "m", "label": "DeepSeek",
    })

    client = _client()
    body = client.post("/api/ai/explain-strategy", json={"strategy_file": "best_B.json"}).json()
    assert body["cached"] is False and body["explanation"] == "新生成解读"
    body2 = client.post("/api/ai/explain-strategy", json={"strategy_file": "best_B.json"}).json()
    assert body2["cached"] is True and body2["explanation"] == "新生成解读"


def test_endpoint_maps_no_key_to_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """resolve_provider 的 ValueError（如未配 key）→ 400 带原文案。"""
    from web.progress import STRATEGIES_DIR

    monkeypatch.setattr("web.progress.STRATEGIES_DIR", tmp_path)
    (tmp_path / "best_C.json").write_text('{"symbol": "T", "formula": [1]}', encoding="utf-8")
    monkeypatch.setattr(ai_mod, "_EXPLAIN_CACHE_PATH", tmp_path / "cache.json")

    def _raise(**kw):
        raise ValueError("请填写 DeepSeek API Key")

    monkeypatch.setattr(ai_mod, "explain_strategy", _raise)  # 端点函数内 from 源模块导入，须 patch 源
    resp = _client().post("/api/ai/explain-strategy", json={"strategy_file": "best_C.json"})
    assert resp.status_code == 400
    assert "DeepSeek" in resp.json()["detail"]
