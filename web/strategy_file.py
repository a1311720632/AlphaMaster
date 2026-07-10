"""Strategy JSON inspection for the web UI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from web.progress import PROJECT_ROOT, _decode_formula

_BEST_NAME_RE = re.compile(r"^best_(.+)\.json$", re.IGNORECASE)


def strategy_path_for_symbol(symbol: str) -> Path:
    return PROJECT_ROOT / "strategies" / f"best_{symbol}.json"


def symbol_from_strategy_path(path: Path) -> str | None:
    m = _BEST_NAME_RE.match(path.name)
    if m:
        return m.group(1)
    return None


def inspect_strategy_file(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"无法解析策略 JSON: {exc}") from exc

    if isinstance(data, list):
        formula = data
        symbol = symbol_from_strategy_path(p)
        best_score = None
        vocab_version = "legacy"
    elif isinstance(data, dict):
        formula = data.get("formula")
        symbol = data.get("symbol") or symbol_from_strategy_path(p)
        best_score = data.get("best_score")
        vocab_version = data.get("vocab_version")
    else:
        raise ValueError("策略文件格式无效")

    if not formula:
        raise ValueError("策略文件缺少 formula 字段")

    formula_decoded = None
    if isinstance(data, dict):
        formula_decoded = data.get("formula_decoded") or _decode_formula(formula)

    return {
        "strategy_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol or "",
        "best_score": best_score,
        "vocab_version": vocab_version,
        "formula_decoded": formula_decoded,
        "valid": True,
        "message": "",
    }


def resolve_strategy_file(
    saved_path: str,
    train_symbol: str | None = None,
) -> str:
    """优先使用已保存路径；否则回退到训练品种对应的 best_{symbol}.json。"""
    if saved_path:
        p = Path(saved_path)
        if p.exists():
            return str(p.resolve())

    if train_symbol:
        default = strategy_path_for_symbol(train_symbol)
        if default.exists():
            return str(default.resolve())

    return saved_path or ""
