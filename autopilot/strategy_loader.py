"""加载 best_{symbol}.json 策略并校验词表版本。

与 web/realtime_manager.py::_load_strategy_meta 同源，但在 autopilot 核心层独立
实现（autopilot 不依赖 web），并强制 FORMULA_VOCAB.verify——词表不匹配即拒绝加载
（model_core/vocab.py R3.7），保证加载的策略与当前特征/算子注册表一致。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from model_core.vocab import FORMULA_VOCAB, VocabVersionMismatchError


class StrategyLoadError(Exception):
    """策略加载失败（文件缺失 / 格式错 / 词表不匹配 / formula 非法）。"""


@dataclass(frozen=True)
class StrategySpec:
    formula: list[int]
    symbol: str
    timeframe: str
    score: float
    vocab_version: str
    path: str

    def __post_init__(self) -> None:
        if not self.formula or not all(isinstance(t, int) for t in self.formula):
            raise StrategyLoadError(f"formula 非法: {self.formula}")
        size = FORMULA_VOCAB.size
        if any(t < 0 or t >= size for t in self.formula):
            raise StrategyLoadError(
                f"formula 含越界 token: {self.formula} (vocab size={size})"
            )


def load_strategy(path: str | Path) -> StrategySpec:
    """读 JSON → FORMULA_VOCAB.verify → 返回 StrategySpec。失败抛 StrategyLoadError。"""
    p = Path(path)
    if not p.exists():
        raise StrategyLoadError(f"策略文件不存在: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyLoadError(f"策略文件读取失败: {exc}") from exc
    if not isinstance(data, dict):
        raise StrategyLoadError("策略文件格式非法（期望 JSON 对象）")

    formula_raw = data.get("formula") or data.get("formula_tokens")
    if not formula_raw:
        raise StrategyLoadError("策略缺少 formula 字段")
    try:
        formula = [int(t) for t in formula_raw]
    except (TypeError, ValueError) as exc:
        raise StrategyLoadError(f"formula 解析失败: {exc}") from exc

    ver = data.get("vocab_version", "unknown")
    try:
        FORMULA_VOCAB.verify(ver)
    except VocabVersionMismatchError as exc:
        raise StrategyLoadError(
            f"词表版本不匹配: 文件 {ver} != 当前 {FORMULA_VOCAB.version}；需重新训练后加载"
        ) from exc

    return StrategySpec(
        formula=formula,
        symbol=str(data.get("symbol") or ""),
        timeframe=str(data.get("timeframe") or ""),
        score=float(data.get("best_score") or data.get("train_best_score") or 0.0),
        vocab_version=ver,
        path=str(p),
    )
