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


# MT5 风格周期（策略文件里写入，来自 Config.get_timeframe 的键）→ 项目统一规范周期
# （CANON_TIMEFRAMES，与 web 数据源 / autopilot engine 的 _CADENCE / _TF_SECONDS 对齐）。
# 训练侧按 MT5 周期记录（H1/M15/D1…），实盘行情源按规范周期识别（1h/15m/1d…）；加载时
# 归一化一次，下游 engine 与 OKXSource 都拿到能识别的键（否则报「OKX 不支持周期 H1」）。
_MT5_TO_CANON: dict[str, str] = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1w", "MN1": "1M",
}


def normalize_timeframe(tf: str) -> str:
    """把策略文件里的周期字符串归一化为项目规范周期（CANON_TIMEFRAMES）。

    幂等：已是规范周期（如 '1h'）原样返回；MT5 风格（如 'H1' / 'M15'）转 '1h' / '15m'；
    未知值原样返回，交给数据源自行报「不支持周期」。大小写与首尾空白容错。
    """
    if not tf:
        return ""
    t = tf.strip()
    return _MT5_TO_CANON.get(t.upper(), t)


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
        timeframe=normalize_timeframe(str(data.get("timeframe") or "")),
        score=float(data.get("best_score") or data.get("train_best_score") or 0.0),
        vocab_version=ver,
        path=str(p),
    )
