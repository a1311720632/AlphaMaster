"""Load training data from a single Parquet K-line file."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.data_manager import MT5DataManager
from model_core.features import MT5FeatureEngineer

_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
_PARQUET_RE = re.compile(
    rf"^(.+)_({'|'.join(_TIMEFRAMES)})\.parquet$",
    re.IGNORECASE,
)


def parse_parquet_filename(path: str | Path) -> tuple[str, str]:
    """Parse ``{symbol}_{timeframe}.parquet`` e.g. ``AAPL_H1.parquet``."""
    name = Path(path).name
    m = _PARQUET_RE.match(name)
    if not m:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet；当前: {name}"
        )
    return m.group(1), m.group(2).upper()


def inspect_parquet_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.suffix.lower() != ".parquet":
        raise ValueError("请选择 .parquet 文件")

    symbol, timeframe = parse_parquet_filename(p)
    df = pd.read_parquet(p)
    bars = len(df)
    if bars < Config.MIN_BARS:
        raise ValueError(
            f"数据不足: {bars} bars（至少需要 {Config.MIN_BARS}）"
        )

    years = round(bars / 6240, 2) if timeframe == "H1" else None
    return {
        "data_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
        "years_h1": years,
        "valid": True,
        "message": "",
    }


class ParquetDataManager:
    """Single-symbol data manager backed by one Parquet file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.symbol, self.timeframe = parse_parquet_filename(self.file_path)
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None

    def load(self) -> None:
        df = pd.read_parquet(self.file_path)
        if len(df) < Config.MIN_BARS:
            raise ValueError(
                f"数据不足: {len(df)} bars（至少需要 {Config.MIN_BARS}）"
            )

        volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        required = ["time", "open", "high", "low", "close", volume_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet 缺少列: {missing}")

        sub = df[required].copy().rename(columns={volume_col: "volume"})
        sub = sub.sort_values("time")
        sub = sub[~sub["time"].duplicated(keep="last")]

        rows = {field: sub[field].values for field in ["open", "high", "low", "close", "volume"]}
        import numpy as np

        raw: dict[str, torch.Tensor] = {
            field: torch.tensor(np.array([rows[field]]), dtype=torch.float32)
            for field in ["open", "high", "low", "close", "volume"]
        }
        raw["time"] = torch.tensor(
            np.array([sub["time"].values.astype("int64")]),
            dtype=torch.int64,
        )

        self._raw_dict = raw
        self._target_ret = MT5DataManager._compute_target_ret(raw["open"])
        logger.info(
            f"[数据] 已加载 {self.symbol} {self.timeframe}，"
            f"共 {raw['open'].shape[1]} 根K线，文件 {self.file_path.name}"
        )

    @property
    def symbols(self) -> list[str]:
        return [self.symbol]

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        if self._raw_dict is None:
            raise RuntimeError("Call load() first")
        return self._raw_dict

    @property
    def feat_tensor(self) -> torch.Tensor:
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self) -> torch.Tensor:
        if self._target_ret is None:
            raise RuntimeError("Call load() first")
        return self._target_ret

    @property
    def bar_time(self) -> torch.Tensor:
        raw = self.raw_dict
        if "time" in raw:
            return raw["time"][:, -1].long()
        return torch.zeros(1, dtype=torch.int64)
