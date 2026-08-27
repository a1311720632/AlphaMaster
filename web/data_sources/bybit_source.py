"""Bybit 数据源（公开 REST v5，USDT 永续 linear）——备用源链第二位（B2/ADR-0007）。

镜像 okx_source 的结构：模块级 _get 可 monkeypatch、纯 urllib、无第三方依赖。
仅作行情备源——执行后端只有 OKXBackend，本源永不下单。
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

BYBIT_BASE = "https://api.bybit.com"

# 项目周期 -> Bybit interval（linear kline）
_TF = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

# interval -> 秒（drop_forming 判定用：Bybit 无 confirm 字段，按时间窗推断）
_TF_SECONDS = {
    "1": 60, "5": 300, "15": 900, "30": 1800,
    "60": 3600, "240": 14400, "D": 86400, "W": 604800, "M": 2592000,
}

_PRESETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "LINKUSDT"]


def _normalize_symbol(symbol: str) -> str:
    """BTCUSDT / BTC-USDT / BTC-USDT-SWAP / BTC/USDT:USDT → BTCUSDT（linear 原生形）。"""
    s = (symbol or "").strip().upper()
    if not s:
        raise DataSourceUnavailable("请填写品种")
    s = s.replace("/", "-")
    if s.endswith("-SWAP"):
        s = s[: -len("-SWAP")]
    s = s.replace("-", "")
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            return s
    raise DataSourceUnavailable(f"无法识别 Bybit 品种：{symbol}（示例 BTCUSDT）")


def _get(path: str, params: dict[str, str], retries: int = 3) -> dict:
    """GET {BYBIT_BASE}{path}?{params} → result 字段（dict）。失败重试指数退避。"""
    query = urllib.parse.urlencode(params)
    url = f"{BYBIT_BASE}{path}?{query}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AlphaMaster/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("retCode") != 0:
                raise RuntimeError(f"Bybit API {body.get('retCode')}: {body.get('retMsg')}")
            return body.get("result") or {}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(min(8, 2**attempt))
    raise DataSourceUnavailable(f"Bybit 请求失败: {last_err}")


class BybitSource(DataSource):
    kind = "bybit"
    label = "Bybit"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        return (True, "公开行情 · USDT 永续（备用源）")

    def supported_timeframes(self) -> list[str]:
        return list(_TF.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in _TF:
            raise DataSourceUnavailable(f"Bybit 不支持周期 {timeframe}")
        interval = _TF[timeframe]
        want = min(max(n + 2, 20), 1000)  # Bybit kline 上限 1000

        with self._lock:
            result = _get(
                "/v5/market/kline",
                {"category": "linear", "symbol": _normalize_symbol(symbol),
                 "interval": interval, "limit": str(want)},
            )
        raw = result.get("list") or []
        if not raw:
            raise DataSourceUnavailable(f"Bybit 无数据：{symbol}")

        interval_s = _TF_SECONDS.get(interval, 3600)
        now_ms = int(time.time() * 1000)
        bars: list[Bar] = []
        for item in raw:
            # [startTs(ms), open, high, low, close, volume, turnover]（字符串，降序）
            start_ms = int(item[0])
            if drop_forming and start_ms + interval_s * 1000 > now_ms:
                continue  # 该 bar 的收盘时刻在未来 → 未收盘，剔除
            bars.append(
                Bar(
                    ts=start_ms // 1000,
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5] or 0.0),
                )
            )
        bars.sort(key=lambda b: b.ts)
        if not bars:
            raise DataSourceUnavailable(f"Bybit 无已收盘 bar：{symbol}")
        return bars[-n:]

    def fetch_ticker(self, symbol: str) -> float:
        result = _get(
            "/v5/market/tickers",
            {"category": "linear", "symbol": _normalize_symbol(symbol)},
            retries=2,
        )
        rows = result.get("list") or []
        if not rows:
            raise DataSourceUnavailable(f"Bybit 无 ticker：{symbol}")
        last = rows[0].get("lastPrice")
        if last is None:
            raise DataSourceUnavailable(f"Bybit ticker 缺 lastPrice：{symbol}")
        return float(last)
