"""币安数据源（公开 REST，USDT 永续/现货 K 线）——备用源链末位（B2/ADR-0007）。

排链尾原因：HK 合规灰区，仅作行情只读备源（无 API key、不下单）；
AUTOPILOT_FALLBACK_CHAIN env 可随时摘除。
镜像 okx_source 的结构：模块级 _get 可 monkeypatch、纯 urllib。
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

# 主域名被限连时的备选（顺序尝试）
BINANCE_HOSTS = ("https://api.binance.com", "https://api1.binance.com", "https://data-api.binance.vision")

# 项目周期 -> Binance interval
_TF = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
    "1M": "1M",
}

_PRESETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "LINKUSDT"]


def _normalize_symbol(symbol: str) -> str:
    """BTCUSDT / BTC-USDT / BTC-USDT-SWAP / BTC/USDT:USDT → BTCUSDT。"""
    s = (symbol or "").strip().upper()
    if not s:
        raise DataSourceUnavailable("请填写品种")
    if ":" in s:  # ccxt 统一永续形 BTC/USDT:USDT
        s = s.split(":")[0]
    s = s.replace("/", "-")
    if s.endswith("-SWAP"):
        s = s[: -len("-SWAP")]
    s = s.replace("-", "")
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            return s
    raise DataSourceUnavailable(f"无法识别 Binance 品种：{symbol}（示例 BTCUSDT）")


def _get(path: str, params: dict[str, str], retries: int = 3) -> list:
    """逐主机尝试 GET → data（list）。失败重试指数退避。"""
    query = urllib.parse.urlencode(params)
    last_err: Exception | None = None
    for attempt in range(retries):
        host = BINANCE_HOSTS[attempt % len(BINANCE_HOSTS)]  # 重试轮换主机
        url = f"{host}{path}?{query}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AlphaMaster/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(min(8, 2**attempt))
    raise DataSourceUnavailable(f"Binance 请求失败: {last_err}")


class BinanceSource(DataSource):
    kind = "binance"
    label = "Binance（仅行情备用）"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        return (True, "公开行情 · 现货 K 线（仅行情备用 · 链尾）")

    def supported_timeframes(self) -> list[str]:
        return list(_TF.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in _TF:
            raise DataSourceUnavailable(f"Binance 不支持周期 {timeframe}")
        want = min(max(n + 2, 20), 1000)

        with self._lock:
            raw = _get("/api/v3/klines", {"symbol": _normalize_symbol(symbol),
                                          "interval": _TF[timeframe], "limit": str(want)})
        if not raw:
            raise DataSourceUnavailable(f"Binance 无数据：{symbol}")

        bars: list[Bar] = []
        for item in raw:
            # [openTs(ms), o, h, l, c, vol, closeTs, ..., trades, takerBase, takerQuote, closed]
            closed = len(item) >= 12 and str(item[11]) == "True"  # 现货 API 的最后字段恒 "True"
            if drop_forming and not closed:
                continue
            bars.append(
                Bar(
                    ts=int(item[0]) // 1000,
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5] or 0.0),
                )
            )
        bars.sort(key=lambda b: b.ts)
        if not bars:
            raise DataSourceUnavailable(f"Binance 无已收盘 bar：{symbol}")
        return bars[-n:]

    def fetch_ticker(self, symbol: str) -> float:
        data = _get("/api/v3/ticker/price", {"symbol": _normalize_symbol(symbol)}, retries=2)
        if not data or "price" not in data:
            raise DataSourceUnavailable(f"Binance 无 ticker：{symbol}")
        return float(data["price"])
