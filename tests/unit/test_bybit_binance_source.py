"""Bybit / Binance 备源测试（B2/ADR-0007）：monkeypatch 模块 _get，全程不触网。

模式照抄 test_okx_source_ticker.py。
"""
from __future__ import annotations

import pytest

from web.data_sources.base import DataSourceUnavailable
from web.data_sources import bybit_source as bys
from web.data_sources import binance_source as bns
from web.data_sources.bybit_source import BybitSource, _normalize_symbol as by_norm
from web.data_sources.binance_source import BinanceSource, _normalize_symbol as bn_norm


# ── Bybit ────────────────────────────────────────────────────────────
def test_bybit_symbol_normalize():
    assert by_norm("BTCUSDT") == "BTCUSDT"
    assert by_norm("BTC-USDT") == "BTCUSDT"
    assert by_norm("BTC-USDT-SWAP") == "BTCUSDT"
    with pytest.raises(DataSourceUnavailable):
        by_norm("")
    with pytest.raises(DataSourceUnavailable):
        by_norm("WHATISTHIS")


def _by_kline(start_ms: int, o=100.0, c=101.0, vol=10.0):
    # Bybit item: [startTs, open, high, low, close, volume, turnover]（字符串）
    return [str(start_ms), str(o), str(max(o, c)), str(min(o, c)), str(c), str(vol), "0"]


def test_bybit_fetch_bars_sorted_and_forming_dropped(monkeypatch):
    """降序入参 → 升序出参；未收盘（收盘时刻在未来）被剔除。"""
    now_s = 1_700_000_000
    now_ms = now_s * 1000
    # 两根已收盘（3600s 窗）+ 一根未收盘（start = now - 600s → 收盘在未来）
    rows = [
        _by_kline(now_ms - 600_000, 102.0, 103.0, vol=3.0),   # forming
        _by_kline(now_ms - 2 * 3600_000, 100.0, 101.0, vol=1.0),
        _by_kline(now_ms - 1 * 3600_000, 101.0, 102.0, vol=2.0),
    ]
    monkeypatch.setattr(bys, "_get", lambda path, params, retries=3: {"list": rows})
    monkeypatch.setattr(bys.time, "time", lambda: float(now_s))
    bars = BybitSource().fetch_bars("BTCUSDT", "1h", 5)
    assert [b.ts for b in bars] == sorted(b.ts for b in bars)
    assert len(bars) == 2  # forming 被剔
    assert bars[-1].close == pytest.approx(102.0)
    assert bars[-1].volume == pytest.approx(2.0)


def test_bybit_fetch_bars_empty_raises(monkeypatch):
    monkeypatch.setattr(bys, "_get", lambda path, params, retries=3: {"list": []})
    with pytest.raises(DataSourceUnavailable):
        BybitSource().fetch_bars("BTCUSDT", "1h", 5)


def test_bybit_fetch_ticker(monkeypatch):
    seen = {}

    def fake(path, params, retries=3):
        seen["symbol"] = params.get("symbol")
        return {"list": [{"lastPrice": "50000.5"}]}

    monkeypatch.setattr(bys, "_get", fake)
    assert BybitSource().fetch_ticker("BTC-USDT") == pytest.approx(50000.5)
    assert seen["symbol"] == "BTCUSDT"


def test_bybit_api_error_raises_unavailable(monkeypatch):
    """_get 抛 DataSourceUnavailable（其重试耗尽后的契约）→ fetch_bars 透传。"""

    def boom(path, params, retries=3):
        raise DataSourceUnavailable("Bybit 请求失败: net down")

    monkeypatch.setattr(bys, "_get", boom)
    with pytest.raises(DataSourceUnavailable):
        BybitSource().fetch_bars("BTCUSDT", "1h", 5)


# ── Binance ──────────────────────────────────────────────────────────
def test_binance_symbol_normalize():
    assert bn_norm("BTCUSDT") == "BTCUSDT"
    assert bn_norm("BTC-USDT-SWAP") == "BTCUSDT"
    assert bn_norm("BTC/USDT:USDT") == "BTCUSDT"


def _bn_kline(start_ms: int, o=100.0, c=101.0, vol=10.0, closed=True):
    # Binance item: [openTs,o,h,l,c,vol,closeTs,qquote,trades,tb,tq,"True"]
    return [start_ms, str(o), str(max(o, c)), str(min(o, c)), str(c), str(vol),
            start_ms + 3_599_000, "0", 100, "0", "0", str(closed)]


def test_binance_fetch_bars_drop_forming(monkeypatch):
    now_s = 1_700_000_000
    rows = [
        _bn_kline((now_s - 600) * 1000, 102.0, 103.0, closed=False),  # forming（API 无 closed=False，防御）
        _bn_kline((now_s - 7200) * 1000, 100.0, 101.0),
        _bn_kline((now_s - 3600) * 1000, 101.0, 102.0),
    ]
    monkeypatch.setattr(bns, "_get", lambda path, params, retries=3: rows)
    bars = BinanceSource().fetch_bars("BTCUSDT", "1h", 5)
    assert [b.ts for b in bars] == sorted(b.ts for b in bars)
    assert len(bars) == 2
    assert bars[-1].close == pytest.approx(102.0)


def test_binance_fetch_ticker(monkeypatch):
    monkeypatch.setattr(bns, "_get", lambda path, params, retries=2: {"price": "61000.25"})
    assert BinanceSource().fetch_ticker("BTCUSDT") == pytest.approx(61000.25)
