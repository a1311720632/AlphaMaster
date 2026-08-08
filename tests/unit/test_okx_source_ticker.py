"""OKXSource.fetch_ticker 单测（monkeypatch _okx_get，全程不触网）。"""
from __future__ import annotations

import pytest

from web.data_sources import okx_source as oks
from web.data_sources.base import DataSourceUnavailable
from web.data_sources.okx_source import OKXSource


def test_fetch_ticker_reads_last(monkeypatch):
    monkeypatch.setattr(oks, "_okx_get", lambda path, params, retries=3: [{"last": "50000.5"}])
    assert OKXSource().fetch_ticker("BTCUSDT") == pytest.approx(50000.5)


def test_fetch_ticker_normalizes_inst_id(monkeypatch):
    seen = {}

    def fake(path, params, retries=3):
        seen["instId"] = params.get("instId")
        return [{"last": "100.0"}]

    monkeypatch.setattr(oks, "_okx_get", fake)
    OKXSource().fetch_ticker("BTC-USDT-SWAP")
    assert seen["instId"] == "BTC-USDT-SWAP"


def test_fetch_ticker_empty_raises(monkeypatch):
    monkeypatch.setattr(oks, "_okx_get", lambda path, params, retries=3: [])
    with pytest.raises(DataSourceUnavailable):
        OKXSource().fetch_ticker("BTCUSDT")


def test_fetch_ticker_propagates_error(monkeypatch):
    def boom(path, params, retries=3):
        raise DataSourceUnavailable("net down")

    monkeypatch.setattr(oks, "_okx_get", boom)
    with pytest.raises(DataSourceUnavailable):
        OKXSource().fetch_ticker("BTCUSDT")
