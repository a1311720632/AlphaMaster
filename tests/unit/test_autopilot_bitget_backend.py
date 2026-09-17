"""BitgetBackend 单元测试（mock 交易所，离线）。

验证两点：(1) 继承链——OKXBackend 的执行行为原样可用；(2) 构造形状——真实构造
走 ccxt.bitget + enableDemoTrading（demo trading 服务），而非 set_sandbox_mode。
真实 Bitget demo 冒烟需凭据，手动（同 OKX Phase C 口径）。
"""
from __future__ import annotations

import types

import pytest

import autopilot.backends as bk
from autopilot.backends import BitgetBackend


class FakeBitgetExchange:
    """记录构造参数与 enableDemoTrading 调用；market/账户配置给最小实现。"""

    last_instance: "FakeBitgetExchange | None" = None

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.demo_calls: list[bool] = []
        self.load_markets_calls = 0
        self.config_calls: list[tuple] = []
        FakeBitgetExchange.last_instance = self

    def enableDemoTrading(self, flag=True):
        self.demo_calls.append(bool(flag))

    def load_markets(self, reload=False):
        self.load_markets_calls += 1
        return {}

    def market(self, symbol):
        return {
            "symbol": symbol, "settle": "USDT", "base": "XRP",
            "contractSize": 1.0,
            "limits": {"amount": {"min": 1.0}, "cost": {"min": 0.0}},
            "precision": {"amount": 8},
        }

    def amount_to_precision(self, symbol, amount):
        return str(round(float(amount), 8))

    def set_position_mode(self, hedged, settle=None):
        self.config_calls.append(("set_position_mode", hedged, settle))

    def set_margin_mode(self, margin_type, settle=None):
        self.config_calls.append(("set_margin_mode", margin_type, settle))

    def fetch_ticker(self, symbol):
        return {"last": 2.0, "ask": 2.0, "bid": 2.0}

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self._last_amount = amount
        return {"id": "bg-1", "status": "closed"}

    def fetch_order(self, order_id, symbol):
        # 回执 filled = 上一张单的请求量（单笔场景足够）
        return {"status": "closed", "average": 2.0,
                "filled": getattr(self, "_last_amount", 0.0)}

    def fetch_balance(self):
        return {"USDT": {"total": 5000.0, "free": 5000.0, "used": 0.0}}


def _patch_ccxt(monkeypatch, cls=FakeBitgetExchange):
    """离线伪造 backends.ccxt.bitget（本机可能未装 ccxt）。"""
    monkeypatch.setattr(bk, "_CCXT_AVAILABLE", True)
    monkeypatch.setattr(bk, "ccxt", types.SimpleNamespace(bitget=cls))


def test_bitget_demo_construction_shape(monkeypatch):
    """sandbox=True → ccxt.bitget + enableDemoTrading(True)；凭据三件套透传。"""
    _patch_ccxt(monkeypatch)
    be = BitgetBackend("XRPUSDT", sandbox=True,
                       api_key="k1", secret="s1", passphrase="p1")
    ex = FakeBitgetExchange.last_instance
    assert isinstance(ex, FakeBitgetExchange)
    assert be._ex is ex
    assert ex.demo_calls == [True]                      # demo trading 服务，非 set_sandbox_mode
    assert ex.load_markets_calls == 1                   # 注入父类前已加载（否则 market() 抛错）
    assert ex.config["apiKey"] == "k1"
    assert ex.config["secret"] == "s1"
    assert ex.config["password"] == "p1"               # ccxt 的 passphrase 键名是 password
    assert ex.config["options"]["defaultType"] == "swap"
    assert be.mode == "testnet"
    assert be._ccxt_symbol == "XRP/USDT:USDT"
    # 单向持仓 + 逐仓配置照走（ccxt 统一接口，继承自 OKXBackend）
    kinds = [c[0] for c in ex.config_calls]
    assert "set_position_mode" in kinds and "set_margin_mode" in kinds


def test_bitget_live_skips_demo_flag(monkeypatch):
    """sandbox=False（live）→ 不调 enableDemoTrading。"""
    _patch_ccxt(monkeypatch)
    be = BitgetBackend("XRPUSDT", sandbox=False,
                       api_key="k", secret="s", passphrase="p")
    assert be.mode == "live"
    assert FakeBitgetExchange.last_instance.demo_calls == []


def test_bitget_without_ccxt_raises_clear_error(monkeypatch):
    """无 ccxt（未装）→ 构造抛清晰 RuntimeError（镜像 OKXBackend 行为）。"""
    monkeypatch.setattr(bk, "_CCXT_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="ccxt 未安装"):
        BitgetBackend("XRPUSDT", sandbox=True, api_key="k", secret="s", passphrase="p")


def test_bitget_inherits_execution_path(monkeypatch):
    """继承链回归：delta→合约换算→下单→回执回读 全走 OKXBackend 实现。"""
    _patch_ccxt(monkeypatch)
    be = BitgetBackend("XRPUSDT", sandbox=True, exchange=FakeBitgetExchange())
    res = be.place_delta_order("XRPUSDT", 100.0)      # 100 / (2.0×1) = 50 张
    assert res.ok and "confirmed" in res.message
    assert res.filled_notional == pytest.approx(100.0)  # 50 × 2.0 × 1
    assert res.price == pytest.approx(2.0)
    assert be.fetch_equity() == pytest.approx(5000.0)
