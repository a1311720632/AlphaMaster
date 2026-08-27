"""OKXBackend 单元测试（mock 交易所，离线）。

验证：品种映射、权益/持仓读取方向、delta→合约→下单的换算与方向、reduceOnly 全平、
最小手过滤。真实 OKX demo 冒烟见计划 Phase C（需凭据，手动）。
"""
from __future__ import annotations

import pytest

from autopilot.backends import OKXBackend


class FakeExchange:
    """最小 mock：记录 create_order，返回罐装 balance/positions/ticker。"""

    def __init__(
        self,
        *,
        equity: float = 1000.0,
        position_contracts: float = 0.0,
        position_side: str | None = None,
        entry_price: float = 100.0,
        last_price: float = 100.0,
        contract_size: float = 1.0,
        min_amount: float = 1.0,
        unrealised: float = 50.0,
    ) -> None:
        self._equity = equity
        self._contracts = position_contracts
        self._side = position_side
        self._entry = entry_price
        self._last = last_price
        self._cs = contract_size
        self._min = min_amount
        self._upnl = unrealised
        self.orders: list[dict] = []
        self.config_calls: list[tuple] = []

    def market(self, symbol):
        return {
            "symbol": symbol, "settle": "USDT", "base": "BTC",
            "contractSize": self._cs,
            "limits": {"amount": {"min": self._min}, "cost": {"min": 0.0}},
            "precision": {"amount": 8},
        }

    def amount_to_precision(self, symbol, amount):
        return str(round(float(amount), 8))

    def set_position_mode(self, hedged, settle=None):
        self.config_calls.append(("set_position_mode", hedged, settle))

    def set_margin_mode(self, margin_type, settle=None):
        self.config_calls.append(("set_margin_mode", margin_type, settle))

    def fetch_balance(self):
        return {"USDT": {"total": self._equity, "free": self._equity, "used": 0.0}}

    def fetch_positions(self, symbols):
        if self._contracts == 0 or self._side is None:
            return []
        return [{
            "symbol": symbols[0], "contracts": self._contracts,
            "entryPrice": self._entry, "side": self._side,
            "unrealisedPnl": self._upnl,
        }]

    def fetch_ticker(self, symbol):
        return {"last": self._last, "ask": self._last, "bid": self._last}

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        order = {"symbol": symbol, "type": type_, "side": side, "amount": amount, "params": params}
        self.orders.append(order)
        return {"id": "fake-1", "status": "closed"}


def _backend(exchange=None, **kw) -> OKXBackend:
    return OKXBackend("BTCUSDT", sandbox=True, exchange=exchange, **kw)


def test_to_ccxt_symbol_mapping():
    assert OKXBackend._to_ccxt_symbol("BTCUSDT") == "BTC/USDT:USDT"
    assert OKXBackend._to_ccxt_symbol("BTC-USDT-SWAP") == "BTC/USDT:USDT"
    assert OKXBackend._to_ccxt_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"


def test_fetch_equity_reads_usdt_total():
    ex = FakeExchange(equity=1234.5)
    be = _backend(ex)
    assert be.fetch_equity() == pytest.approx(1234.5)


def test_fetch_position_notional_direction():
    # 多头：+contracts*entry*cs
    ex = FakeExchange(position_contracts=2.0, position_side="long", entry_price=100.0)
    be = _backend(ex)
    assert be.fetch_position_notional("BTCUSDT") == pytest.approx(200.0)
    # 空头：负
    ex2 = FakeExchange(position_contracts=2.0, position_side="short", entry_price=100.0)
    be2 = _backend(ex2)
    assert be2.fetch_position_notional("BTCUSDT") == pytest.approx(-200.0)
    # 空仓
    ex3 = FakeExchange()
    assert _backend(ex3).fetch_position_notional("BTCUSDT") == 0.0


def test_place_delta_buy_converts_notional_to_contracts():
    ex = FakeExchange(last_price=100.0, contract_size=1.0, min_amount=1.0)
    be = _backend(ex)
    res = be.place_delta_order("BTCUSDT", 500.0)   # 500 / (100×1) = 5 合约
    assert res.ok
    assert ex.orders and ex.orders[0]["side"] == "buy" and ex.orders[0]["amount"] == 5
    assert res.filled_notional == pytest.approx(500.0)


def test_place_delta_sell_direction():
    ex = FakeExchange(last_price=100.0, contract_size=1.0, min_amount=1.0)
    be = _backend(ex)
    be.place_delta_order("BTCUSDT", -300.0)        # 卖 3 合约
    assert ex.orders[0]["side"] == "sell" and ex.orders[0]["amount"] == 3


def test_place_delta_below_min_amount_no_order():
    ex = FakeExchange(last_price=100.0, contract_size=1.0, min_amount=1.0)
    be = _backend(ex)
    res = be.place_delta_order("BTCUSDT", 50.0)    # 0.5 合约 < min 1 → 跳过
    assert res.ok and res.filled_notional == 0.0
    assert ex.orders == []


def test_contract_size_scaling():
    # OKX BTC 永续 contractSize=0.01：delta_notional=500、price=50000 → 500/(50000×0.01)=1 合约
    ex = FakeExchange(last_price=50000.0, contract_size=0.01, min_amount=1.0)
    be = _backend(ex)
    be.place_delta_order("BTCUSDT", 500.0)
    assert ex.orders[0]["amount"] == 1


def test_flatten_all_uses_reduceonly_opposite_side():
    # 持 2 合约多头（+200 名义），全平 → reduceOnly 卖 2
    ex = FakeExchange(position_contracts=2.0, position_side="long", entry_price=100.0,
                      last_price=100.0, contract_size=1.0, min_amount=1.0)
    be = _backend(ex)
    res = be.flatten_all("BTCUSDT")
    assert res.ok
    assert ex.orders[0]["side"] == "sell"
    assert ex.orders[0]["amount"] == 2
    assert (ex.orders[0]["params"] or {}).get("reduceOnly") is True


def test_configure_account_attempts_one_way_isolated():
    ex = FakeExchange()
    _backend(ex)
    methods = {c[0] for c in ex.config_calls}
    assert "set_position_mode" in methods
    assert "set_margin_mode" in methods


def test_fetch_position_detail_returns_triple():
    ex = FakeExchange(position_contracts=2.0, position_side="long",
                      entry_price=100.0, unrealised=50.0)
    be = _backend(ex)
    notional, entry, unreal = be.fetch_position_detail("BTCUSDT")
    assert notional == pytest.approx(200.0)
    assert entry == pytest.approx(100.0)
    assert unreal == pytest.approx(50.0)


def test_fetch_position_detail_short_signs():
    ex = FakeExchange(position_contracts=2.0, position_side="short",
                      entry_price=100.0, unrealised=-30.0)
    be = _backend(ex)
    notional, entry, unreal = be.fetch_position_detail("BTCUSDT")
    assert notional == pytest.approx(-200.0)
    assert entry == pytest.approx(100.0)
    assert unreal == pytest.approx(-30.0)


def test_fetch_position_detail_flat():
    be = _backend(FakeExchange())  # 无持仓
    assert be.fetch_position_detail("BTCUSDT") == (0.0, 0.0, 0.0)


# ── 成交回执回读（下单后 fetch_order 拿真实均价/成交/手续费）──────────────────
class ReceiptExchange(FakeExchange):
    """create_order 返回订单 id，fetch_order 返回罐装成交回执。"""

    def __init__(self, receipt: dict, **kw):
        super().__init__(**kw)
        self._receipt = receipt
        self.fetch_calls: list[tuple] = []

    def fetch_order(self, order_id, symbol):
        self.fetch_calls.append((order_id, symbol))
        return dict(self._receipt)


def test_place_delta_uses_real_fill_receipt():
    """closed 回执：filled_notional/price/fee 全部来自交易所回执，不再估算。"""
    ex = ReceiptExchange(
        {"status": "closed", "average": 101.0, "filled": 5,
         "fee": {"cost": 2.5, "currency": "USDT"}},
        last_price=100.0, contract_size=1.0, min_amount=1.0,
    )
    be = _backend(ex)
    res = be.place_delta_order("BTCUSDT", 500.0)   # 请求买 5 张
    assert res.ok and "confirmed" in res.message
    assert res.filled_notional == pytest.approx(505.0)   # 5 × 101 × 1（真实均价）
    assert res.price == pytest.approx(101.0)
    assert res.fee == pytest.approx(2.5)
    assert ex.fetch_calls and ex.fetch_calls[0][1] == "BTC/USDT:USDT"


def test_place_delta_receipt_sell_signs_negative():
    """卖方向回执：filled_notional 带负号。"""
    ex = ReceiptExchange({"status": "closed", "average": 99.0, "filled": 3},
                         last_price=100.0, contract_size=1.0, min_amount=1.0)
    res = _backend(ex).place_delta_order("BTCUSDT", -300.0)
    assert res.ok and res.filled_notional == pytest.approx(-297.0)
    assert res.price == pytest.approx(99.0)


def test_place_delta_rejected_reports_failure():
    """回执 canceled/rejected 且零成交 → ok=False（engine 走执行熔断计数）。"""
    ex = ReceiptExchange({"status": "canceled", "filled": 0},
                         last_price=100.0, min_amount=1.0)
    res = _backend(ex).place_delta_order("BTCUSDT", 500.0)
    assert not res.ok
    assert "未成交" in res.message


def test_place_delta_partial_fill_records_part():
    """部分成交（open/超时未完结）：ok=True 只记已成交部分并注明。"""
    ex = ReceiptExchange({"status": "open", "average": 100.5, "filled": 2},
                         last_price=100.0, contract_size=1.0, min_amount=1.0)
    import autopilot.backends as bk

    orig = bk.time.sleep
    bk.time.sleep = lambda s: None  # 部分成交会轮询满 6 次，测试免真等
    try:
        res = _backend(ex).place_delta_order("BTCUSDT", 500.0)
    finally:
        bk.time.sleep = orig
    assert res.ok
    assert res.filled_notional == pytest.approx(201.0)   # 只记已成交 2 张
    assert "部分成交" in res.message


def test_fee_converts_base_currency_to_quote():
    """手续费以 base 币（BTC）计价 → × 均价折算成 USDT 口径。"""
    ex = ReceiptExchange(
        {"status": "closed", "average": 101.0, "filled": 5,
         "fee": {"cost": 0.01, "currency": "BTC"}},
        last_price=100.0, contract_size=1.0, min_amount=1.0,
    )
    res = _backend(ex).place_delta_order("BTCUSDT", 500.0)
    assert res.fee == pytest.approx(1.01)   # 0.01 BTC × 101


def test_no_receipt_support_falls_back_to_estimate(monkeypatch):
    """交易所不支持 fetch_order / 持续失败 → 回落 ticker 估算（旧行为）。"""
    import autopilot.backends as bk

    monkeypatch.setattr(bk.time, "sleep", lambda s: None)

    class NoFetch(FakeExchange):
        pass  # 无 fetch_order 属性

    ex = NoFetch(last_price=100.0, contract_size=1.0, min_amount=1.0)
    res = _backend(ex).place_delta_order("BTCUSDT", 500.0)
    assert res.ok and res.filled_notional == pytest.approx(500.0)
    assert "estimated" in res.message

    class AlwaysRaise(FakeExchange):
        def fetch_order(self, oid, symbol):
            raise RuntimeError("net down")

    res2 = _backend(AlwaysRaise(last_price=100.0, min_amount=1.0)).place_delta_order(
        "BTCUSDT", 500.0)
    assert res2.ok and res2.filled_notional == pytest.approx(500.0)
    assert "estimated" in res2.message
