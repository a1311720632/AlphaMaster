"""执行后端：paper / testnet / live 三模式共享同一接口。

- SimBackend (paper):    不打交易所。close-to-close 模拟成交 + 追踪权益。
- OKXBackend (testnet/live): ccxt 真实下单/持仓/余额。骨架在此，完整实现见 Phase C
  （docs/adr/0004：OKX via ccxt，后端交易所无关）。

奇偶性：后端只把 engine 算出的 delta 落地，**不参与**信号/仓位计算。
ADR-0006：fetch_position_notional 以交易所为准，使连续调仓靠下根 bar 的 delta 自愈。
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ccxt 可选导入（镜像 config.py 对 MetaTrader5 的 stub 套路）：CI 无 ccxt 时
# OKXBackend 构造抛清晰错误，SimBackend 与整个导入图不受影响。
try:  # pragma: no cover - 环境相关
    import ccxt  # type: ignore
    _CCXT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ccxt = None  # type: ignore
    _CCXT_AVAILABLE = False


@dataclass
class OrderResult:
    """下单结果。filled_notional = 实际成交的带方向名义（正多/负空）。"""

    ok: bool
    filled_notional: float = 0.0
    message: str = ""


class ExecutionBackend(ABC):
    """三模式执行后端统一接口。"""

    mode: str = ""

    @abstractmethod
    def fetch_equity(self) -> float:
        """账户权益。paper=追踪值；ccxt=fetch_balance 的 USDT 总权益。"""

    @abstractmethod
    def fetch_position_notional(self, symbol: str) -> float:
        """当前持仓的带方向名义（多+ / 空-）。ADR-0006：以交易所实际持仓为准。"""

    @abstractmethod
    def place_delta_order(self, symbol: str, delta_notional: float) -> OrderResult:
        """市价单把名义持仓推进 delta_notional（正=加多/减空，负=反向）。"""

    @abstractmethod
    def flatten_all(self, symbol: str) -> OrderResult:
        """全平该品种（运营熔断触发时调用）。"""

    def mark_to_market(self, prev_close: float, cur_close: float) -> float:
        """按上一期持仓更新内部权益。

        SimBackend 实现以推进模拟权益；交易所后端 no-op（由交易所记账，每 bar
        直接 fetch_equity 读真实值）。返回本期实现盈亏。
        """
        return 0.0

    def close(self) -> None:  # noqa: B027 - 可选
        """释放资源（如 ccxt 连接）。"""


class SimBackend(ExecutionBackend):
    """paper 模式：本地模拟。

    不打交易所。mark-to-market 用 close-to-close 实现收益（**仅展示**，非回测
    target_ret 口径——paper 的奇偶职责是信号序列与回测一致，不是盈亏口径）。
    成本按 |delta| × cost_rate 从权益扣除，与回测 turnover × cost_rate 同口径。
    """

    mode = "paper"

    def __init__(self, start_equity: float, cost_rate: float) -> None:
        self._equity = float(start_equity)
        self._cost_rate = float(cost_rate)
        self._position_notional = 0.0  # 带方向名义（多+ / 空-）

    def fetch_equity(self) -> float:
        return self._equity

    def fetch_position_notional(self, symbol: str) -> float:
        return self._position_notional

    def mark_to_market(self, prev_close: float, cur_close: float) -> float:
        """上一期持仓（= 上期 target）× close-to-close 收益 → 计入权益。"""
        if prev_close <= 0 or cur_close <= 0:
            return 0.0
        ret = cur_close / prev_close - 1.0
        pnl = self._position_notional * ret  # 名义 × 收益率 = 盈亏（带方向）
        self._equity += pnl
        return pnl

    def place_delta_order(self, symbol: str, delta_notional: float) -> OrderResult:
        # paper 全额成交（无部分成交/滑点）；成本按名义 × cost_rate 扣权益
        cost = abs(delta_notional) * self._cost_rate
        self._equity -= cost
        self._position_notional += delta_notional
        return OrderResult(ok=True, filled_notional=delta_notional, message="sim fill")

    def flatten_all(self, symbol: str) -> OrderResult:
        return self.place_delta_order(symbol, -self._position_notional)


class OKXBackend(ExecutionBackend):
    """testnet / live 模式：OKX via ccxt（ADR-0004）。

    下单链路：fetch_balance（权益）→ fetch_positions（实际持仓，ADR-0006 以交易所为准）
    → create_order(market)（按 delta 推进名义持仓）。构造时显式设定单向持仓 + 逐仓保证金，
    失败则容忍（用户可能已在账户侧预设）。

    `exchange` 可注入：测试传 mock 交易所；run_autopilot 不传 → 构造真实 ccxt.okx。
    本地无 ccxt/凭据时无法做真实冒烟——需在 OKX demo trading 手动验证（见计划 Phase C）。
    """

    def __init__(
        self,
        symbol: str,
        sandbox: bool,
        api_key: str = "",
        secret: str = "",
        passphrase: str = "",
        exchange: object | None = None,
    ) -> None:
        self._ccxt_symbol = self._to_ccxt_symbol(symbol)
        self._raw_symbol = symbol
        self._sandbox = sandbox
        self.mode = "testnet" if sandbox else "live"

        if exchange is not None:
            self._ex = exchange  # 测试注入
        else:
            if not _CCXT_AVAILABLE:
                raise RuntimeError(
                    "ccxt 未安装；testnet/live 模式需要它。"
                    "运行 python -m pip install ccxt，或使用 --mode paper。"
                )
            self._ex = ccxt.okx({  # type: ignore[union-attr]
                "apiKey": api_key,
                "secret": secret,
                "password": passphrase,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            })
            if sandbox:
                self._ex.set_sandbox_mode(True)  # type: ignore[union-attr]
            self._ex.load_markets()  # type: ignore[union-attr]  # 拉取合约规格（contractSize/limits）

        self._market = self._ex.market(self._ccxt_symbol)  # type: ignore[union-attr]
        self._configure_account()

    # ── 账户配置：单向持仓 + 逐仓保证金（容忍失败）──────────────────────
    def _configure_account(self) -> None:
        if not _CCXT_AVAILABLE:
            return
        settle = (self._market or {}).get("settle", "USDT")
        for method_name, args in (
            ("set_position_mode", (False, settle)),   # False = 单向（one-way）
            ("set_margin_mode", ("isolated", settle)),
        ):
            method = getattr(self._ex, method_name, None)
            if method is None:
                continue
            try:
                method(*args)
            except Exception:  # noqa: BLE001 - 账户可能已预设；失败不阻断
                pass

    @staticmethod
    def _to_ccxt_symbol(symbol: str) -> str:
        """BTCUSDT / BTC-USDT-SWAP / BTC-USDT / BTC/USDT → ccxt 永续 BASE/QUOTE:QUOTE。"""
        raw_up = symbol.upper()
        if ":" in raw_up:  # 已是 ccxt 统一永续形式 → 原样返回（保留 /）
            return raw_up
        s = raw_up.replace("/", "-").replace("_", "-")
        parts = [p for p in s.split("-") if p]
        if parts and parts[-1] == "SWAP":  # OKX inst id 的永续后缀
            parts = parts[:-1]
        if len(parts) >= 2:
            base, quote = parts[0], parts[1]
            return f"{base}/{quote}:{quote}"
        # 无分隔符：按计价后缀识别（BTCUSDT → BTC/USDT:USDT）
        raw = parts[0] if parts else s
        for quote in ("USDT", "USDC", "USD"):
            if raw.endswith(quote) and len(raw) > len(quote):
                return f"{raw[: -len(quote)]}/{quote}:{quote}"
        raise ValueError(f"无法识别 OKX 品种: {symbol}")

    # ── 数值辅助 ─────────────────────────────────────────────────────────
    def _contract_size(self) -> float:
        return float((self._market or {}).get("contractSize", 1) or 1)

    def _min_amount(self) -> float:
        limits = (self._market or {}).get("limits", {}) or {}
        return float((limits.get("amount", {}) or {}).get("min") or 0.0)

    def _notional_to_contracts(self, delta_notional: float, price: float) -> float:
        """名义 → 合约数（带符号，遵守精度与最小手）。"""
        if price <= 0 or delta_notional == 0:
            return 0.0
        cs = self._contract_size()
        raw = abs(delta_notional) / (price * cs)
        try:
            raw = float(self._ex.amount_to_precision(self._ccxt_symbol, raw))  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
        raw = abs(raw)
        if raw < self._min_amount():
            return 0.0
        return math.copysign(raw, delta_notional)

    def _current_price(self) -> float:
        try:
            t = self._ex.fetch_ticker(self._ccxt_symbol)  # type: ignore[union-attr]
            return float(t.get("last") or t.get("ask") or t.get("bid") or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    # ── ExecutionBackend 实现 ───────────────────────────────────────────
    def fetch_equity(self) -> float:
        bal = self._ex.fetch_balance()  # type: ignore[union-attr]
        usdt = (bal or {}).get("USDT", {}) or {}
        total = usdt.get("total")
        if total is None:
            total = (usdt.get("free") or 0.0) + (usdt.get("used") or 0.0)
        return float(total or 0.0)

    def fetch_position_notional(self, symbol: str) -> float:
        ccxt_sym = self._to_ccxt_symbol(symbol)
        try:
            positions = self._ex.fetch_positions([ccxt_sym])  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return 0.0
        cs = self._contract_size()
        for p in positions or []:
            if p.get("symbol") != ccxt_sym:
                continue
            contracts = float(p.get("contracts") or 0.0)
            ref = p.get("entryPrice") or p.get("markPrice") or 0.0
            ref = float(ref) or 0.0
            notional = contracts * ref * cs
            return -notional if p.get("side") == "short" else notional
        return 0.0

    def place_delta_order(self, symbol: str, delta_notional: float) -> OrderResult:
        price = self._current_price()
        contracts = self._notional_to_contracts(delta_notional, price)
        if contracts == 0:
            return OrderResult(ok=True, filled_notional=0.0, message="delta 低于最小手")
        side = "buy" if contracts > 0 else "sell"
        try:
            self._ex.create_order(  # type: ignore[union-attr]
                self._ccxt_symbol, "market", side, abs(contracts)
            )
        except Exception as exc:  # noqa: BLE001
            return OrderResult(ok=False, filled_notional=0.0, message=f"下单失败: {exc}")
        cs = self._contract_size()
        return OrderResult(
            ok=True, filled_notional=contracts * price * cs, message="market fill"
        )

    def flatten_all(self, symbol: str) -> OrderResult:
        # reduceOnly 市价全平：反向下等量合约
        notional = self.fetch_position_notional(symbol)
        if abs(notional) < 1e-9:
            return OrderResult(ok=True, filled_notional=0.0, message="已空仓")
        price = self._current_price()
        contracts = self._notional_to_contracts(-notional, price)  # 反向
        if contracts == 0:
            return OrderResult(ok=False, message="平仓量低于最小手")
        side = "buy" if contracts > 0 else "sell"
        try:
            self._ex.create_order(  # type: ignore[union-attr]
                self._ccxt_symbol, "market", side, abs(contracts),
                None, {"reduceOnly": True},
            )
        except Exception as exc:  # noqa: BLE001 - reduceOnly 不被支持时退化为普通市价
            try:
                self._ex.create_order(self._ccxt_symbol, "market", side, abs(contracts))  # type: ignore[union-attr]
            except Exception as exc2:  # noqa: BLE001
                return OrderResult(ok=False, message=f"平仓失败: {exc2}")
        return OrderResult(ok=True, filled_notional=-notional, message="flatten")

    def close(self) -> None:  # noqa: B027
        pass
