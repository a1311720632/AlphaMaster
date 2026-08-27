"""执行后端：paper / testnet / live 三模式共享同一接口。

- SimBackend (paper):    不打交易所。close-to-close 模拟成交 + 追踪权益。
- OKXBackend (testnet/live): ccxt 真实下单/持仓/余额。骨架在此，完整实现见 Phase C
  （docs/adr/0004：OKX via ccxt，后端交易所无关）。

奇偶性：后端只把 engine 算出的 delta 落地，**不参与**信号/仓位计算。
ADR-0006：fetch_position_notional 以交易所为准，使连续调仓靠下根 bar 的 delta 自愈。
"""
from __future__ import annotations

import math
import time
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
    price: float = 0.0   # 成交价（paper=当根收盘，live=ticker）
    fee: float = 0.0     # 手续费（paper=|delta|×cost_rate；live 占位 0.0）


class ExecutionBackend(ABC):
    """三模式执行后端统一接口。"""

    mode: str = ""

    @abstractmethod
    def fetch_equity(self) -> float:
        """账户权益。paper=追踪值；ccxt=fetch_balance 的 USDT 总权益。"""

    @abstractmethod
    def fetch_position_notional(self, symbol: str) -> float:
        """当前持仓的带方向名义（多+ / 空-）。ADR-0006：以交易所实际持仓为准。"""

    def fetch_position_detail(self, symbol: str) -> tuple[float, float, float]:
        """当前持仓明细：(带方向名义, 开仓均价, 未实现盈亏)。

        默认实现只给名义（开仓价/未实现盈亏=0）；SimBackend/OKXBackend 各自 override。
        engine 用它一次性取齐 BarRecord 的 entry/unrealized，避免多次拉取。
        """
        return (self.fetch_position_notional(symbol), 0.0, 0.0)

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

    def restore(
        self, position_notional: float, equity: float, entry_price: float, last_close: float
    ) -> None:
        """从持久化状态恢复后端内存字段。

        仅 paper SimBackend 需要（持仓/权益是内存变量，restart 后须从 state 末根喂回）；
        交易所后端 no-op——持仓/权益以交易所为准（ADR-0006），每根 bar 直接 fetch 读真实值。
        """


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
        self._last_close = 0.0         # 最近 mark_to_market 的收盘价，作 place_delta 的 fill 价
        self._entry_price = 0.0        # 当前持仓移动加权开仓均价（≥0 的 magnitude）

    def fetch_equity(self) -> float:
        return self._equity

    def fetch_position_notional(self, symbol: str) -> float:
        return self._position_notional

    def fetch_position_detail(self, symbol: str) -> tuple[float, float, float]:
        """(带方向名义, 开仓均价, 截至最近收盘的未实现盈亏)。

        未实现盈亏用与 mark_to_market 一致的收益率口径 pos×(close/entry−1)；
        entry/close 未就绪返回 0.0，避免 ZeroDivision。
        """
        if self._entry_price > 0 and self._last_close > 0:
            unreal = self._position_notional * (self._last_close / self._entry_price - 1.0)
        else:
            unreal = 0.0
        return (self._position_notional, self._entry_price, unreal)

    def mark_to_market(self, prev_close: float, cur_close: float) -> float:
        """上一期持仓（= 上期 target）× close-to-close 收益 → 计入权益。"""
        if cur_close > 0:
            self._last_close = float(cur_close)  # 供 place_delta_order 作 fill 价
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
        fill_price = self._last_close
        self._update_entry(delta_notional, fill_price)  # 用更新前的 _position_notional
        self._position_notional += delta_notional
        return OrderResult(
            ok=True, filled_notional=delta_notional, price=fill_price, fee=cost,
            message="sim fill",
        )

    def _update_entry(self, delta_notional: float, fill_price: float) -> None:
        """移动加权更新开仓均价。

        _entry_price 始终是 ≥0 的 magnitude，方向由 _position_notional 的符号体现。
        开仓→fill；同向加仓→名义加权；减仓不穿零→不变；穿零/翻向→fill；平到空仓→0。
        fill_price<=0（首根 tick 未初始化等防御）→ 跳过加权保持原值。
        """
        if fill_price <= 0:
            return
        old = self._position_notional
        new = old + delta_notional
        if new == 0:
            self._entry_price = 0.0
        elif old == 0:
            self._entry_price = fill_price
        elif (new > 0) == (old > 0):
            if (delta_notional > 0) == (old > 0):
                # 同向加仓 → 名义加权
                self._entry_price = (
                    abs(old) * self._entry_price + abs(delta_notional) * fill_price
                ) / (abs(old) + abs(delta_notional))
            # 反向减仓不穿零 → entry 不变
        else:
            # 穿零/翻向 → 新方向以 fill 重新计
            self._entry_price = fill_price

    def flatten_all(self, symbol: str) -> OrderResult:
        return self.place_delta_order(symbol, -self._position_notional)

    def restore(
        self, position_notional: float, equity: float, entry_price: float, last_close: float
    ) -> None:
        """从 state 末根整体覆写内存字段（不走 _update_entry 的移动加权；恢复是整体回填）。"""
        self._position_notional = float(position_notional)
        self._equity = float(equity)
        self._entry_price = float(entry_price)
        self._last_close = float(last_close)


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

    def fetch_position_detail(self, symbol: str) -> tuple[float, float, float]:
        """(带方向名义, 开仓均价, 未实现盈亏)。未实现盈亏读 ccxt unrealisedPnl（两种拼写都容错）。"""
        ccxt_sym = self._to_ccxt_symbol(symbol)
        try:
            positions = self._ex.fetch_positions([ccxt_sym])  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return (0.0, 0.0, 0.0)
        cs = self._contract_size()
        for p in positions or []:
            if p.get("symbol") != ccxt_sym:
                continue
            contracts = float(p.get("contracts") or 0.0)
            entry = float(p.get("entryPrice") or 0.0)
            ref = entry or float(p.get("markPrice") or 0.0) or 0.0
            notional = contracts * ref * cs
            notional = -notional if p.get("side") == "short" else notional
            unreal = float(p.get("unrealizedPnl") or p.get("unrealisedPnl") or 0.0)
            return (notional, entry, unreal)
        return (0.0, 0.0, 0.0)

    def fetch_position_notional(self, symbol: str) -> float:
        return self.fetch_position_detail(symbol)[0]

    def place_delta_order(self, symbol: str, delta_notional: float) -> OrderResult:
        price = self._current_price()
        contracts = self._notional_to_contracts(delta_notional, price)
        if contracts == 0:
            return OrderResult(ok=True, filled_notional=0.0, message="delta 低于最小手")
        side = "buy" if contracts > 0 else "sell"
        return self._market_order_confirmed(side, contracts, price)

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
        result = self._market_order_confirmed(side, contracts, price, reduce_only=True)
        if result.ok:
            result.message = f"flatten {result.message}".strip()
        else:
            result.message = f"平仓{result.message}"
        return result

    # ── 下单 + 成交回执回读（账本真实化）────────────────────────────────
    def _market_order_confirmed(
        self, side: str, contracts: float, fallback_price: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """市价单 → 回读成交回执（fetch_order 轮询），用真实均价/成交张数/手续费落账。

        三态：
          - 回执 closed      → ok=True，filled/price/fee 全真实
          - 回执未完结(部分成交/超时) → ok=True 只记已成交部分，message 注明
          - 回执 canceled/rejected 且零成交 → ok=False
        无回读能力（mock 未实现 fetch_order / create_order 无 id）→ 回落 ticker 估算，
        与旧行为一致；持仓真值仍由 ADR-0006 对账兜底，估算只影响审计精度不影响仓位。
        """
        direction = 1.0 if side == "buy" else -1.0
        params = {"reduceOnly": True} if reduce_only else None
        try:
            res = self._ex.create_order(  # type: ignore[union-attr]
                self._ccxt_symbol, "market", side, abs(contracts), None, params
            )
        except Exception as exc:  # noqa: BLE001 - reduceOnly 不被支持时退化为普通市价
            if not reduce_only:
                return OrderResult(ok=False, filled_notional=0.0, message=f"下单失败: {exc}")
            try:
                res = self._ex.create_order(  # type: ignore[union-attr]
                    self._ccxt_symbol, "market", side, abs(contracts)
                )
            except Exception as exc2:  # noqa: BLE001
                return OrderResult(ok=False, filled_notional=0.0, message=f"下单失败: {exc2}")
        cs = self._contract_size()
        oid = res.get("id") if isinstance(res, dict) else None
        receipt = self._read_fill_receipt(oid)
        if receipt is None:
            # 无回读：按请求量 × ticker 价估算（fee 占位 0）
            return OrderResult(
                ok=True, filled_notional=direction * abs(contracts) * fallback_price * cs,
                price=fallback_price, message="market fill (estimated)",
            )
        status = str(receipt.get("status") or "")
        filled_c = abs(float(receipt.get("filled") or 0.0))
        avg = float(receipt.get("average") or receipt.get("price") or 0.0) or fallback_price
        if filled_c <= 1e-12 and status != "closed":
            return OrderResult(ok=False, filled_notional=0.0,
                               message=f"订单未成交: {status or 'unknown'}")
        note = "" if status == "closed" else f" (部分成交 {status})"
        return OrderResult(
            ok=True,
            filled_notional=direction * filled_c * avg * cs,
            price=avg,
            fee=self._fee_to_quote(receipt, avg),
            message=f"confirmed {status}{note}".strip(),
        )

    def _read_fill_receipt(self, order_id: object) -> dict | None:
        """轮询 fetch_order 拿成交回执。无回读/回读持续失败 → None（回落估算）。

        OKX 市价单通常毫秒级完结；6 次 × 0.4s 只是极端拥堵时的上限。
        """
        if not order_id:
            return None
        fetch = getattr(self._ex, "fetch_order", None)
        if fetch is None:
            return None
        last: dict | None = None
        for _ in range(6):
            try:
                last = fetch(order_id, self._ccxt_symbol)  # type: ignore[operator]
            except Exception:  # noqa: BLE001 - 单次回读失败重试，持续失败回落估算
                time.sleep(0.3)
                continue
            status = str((last or {}).get("status") or "")
            if status in ("closed", "canceled", "rejected", "expired"):
                return last
            time.sleep(0.4)
        return last  # open/partial：调用方按已成交部分记账

    def _fee_to_quote(self, receipt: dict, avg_price: float) -> float:
        """手续费折算成计价币（USDT）：settle 直接用；base 计价 × 均价近似。"""
        fee = receipt.get("fee") or {}
        try:
            cost = abs(float(fee.get("cost") or 0.0))
        except (TypeError, ValueError):
            return 0.0
        cur = str(fee.get("currency") or "").upper()
        settle = str((self._market or {}).get("settle") or "USDT").upper()
        base = str((self._market or {}).get("base") or "").upper()
        if not cur or cur == settle:
            return cost
        if base and cur == base and avg_price > 0:
            return cost * avg_price
        return cost

    def close(self) -> None:  # noqa: B027
        pass
