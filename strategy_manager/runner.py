"""
strategy_manager/runner.py — MT5 策略主循环控制器（回测对标版）

核心改动（vs 旧版本）：
  1. 信号改为 tanh 连续仓位，与 backtest.py 完全一致（Config.SIGNAL_MODE）
  2. 入场/出场统一为「信号翻转驱动」(_reconcile_positions)
  3. 支持做空，多/空均可反手
  4. K 线收盘触发调仓（REBALANCE_ON_BAR_CLOSE=True），消除时间偏差
  5. EXIT_MODE 控制是否叠加风控层（signal / risk / hybrid）
  6. MAX_OPEN_POSITIONS=None 表示不限制，严格对标回测
"""
from __future__ import annotations

import json
import os
import sys
import time
from numbers import Real

import torch
from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

    class _MT5Stub:
        def shutdown(self) -> None:
            pass

    mt5 = _MT5Stub()  # type: ignore[assignment]

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from execution.price_feed import MT5PriceFeed
from execution.trader import MT5Trader
from model_core.vm import StackVM
from strategy_manager.portfolio import MT5PortfolioManager
from strategy_manager.risk import MT5RiskEngine
from strategy_manager.signal import (
    compute_target_positions,
    target_to_direction,
    reconcile_action,
    HOLD, OPEN_LONG, OPEN_SHORT, CLOSE, REVERSE_TO_LONG, REVERSE_TO_SHORT,
)

_LOOP_INTERVAL: int = 60


class MT5StrategyRunner:
    """同步策略主循环控制器（回测对标版）。

    与旧版本关键差异：
    - 使用 compute_target_positions() 替代 sigmoid+阈值
    - _reconcile_positions() 替代 _scan_for_entries()
    - K 线收盘触发，消除回测-实盘时间偏差
    - 支持做空与反手
    - EXIT_MODE 控制风控叠加
    """

    def __init__(self) -> None:
        from model_core.vocab import VOCAB_VERSION as _CURRENT_VER
        from pathlib import Path as _Path

        # ── 加载策略：支持每品种多公式（信号取平均合并）──────────────
        # symbol_formulas_multi: {sym: [[token,...], [token,...], ...]}
        # 每品种可对应多条公式，信号层取平均后合并为单一仓位方向。
        self.symbol_formulas_multi: dict[str, list[list[int]]] = {}
        # 向后兼容：self.symbol_formulas 保留为"每品种第一条公式"（供旧代码引用）
        self.symbol_formulas: dict[str, list[int]] = {}

        strategies_dir = _Path("strategies")
        archive_dir    = strategies_dir / "archive"

        def _load_formula(path: "_Path") -> "list[int] | None":
            """加载单个策略文件，返回 formula token 列表，失败返回 None。"""
            if not path.exists():
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or "formula" not in data:
                    return None
                ver = data.get("vocab_version", "unknown")
                if ver != _CURRENT_VER:
                    logger.warning(f"[Runner] {path.name}: vocab_version={ver} != {_CURRENT_VER}, skip")
                    return None
                score = data.get("best_score", 0.0)
                if score <= 0.0:
                    logger.warning(f"[Runner] {path.name}: best_score={score:.4f} <= 0, skip (invalid strategy)")
                    return None
                return [int(t) for t in data["formula"]]
            except Exception as exc:
                logger.warning(f"[Runner] {path.name}: 加载失败 {exc}")
                return None

        # ── 为每品种收集所有有效公式（支持多条）────────────────────
        # 查找顺序：
        #   1. strategies/best_{sym}.json（per-symbol 主策略）
        #   2. strategies/best_forex.json（组策略，若品种在 forex 组）
        #   3. strategies/archive/best_forex_*.json（归档的历史最优版本）
        forex_group = ["EURUSD", "USDJPY"]

        for sym in Config.SYMBOLS:
            formulas_for_sym: list[list[int]] = []
            seen: set[str] = set()  # 去重，避免同一公式加两次

            def _add(f: "list[int] | None", label: str) -> None:
                if f is None:
                    return
                key = str(f)
                if key in seen:
                    return
                seen.add(key)
                formulas_for_sym.append(f)
                logger.info(f"[Runner] {sym}: 加载公式 [{label}] {f}")

            # 1. per-symbol 策略文件
            _add(_load_formula(strategies_dir / f"best_{sym}.json"), f"best_{sym}")

            # 2. forex 组共享策略（forex_v2）
            if sym in forex_group:
                _add(_load_formula(strategies_dir / "best_forex.json"), "best_forex(v2)")

            # 3. 归档版本（forex_v1）
            if sym in forex_group:
                _add(_load_formula(archive_dir / "best_forex_20250705_pre_refactor.json"),
                     "archive_forex_v1")

            if formulas_for_sym:
                self.symbol_formulas_multi[sym] = formulas_for_sym
                self.symbol_formulas[sym] = formulas_for_sym[0]  # 向后兼容
            else:
                logger.warning(f"[Runner] {sym}: 无有效公式，该品种将保持空仓")

        # ── 若多因子均无，回退到 best_mt5_strategy.json ──────────────
        if not self.symbol_formulas_multi:
            strategy_path = Config.STRATEGY_FILE
            if not os.path.exists(strategy_path):
                logger.critical(
                    f"未找到任何策略文件（strategies/best_*.json 或 {strategy_path}）。"
                    "请先运行 main.py 训练。"
                )
                sys.exit(1)
            try:
                with open(strategy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.critical(f"加载策略失败: {exc}")
                sys.exit(1)

            if isinstance(data, list):
                logger.critical(
                    "[Runner] 不支持旧格式策略（vocab v2.0 后特征顺序已变）。"
                    "请重新训练（python main.py）。"
                )
                sys.exit(1)
            elif isinstance(data, dict) and "formula" in data:
                ver = data.get("vocab_version", "unknown")
                if ver != _CURRENT_VER:
                    logger.critical(
                        f"[Runner] vocab_version={ver} != {_CURRENT_VER}，请重新训练。"
                    )
                    sys.exit(1)
                formula = [int(t) for t in data["formula"]]
                for sym in Config.SYMBOLS:
                    self.symbol_formulas_multi[sym] = [formula]
                    self.symbol_formulas[sym] = formula
                logger.info("[Runner] 使用单公式回退模式（所有品种共用）")
            else:
                logger.critical("策略文件格式不支持。")
                sys.exit(1)

        # ── 打印加载汇总 ─────────────────────────────────────────────
        for sym, fmls in self.symbol_formulas_multi.items():
            logger.success(f"[Runner] {sym}: {len(fmls)} 条公式已加载")

        # 向后兼容：取第一个品种的第一条公式
        self.formula = next(iter(self.symbol_formulas.values()))

        self.vm        = StackVM()
        self.portfolio = MT5PortfolioManager()
        self.risk      = MT5RiskEngine()
        self.trader    = MT5Trader()

        self._fetcher: MT5DataFetcher | None       = None
        self._data_manager: MT5DataManager | None  = None
        self._last_refresh: float                   = 0.0
        self._last_bar_time: torch.Tensor | None    = None


    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """同步主循环。

        流程：
            1. 连接 MT5 终端
            2. while True:
               a. 检查停止信号
               b. 按需刷新数据
               c. 若 REBALANCE_ON_BAR_CLOSE=True，只在新 K 线收盘时调仓
               d. 同步 MT5 仓位
               e. 调仓（_reconcile_positions）
               f. 若 EXIT_MODE in ('risk','hybrid')，叠加风控监控
               g. 休眠
        """
        logger.info("[Runner] Starting MT5StrategyRunner (backtest-parity mode)...")
        logger.info(f"  SIGNAL_MODE={Config.SIGNAL_MODE}  EXIT_MODE={Config.EXIT_MODE}  "
                    f"MAX_OPEN_POSITIONS={Config.MAX_OPEN_POSITIONS}  "
                    f"REBALANCE_ON_BAR_CLOSE={Config.REBALANCE_ON_BAR_CLOSE}")

        try:
            self.trader.connect()
        except (ConnectionError, RuntimeError) as exc:
            logger.critical(f"[Runner] Cannot connect MT5 trader: {exc}")
            sys.exit(1)

        self._fetcher = MT5DataFetcher()
        try:
            self._fetcher.connect()
        except ConnectionError as exc:
            logger.critical(f"[Runner] Cannot connect MT5 fetcher: {exc}")
            sys.exit(1)

        self._data_manager = MT5DataManager(self._fetcher)
        try:
            self._data_manager.load()
            self._last_refresh = time.time()
        except Exception as exc:
            logger.error(f"[Runner] Initial data load failed: {exc}")

        logger.info("[Runner] MT5 connections established. Entering main loop.")

        while True:
            loop_start = time.time()

            # a. 停止信号
            if self._handle_stop_signal():
                logger.info("[Runner] Stop signal detected. Exiting.")
                break

            # b. 数据刷新
            if time.time() - self._last_refresh >= Config.DATA_REFRESH_INTERVAL:
                try:
                    self._data_manager.reload()
                    self._last_refresh = time.time()
                    logger.info("[Runner] Data refreshed.")
                except Exception as exc:
                    logger.error(f"[Runner] Data reload failed: {exc}")

            # c. 检测新 K 线收盘
            new_bar = True
            if Config.REBALANCE_ON_BAR_CLOSE and self._data_manager is not None:
                try:
                    cur_bar_time = self._data_manager.bar_time   # [N]
                    if (self._last_bar_time is not None and
                            cur_bar_time.shape == self._last_bar_time.shape and
                            (cur_bar_time == self._last_bar_time).all()):
                        new_bar = False
                    else:
                        self._last_bar_time = cur_bar_time.clone()
                except Exception as exc:
                    logger.warning(f"[Runner] bar_time check failed: {exc}")

            # d. 同步 MT5 仓位
            try:
                self.portfolio.sync_from_mt5()
            except Exception as exc:
                logger.warning(f"[Runner] Portfolio sync failed: {exc}")

            if new_bar:
                # e. 计算信号并对账调仓
                targets = self._compute_targets()
                if targets is not None:
                    try:
                        self._reconcile_positions(targets)
                    except Exception as exc:
                        logger.error(f"[Runner] _reconcile_positions raised: {exc}")
            else:
                logger.debug("[Runner] Same bar, skipping rebalance.")

            # f. 风控监控（可选叠加层）
            if Config.EXIT_MODE in ("risk", "hybrid"):
                try:
                    self._monitor_positions()
                except Exception as exc:
                    logger.error(f"[Runner] _monitor_positions raised: {exc}")

            # g. 休眠
            elapsed = time.time() - loop_start
            sleep_t = max(10, _LOOP_INTERVAL - elapsed)
            logger.info(f"[Runner] Cycle {elapsed:.2f}s. Sleep {sleep_t:.2f}s.")
            time.sleep(sleep_t)

    def shutdown(self) -> None:
        logger.info("[Runner] Shutting down...")
        try:
            if self._fetcher is not None:
                self._fetcher.shutdown()
        except Exception as exc:
            logger.warning(f"[Runner] fetcher.shutdown() raised: {exc}")
        mt5.shutdown()
        logger.info("[Runner] Stopped.")


    # ──────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────

    def _handle_stop_signal(self) -> bool:
        stop_path = Config.STOP_SIGNAL
        if not os.path.exists(stop_path):
            return False
        logger.warning(f"[Runner] STOP_SIGNAL detected at '{stop_path}'.")
        try:
            with open(stop_path, "w", encoding="utf-8") as f:
                f.write("STOPPED")
        except OSError as exc:
            logger.warning(f"[Runner] Failed to mark stop signal: {exc}")
        return True

    def _compute_targets(self) -> torch.Tensor | None:
        """为每个品种计算合并后的目标仓位 [-1, +1]，形状 [N]。

        多公式合并逻辑（信号平均）：
          - 每个有效公式独立执行 StackVM，得到最新 bar 的因子值
          - 对所有公式的 tanh(factor) 取算术平均，作为最终仓位信号
          - 若两条公式方向相反，信号相互抵消 → 趋近于 0 → 不开仓
          - 若两条方向一致，信号叠加强化 → 更大仓位比例
        这是标准 alpha 合成方法，安全且符合回测逻辑。
        """
        if self._data_manager is None:
            return None
        try:
            from model_core.features import MT5FeatureEngineer
            raw_dict = self._data_manager.raw_dict
            symbols  = self._data_manager.symbols
            N        = len(symbols)
            feat_all = MT5FeatureEngineer.compute_features(raw_dict)  # [N, F, T]

            targets   = torch.zeros(N, dtype=torch.float32)
            prev_dirs = torch.zeros(N, dtype=torch.float32)
            for i, sym in enumerate(symbols):
                prev_dirs[i] = float(self.portfolio.get_direction(sym))

            for i, sym in enumerate(symbols):
                formulas = self.symbol_formulas_multi.get(sym)
                if not formulas:
                    logger.warning(f"[Runner] {sym}: 无策略公式，保持空仓")
                    continue

                feat_i = feat_all[i:i+1]   # [1, F, T]

                # ── 多公式信号平均 ────────────────────────────────────
                valid_signals: list[float] = []
                for fi, formula in enumerate(formulas):
                    raw_i = self.vm.execute(formula, feat_i)   # [1, T] or None
                    if raw_i is None:
                        logger.warning(f"[Runner] {sym} formula[{fi}]: StackVM 返回 None，跳过")
                        continue
                    latest_val = raw_i[0, -1].item()   # 最新 bar 因子值（标量）
                    signal = float(torch.tanh(torch.tensor(latest_val)).item())
                    valid_signals.append(signal)
                    logger.debug(f"[Runner] {sym} formula[{fi}]: factor={latest_val:+.4f} → tanh={signal:+.4f}")

                if not valid_signals:
                    logger.error(f"[Runner] {sym}: 所有公式均失败，保持空仓")
                    continue

                # 算术平均（两条反向信号相互抵消，同向信号叠加）
                avg_signal = sum(valid_signals) / len(valid_signals)

                # 应用 MIN_TRADE_EXPOSURE 门槛（与回测一致）
                from config import Config as _Cfg
                min_exp = getattr(_Cfg, "MIN_TRADE_EXPOSURE", 0.05)
                if abs(avg_signal) < min_exp:
                    avg_signal = 0.0

                targets[i] = avg_signal

                # 详细日志：显示各公式信号和合并结果
                signals_str = " / ".join(f"{s:+.3f}" for s in valid_signals)
                direction_str = "多" if avg_signal > 0 else ("空" if avg_signal < 0 else "空仓")
                logger.info(
                    f"[Runner] {sym}: 各公式信号=[{signals_str}] → "
                    f"均值={avg_signal:+.4f} ({direction_str})"
                )

            logger.info(
                "[Runner] 最终目标仓位: " +
                " | ".join(
                    f"{sym}={targets[i].item():+.3f}"
                    for i, sym in enumerate(symbols)
                )
            )
            return targets.float()

        except Exception as exc:
            logger.error(f"[Runner] _compute_targets failed: {exc}")
            return None

    def _reconcile_positions(self, targets: torch.Tensor) -> None:
        """对每个品种对账并执行调仓（替代旧版 _scan_for_entries）。

        对账逻辑（严格对标回测）：
            current = portfolio.get_direction(symbol)  # +1 / -1 / 0
            target  = sign(targets[i]) with min exposure band
            action  = reconcile_action(current, target)

        根据 action 执行对应 MT5 订单。
        """
        if self._data_manager is None:
            return

        symbols = self._data_manager.symbols
        n = min(len(symbols), len(targets))

        for idx in range(n):
            symbol       = symbols[idx]
            target_value = float(targets[idx].item())
            target       = target_to_direction(target_value)
            exposure     = abs(target_value) if target != 0 else 0.0
            current      = self.portfolio.get_direction(symbol)
            action       = reconcile_action(current, target)

            if action == HOLD:
                logger.debug(
                    f"[Reconcile] {symbol}: HOLD "
                    f"(dir={current}, target={target_value:+.2f})"
                )
                continue

            # MAX_OPEN_POSITIONS 约束（None 表示不限）
            max_pos = Config.MAX_OPEN_POSITIONS
            if max_pos is not None and action in (OPEN_LONG, OPEN_SHORT):
                if self.portfolio.get_open_count() >= max_pos:
                    logger.info(
                        f"[Reconcile] {symbol}: skip {action} — "
                        f"max_positions={max_pos} reached"
                    )
                    continue

            logger.info(
                f"[Reconcile] {symbol}: {action}  current={current}→target={target} "
                f"raw={target_value:+.2f}"
            )

            pos = self.portfolio.positions.get(symbol)
            ticket = pos.ticket if pos is not None else 0

            # ── 执行动作 ────────────────────────────────────────────
            if action == OPEN_LONG:
                lot = self._calc_lot(symbol, exposure)
                if lot <= 0:
                    logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                    continue
                ok = self.trader.buy(symbol, lot)
                if ok:
                    price = self._get_price(symbol)
                    self.portfolio.add_position(symbol, 0, price, lot, "BUY")

            elif action == OPEN_SHORT:
                lot = self._calc_lot(symbol, exposure)
                if lot <= 0:
                    logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                    continue
                ok = self.trader.open_short(symbol, lot)
                if ok:
                    price = self._get_price(symbol)
                    self.portfolio.add_position(symbol, 0, price, lot, "SELL")

            elif action == CLOSE:
                direction = pos.direction if pos else "BUY"
                ok = self.trader.close_position(symbol, pos.lot_size if pos else lot,
                                                direction, ticket)
                if ok:
                    self.portfolio.close_position(symbol)

            elif action == REVERSE_TO_LONG:
                lot = self._calc_lot(symbol, exposure)
                if lot <= 0:
                    logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                    continue
                # 先平空，再开多
                if pos:
                    ok_close = self.trader.close_position(
                        symbol, pos.lot_size, pos.direction, ticket
                    )
                    if ok_close:
                        self.portfolio.close_position(symbol)
                ok_open = self.trader.buy(symbol, lot)
                if ok_open:
                    price = self._get_price(symbol)
                    self.portfolio.add_position(symbol, 0, price, lot, "BUY")

            elif action == REVERSE_TO_SHORT:
                lot = self._calc_lot(symbol, exposure)
                if lot <= 0:
                    logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                    continue
                # 先平多，再开空
                if pos:
                    ok_close = self.trader.close_position(
                        symbol, pos.lot_size, pos.direction, ticket
                    )
                    if ok_close:
                        self.portfolio.close_position(symbol)
                ok_open = self.trader.open_short(symbol, lot)
                if ok_open:
                    price = self._get_price(symbol)
                    self.portfolio.add_position(symbol, 0, price, lot, "SELL")

    def _monitor_positions(self) -> None:
        """可选风控层（EXIT_MODE='risk' 或 'hybrid'）。

        多头：profit = current/entry - 1
        空头：profit = entry/current - 1（方向相反）
        止损、部分止盈、追踪止损逻辑同旧版，但空头追踪最低价。

        hybrid 模式下仅做紧急熔断（止损），不做部分止盈/追踪止损。
        """
        for symbol, pos in list(self.portfolio.positions.items()):
            tick = MT5PriceFeed.get_tick(symbol)
            if tick is None:
                logger.warning(f"[Monitor] Cannot fetch price for {symbol}.")
                continue

            current_price: float = tick["mid"]
            self.portfolio.update_price(symbol, current_price)

            if pos.entry_price <= 0:
                continue

            if pos.direction == "BUY":
                profit = current_price / pos.entry_price - 1.0
            else:  # SELL（空头）
                profit = pos.entry_price / current_price - 1.0

            # ── 止损（所有模式）────────────────────────────────────
            if profit < Config.STOP_LOSS_PCT:
                logger.warning(
                    f"[Monitor] STOP LOSS: {symbol} {pos.direction} "
                    f"profit={profit:.2%}"
                )
                ok = self.trader.close_position(
                    symbol, pos.lot_size, pos.direction, pos.ticket
                )
                if ok:
                    self.portfolio.close_position(symbol)
                continue

            # hybrid 模式只做止损，跳过下面的止盈/追踪
            if Config.EXIT_MODE == "hybrid":
                continue

            # ── 部分止盈（risk 模式）────────────────────────────────
            if profit > Config.TAKE_PROFIT_PCT and not pos.is_partial_closed:
                half = round(pos.lot_size / 2, 2)
                if half > 0:
                    logger.info(f"[Monitor] Partial TP: {symbol} profit={profit:.2%}")
                    ok = self.trader.close_position(
                        symbol, half, pos.direction, pos.ticket
                    )
                    if ok:
                        pos.is_partial_closed = True
                        self.portfolio.save_state()
                continue

            # ── 追踪止损（risk 模式，多头用最高价，空头用最低价）──
            if profit > Config.TRAILING_ACTIVATION:
                if pos.direction == "BUY" and pos.highest_price > 0:
                    drawdown = (pos.highest_price - current_price) / pos.highest_price
                    if drawdown > Config.TRAILING_DROP:
                        logger.warning(
                            f"[Monitor] TRAILING STOP (long): {symbol} "
                            f"dd={drawdown:.2%}"
                        )
                        ok = self.trader.close_position(
                            symbol, pos.lot_size, pos.direction, pos.ticket
                        )
                        if ok:
                            self.portfolio.close_position(symbol)
                elif pos.direction == "SELL" and pos.lowest_price > 0:
                    # 空头：从最低价反弹超过 TRAILING_DROP 则止损
                    rebound = (current_price - pos.lowest_price) / pos.lowest_price
                    if rebound > Config.TRAILING_DROP:
                        logger.warning(
                            f"[Monitor] TRAILING STOP (short): {symbol} "
                            f"rebound={rebound:.2%}"
                        )
                        ok = self.trader.close_position(
                            symbol, pos.lot_size, pos.direction, pos.ticket
                        )
                        if ok:
                            self.portfolio.close_position(symbol)

    # ──────────────────────────────────────────────────────────────────────
    # 辅助
    # ──────────────────────────────────────────────────────────────────────

    def _calc_lot(self, symbol: str, exposure: float = 1.0) -> float:
        """基于 ATR 波动率目标计算手数，让各品种盈亏金额均衡。

        使用最新 14-bar ATR 作为波动参考，目标每笔 1个ATR波动 = equity × RISK_PER_TRADE。
        exposure 来自连续目标仓位绝对值，用于把 tanh 强弱映射到实际风险金额。
        """
        exposure = max(0.0, min(1.0, float(exposure)))
        if exposure <= 0:
            return 0.0
        account = self.trader.get_account_info()
        if account is None:
            return 0.0
        equity = account["equity"]

        # 从当前数据中取该品种最近 14 根 K 线的 ATR
        atr_price = self._get_atr(symbol)
        if not isinstance(atr_price, Real) or atr_price <= 0:
            # ATR 获取失败：回退到最小手数，避免不交易
            logger.warning(f"[_calc_lot] {symbol}: ATR 获取失败，使用最小手数")
            try:
                import MetaTrader5 as mt5
                info = mt5.symbol_info(symbol)
                return info.volume_min if info else 0.01
            except Exception:
                return 0.01

        max_lot = getattr(Config, "MAX_LOT_PER_TRADE", 0.1)
        base_risk = getattr(self.risk, "risk_per_trade", Config.RISK_PER_TRADE)
        if not isinstance(base_risk, Real) or base_risk <= 0:
            base_risk = Config.RISK_PER_TRADE
        lot = self.risk.calculate_lot_by_atr(
            symbol=symbol,
            equity=equity,
            atr_price=atr_price,
            target_risk_pct=base_risk * exposure,
            max_lot=max_lot,
        )
        return lot

    def _get_atr(self, symbol: str, period: int = 14) -> float | None:
        """从已加载数据中读取该品种最近 period 根 K 线的 ATR。"""
        if self._data_manager is None:
            return None
        try:
            raw   = self._data_manager.raw_dict
            syms  = self._data_manager.symbols
            if symbol not in syms:
                return None
            idx   = syms.index(symbol)
            hi    = raw["high"][idx, -period:].float()
            lo    = raw["low"][idx,  -period:].float()
            cl    = raw["close"][idx, -period:].float()
            # 简化 ATR：high-low 均值（因果，不看前一根收盘）
            atr   = (hi - lo).mean().item()
            return atr
        except Exception as exc:
            logger.warning(f"[_get_atr] {symbol}: {exc}")
            return None

    def _get_price(self, symbol: str) -> float:
        """获取当前中间价，失败返回 0.0。"""
        tick = MT5PriceFeed.get_tick(symbol)
        return tick["mid"] if tick else 0.0
