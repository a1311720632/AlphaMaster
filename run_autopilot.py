"""自动驾驶 CLI 入口（第四步）。

装配 策略 + 行情源(OKXSource) + 执行后端(SimBackend/OKXBackend) + AutopilotEngine，
进入调仓主循环。stdout 输出结构化日志（web 的 AutopilotManager 以 subprocess 方式
拉起本脚本并捕获 stdout 落盘）。

用法：
  # paper 模式（不打交易所，回测核心吃实时 OKX bar）
  python run_autopilot.py --strategy-file strategies/best_BTCUSDT.json --mode paper

  # 冒烟：处理完 2 根新 bar 即退出
  python run_autopilot.py --strategy-file strategies/best_BTCUSDT.json --mode paper --max-bars 2

  # testnet / live（需 ccxt + OKX 凭据；Phase C 完整启用）
  python run_autopilot.py --strategy-file strategies/best_BTCUSDT.json --mode testnet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from autopilot import load_strategy
from autopilot.backends import OKXBackend, SimBackend
from autopilot.engine import AutopilotEngine
from web.data_sources.okx_source import OKXSource

_MODES = ("paper", "testnet", "live")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _build_backend(mode: str, symbol: str) -> object:
    if mode == "paper":
        return SimBackend(
            start_equity=Config.AUTOPILOT_PAPER_START_EQUITY,
            cost_rate=Config.COST_RATE,
        )
    sandbox = mode == "testnet"
    return OKXBackend(
        symbol=symbol,
        sandbox=sandbox,
        api_key=Config.OKX_API_KEY,
        secret=Config.OKX_SECRET_KEY,
        passphrase=Config.OKX_PASSPHRASE,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AlphaMaster 自动驾驶（第四步）")
    p.add_argument("--strategy-file", required=True, help="策略 best_{symbol}.json 路径")
    p.add_argument(
        "--mode", default=Config.AUTOPILOT_MODE, choices=_MODES, help="paper/testnet/live"
    )
    p.add_argument("--symbol", default=None, help="覆盖策略内品种（默认用策略文件里的）")
    p.add_argument("--timeframe", default=None, help="覆盖策略内周期（默认用策略文件里的）")
    p.add_argument("--exchange", default=Config.AUTOPILOT_EXCHANGE, help="交易所（v1=okx）")
    p.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="处理完这么多根新 bar 后退出（冒烟测试用；不填=永久运行）",
    )
    args = p.parse_args(argv)

    # 1. 策略
    try:
        strategy = load_strategy(args.strategy_file)
    except Exception as exc:  # noqa: BLE001
        _log(f"[autopilot] 策略加载失败: {exc}")
        return 2

    # 命令行覆盖品种/周期（否则用策略文件里的）
    symbol = args.symbol or strategy.symbol
    timeframe = args.timeframe or strategy.timeframe or "1h"
    # 同步回 strategy 供 engine 用其 timeframe 决定 cadence
    from autopilot.strategy_loader import StrategySpec

    strategy = StrategySpec(
        formula=strategy.formula,
        symbol=symbol,
        timeframe=timeframe,
        score=strategy.score,
        vocab_version=strategy.vocab_version,
        path=strategy.path,
    )

    _log(
        f"[autopilot] 策略就绪: symbol={symbol} tf={timeframe} "
        f"vocab={strategy.vocab_version} score={strategy.score:.4f} "
        f"formula_len={len(strategy.formula)}"
    )

    # 2. 行情源
    if args.exchange != "okx":
        _log(f"[autopilot] 暂不支持交易所 {args.exchange}（v1 仅 okx）")
        return 2
    datasource = OKXSource()

    # 3. 执行后端
    try:
        backend = _build_backend(args.mode, symbol)
    except Exception as exc:  # noqa: BLE001
        _log(f"[autopilot] 后端初始化失败: {exc}")
        return 2

    # 4. 引擎
    engine = AutopilotEngine(
        strategy=strategy,
        datasource=datasource,
        backend=backend,
        lookback_bars=Config.AUTOPILOT_LOOKBACK_BARS,
        breaker_max_drawdown_pct=Config.AUTOPILOT_BREAKER_MAX_DRAWDOWN_PCT,
        breaker_max_bars_stale=Config.AUTOPILOT_BREAKER_MAX_BARS_STALE,
        min_notional_delta=Config.AUTOPILOT_MIN_NOTIONAL_DELTA,
        state_path=Config.AUTOPILOT_STATE_FILE,
        stop_signal_paths=[Config.AUTOPILOT_STOP_SIGNAL, Config.STOP_SIGNAL],
        max_bars=args.max_bars,
        log=_log,
    )
    reason = engine.run_forever()
    clean = reason.startswith("STOP_SIGNAL") or reason.startswith("max_bars") or reason == "stopped"
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
