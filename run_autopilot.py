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


def _build_datasource() -> object:
    """备用源链装配（B2/ADR-0007）：链配置 AUTOPILOT_FALLBACK_CHAIN，默认 okx,bybit,binance。"""
    from web.data_sources.fallback_source import FallbackDataSource

    kinds = [k.strip().lower() for k in Config.AUTOPILOT_FALLBACK_CHAIN.split(",") if k.strip()]
    sources = []
    for kind in kinds:
        try:
            if kind == "okx":
                sources.append(OKXSource())
            elif kind == "bybit":
                from web.data_sources.bybit_source import BybitSource

                sources.append(BybitSource())
            elif kind == "binance":
                from web.data_sources.binance_source import BinanceSource

                sources.append(BinanceSource())
            else:
                _log(f"[autopilot] 未知备源 kind={kind}，跳过")
        except Exception as exc:  # noqa: BLE001 - 单源构造失败不阻断整链
            _log(f"[autopilot] 备源 {kind} 构造失败，从链中摘除: {exc}")
    if not sources:
        _log("[autopilot] 备源链为空，回落单源 OKX")
        return OKXSource()
    if len(sources) == 1:
        return sources[0]
    return FallbackDataSource(sources, max_fails_per_source=Config.AUTOPILOT_FALLBACK_MAX_FAILS)


def _resolve_drawdown_breaker() -> tuple[bool, float]:
    """回撤熔断生效配置 (enabled, 正小数 pct)。真源在 web.settings.drawdown_breaker_config
    （web_settings 显式键 > Config env；env 不可达阈值 = 事实关闭）。包一层便于单测。"""
    from web.settings import drawdown_breaker_config

    return drawdown_breaker_config()


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
    p.add_argument(
        "--state-file",
        default=None,
        help="覆盖 state 文件路径（默认 Config.AUTOPILOT_STATE_FILE）。"
             "D2 离线补算 paper 对照时传独立文件，避免覆盖 testnet 账本/触发归档",
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
    # 同步回 strategy 供 engine 用其 timeframe 决定 cadence
    from autopilot.strategy_loader import StrategySpec, normalize_timeframe

    # --timeframe 容忍 MT5 风格写法（H1/M15…）；策略内周期已在 load_strategy 归一化为规范周期
    timeframe = (
        normalize_timeframe(args.timeframe) if args.timeframe else (strategy.timeframe or "1h")
    )
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

    # 2. 行情源（备用源链 B2/ADR-0007：OKX → Bybit → Binance，全挂才断连熔断）
    if args.exchange != "okx":
        _log(f"[autopilot] 暂不支持交易所 {args.exchange}（v1 仅 okx）")
        return 2
    datasource = _build_datasource()

    # 3. 执行后端
    try:
        backend = _build_backend(args.mode, symbol)
    except Exception as exc:  # noqa: BLE001
        _log(f"[autopilot] 后端初始化失败: {exc}")
        return 2

    # 4. 告警（B4/ADR-0007）：引擎内直调飞书；webhook 未配置时 Alerter 只 log
    from autopilot.alerts import Alerter

    alerter = Alerter(log=_log)

    # 4.5 回撤熔断配置（web 设置 > Config env；本次启动读一次，进程内不再变）
    breaker_enabled, breaker_pct = _resolve_drawdown_breaker()
    _log(
        f"[autopilot] 回撤熔断: {'启用' if breaker_enabled else '关闭'}"
        + (f"（峰值回撤 ≤ {breaker_pct * 100:.2f}% 全平停机）" if breaker_enabled else "")
    )

    # 5. 引擎
    state_path = args.state_file or Config.AUTOPILOT_STATE_FILE
    # D2 隔离：--state-file 补算时账本放 state 文件旁（.ledger.jsonl），
    # 不与生产冷账本混写（否则污染审计与 readiness 门槛计数）
    ledger_file = (
        str(Path(state_path).with_suffix(".ledger.jsonl"))
        if args.state_file
        else None
    )
    engine = AutopilotEngine(
        strategy=strategy,
        datasource=datasource,
        backend=backend,
        lookback_bars=Config.AUTOPILOT_LOOKBACK_BARS,
        breaker_max_drawdown_pct=-breaker_pct,  # 引擎侧约定负数阈值
        breaker_drawdown_enabled=breaker_enabled,
        breaker_max_bars_stale=Config.AUTOPILOT_BREAKER_MAX_BARS_STALE,
        min_notional_delta=Config.AUTOPILOT_MIN_NOTIONAL_DELTA,
        state_path=state_path,
        stop_signal_paths=[Config.AUTOPILOT_STOP_SIGNAL, Config.STOP_SIGNAL],
        max_bars=args.max_bars,
        ledger_dir=Config.AUTOPILOT_LEDGER_DIR or None,
        ledger_file=ledger_file,
        alerter=alerter,
        heartbeat_url=Config.AUTOPILOT_HEARTBEAT_URL,
        heartbeat_max_silent_s=Config.AUTOPILOT_HEARTBEAT_MAX_SILENT_S,
        log=_log,
    )
    reason = engine.run_forever()
    clean = (
        reason.startswith("STOP_SIGNAL")
        or reason.startswith("max_bars")
        or reason == "stopped"
    )
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
