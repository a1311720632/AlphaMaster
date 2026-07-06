"""
live_trade.py — 自动交易启动脚本

使用方法：
    python live_trade.py                    # forex 双因子模式（forex_v1 + forex_v2，默认）
    python live_trade.py --dry-run          # 模拟运行（不下单，只打印信号）
    python live_trade.py --symbols EURUSD   # 只交易指定品种
    python live_trade.py --single           # 使用 best_mt5_strategy.json 单公式（回退）

已启用的有效策略（forex 双因子）：
    forex_v1：strategies/archive/best_forex_20250705_pre_refactor.json
              公式：AROON_OSC_25 → TS_MIN_10 → EMA_20 → CS_SCALE → SIGN → CS_SCALE → EMA_5 → MOMENTUM_5
              回测：8年，Sharpe 0.879，MDD 7.03%，3折WF全正，2x成本仍盈利
    forex_v2：strategies/best_forex.json
              公式：DMI_DIFF_14 → HL_RANGE → TS_MIN_20 → SUB → TS_MIN_20 → TS_MEAN_20 → CS_SCALE → WMA
              回测：8年，Sharpe 0.684，MDD 6.88%，2x成本仍盈利

信号合并规则：
    两条因子信号取算术平均（tanh 压缩后）。
    若两条信号方向相反 → 均值趋近 0 → 不开仓（等待两者达成一致）。
    若两条信号方向一致 → 均值绝对值更大 → 较大仓位比例。

注意：
  - 需要 MT5 终端已登录并允许自动交易
  - 确保 .env 中配置了 MT5_LOGIN / MT5_PASSWORD / MT5_SERVER（若需要登录）
  - 停止方法：在当前目录创建 STOP_SIGNAL 文件，或直接 Ctrl+C
"""

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import Config
from strategy_manager.runner import MT5StrategyRunner
from loguru import logger

# ── 默认只交易 forex 组（唯一有效策略的品种）──────────────────────────────────
_FOREX_SYMBOLS = ["EURUSD", "USDJPY"]


def main():
    # 命令行参数处理
    dry_run = "--dry-run" in sys.argv
    single  = "--single"  in sys.argv
    sym_override = None
    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        sym_override = sys.argv[idx+1:]

    # 默认限定 forex 组品种（index/metals 无有效策略，避免空仓噪声）
    if sym_override:
        Config.SYMBOLS = sym_override
        logger.info(f"[live_trade] 品种覆盖: {Config.SYMBOLS}")
    else:
        Config.SYMBOLS = _FOREX_SYMBOLS
        logger.info(f"[live_trade] 使用默认 forex 品种: {Config.SYMBOLS}")

    if dry_run:
        logger.info("[live_trade] DRY RUN 模式：只打印信号，不下单")

    if single:
        logger.info("[live_trade] 单公式模式：所有品种共用 best_mt5_strategy.json")

    logger.info("=" * 60)
    logger.info("  AlphaGPT 自动交易启动  [forex 双因子模式]")
    logger.info(f"  品种:       {Config.SYMBOLS}")
    logger.info(f"  周期:       H1")
    logger.info(f"  因子策略:   forex_v1 (archive) + forex_v2 (best_forex)")
    logger.info(f"  信号合并:   双因子均值（反向时相互抵消，同向时叠加）")
    logger.info(f"  信号模式:   {Config.SIGNAL_MODE}")
    logger.info(f"  出场模式:   {Config.EXIT_MODE}")
    logger.info("=" * 60)

    runner = MT5StrategyRunner()

    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("[live_trade] 收到 Ctrl+C，正在停止...")
    finally:
        runner.shutdown()
        logger.info("[live_trade] 已停止。")


if __name__ == "__main__":
    main()
