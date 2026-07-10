"""
run_backtest.py — 多因子组合回测（含真实点差、夏普、资金曲线）

用法：
    python run_backtest.py              # 多因子模式（每品种独立公式）
    python run_backtest.py --single     # 单公式兼容模式
"""

import json, sys, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from backtest_viz import BacktestEngine, BacktestChart, BacktestReport
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless

# ── 各品种真实点差（从 MT5 实时获取，单边 log cost）──────────────────────────
# 运行时自动刷新；若 MT5 不可用则用保守默认值
DEFAULT_COST_RATES = {
    "EURUSDm": 0.000035,
    "USDJPYm": 0.000031,
    "XAUUSDm": 0.000031,
    "USTECm":  0.000041,
    "US500m":  0.000048,
}
_H1_PER_YEAR = 6240


def get_live_spreads() -> dict[str, float]:
    """从 MT5 实时获取各品种点差，返回 {symbol: log_cost_rate}。"""
    try:
        import MetaTrader5 as mt5
        mt5.initialize()
        costs = {}
        for sym in Config.SYMBOLS:
            tick = mt5.symbol_info_tick(sym)
            if tick and tick.ask > 0:
                mid = (tick.ask + tick.bid) / 2
                costs[sym] = (tick.ask - tick.bid) / mid / 2   # 单边
            else:
                costs[sym] = DEFAULT_COST_RATES.get(sym, 0.0001)
        mt5.shutdown()
        return costs
    except Exception:
        return dict(DEFAULT_COST_RATES)


def decode_formula(tokens: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in tokens)


def load_strategy(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"formula": data, "vocab_version": "legacy", "symbol": None}
    return data


# ── 统计指标 ──────────────────────────────────────────────────────────────────

def calc_sharpe(pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """年化 Sharpe（无风险利率=0）。"""
    m = pnl.mean()
    s = pnl.std(ddof=0)
    if s < 1e-10:
        return 0.0
    return float(m / s * math.sqrt(periods_per_year))


def calc_sortino(pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """年化 Sortino（下行标准差）。"""
    m    = pnl.mean()
    down = pnl[pnl < 0]
    ds   = down.std(ddof=0) if len(down) > 0 else 1e-10
    ds   = max(ds, abs(m), 1e-10)
    return float(np.clip(m / ds * math.sqrt(periods_per_year), -20, 20))


def calc_max_drawdown(cum_pnl: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_pnl)
    return float((peak - cum_pnl).max())


def calc_calmar(cum_pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """Calmar = 年化收益 / 最大回撤。"""
    T      = len(cum_pnl)
    ann    = cum_pnl[-1] * periods_per_year / T if T > 0 else 0
    mdd    = calc_max_drawdown(cum_pnl)
    return float(ann / mdd) if mdd > 1e-8 else 0.0


# ── 资金曲线图 ────────────────────────────────────────────────────────────────

def plot_equity_curves(results_map: dict, output_dir: str, times_arr: np.ndarray | None = None):
    """绘制各品种 + 等权组合的资金曲线。

    Args:
        results_map: {symbol: {"pnl": np.array, "cum_pnl": np.array, ...}}
        output_dir:  输出目录
        times_arr:   时间戳数组（Unix秒），用于 X 轴刻度
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    syms   = list(results_map.keys())
    n_syms = len(syms)

    fig = plt.figure(figsize=(18, 10), dpi=110)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax_eq  = fig.add_subplot(gs[0])   # 资金曲线
    ax_dd  = fig.add_subplot(gs[1], sharex=ax_eq)   # 组合回撤

    colors = ["#1565c0", "#00897b", "#e65100", "#6a1b9a", "#558b2f", "#b71c1c"]

    # 等权组合 PnL
    all_pnls = np.stack([results_map[s]["pnl"] for s in syms], axis=0)
    port_pnl = all_pnls.mean(axis=0)
    port_cum = np.cumsum(port_pnl)

    T = len(port_cum)
    x = np.arange(T)

    # 各品种曲线
    for i, sym in enumerate(syms):
        cum = results_map[sym]["cum_pnl"]
        ax_eq.plot(x, cum, linewidth=0.8, alpha=0.65, color=colors[i % len(colors)],
                   label=f"{sym} ({results_map[sym]['sortino']:+.2f})")

    # 组合曲线（加粗）
    ax_eq.plot(x, port_cum, linewidth=2.2, color="black", label=f"Portfolio ({calc_sortino(port_pnl):+.2f})")
    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.fill_between(x, port_cum, 0, where=port_cum >= 0, alpha=0.06, color="#1565c0")
    ax_eq.fill_between(x, port_cum, 0, where=port_cum < 0,  alpha=0.06, color="#b71c1c")
    ax_eq.set_ylabel("Cumulative Log Return", fontsize=9)
    ax_eq.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(
        f"Multi-Factor Portfolio  |  "
        f"TotalRet={port_cum[-1]:+.3f}  "
        f"Sharpe={calc_sharpe(port_pnl):+.2f}  "
        f"Sortino={calc_sortino(port_pnl):+.2f}  "
        f"MaxDD={calc_max_drawdown(port_cum):.3f}  "
        f"Calmar={calc_calmar(port_cum):+.2f}",
        fontsize=10, pad=6,
    )

    # 组合回撤
    peak = np.maximum.accumulate(port_cum)
    dd   = port_cum - peak
    ax_dd.fill_between(x, dd, 0, alpha=0.5, color="#b71c1c")
    ax_dd.axhline(0, color="gray", linewidth=0.5)
    ax_dd.set_ylabel("Drawdown", fontsize=8)
    ax_dd.grid(alpha=0.2)

    # X 轴时间刻度
    if times_arr is not None and len(times_arr) == T:
        from datetime import datetime, timezone
        step  = max(1, T // 10)
        ticks = x[::step]
        labels = [
            datetime.fromtimestamp(int(times_arr[i]), tz=timezone.utc).strftime("%y-%m-%d")
            for i in range(0, T, step)
        ]
        ax_dd.set_xticks(ticks)
        ax_dd.set_xticklabels(labels[:len(ticks)], fontsize=7, rotation=20)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    path = str(Path(output_dir) / "portfolio_equity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  资金曲线图已保存 → {path}")
    return path


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR  = "backtest_output"
    single_mode = "--single" in sys.argv
    offline_mode = "--offline" in sys.argv

    strategy_file = None
    data_file_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == "--strategy-file" and i + 1 < len(sys.argv):
            strategy_file = sys.argv[i + 1]
        elif arg == "--data-file" and i + 1 < len(sys.argv):
            data_file_arg = sys.argv[i + 1]

    # ── 1. 获取真实点差 ──────────────────────────────────────────────
    if offline_mode:
        print("\n[离线模式] 使用默认点差，不连接 MT5")
        cost_rates = dict(DEFAULT_COST_RATES)
    else:
        print("\n获取实时点差...")
        cost_rates = get_live_spreads()
    for sym, c in cost_rates.items():
        print(f"  {sym:12s}: cost_rate={c:.6f}")

    # ── 2. 加载策略 ─────────────────────────────────────────────────
    symbols_to_load: list[str] | None = None
    print(f"\n{'='*62}")
    if strategy_file:
        data = load_strategy(Path(strategy_file))
        if data is None:
            print(f"[ERROR] 找不到: {strategy_file}"); sys.exit(1)
        sym = data.get("symbol")
        if not sym:
            stem = Path(strategy_file).stem
            if stem.startswith("best_"):
                sym = stem.replace("best_", "", 1)
            elif stem.startswith("strategy_"):
                sym = stem.replace("strategy_", "", 1)
        if not sym:
            print("[ERROR] 策略文件未包含 symbol，且无法从文件名识别"); sys.exit(1)
        symbol_formulas = {sym: data["formula"]}
        symbols_to_load = [sym]
        sc = data.get("best_score", "N/A")
        score_txt = f"{sc:.3f}" if isinstance(sc, (int, float)) else str(sc)
        print(f"  模式: 单策略文件 ({Path(strategy_file).name})")
        print(f"  {sym}: score={score_txt}  {decode_formula(data['formula'])}")
    elif single_mode:
        data = load_strategy(Path(Config.STRATEGY_FILE))
        if data is None:
            print(f"[ERROR] 找不到: {Config.STRATEGY_FILE}"); sys.exit(1)
        symbol_formulas = {sym: data["formula"] for sym in Config.SYMBOLS}
        print("  模式: 单公式（所有品种共用）")
    else:
        symbol_formulas = {}
        for sym in Config.SYMBOLS:
            path = Path("strategies") / f"best_{sym}.json"
            data = load_strategy(path)
            if data is None:
                print(f"  [缺失] {sym}")
                continue
            ver = data.get("vocab_version", "unknown")
            if ver != VOCAB_VERSION:
                print(f"  [跳过] {sym}: vocab_version 不符 ({ver} vs {VOCAB_VERSION})")
                continue
            symbol_formulas[sym] = data["formula"]
            sc = data.get("best_score", "N/A")
            print(f"  {sym}: score={sc:.3f}  {decode_formula(data['formula'])}")

    if not symbol_formulas:
        print("[ERROR] 没有有效策略，请先运行 main.py"); sys.exit(1)
    print(f"{'='*62}\n")

    # ── 3. 加载数据 ───────────────────────────────────────────────────
    if data_file_arg:
        from data_pipeline.parquet_manager import ParquetDataManager

        print(f"正在加载数据（Parquet: {Path(data_file_arg).name}）...")
        pm = ParquetDataManager(data_file_arg)
        pm.load()
        raw_dict = pm.raw_dict
        syms = pm.symbols
    else:
        print(f"正在加载数据{'（离线缓存）' if offline_mode else '（连接 MT5）'}...")
        with MT5DataFetcher(offline=offline_mode) as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load(symbols=symbols_to_load)
            raw_dict = mgr.raw_dict
            syms = mgr.symbols

    T = raw_dict["open"].shape[1]
    times_all = raw_dict.get("time", None)
    print(f"  品种: {syms}  T={T} bars\n")

    # ── 4. 为每品种计算因子 + 回测 ───────────────────────────────
    vm   = StackVM()
    # 因果特征化：_robust_norm 现为滚动窗口实现，传入全量序列是安全的
    # 每个时间步 t 的归一化参数只依赖 [t-w+1..t]，无 look-ahead
    feat = MT5FeatureEngineer.compute_features(raw_dict)  # [N, F, T]，因果安全

    results_map = {}
    backtest_results = []

    for i, sym in enumerate(syms):
        if sym not in symbol_formulas:
            print(f"  [跳过] {sym}（无策略）")
            continue

        formula   = symbol_formulas[sym]
        cost_rate = cost_rates.get(sym, 0.0001)
        feat_i    = feat[i:i+1]
        raw_i     = {k: v[i:i+1] for k, v in raw_dict.items()}

        # 用真实点差运行 BacktestEngine
        engine    = BacktestEngine(formula=formula, cost_rate=cost_rate)
        sym_res   = engine.run(raw_i, feat_i, [sym])
        backtest_results.extend(sym_res)

        r = sym_res[0]
        pnl_arr = r.pnl
        cum_arr = r.cum_pnl
        sharpe  = calc_sharpe(pnl_arr)
        sortino = calc_sortino(pnl_arr)
        mdd     = calc_max_drawdown(cum_arr)
        calmar  = calc_calmar(cum_arr)

        results_map[sym] = {
            "pnl":          pnl_arr,
            "cum_pnl":      cum_arr,
            "total_return": r.total_return,
            "sharpe":       sharpe,
            "sortino":      sortino,
            "max_drawdown": mdd,
            "calmar":       calmar,
            "n_trades":     r.n_trades,
            "win_rate":     r.win_rate,
            "avg_hold":     r.avg_hold_bars,
            "cost_rate":    cost_rate,
        }

    # ── 5. 打印各品种统计 ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  多因子回测报告（真实点差）")
    print(f"{'='*62}")
    header = f"{'品种':12s} {'PnL':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'Trades':>7} {'WinRate':>8} {'AvgH':>6}"
    print(f"  {header}")
    print(f"  {'─'*80}")
    for sym, d in results_map.items():
        print(f"  {sym:12s} "
              f"{d['total_return']:+8.3f} "
              f"{d['sharpe']:+8.3f} "
              f"{d['sortino']:+8.3f} "
              f"{d['max_drawdown']:8.3f} "
              f"{d['calmar']:+8.2f} "
              f"{d['n_trades']:7d} "
              f"{d['win_rate']:8.1%} "
              f"{d['avg_hold']:6.1f}h")

    # 等权组合
    if results_map:
        all_pnls = np.stack([d["pnl"] for d in results_map.values()], axis=0)
        port_pnl = all_pnls.mean(axis=0)
        port_cum = np.cumsum(port_pnl)
        p_sharpe  = calc_sharpe(port_pnl)
        p_sortino = calc_sortino(port_pnl)
        p_mdd     = calc_max_drawdown(port_cum)
        p_calmar  = calc_calmar(port_cum)
        print(f"  {'─'*80}")
        print(f"  {'Portfolio':12s} "
              f"{port_cum[-1]:+8.3f} "
              f"{p_sharpe:+8.3f} "
              f"{p_sortino:+8.3f} "
              f"{p_mdd:8.3f} "
              f"{p_calmar:+8.2f}")
        print(f"\n  正收益品种: {sum(1 for d in results_map.values() if d['total_return']>0)}/{len(results_map)}")
        print(f"  Sharpe>1 品种: {sum(1 for d in results_map.values() if d['sharpe']>1)}/{len(results_map)}")
    print(f"{'='*62}\n")

    # ── 6. 资金曲线图 ─────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if results_map:
        times_np = times_all[0].numpy() if times_all is not None else None
        plot_equity_curves(results_map, OUTPUT_DIR, times_np)

    # ── 7. K 线 + 交易图 ─────────────────────────────────────────────
    # 用 try-except 保护，避免个别品种 NaN 导致整个回测崩溃
    try:
        print("生成 K 线图（最近 120 根）...")
        chart = BacktestChart(max_bars=120)
        chart.plot_all(backtest_results, output_dir=OUTPUT_DIR)
        for r in backtest_results:
            try:
                saved = chart.plot_all_trade_zooms(r, output_dir=OUTPUT_DIR,
                                                   pre_bars=25, post_bars=12, max_trades=8)
                print(f"  {r.symbol}: {len(saved)} 张缩放图")
            except Exception as e:
                print(f"  {r.symbol}: 缩放图生成失败（{e}），跳过")
    except Exception as e:
        print(f"[警告] K 线图生成失败（{e}），跳过画图，不影响回测结果")

    # ── 8. 保存 JSON 报告 ─────────────────────────────────────────────
    report = {
        "mode": "single" if single_mode else "multi_factor",
        "cost_rates": cost_rates,
        "symbols": {},
        "portfolio": {},
    }
    for sym, d in results_map.items():
        formula = symbol_formulas.get(sym, [])
        report["symbols"][sym] = {
            "formula":      formula,
            "readable":     decode_formula(formula),
            "cost_rate":    d["cost_rate"],
            "total_return": round(d["total_return"], 6),
            "sharpe":       round(d["sharpe"], 4),
            "sortino":      round(d["sortino"], 4),
            "max_drawdown": round(d["max_drawdown"], 6),
            "calmar":       round(d["calmar"], 4),
            "n_trades":     d["n_trades"],
            "win_rate":     round(d["win_rate"], 4),
            "avg_hold_bars":round(d["avg_hold"], 2),
        }
    if results_map:
        report["portfolio"] = {
            "total_return": round(float(port_cum[-1]), 6),
            "sharpe":       round(p_sharpe, 4),
            "sortino":      round(p_sortino, 4),
            "max_drawdown": round(p_mdd, 6),
            "calmar":       round(p_calmar, 4),
        }
    rp = f"{OUTPUT_DIR}/multi_factor_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 报告已保存 → {rp}")
    print("完成。\n")


if __name__ == "__main__":
    main()
