# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AlphaMaster is a **reinforcement-learning factor-mining pipeline for quantitative trading** (targets MetaTrader 5 / CFDs: forex, metals, US & JP indices). An RL Transformer ("AlphaGPT") generates **interpretable factor formulas as token sequences** (features + operators); a StackVM executes them into a scalar factor; `position = tanh(factor)` gives a continuous position in (-1, 1); a backtester scores the result and the score trains the generator. The same signal logic is shared across training, backtest, and live trading — this parity is the central invariant of the codebase.

Comments and docstrings are largely in Chinese; user-facing output is Chinese.

## Common commands

```bash
# Install (Chinese Windows: prefer `python -m pip` over bare `pip` — avoids GBK encoding errors)
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt   # matplotlib/seaborn for lord/experiment.py

# Web console (recommended entry): http://127.0.0.1:8765
python run_web.py --port 8765
start_web.bat                  # Windows launcher: frees the port, warms APIs, opens browser

# Train from a single Parquet (forces OFFLINE Parquet, never connects to MT5)
python train_file.py --data-file "D:\K线数据\XAUUSD_H1.parquet"
python train_file.py --data-file "D:\K线数据\XAUUSD_H1.parquet" --from-scratch   # clear checkpoints, retrain

# Train via MT5 / local cache (group/single/cross-section modes)
python main.py                 # per-symbol training over Config.TRAINABLE_SYMBOLS (current default)
python main.py --offline       # local cache only, no MT5
python main.py --single XAUUSD | --cross-section | --group risk

# Live MT5 trading loop (requires .env + a trained strategies/best_*.json + MT5 terminal logged in)
python run.py

# Tests
pytest tests/                  # unit/ + smoke/ + property/ (uses Hypothesis for property tests)
pytest tests/unit/test_walk_forward_gap.py -q
pytest tests/unit/test_backtest.py::TestX::test_case -q
```

Parquet files must be named `{symbol}_{timeframe}.parquet` (e.g. `US30.cash_H1.parquet`); `inspect_parquet_file` parses symbol/timeframe from the filename.

## Architecture

### The core paradigm — formulas are token sequences
A strategy is a list of integer tokens (`formula`) saved to `strategies/best_{symbol}.json`. Tokens are split into two disjoint ranges: feature ids `[0, F)` and operator ids `[F, F+O)`, where `F = feature_count`. The `AlphaGPT` Transformer autoregressively samples a token sequence; `StackVM` (`model_core/vm.py`) compiles+executes it over the feature tensor to produce a `[N, T]` factor; `strategy_manager/signal.py::compute_target_positions` applies `tanh` (+ a `MIN_TRADE_EXPOSURE` floor → flat when weak) to get positions. The vocab is registry-driven: `model_core/features.py::FEATURE_REGISTRY` and `model_core/ops.py::OPERATOR_REGISTRY` feed `model_core/vocab.py`, which deterministically derives `VOCAB_VERSION = "v" + sha256(token_names)[:12]`. **Adding/removing/renaming any feature or operator changes `VOCAB_VERSION` and invalidates every saved strategy** (`VocabVersionMismatchError` on load).

### Module map
- **`model_core/`** — the brain. `alphagpt.py` (RL generator + stability monitors), `engine.py` (`AlphaEngine` training loop: `ConstrainedSampler` guarantees 100% legal formulas, walk-forward folds, elite replay pool, entropy-collapse detection + restart, adaptive noise), `vm.py` (`StackVM` + the positive-only / sign-restore operator classification that prevents beta-factor collapse), `backtest.py` (scoring), `evaluator.py`, `island_engine.py` (multi-start, off by default).
- **`data_pipeline/`** — `fetcher.py` (MT5 or offline), `data_manager.py` (multi-symbol, time-aligned), `single_symbol_manager.py`, `parquet_manager.py` (offline single-file), `kline_cache.py`.
- **`strategy_manager/`** — live side. `signal.py` (the **shared** `compute_target_positions`), `runner.py` (`MT5StrategyRunner` main loop, loads `best_*.json`), `risk.py`, `portfolio.py`, `live_signal.py`.
- **`execution/`** — MT5 order placement + price feed (`MetaTrader5` not installed in CI → stubbed).
- **`autopilot/`** — step-4 live trader (crypto, v1 OKX via ccxt). `engine.py` (`AutopilotEngine` main loop; reuses `compute_target_positions_stateless` for backtest parity), `backends.py` (`SimBackend` for paper, `OKXBackend` for testnet/live; ccxt optional-import like MT5), `breakers.py` (drawdown + connectivity circuit breakers + alert monitors), `state.py` (`autopilot_state.json` ledger, separate from MT5 `portfolio_state.json`), `strategy_loader.py` (`FORMULA_VOCAB.verify`-gated). Launched by root `run_autopilot.py` via `web/autopilot_manager.py` subprocess. Decisions in `docs/adr/0001…0006` + `CONTEXT.md`.
- **`web/`** — FastAPI app (`web/app.py`) wrapping `training_manager.py`, `backtest_manager.py`, `realtime_manager.py`, `autopilot_manager.py`; `data_sources/` pluggable feeds (MT5/OKX/TradingView/通达信); `ai_analyze.py`+`ai_providers.py` for optional LLM analysis; `feishu_notify.py` for direction-turn alerts.
- **`backtest_viz/`** — a separate backtest engine + charts/report (distinct from `model_core/backtest.py`).
- Many root-level `*.py` (`backtest_*.py`, `train_*.py`, `analyze_*.py`, `verify_all_strategies.py`, `download_*.py`) are **standalone research/ops scripts**, not the main pipeline.

### Data flow (train → backtest → live)
`data_pipeline` loads OHLCV → `model_core` trains & writes `strategies/best_{symbol}.json` (and `checkpoints/ckpt_{symbol}_step_*.pt`) → `strategy_manager` loads the formula, recomputes the factor on closed bars, and emits `tanh` positions → `execution` places MT5 orders; portfolio state persists to `portfolio_state.json`. `REBALANCE_ON_BAR_CLOSE=True` + `EXECUTION_LAG_BARS=1` keep live timing aligned with the backtest's `target_ret`.

## Critical invariants & gotchas

- **Two config sources, one is authoritative.** Root `config.py::Config` holds global settings (symbols, paths, risk, signal/exit modes). `model_core/config.py::ModelConfig` is the **authoritative** source for training hyperparameters — the `Config.INPUT_DIM/BATCH_SIZE/TRAIN_STEPS/DEVICE` values are explicitly "for reference only" and do not take effect.
- **CPU is forced for training** (`ModelConfig.DEVICE = cpu`). Benchmarked ~2.3× faster than CUDA because per-op tensors are tiny (kernel-launch latency dominates). Do not switch to `cuda` without re-benchmarking.
- **Reward mode is set in code, not config.** `train_file.py`'s `__main__` sets `ModelConfig.REWARD_MODE = "ftmo"` (the others are `"standard"` / `"forex"`). `Config.REWARD_MODE` is a separate field; check which one your entry point actually mutates.
- **Never overwrite a better on-disk strategy.** `train_file.py::_save_strategy` keeps the higher-scoring `best_{symbol}.json`; `--from-scratch` additionally seeds `best_formula`/`best_score` from the existing file as a floor. Respect this when touching save logic.
- **`strategies/` and `checkpoints/` are gitignored** (regenerated by training). The repo ships no strategies — train first. `.env` is gitignored; copy `.env.example`.
- **Stale absolute paths.** Some `tests/unit/*.py` and all `_launch_*.bat` hardcode the repo's old location `D:\cl\MT5_AlphaGPT` (e.g. `sys.path.insert(0, r'd:\cl\MT5_AlphaGPT')` and `cd /d D:\cl\MT5_AlphaGPT`). The repo now lives at `D:\code\AlphaMaster`. These will break when run as-is — fix the path or run via the module path from the repo root instead of trusting the hardcoded insert.
- **`KLINE_CACHE_DIR` defaults to a Chinese path** `D:\K线数据` (override with the `KLINE_CACHE_DIR` env var). MT5 credentials come from `MT5_LOGIN/MT5_PASSWORD/MT5_SERVER` env vars / `.env`.
- **`MetaTrader5` is optional.** `config.py` stubs the MT5 constants when the package is absent, so the import graph and most tests work without a live terminal.

## Testing notes

- `tests/conftest.py` has an **autouse** fixture that redirects `web.settings.SETTINGS_PATH` into `tmp_path` — unit tests can never clobber the real `web_settings.json`. Preserve this isolation when adding web-touching tests.
- Layout: `tests/unit/` (focused logic), `tests/smoke/` (config-field / requirements sanity), `tests/property/` (Hypothesis-based invariants for backtest, features, ops, risk, portfolio, runner). Prefer the property suite when verifying numeric/signal invariants.
- MT5-dependent paths are stubbed; tests should not require a broker connection.
