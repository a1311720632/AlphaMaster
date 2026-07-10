"""FastAPI application for AlphaMaster training UI."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.parquet_manager import inspect_parquet_file
from model_core.config import ModelConfig
from web.file_dialog import pick_parquet_file, pick_strategy_file
from web.progress import get_symbol_progress, get_strategy_for_export, list_strategies
from web.server_log import (
    debug_snapshot,
    get_logger,
    is_debug_mode,
    log_error,
    set_debug_mode,
    setup_logging,
)
from web.settings import load_settings, save_settings
from web.strategy_file import inspect_strategy_file, resolve_strategy_file
from web.training_manager import training_manager
from web.training_package import build_training_export_zip, import_training_package
from web.backtest_manager import backtest_manager

STATIC_DIR = Path(__file__).resolve().parent / "static"
BACKTEST_OUTPUT_DIR = ROOT / "backtest_output"

setup_logging()
logger = get_logger()

app = FastAPI(title="AlphaMaster Training", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartTrainingRequest(BaseModel):
    data_file: str


class ClientLogRequest(BaseModel):
    level: str = "error"
    message: str
    context: dict[str, Any] | None = None


class SettingsRequest(BaseModel):
    last_data_file: str | None = None
    last_strategy_file: str | None = None
    debug_mode: bool | None = None


class StartBacktestRequest(BaseModel):
    strategy_file: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_error(f"{request.method} {request.url.path} unhandled", exc)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    if is_debug_mode():
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    if response.status_code >= 400:
        log_error(f"{request.method} {request.url.path} -> HTTP {response.status_code}")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_error(f"{request.method} {request.url.path} HTTP {exc.status_code}: {exc.detail}")
    detail = exc.detail
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_error(f"{request.method} {request.url.path} crashed", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()},
    )


def _inspect_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_parquet_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _browse_data_file() -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening native file picker")
    try:
        path = pick_parquet_file()
    except Exception as exc:
        log_error("File picker failed", exc)
        raise HTTPException(500, f"文件选择失败: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("File picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected file: %s", path)
    info = _inspect_or_http(path)
    save_settings({"last_data_file": info["data_file"]})
    return {"ok": True, "cancelled": False, **info}


def _strategy_context() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    train_symbol = None
    if data_file:
        try:
            train_symbol = inspect_parquet_file(data_file).get("symbol")
        except Exception:
            pass

    resolved = resolve_strategy_file(
        settings.get("last_strategy_file") or "",
        train_symbol,
    )
    strategy_info = None
    if resolved:
        try:
            strategy_info = inspect_strategy_file(resolved)
        except Exception as e:
            strategy_info = {
                "strategy_file": resolved,
                "valid": False,
                "message": str(e),
            }
    return {
        "last_strategy_file": resolved,
        "strategy_file": strategy_info,
        "train_symbol": train_symbol,
    }


def _browse_strategy_file() -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening strategy file picker")
    try:
        path = pick_strategy_file()
    except Exception as exc:
        log_error("Strategy file picker failed", exc)
        raise HTTPException(500, f"文件选择失败: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("Strategy file picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected strategy: %s", path)
    info = _inspect_strategy_or_http(path)
    save_settings({"last_strategy_file": info["strategy_file"]})
    return {"ok": True, "cancelled": False, **info}


def _inspect_strategy_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_strategy_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.1.0"}


@app.get("/api/routes")
def api_routes() -> dict[str, Any]:
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path and methods:
            routes.append({"path": path, "methods": sorted(methods)})
    return {"routes": sorted(routes, key=lambda r: r["path"])}


@app.get("/api/debug/logs")
def api_debug_logs(lines: int = 200) -> dict[str, Any]:
    return debug_snapshot(lines)


@app.post("/api/debug/client-log")
def api_client_log(req: ClientLogRequest) -> dict[str, bool]:
    msg = req.message
    if req.context:
        msg = f"{msg} | context={req.context}"
    if req.level == "error":
        log_error(f"[client] {msg}")
    elif is_debug_mode():
        logger.info("[client] %s", msg)
    return {"ok": True}


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return load_settings()


@app.put("/api/settings")
def api_put_settings(req: SettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.last_data_file is not None:
        payload["last_data_file"] = req.last_data_file
    if req.last_strategy_file is not None:
        payload["last_strategy_file"] = req.last_strategy_file
    if req.debug_mode is not None:
        payload["debug_mode"] = req.debug_mode
    saved = save_settings(payload)
    if req.debug_mode is not None:
        set_debug_mode(req.debug_mode)
    return {"ok": True, **saved}


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    if data_file:
        try:
            file_info = inspect_parquet_file(data_file)
        except Exception as e:
            file_info = {
                "data_file": data_file,
                "valid": False,
                "message": str(e),
            }
    snap = debug_snapshot(1)
    strat_ctx = _strategy_context()
    return {
        "train_steps": ModelConfig.TRAIN_STEPS,
        "batch_size": ModelConfig.BATCH_SIZE,
        "reward_mode": ModelConfig.REWARD_MODE,
        "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
        "device": str(ModelConfig.DEVICE),
        "last_data_file": data_file,
        "data_file": file_info,
        "last_strategy_file": strat_ctx["last_strategy_file"],
        "strategy_file": strat_ctx["strategy_file"],
        "debug_mode": load_settings().get("debug_mode", False),
        "server_log": snap["server_log"],
        "error_log": snap["error_log"],
    }


@app.post("/api/data-file/browse")
@app.get("/api/data-file/browse")
def api_browse_data_file() -> dict[str, Any]:
    return _browse_data_file()


@app.post("/api/strategy-file/browse")
@app.get("/api/strategy-file/browse")
def api_browse_strategy_file() -> dict[str, Any]:
    return _browse_strategy_file()


def _progress_with_live_step(symbol: str, active: bool) -> dict[str, Any]:
    p = get_symbol_progress(symbol)
    current_step = p.current_step
    if active:
        live = training_manager.parse_step_from_log()
        if live is not None:
            current_step = max(current_step, live)
    train_steps = p.train_steps
    progress_pct = min(100.0, 100.0 * current_step / train_steps) if train_steps > 0 else 0.0
    return {
        "symbol": p.symbol,
        "current_step": current_step,
        "train_steps": train_steps,
        "progress_pct": round(progress_pct, 1),
        "best_score": p.best_score,
        "formula_decoded": p.formula_decoded,
        "status": p.status,
        "history": p.history,
        "has_checkpoint": bool(p.checkpoint_path),
        "has_strategy": p.has_strategy,
    }


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    progress = None

    if data_file:
        try:
            file_info = inspect_parquet_file(data_file)
            row = _progress_with_live_step(file_info["symbol"], active=False)
            progress = {
                "symbol": row["symbol"],
                "status": row["status"],
                "current_step": row["current_step"],
                "train_steps": row["train_steps"],
                "progress_pct": row["progress_pct"],
                "best_score": row["best_score"],
                "formula_decoded": row["formula_decoded"],
                "has_checkpoint": row.get("has_checkpoint", False),
                "has_strategy": row.get("has_strategy", False),
            }
        except Exception as e:
            file_info = {"data_file": data_file, "valid": False, "message": str(e)}

    training = training_manager.status()
    job = training.get("job")
    if job and job.get("symbol") and training.get("active"):
        row = _progress_with_live_step(job["symbol"], active=True)
        progress = {
            "symbol": row["symbol"],
            "status": "running_job",
            "current_step": row["current_step"],
            "train_steps": row["train_steps"],
            "progress_pct": row["progress_pct"],
            "best_score": row["best_score"],
            "formula_decoded": row["formula_decoded"],
            "has_checkpoint": row.get("has_checkpoint", False),
            "has_strategy": row.get("has_strategy", False),
        }

    return {
        "data_file": file_info,
        "progress": progress,
        "training": training,
    }


@app.get("/api/symbols/{symbol}")
def api_symbol(symbol: str) -> dict[str, Any]:
    p = get_symbol_progress(symbol)
    return {
        "symbol": p.symbol,
        "status": p.status,
        "current_step": p.current_step,
        "train_steps": p.train_steps,
        "progress_pct": round(p.progress_pct, 1),
        "best_score": p.best_score,
        "best_formula": p.best_formula,
        "formula_decoded": p.formula_decoded,
        "has_strategy": p.has_strategy,
        "strategy_score": p.strategy_score,
        "checkpoint_path": p.checkpoint_path,
        "history": p.history,
    }


@app.get("/api/strategies")
def api_strategies() -> dict[str, Any]:
    return {"strategies": list_strategies()}


@app.get("/api/strategies/{symbol}/export")
def api_export_strategy(symbol: str):
    import json

    from fastapi.responses import Response

    try:
        payload = get_strategy_for_export(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    safe = symbol.replace(".", "_")
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="strategy_{safe}.json"',
        },
    )


@app.get("/api/training/{symbol}/export")
def api_export_training(symbol: str):
    from fastapi.responses import Response

    try:
        body, zip_name = build_training_export_zip(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=body,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.post("/api/training/import")
async def api_import_training(
    file: UploadFile = File(...),
    symbol: str | None = Query(None, description="当前选择的品种，用于校验导入包是否一致"),
) -> dict[str, Any]:
    if training_manager.status().get("active"):
        raise HTTPException(409, "训练进行中，请先停止再导入")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "上传文件为空")

    try:
        return import_training_package(
            raw,
            file.filename or "upload.zip",
            expected_symbol=symbol or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/training/status")
def api_training_status() -> dict[str, Any]:
    status = training_manager.status()
    status["log_tail"] = training_manager.tail_log(150)
    return status


@app.post("/api/training/start")
def api_training_start(req: StartTrainingRequest) -> dict[str, Any]:
    info = _inspect_or_http(req.data_file)
    save_settings({"last_data_file": info["data_file"]})
    try:
        job = training_manager.start(
            data_file=info["data_file"],
            symbol=info["symbol"],
            timeframe=info["timeframe"],
            mode="ftmo",
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict(), "data_file": info}


@app.post("/api/training/stop")
def api_training_stop() -> dict[str, Any]:
    stopped = training_manager.stop()
    return {"ok": stopped, "training": training_manager.status()}


# ─────────────────────────────────────────────────────────────────────
# 回测 API
# ─────────────────────────────────────────────────────────────────────

_METRIC_KEYS = (
    "total_return", "sharpe", "sortino", "max_drawdown",
    "calmar", "n_trades", "win_rate", "avg_hold_bars",
)


def _load_backtest_report() -> dict[str, Any] | None:
    import json

    report_path = BACKTEST_OUTPUT_DIR / "multi_factor_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _backtest_focus_symbol(symbol: str | None = None) -> str | None:
    """Resolve the symbol used to filter backtest charts/report for the web UI."""
    if symbol:
        return symbol.strip() or None

    job = backtest_manager.status().get("job") or {}
    if job.get("symbol"):
        return str(job["symbol"])

    strat = _strategy_context().get("strategy_file") or {}
    if strat.get("symbol"):
        return str(strat["symbol"])

    report = _load_backtest_report()
    if report:
        keys = list((report.get("symbols") or {}).keys())
        if len(keys) == 1:
            return keys[0]
    return None


def _filter_report_for_symbol(report: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbols = report.get("symbols") or {}
    if symbol not in symbols:
        return report

    sym_data = symbols[symbol]
    return {
        **report,
        "focus_symbol": symbol,
        "symbols": {symbol: sym_data},
        "portfolio": {
            "total_return": sym_data.get("total_return"),
            "sharpe": sym_data.get("sharpe"),
            "sortino": sym_data.get("sortino"),
            "max_drawdown": sym_data.get("max_drawdown"),
            "calmar": sym_data.get("calmar"),
            "n_trades": sym_data.get("n_trades"),
            "win_rate": sym_data.get("win_rate"),
        },
    }


def _list_backtest_charts(symbol: str | None = None) -> list[dict[str, str]]:
    """列出回测输出目录下的图表；单品种模式只返回该品种相关文件。"""
    if not BACKTEST_OUTPUT_DIR.exists():
        return []

    if symbol:
        charts: list[dict[str, str]] = []
        equity = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
        if equity.exists():
            charts.append(
                {"name": equity.name, "label": f"{symbol} 资金曲线", "kind": "equity"}
            )
        main = BACKTEST_OUTPUT_DIR / f"{symbol}.png"
        if main.exists():
            charts.append(
                {"name": main.name, "label": f"{symbol} K线与交易", "kind": "symbol"}
            )
        for path in sorted(BACKTEST_OUTPUT_DIR.glob(f"{symbol}_trade*_zoom.png")):
            stem = path.stem
            trade_no = stem.replace(f"{symbol}_trade", "").replace("_zoom", "")
            label = f"{symbol} 交易 #{trade_no}" if trade_no.isdigit() else stem
            charts.append({"name": path.name, "label": label, "kind": "trade"})
        return charts

    charts = []
    portfolio = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
    if portfolio.exists():
        charts.append({"name": "portfolio_equity.png", "label": "组合资金曲线", "kind": "portfolio"})
    for path in sorted(BACKTEST_OUTPUT_DIR.glob("equity_*.png")):
        sym = path.stem.replace("equity_", "", 1)
        charts.append({"name": path.name, "label": f"{sym} 资金曲线", "kind": "symbol"})
    return charts


@app.get("/api/backtest/status")
def api_backtest_status() -> dict[str, Any]:
    status = backtest_manager.status()
    status["log_tail"] = backtest_manager.tail_log(200)
    return status


@app.post("/api/backtest/start")
def api_backtest_start(req: StartBacktestRequest) -> dict[str, Any]:
    info = _inspect_strategy_or_http(req.strategy_file)
    save_settings({"last_strategy_file": info["strategy_file"]})

    data_file: str | None = None
    last_data = load_settings().get("last_data_file") or ""
    if last_data:
        try:
            pf = inspect_parquet_file(last_data)
            if pf.get("symbol") == info.get("symbol"):
                data_file = pf["data_file"]
        except Exception:
            pass

    try:
        job = backtest_manager.start(
            strategy_file=info["strategy_file"],
            data_file=data_file,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict(), "strategy_file": info}


@app.post("/api/backtest/stop")
def api_backtest_stop() -> dict[str, Any]:
    stopped = backtest_manager.stop()
    return {"ok": stopped, "backtest": backtest_manager.status()}


@app.get("/api/backtest/report")
def api_backtest_report(symbol: str | None = None) -> dict[str, Any]:
    report = _load_backtest_report()
    focus = _backtest_focus_symbol(symbol)
    if report and focus:
        report = _filter_report_for_symbol(report, focus)
    return {
        "available": report is not None,
        "report": report,
        "charts": _list_backtest_charts(focus),
        "focus_symbol": focus,
    }


@app.get("/api/backtest/chart/{name}")
def api_backtest_chart(name: str):
    # 防止路径穿越：仅允许输出目录内的 png 文件
    if "/" in name or "\\" in name or ".." in name or not name.lower().endswith(".png"):
        raise HTTPException(400, "非法文件名")
    path = (BACKTEST_OUTPUT_DIR / name).resolve()
    try:
        path.relative_to(BACKTEST_OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "非法路径") from None
    if not path.exists():
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
