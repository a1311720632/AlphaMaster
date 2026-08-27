"""FastAPI application for AlphaMaster training UI."""
from __future__ import annotations

import json
import os
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
from web.progress import (
    get_symbol_progress,
    get_strategy_for_export,
    invalidate_checkpoint_cache,
    list_strategies,
    build_strategy_export_filename,
)
from web.server_log import (
    debug_snapshot,
    get_logger,
    is_debug_mode,
    log_error,
    set_debug_mode,
    setup_logging,
)
from web.settings import load_settings, save_settings
from web.strategy_file import (
    inspect_strategy_file,
    resolve_strategy_file,
    strategy_path_for_symbol,
    sync_best_strategy_for_symbol,
)
from web.training_manager import training_manager
from web.training_time import get_training_time_summary
from web.training_package import build_training_export_zip, import_training_package
from web.backtest_manager import backtest_manager
from web.realtime_manager import realtime_manager
from web.autopilot_manager import autopilot_manager
from web.data_sources.factory import get_source, list_sources
from strategy_manager.live_signal import min_exposure

STATIC_DIR = Path(__file__).resolve().parent / "static"
BACKTEST_OUTPUT_DIR = ROOT / "backtest_output"


class NoCacheStaticFiles(StaticFiles):
    """静态资源禁用启发式缓存：浏览器每次必须带 ETag 向服务器校验
    （未变→304 走缓存，变了→立即拉新），避免改了 app.js 后旧脚本被复用。"""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response

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
    from_scratch: bool = False


class ClientLogRequest(BaseModel):
    level: str = "error"
    message: str
    context: dict[str, Any] | None = None


class SettingsRequest(BaseModel):
    last_data_file: str | None = None
    last_strategy_file: str | None = None
    debug_mode: bool | None = None
    ai_provider: str | None = None
    ai_api_key: str | None = None
    bt_commission_pct: float | None = None
    bt_slippage_pct: float | None = None


class AnalyzeTrainingRequest(BaseModel):
    provider: str | None = None
    api_key: str | None = None
    symbol: str | None = None


class StartBacktestRequest(BaseModel):
    strategy_file: str
    commission_pct: float | None = None
    slippage_pct: float | None = None


class StartAutopilotRequest(BaseModel):
    strategy_file: str
    mode: str = "paper"  # paper / testnet / live
    symbol: str | None = None
    timeframe: str | None = None


class AutopilotPreflightRequest(BaseModel):
    strategy_file: str
    mode: str  # testnet | live（paper 无需预检）
    symbol: str | None = None
    timeframe: str | None = None


class AddWatchRequest(BaseModel):
    source: str
    symbol: str
    timeframe: str
    strategy_file: str


class RemoveWatchRequest(BaseModel):
    id: str


class FeishuSettingsRequest(BaseModel):
    enabled: bool | None = None               # 实时分析转折提醒开关（feishu_enabled）
    autopilot_enabled: bool | None = None     # 自动驾驶告警开关（feishu_autopilot_enabled）
    webhook_url: str | None = None
    secret: str | None = None


class FeishuTestRequest(BaseModel):
    webhook_url: str | None = None
    secret: str | None = None
    source: str | None = None  # "autopilot" → 推送文案换成自动驾驶语境


class StrategyFileRequest(BaseModel):
    file: str  # 策略文件名（best_*.json），多周期/多条后 symbol 不再唯一，用文件名定位


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
        # headless（无 tkinter / 无 DISPLAY）：当作取消 + 标记，前端转下拉选择
        log_error("File picker unavailable (headless?)", exc)
        return {"ok": False, "cancelled": True, "native_picker": False}

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
    train_timeframe = None
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            train_symbol = info.get("symbol")
            train_timeframe = info.get("timeframe")
        except Exception:
            pass

    resolved = resolve_strategy_file(
        settings.get("last_strategy_file") or "",
        train_symbol,
        train_timeframe,
    )
    strategy_info = None
    if resolved:
        try:
            strategy_info = inspect_strategy_file(
                resolved,
                data_file_hint=settings.get("last_data_file") or None,
            )
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
        # headless（无 tkinter / 无 DISPLAY）：当作取消 + 标记，前端转下拉选择
        log_error("Strategy file picker unavailable (headless?)", exc)
        return {"ok": False, "cancelled": True, "native_picker": False}

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


def _resolve_train_symbol(symbol: str | None = None) -> str | None:
    if symbol:
        return symbol.strip() or None
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    if not data_file:
        return None
    try:
        return inspect_parquet_file(data_file).get("symbol")
    except Exception:
        return None


def _wait_training_idle(timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        training_manager.status()
        if not training_manager.status().get("active"):
            return
        time.sleep(0.2)


def _sync_and_persist_best_strategy(
    symbol: str,
    *,
    data_file_hint: str | None = None,
) -> dict[str, Any] | None:
    invalidate_checkpoint_cache()
    hint = data_file_hint
    if not hint:
        job = training_manager.status().get("job") or {}
        if str(job.get("symbol") or "") == symbol:
            hint = job.get("data_file") or None
    if not hint:
        hint = load_settings().get("last_data_file") or None
    info = sync_best_strategy_for_symbol(symbol, data_file_hint=hint)
    if info:
        save_settings({"last_strategy_file": info["strategy_file"]})
    return info


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
    if req.ai_provider is not None:
        payload["ai_provider"] = req.ai_provider
    if req.ai_api_key is not None:
        payload["ai_api_key"] = req.ai_api_key
    if req.bt_commission_pct is not None:
        payload["bt_commission_pct"] = req.bt_commission_pct
    if req.bt_slippage_pct is not None:
        payload["bt_slippage_pct"] = req.bt_slippage_pct
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
        "ai_provider": load_settings().get("ai_provider", "deepseek"),
        "ai_api_key": load_settings().get("ai_api_key", ""),
        "bt_commission_pct": settings.get("bt_commission_pct", 0.02),
        "bt_slippage_pct": settings.get("bt_slippage_pct", 0.01),
        "server_log": snap["server_log"],
        "error_log": snap["error_log"],
    }


@app.get("/api/ai/providers")
def api_ai_providers() -> dict[str, Any]:
    from web.ai_providers import provider_status

    status = provider_status()
    settings = load_settings()
    status["selected"] = settings.get("ai_provider", "deepseek")
    status["has_api_key"] = bool(settings.get("ai_api_key"))
    return status


@app.post("/api/ai/analyze-training")
def api_ai_analyze_training(req: AnalyzeTrainingRequest):
    from fastapi.responses import StreamingResponse

    from web.ai_analyze import analyze_training_stream

    settings = load_settings()
    raw_key = req.api_key if req.api_key is not None else settings.get("ai_api_key") or ""
    key_lower = str(raw_key).strip().lower()

    # openclaw_wb 必须先于 openclaw 判断
    if key_lower in ("openclaw_wb",) or key_lower.startswith("openclaw_wb/"):
        provider = "openclaw_wb"
    elif key_lower in ("openclaw",) or key_lower.startswith("openclaw/"):
        provider = "openclaw"
    else:
        provider = (req.provider or settings.get("ai_provider") or "deepseek").strip()

    save_settings({
        "ai_provider": provider,
        "ai_api_key": str(raw_key).strip(),
    })

    def event_gen():
        try:
            for event in analyze_training_stream(
                provider=provider,
                api_key=str(raw_key).strip() or None,
                symbol=req.symbol,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/data-file/browse")
@app.get("/api/data-file/browse")
def api_browse_data_file() -> dict[str, Any]:
    return _browse_data_file()


@app.get("/api/data-file/list")
def api_list_data_files() -> dict[str, Any]:
    """headless 友好：列出 KLINE_CACHE_DIR + 项目根下的 *.parquet（下拉选，替代弹不了的本地选框）。
    复用 inspect_parquet_file 取 symbol/timeframe/bars，让前端 renderDataFileCard 直接可用。"""
    from config import Config

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in (Path(Config.KLINE_CACHE_DIR), ROOT):
        try:
            base = Path(d).resolve()
        except OSError:
            continue
        if not base.exists():
            continue
        for p in sorted(base.glob("*.parquet")):
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            try:
                info = inspect_parquet_file(str(p))
            except Exception as exc:  # noqa: BLE001 - 单个文件坏不影响整列
                # inspect 抛错（如日线 <3000 根"数据不足"）时，仍从文件名解析
                # symbol/timeframe，让前端下拉正常显示（如 BTCUSDT D1）而非 "?"
                stem = p.stem
                parts = stem.rsplit("_", 1)
                info = {
                    "data_file": str(p),
                    "filename": p.name,
                    "symbol": parts[0] if len(parts) == 2 else stem,
                    "timeframe": parts[1] if len(parts) == 2 else "",
                    "valid": False,
                    "message": f"读取失败: {exc}",
                }
            info.setdefault("filename", p.name)
            out.append(info)
    return {"files": out}


@app.post("/api/strategy-file/browse")
@app.get("/api/strategy-file/browse")
def api_browse_strategy_file() -> dict[str, Any]:
    return _browse_strategy_file()


@app.post("/api/strategy-file/sync-best")
@app.get("/api/strategy-file/sync-best")
def api_sync_best_strategy(symbol: str | None = None) -> dict[str, Any]:
    sym = _resolve_train_symbol(symbol)
    if not sym:
        raise HTTPException(400, "请先选择训练数据文件或指定品种")
    info = _sync_and_persist_best_strategy(sym)
    if not info:
        raise HTTPException(404, f"未找到 {sym} 的可用策略")
    return {"ok": True, **info}


def _progress_with_live_step(symbol: str, active: bool) -> dict[str, Any]:
    p = get_symbol_progress(symbol)
    current_step = p.current_step
    if active:
        live = training_manager.parse_step_from_log()
        if live is not None:
            current_step = max(current_step, live)
    train_steps = p.train_steps
    progress_pct = min(100.0, 100.0 * current_step / train_steps) if train_steps > 0 else 0.0
    val_score = None
    hist = p.history or {}
    vals = hist.get("val_score") or []
    if vals:
        try:
            val_score = float(vals[-1])
        except (TypeError, ValueError):
            val_score = None
    return {
        "symbol": p.symbol,
        "current_step": current_step,
        "train_steps": train_steps,
        "progress_pct": round(progress_pct, 1),
        "best_score": p.best_score,
        "val_score": val_score,
        "formula_decoded": p.formula_decoded,
        "status": p.status,
        "history": p.history,
        "has_checkpoint": bool(p.checkpoint_path),
        "has_strategy": p.has_strategy,
    }


def _attach_training_time(
    row: dict[str, Any] | None,
    *,
    symbol: str | None,
    job: dict[str, Any] | None,
    active: bool,
) -> dict[str, Any] | None:
    if not row or not symbol:
        return row
    summary = get_training_time_summary(symbol, job=job, active=active)
    row = dict(row)
    row["session_seconds"] = summary.session_seconds
    row["history_total_seconds"] = summary.history_total_seconds
    return row


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    progress = None

    training = training_manager.status()
    job = training.get("job")
    active = bool(training.get("active"))

    if data_file:
        try:
            file_info = inspect_parquet_file(data_file)
            sym = file_info.get("symbol")
            row = _progress_with_live_step(sym, active=False)
            progress = {
                "symbol": row["symbol"],
                "status": row["status"],
                "current_step": row["current_step"],
                "train_steps": row["train_steps"],
                "progress_pct": row["progress_pct"],
                "best_score": row["best_score"],
                "val_score": row.get("val_score"),
                "formula_decoded": row["formula_decoded"],
                "has_checkpoint": row.get("has_checkpoint", False),
                "has_strategy": row.get("has_strategy", False),
            }
            progress = _attach_training_time(
                progress, symbol=sym, job=job, active=active and job and job.get("symbol") == sym
            )
        except Exception as e:
            file_info = {"data_file": data_file, "valid": False, "message": str(e)}

    if job and job.get("symbol") and active:
        sym = job["symbol"]
        row = _progress_with_live_step(sym, active=True)
        progress = {
            "symbol": row["symbol"],
            "status": "running_job",
            "current_step": row["current_step"],
            "train_steps": row["train_steps"],
            "progress_pct": row["progress_pct"],
            "best_score": row["best_score"],
            "val_score": row.get("val_score"),
            "formula_decoded": row["formula_decoded"],
            "has_checkpoint": row.get("has_checkpoint", False),
            "has_strategy": row.get("has_strategy", False),
        }
        progress = _attach_training_time(progress, symbol=sym, job=job, active=True)

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


def _resolve_strategy_file_safe(file: str) -> Path | None:
    """策略文件名安全解析：必须 best_*.json 且落在 STRATEGIES_DIR 内（防路径穿越）。"""
    from web.progress import STRATEGIES_DIR

    name = Path(file).name  # 剥任何路径前缀，只留文件名
    if not name.startswith("best_") or not name.endswith(".json"):
        return None
    path = (STRATEGIES_DIR / name).resolve()
    try:
        path.relative_to(STRATEGIES_DIR.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


@app.post("/api/strategies/export")
def api_export_strategy(req: StrategyFileRequest):
    """按文件名导出策略 JSON（多周期/多条后 symbol 不再唯一，用 file 定位）。"""
    import json

    from fastapi.responses import Response

    path = _resolve_strategy_file_safe(req.file)
    if path is None:
        raise HTTPException(status_code=404, detail=f"策略不存在: {req.file}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"读取失败: {exc}") from exc
    symbol = payload.get("symbol") or path.stem
    score = payload.get("best_score")
    step = int(payload.get("trained_step") or payload.get("train_steps") or 0)
    filename = build_strategy_export_filename(symbol, step, score)
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/strategies")
def api_delete_strategy(req: StrategyFileRequest):
    """删除指定策略文件（按文件名，路径穿越防护）。"""
    path = _resolve_strategy_file_safe(req.file)
    if path is None:
        raise HTTPException(status_code=404, detail=f"策略不存在: {req.file}")
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"删除失败: {exc}") from exc
    return {"ok": True, "strategies": list_strategies()}


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
            from_scratch=bool(req.from_scratch),
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    if req.from_scratch:
        invalidate_checkpoint_cache()
    return {
        "ok": True,
        "job": job.to_dict(),
        "data_file": info,
        "from_scratch": bool(req.from_scratch),
    }


@app.post("/api/training/stop")
def api_training_stop() -> dict[str, Any]:
    job = training_manager.status().get("job") or {}
    symbol = job.get("symbol")
    data_file_hint = job.get("data_file")
    stopped = training_manager.stop()
    strategy_file = None
    if symbol:
        _wait_training_idle()
        strategy_file = _sync_and_persist_best_strategy(
            symbol,
            data_file_hint=data_file_hint,
        )
    return {
        "ok": stopped,
        "training": training_manager.status(),
        "strategy_file": strategy_file,
    }


# ─────────────────────────────────────────────────────────────────────
# 回测 API
# ─────────────────────────────────────────────────────────────────────

_METRIC_KEYS = (
    "total_return", "sharpe", "sortino", "profit_loss_ratio",
    "n_trades", "win_rate", "avg_hold_bars",
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
            "profit_loss_ratio": sym_data.get("profit_loss_ratio"),
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


def _resolve_data_file_fallback(symbol: str | None, timeframe: str | None) -> str | None:
    """策略记录的 data_file 在当前环境加载不了时（典型：服务器上 strategy JSON 仍存着
    Windows 训练路径），按 {SYMBOL}_{TIMEFRAME}.parquet 从 KLINE_CACHE_DIR + 项目根兜底。"""
    from config import Config

    if not symbol or not timeframe:
        return None
    names = [f"{symbol}_{timeframe}.parquet"]
    low = timeframe.lower()
    if low != timeframe:
        names.append(f"{symbol}_{low}.parquet")
    for d in (Path(Config.KLINE_CACHE_DIR), ROOT):
        try:
            base = Path(d).resolve()
        except OSError:
            continue
        if not base.exists():
            continue
        for name in names:
            p = base / name
            if p.is_file():
                return str(p)
    return None


@app.post("/api/backtest/start")
def api_backtest_start(req: StartBacktestRequest) -> dict[str, Any]:
    info = _inspect_strategy_or_http(req.strategy_file)
    settings = load_settings()
    commission = (
        float(req.commission_pct)
        if req.commission_pct is not None
        else float(settings.get("bt_commission_pct", 0.02))
    )
    slippage = (
        float(req.slippage_pct)
        if req.slippage_pct is not None
        else float(settings.get("bt_slippage_pct", 0.01))
    )
    if commission < 0 or slippage < 0:
        raise HTTPException(400, "手续费和滑点不能为负数")

    save_settings({
        "last_strategy_file": info["strategy_file"],
        "bt_commission_pct": commission,
        "bt_slippage_pct": slippage,
    })

    data_file: str | None = None
    strat_load_error: str | None = None
    # 1) 优先用策略 JSON 里记录的训练数据路径
    strat_data = (info.get("data_file") or "").strip()
    if strat_data:
        try:
            pf = inspect_parquet_file(strat_data)
            if pf.get("valid") is False:
                raise HTTPException(
                    400,
                    f"策略记录的数据文件无效: {pf.get('message') or strat_data}",
                )
            data_file = pf["data_file"]
        except HTTPException:
            raise
        except Exception as e:
            # 策略记录的是本机训练路径，当前环境（如服务器）可能找不到——
            # 不在此直接报错，记下来落到下面按 symbol+tf 兜底
            strat_load_error = f"{strat_data}（{e}）"
    else:
        # 2) 回退：训练页最近选择的、同品种 Parquet
        last_data = settings.get("last_data_file") or ""
        if last_data:
            try:
                pf = inspect_parquet_file(last_data)
                if pf.get("symbol") == info.get("symbol") and pf.get("valid") is not False:
                    data_file = pf["data_file"]
            except Exception:
                pass

    # 3) 仍没有 → 按 {SYMBOL}_{TIMEFRAME}.parquet 从 KLINE_CACHE_DIR/项目根兜底
    #    （服务器上 strategy JSON 常存着 Windows 训练路径，这里改用本地数据）
    if not data_file:
        data_file = _resolve_data_file_fallback(info.get("symbol"), info.get("timeframe"))

    if not data_file:
        hint = f"（策略记录的 data_file 加载失败: {strat_load_error}）" if strat_load_error else ""
        raise HTTPException(
            400,
            "该策略未记录可用数据文件路径，且未按 symbol+周期在 KLINE_CACHE_DIR 找到同名 Parquet。"
            f"{hint} 请在「模型训练」页选择对应品种的 Parquet 再回测。",
        )

    save_settings({"last_data_file": data_file})

    try:
        job = backtest_manager.start(
            strategy_file=info["strategy_file"],
            data_file=data_file,
            commission_pct=commission,
            slippage_pct=slippage,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict(), "strategy_file": info, "data_file": data_file}


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


@app.get("/api/backtest/equity")
def api_backtest_equity(symbol: str | None = None) -> dict[str, Any]:
    """资金曲线原始数据（供前端渲染交互式 HTML 图表）。"""
    import json

    path = BACKTEST_OUTPUT_DIR / "equity_curve.json"
    focus = _backtest_focus_symbol(symbol)
    if not path.exists():
        return {"available": False, "focus_symbol": focus, "data": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available": False, "focus_symbol": focus, "data": None}

    # 单品种模式：只保留聚焦品种，去掉无关序列
    if focus and isinstance(data.get("symbols"), dict) and focus in data["symbols"]:
        data = {
            **data,
            "symbols": {focus: data["symbols"][focus]},
        }
        data.pop("portfolio", None)

    return {"available": True, "focus_symbol": focus, "data": data}


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


# ─────────────────────────────────────────────────────────────────────
# 实时行情分析 API
# ─────────────────────────────────────────────────────────────────────


@app.on_event("startup")
def _startup_realtime() -> None:
    try:
        realtime_manager.load_persisted()
    except Exception as exc:  # noqa: BLE001
        log_error("realtime load_persisted failed", exc)


@app.on_event("startup")
def _startup_autopilot_relaunch() -> None:
    # β：web 启动时若 autopilot_intended_running=True，从持久化设置重拉 autopilot
    try:
        autopilot_manager.relaunch_if_intended()
    except Exception as exc:  # noqa: BLE001
        log_error("autopilot boot relaunch handler failed", exc)


@app.on_event("shutdown")
def _shutdown_autopilot_watcher() -> None:
    try:
        autopilot_manager.shutdown()
    except Exception as exc:  # noqa: BLE001
        log_error("autopilot watcher shutdown failed", exc)


@app.get("/api/realtime/sources")
def api_realtime_sources() -> dict[str, Any]:
    return {"sources": list_sources(), "min_exposure": min_exposure()}


@app.post("/api/realtime/tradingview/probe")
def api_realtime_tradingview_probe() -> dict[str, Any]:
    """Probe TradingView reachability (same behavior as PA_Agent before fetch)."""
    from web.data_sources.tradingview_connectivity import (
        TV_CLOUD_SERVER_WIKI_URL,
        TV_CONNECTIVITY_MESSAGE,
        check_tradingview_connectivity,
    )

    ok, detail = check_tradingview_connectivity(
        timeout_s=15.0, max_attempts=2, retry_delay_s=2.0
    )
    return {
        "ok": ok,
        "detail": detail,
        "blocked": not ok,
        "title": "无法使用 TradingView",
        "message": None if ok else TV_CONNECTIVITY_MESSAGE,
        "wiki_url": TV_CLOUD_SERVER_WIKI_URL,
    }


@app.get("/api/realtime/strategies")
def api_realtime_strategies() -> dict[str, Any]:
    """已保存的 best_*.json 策略，供因子来源下拉。"""
    rows = []
    for s in list_strategies():
        sym = s.get("symbol")
        spath = s.get("path")
        if not sym or not spath:
            continue
        rows.append(
            {
                "symbol": sym,
                "timeframe": s.get("timeframe"),
                "best_score": s.get("best_score"),
                "formula_decoded": s.get("formula_decoded"),
                "strategy_file": spath,
            }
        )
    return {"strategies": rows}


@app.get("/api/realtime/status")
def api_realtime_status() -> dict[str, Any]:
    return realtime_manager.status()


@app.post("/api/realtime/watch")
def api_realtime_watch(req: AddWatchRequest) -> dict[str, Any]:
    try:
        watch = realtime_manager.add_watch(
            req.source, req.symbol, req.timeframe, req.strategy_file
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "watch": watch}


@app.post("/api/realtime/unwatch")
def api_realtime_unwatch(req: RemoveWatchRequest) -> dict[str, Any]:
    removed = realtime_manager.remove_watch(req.id)
    return {"ok": removed}


@app.post("/api/realtime/start")
def api_realtime_start() -> dict[str, Any]:
    realtime_manager.start()
    return {"ok": True, **realtime_manager.status()}


@app.post("/api/realtime/stop")
def api_realtime_stop() -> dict[str, Any]:
    realtime_manager.stop()
    return {"ok": True, "running": False}


@app.get("/api/realtime/feishu")
def api_realtime_feishu_get() -> dict[str, Any]:
    s = load_settings()
    return {
        "enabled": bool(s.get("feishu_enabled")),
        "autopilot_enabled": bool(s.get("feishu_autopilot_enabled", True)),
        "webhook_url": s.get("feishu_webhook_url") or "",
        "secret": s.get("feishu_secret") or "",
    }


@app.put("/api/realtime/feishu")
def api_realtime_feishu_put(req: FeishuSettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.enabled is not None:
        payload["feishu_enabled"] = bool(req.enabled)
    if req.autopilot_enabled is not None:
        payload["feishu_autopilot_enabled"] = bool(req.autopilot_enabled)
    if req.webhook_url is not None:
        payload["feishu_webhook_url"] = req.webhook_url
    if req.secret is not None:
        payload["feishu_secret"] = req.secret
    saved = save_settings(payload)
    return {
        "ok": True,
        "enabled": bool(saved.get("feishu_enabled")),
        "autopilot_enabled": bool(saved.get("feishu_autopilot_enabled", True)),
        "webhook_url": saved.get("feishu_webhook_url") or "",
        "secret": saved.get("feishu_secret") or "",
    }


@app.post("/api/realtime/feishu/test")
def api_realtime_feishu_test(req: FeishuTestRequest) -> dict[str, Any]:
    from web.feishu_notify import send_text

    url = (req.webhook_url or "").strip()
    if not url:
        url = (load_settings().get("feishu_webhook_url") or "").strip()
    if not url:
        raise HTTPException(400, "请先填写 Webhook URL")
    secret = req.secret
    if secret is None:
        secret = load_settings().get("feishu_secret") or ""
    text = (
        "✅ AlphaMaster 飞书通知测试：配置正常。熔断/源切换/日报等自动驾驶告警会推送到本群。"
        if (req.source or "").strip() == "autopilot"
        else "✅ AlphaMaster 飞书通知测试：配置正常。信号方向转折时会推送提醒。"
    )
    ok, msg = send_text(text, webhook_url=url, secret=secret or "")
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


# ─────────────────────────────────────────────────────────────────────
# 自动驾驶 API（第四步：paper / testnet / live）
# ─────────────────────────────────────────────────────────────────────
# 实时现价缓存：symbol -> (monotonic_ts, price)。照搬 realtime_manager._bar_cache 的 TTL 套路。
_ticker_cache: dict[str, tuple[float, float]] = {}
_TICKER_TTL = 10.0


def _get_live_price(symbol: str) -> float | None:
    """带 TTL 缓存的 OKX 现价（web 层每 ~10s 拉一次；失败返 None 不写缓存）。"""
    if not symbol:
        return None
    now = time.monotonic()
    cached = _ticker_cache.get(symbol)
    if cached and (now - cached[0]) < _TICKER_TTL:
        return cached[1]
    try:
        price = float(get_source("okx").fetch_ticker(symbol))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - ticker 失败不能让 status 端点 500
        return None
    if price > 0:
        _ticker_cache[symbol] = (now, price)
        return price
    return None


def _autopilot_state_dict() -> dict[str, Any]:
    """读 autopilot_state.json + 叠加实时现价，回传前端展示所需的账本与实时字段。"""
    try:
        from config import Config
        from autopilot.state import AutopilotState

        st = AutopilotState.load(Config.AUTOPILOT_STATE_FILE)
        if not st:
            return {}
        out = st.to_dict()
        lb = st.history[-1] if st.history else None
        out["last_bar"] = lb
        # 健康面板数据（C3/ADR-0007）：心跳时刻 + 最近运营事件（读冷账本 event 行）
        out["heartbeat_last_ok_ts"] = int(st.heartbeat_last_ok_ts or 0)
        try:
            from autopilot.ledger import Ledger, ledger_path

            lg = Ledger(ledger_path(st.symbol, st.mode, Config.AUTOPILOT_LEDGER_DIR or None))
            out["events_recent"] = list(reversed(lg.tail(30, types={"event"})))
        except Exception:  # noqa: BLE001
            out["events_recent"] = []

        # 实时现价层（决策 #7/#8）：web 层拉 ticker，基于 last_bar 现算浮盈/实时权益。
        # 两个基准不能混——持仓浮盈用 entry(自开仓)，实时权益增量用 close(自上根收盘)。
        last_price = _get_live_price(st.symbol) if lb else None
        out["last_price"] = last_price
        if lb is not None and last_price is not None:
            actual = float(lb.get("actual_notional") or 0.0)
            entry = float(lb.get("entry_price") or 0.0)
            close = float(lb.get("close") or 0.0)
            eq = float(lb.get("equity") or 0.0)
            # 持仓浮盈：自开仓口径（用 entry；entry<=0/空仓→0）
            out["unrealized_pnl_live"] = (
                actual * (last_price / entry - 1.0) if entry > 0 else 0.0
            )
            # 实时权益：自上根收盘口径（用 close；equity 已 mark 到 close）
            out["realtime_equity"] = (
                eq + actual * (last_price / close - 1.0) if close > 0 else eq
            )
            se = float(st.start_equity or 0.0)
            out["realtime_total_return"] = (
                (out["realtime_equity"] - se) / se if se > 0 else None
            )
        else:
            out["unrealized_pnl_live"] = None
            out["realtime_equity"] = None
            out["realtime_total_return"] = None
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@app.get("/api/autopilot/status")
def api_autopilot_status() -> dict[str, Any]:
    status = autopilot_manager.status()
    status["log_tail"] = autopilot_manager.tail_log(150)
    status["state"] = _autopilot_state_dict()
    return status


def _autopilot_preflight_checks(mode: str, symbol: str | None, timeframe: str | None) -> dict[str, Any]:
    """testnet/live 启动预检（C1/D1/ADR-0007）。单项独立 try，互不影响。

    返回 {"ok", "checks": [{id,label,status,detail}], "armed"}。
    """
    from config import Config

    checks: list[dict[str, Any]] = []

    def add(cid: str, label: str, status: str, detail: str = "") -> None:
        checks.append({"id": cid, "label": label, "status": status, "detail": detail})

    # 1. ccxt 可导入
    try:
        import ccxt  # noqa: F401

        add("ccxt_installed", "ccxt 已安装", "pass")
    except ImportError:
        add("ccxt_installed", "ccxt 已安装", "fail", "python -m pip install ccxt")

    # 2. 凭据非空
    missing = [
        name for name, v in (
            ("OKX_API_KEY", Config.OKX_API_KEY),
            ("OKX_SECRET_KEY", Config.OKX_SECRET_KEY),
            ("OKX_PASSPHRASE", Config.OKX_PASSPHRASE),
        ) if not (v or "").strip()
    ]
    if missing:
        add("credentials", "OKX 凭据已配置", "fail", f"缺 {', '.join(missing)}（.env）")
    else:
        add("credentials", "OKX 凭据已配置", "pass")

    # 3. 凭据有效（实测 fetch_balance；同时验证 sandbox 指向——配错会 401/1100xx）
    if not missing:
        try:
            from autopilot.backends import OKXBackend

            be = OKXBackend(
                symbol=symbol or "BTCUSDT", sandbox=(mode == "testnet"),
                api_key=Config.OKX_API_KEY, secret=Config.OKX_SECRET_KEY,
                passphrase=Config.OKX_PASSPHRASE,
            )
            try:
                eq = be.fetch_equity()
                if eq > 0:
                    target = "模拟盘(demo)" if mode == "testnet" else "真实账户"
                    add("credentials_ok", f"凭据有效 · {target} 权益 {eq:.2f} USDT", "pass")
                else:
                    add("credentials_ok", "凭据有效", "fail", f"权益异常: {eq}")
            finally:
                be.close()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            hint = "检查 key 与 sandbox 指向（testnet 需 demo 专用 key）" if (
                "401" in msg or "1100" in msg or "AuthenticationError" in msg
            ) else msg
            add("credentials_ok", "凭据有效", "fail", hint[:200])
    else:
        add("credentials_ok", "凭据有效", "fail", "凭据缺失，跳过实测")

    # 4. state 三元组匹配（不匹配 → warn：启动时自动归档新建，C2）
    try:
        from config import Config as _C

        from autopilot.state import AutopilotState

        st = AutopilotState.load(_C.AUTOPILOT_STATE_FILE)
        if st is None:
            add("state_match", "state 三元组匹配", "pass", "无历史账本，将新建")
        elif (st.symbol, st.timeframe, st.mode) != (
            symbol or st.symbol, timeframe or st.timeframe, mode
        ):
            add(
                "state_match", "state 三元组匹配", "warn",
                f"检测到 {st.symbol}/{st.timeframe}/{st.mode} 的旧账本，启动时将归档后新建",
            )
        else:
            add("state_match", "state 三元组匹配", "pass")
    except Exception as exc:  # noqa: BLE001
        add("state_match", "state 三元组匹配", "warn", f"读取失败: {exc}")

    # 5. testnet 达标清单（仅 live；D1：2-4 周长跑机检）
    if mode == "live":
        try:
            from autopilot.ledger import ledger_path
            from autopilot.readiness import live_readiness, summarize_testnet_run

            sym = symbol or "BTCUSDT"
            lg = ledger_path(sym, "testnet", Config.AUTOPILOT_LEDGER_DIR or None)
            summary = summarize_testnet_run(lg)
            ready, sub = live_readiness(summary)
            add(
                "testnet_ready",
                "testnet 长跑达标（live 门槛）",
                "pass" if ready else "fail",
                "; ".join(f"{c['label']}: {c['detail']}" for c in sub if c["status"] != "pass"),
            )
        except Exception as exc:  # noqa: BLE001
            add("testnet_ready", "testnet 长跑达标（live 门槛）", "fail", str(exc)[:200])
    else:
        add("testnet_ready", "testnet 长跑达标（live 门槛）", "pass", "testnet 模式免检")

    # armed 预留（硬门槛 env 开关位，金额变大时启用）
    armed = bool(os.getenv("AUTOPILOT_LIVE_ARMED", "1").strip().lower() not in ("0", "false"))
    if mode == "live" and not armed:
        add("armed", "live 硬门槛（AUTOPILOT_LIVE_ARMED）", "fail", "未武装")

    ok = all(c["status"] != "fail" for c in checks)
    return {"ok": ok, "checks": checks, "armed": armed}


@app.post("/api/autopilot/preflight")
def api_autopilot_preflight(req: AutopilotPreflightRequest) -> dict[str, Any]:
    if req.mode not in ("testnet", "live"):
        raise HTTPException(400, "preflight 仅用于 testnet/live（paper 免检）")
    return _autopilot_preflight_checks(req.mode, req.symbol, req.timeframe)


@app.post("/api/autopilot/start")
def api_autopilot_start(req: StartAutopilotRequest) -> dict[str, Any]:
    sf = resolve_strategy_file(req.strategy_file) if req.strategy_file else ""
    if not sf:
        raise HTTPException(400, "缺少 strategy_file")
    # testnet/live 快检（C1/ADR-0007）：ccxt + 凭据两项，防 TOCTOU；完整清单走 preflight
    if req.mode in ("testnet", "live"):
        pre = _autopilot_preflight_checks(req.mode, req.symbol, req.timeframe)
        if not pre["ok"]:
            fails = "; ".join(
                f"{c['label']}: {c['detail']}" for c in pre["checks"] if c["status"] == "fail"
            )
            raise HTTPException(409, f"预检未过: {fails}")
    save_settings({"autopilot_mode": req.mode, "autopilot_last_strategy": sf,
                   "autopilot_symbol": req.symbol or "", "autopilot_timeframe": req.timeframe or "",
                   "autopilot_intended_running": True})
    try:
        job = autopilot_manager.start(
            strategy_file=sf,
            mode=req.mode,
            symbol=req.symbol,
            timeframe=req.timeframe,
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict()}


@app.get("/api/autopilot/history")
def api_autopilot_history(type: str = "all", limit: int = 2000) -> dict[str, Any]:
    """冷账本读侧（E1/ADR-0007）：审计视图（日历/曲线/成交/事件时间线）的数据源。

    symbol/mode 取自当前 hot state 三元组 → ledger 路径；无 state → 空。
    """
    from config import Config

    from autopilot.ledger import Ledger, ledger_path
    from autopilot.state import AutopilotState

    st = AutopilotState.load(Config.AUTOPILOT_STATE_FILE)
    if not st:
        return {"rows": []}
    limit = max(1, min(int(limit), 10000))
    types = {type} if type in ("bar", "trade", "event") else None
    lg = Ledger(ledger_path(st.symbol, st.mode, Config.AUTOPILOT_LEDGER_DIR or None))
    return {"rows": lg.tail(limit, types=types)}


@app.post("/api/autopilot/stop")
def api_autopilot_stop() -> dict[str, Any]:
    # 先落"不再意图运行"标志（β），再停——保证 web 重启后不会把它再拉起来
    try:
        save_settings({"autopilot_intended_running": False})
    except Exception:  # noqa: BLE001
        pass
    stopped = autopilot_manager.stop()
    return {"ok": stopped, "autopilot": autopilot_manager.status()}


@app.post("/api/autopilot/reset")
def api_autopilot_reset() -> dict[str, Any]:
    """一键清除：停止自动驾驶 + 删 autopilot_state.json（历史/成交/权益全清，回初始）。"""
    try:
        save_settings({"autopilot_intended_running": False})
    except Exception:  # noqa: BLE001
        pass
    result = autopilot_manager.reset()
    return {"ok": True, **result, "autopilot": autopilot_manager.status()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")
