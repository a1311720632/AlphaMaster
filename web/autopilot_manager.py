"""Subprocess manager for run_autopilot.py jobs（第四步：自动驾驶）。

镜像 web/training_manager.py：JobState / AutopilotJob / start / stop / status / tail_log。
以 subprocess 方式拉起 run_autopilot.py，stdout/stderr 落 logs/autopilot_*.log。
进程隔离：自动驾驶崩溃不会拖垮 web 进程；状态文件 autopilot_state.json 供前端读取。
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.train_logging import strip_ansi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_MODES = ("paper", "testnet", "live")

_PROJECT_ROOT_RESOLVED = PROJECT_ROOT.resolve()
_LOG_DIR_RESOLVED = LOG_DIR.resolve()


def _validate_strategy_path(strategy_file: str) -> str:
    """Defense-in-depth：策略路径必须存在、为 .json、且在项目根目录内。

    本工具可触发真实下单（比回测更敏感），故把策略路径限定在 PROJECT_ROOT 内，
    防止路径穿越读取任意文件。训练产物正常位于 strategies/（在 PROJECT_ROOT 内）。
    与回测的 browse-anywhere 不同——这是有意的更紧约束。
    """
    p = Path(strategy_file).expanduser()
    if not p.is_file():
        raise ValueError(f"策略文件不存在或不是普通文件: {strategy_file}")
    if p.suffix.lower() != ".json":
        raise ValueError(f"策略文件必须是 .json: {strategy_file}")
    try:
        resolved = p.resolve()
        resolved.relative_to(_PROJECT_ROOT_RESOLVED)
    except ValueError as exc:
        raise ValueError("策略文件必须在项目根目录内（防止路径穿越）") from exc
    return str(resolved)


def _safe_log_symbol(symbol: str | None) -> str:
    """净化 symbol 为日志文件名安全片段（剥离所有路径分隔符/特殊字符，限长）。"""
    clean = re.sub(r"[^\w\-]", "_", symbol or "auto")
    return (clean[:50] or "auto")


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class AutopilotJob:
    strategy_file: str
    mode: str
    symbol: str
    timeframe: str
    state: JobState = JobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_file": self.strategy_file,
            "mode": self.mode,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "state": self.state.value,
            "pid": self.pid,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
        }


class AutopilotManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: AutopilotJob | None = None
        self._log_fp = None
        self._stopped_by_user = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            return {
                "active": self._job is not None and self._job.state == JobState.RUNNING,
                "job": self._job.to_dict() if self._job else None,
            }

    def start(
        self,
        *,
        strategy_file: str,
        mode: str = "paper",
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> AutopilotJob:
        if mode not in _MODES:
            raise ValueError(f"未知 mode: {mode}（可选 {list(_MODES)}）")
        strategy_file = _validate_strategy_path(strategy_file)  # 路径穿越防护
        with self._lock:
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("已有自动驾驶任务在运行")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            clean_symbol = _safe_log_symbol(symbol)  # 剥离路径分隔符，防日志文件穿越
            log_path = (LOG_DIR / f"autopilot_{clean_symbol}_{ts}.log").resolve()
            if log_path.parent != _LOG_DIR_RESOLVED:
                raise ValueError("日志路径越界")

            cmd = [
                sys.executable, "-u", "run_autopilot.py",
                "--strategy-file", strategy_file,
                "--mode", mode,
            ]
            if symbol:
                cmd += ["--symbol", symbol]
            if timeframe:
                cmd += ["--timeframe", timeframe]

            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            # 子进程是第一方 run_autopilot.py，testnet/live 需要 OKX_*/MT5_* 等凭据，
            # ccxt 也可能依赖 HTTP_PROXY/SSL_CERT_FILE 等。沿用 training_manager 的
            # os.environ.copy() 模式（仓库既定约定），不做更紧的 env 白名单以免破坏配置加载。
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["LOGURU_COLORIZE"] = "0"

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._stopped_by_user = False
            self._proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            self._job = AutopilotJob(
                strategy_file=strategy_file,
                mode=mode,
                symbol=symbol or "",
                timeframe=timeframe or "",
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            return self._job

    def stop(self) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
            self._stopped_by_user = True
            try:
                self._proc.terminate()
            except Exception:
                self._proc.kill()
            return True

    def reset(self) -> dict[str, Any]:
        """一键清除：停止子进程 + 删除 autopilot_state.json（history/trades/权益记录全清）。

        下次启动 engine 新建 state：paper 权益回到起点（10000），live 重记真实余额。
        """
        from config import Config

        with self._lock:
            stopped = False
            if self._proc is not None and self._proc.poll() is None:
                self._stopped_by_user = True
                try:
                    self._proc.terminate()
                except Exception:
                    self._proc.kill()
                stopped = True
            # 等子进程退出，避免它退出前再写回 state
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
                self._refresh_state()
            # 删 state 文件（防越界校验）
            state_path = (PROJECT_ROOT / Config.AUTOPILOT_STATE_FILE).resolve()
            deleted = False
            try:
                state_path.relative_to(_PROJECT_ROOT_RESOLVED)
                if state_path.is_file():
                    state_path.unlink(missing_ok=True)
                    deleted = True
            except (ValueError, OSError):
                pass
            return {"stopped": stopped, "deleted": deleted}

    def tail_log(self, lines: int = 200) -> list[str]:
        with self._lock:
            if not self._job or not self._job.log_path:
                return []
            path = PROJECT_ROOT / self._job.log_path
            if not path.exists():
                return []
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return []
            return [strip_ansi(line) for line in content.splitlines()[-lines:]]

    def _refresh_state(self) -> None:
        if self._proc is None or self._job is None:
            return
        code = self._proc.poll()
        if code is None:
            return
        self._job.exit_code = code
        self._job.finished_at = datetime.now(timezone.utc).isoformat()
        if self._job.state == JobState.RUNNING:
            if self._stopped_by_user:
                self._job.state = JobState.STOPPED
            elif code == 0:
                self._job.state = JobState.COMPLETED
            elif code < 0:
                self._job.state = JobState.STOPPED
            else:
                self._job.state = JobState.FAILED
        if self._job.state == JobState.FAILED and self._job.error is None:
            self._job.error = f"自动驾驶进程异常退出 (exit_code={code})"
            try:
                if self._job.log_path:
                    path = PROJECT_ROOT / self._job.log_path
                    with path.open("a", encoding="utf-8") as fp:
                        fp.write(f"\n[Web] 自动驾驶进程已结束，退出码: {code}\n")
            except OSError:
                pass
        if self._log_fp:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        self._proc = None


autopilot_manager = AutopilotManager()
