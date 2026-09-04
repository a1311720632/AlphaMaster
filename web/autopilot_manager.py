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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.train_logging import strip_ansi
from web.settings import load_settings, save_settings

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


# β 自动续命用的 PID 文件（与 autopilot_state.json 同目录，在 PROJECT_ROOT 内）
_PID_PATH = PROJECT_ROOT / "autopilot_child.pid"

# 周期 → 秒（镜像 autopilot/engine._TF_SECONDS，避免把整个 engine 拉进 manager）
_TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000,
}


def _log(msg: str) -> None:
    """β 续命决策日志（走 web.server_log，与 _startup_realtime 同口径；失败回落 stderr）。"""
    try:
        from web.server_log import get_logger
        get_logger().info(msg)
    except Exception:  # noqa: BLE001
        print(f"[autopilot-manager] {msg}", file=sys.stderr, flush=True)


def _is_pid_alive(pid: int) -> bool:
    """跨平台判活。Windows 用 OpenProcess(QUERY_LIMITED_INFORMATION)，POSIX 用 os.kill(pid,0)。"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 进程存在但无权限
    except OSError:
        return False
    return True


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
        # β 自动续命（镜像 realtime_manager._loaded 的幂等套路）
        self._boot_relaunch_done = False        # 防开机多次 fire
        self._watcher_thread: threading.Thread | None = None
        self._watcher_evt = threading.Event()    # 可中断睡眠 + 关停信号
        self._relaunch_attempt = 0               # 退避计数（10→60→300，5 次封顶）
        self._last_start_monotonic = 0.0         # 稳定运行 >120s 则重置退避
        self._last_handled_finished_at: str | None = None  # 去重：哪次退出已处理
        self._last_exit_reason = ""              # _refresh_state 在终结态瞬间捕获

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
            # β 自动续命：记 PID（孤儿守护用）+ 重置退避 + 起 watcher
            self._write_pid(self._proc.pid)
            self._relaunch_attempt = 0
            self._last_start_monotonic = time.monotonic()
            self._last_handled_finished_at = None
            self._ensure_watcher()
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
        """一键清除：停止子进程 + 删 autopilot_state.json + 归档冷账本（历史全清回初始）。

        下次启动 engine 新建 state：paper 权益回到起点（10000），live 重记真实余额。
        冷账本（ledger JSONL）一并归档清零（2026-09-04）——只删 state 时，下次启动
        新 state 三元组又指向同一 ledger 文件，旧成交/曲线全部回来，"清除"落空。
        归档而非直删：append-only 审计语义（ADR-0007 ⑧），.cleared_* 先例已有。
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
            # 归档冷账本：三元组取自 web_settings（UI 最后一次启动的配置）
            archived = self._archive_ledger(load_settings())
            return {"stopped": stopped, "deleted": deleted, "ledger_archived": archived}

    def _archive_ledger(self, settings: dict[str, Any]) -> str:
        """按 settings 三元组归档 ledger 主文件 → {name}.cleared_{YYYYMMDD}[_n]。

        symbol/mode 缺失或文件不存在 → 空串（无账可清，不算失败）。归档失败（占用
        等）也返回空串——state 已删，账本残留无害，别阻断一键清除。
        """
        symbol = str(settings.get("autopilot_symbol") or "")
        mode = str(settings.get("autopilot_mode") or "")
        if not symbol or mode not in _MODES:
            return ""
        from config import Config

        from autopilot.ledger import ledger_path

        lp = ledger_path(symbol, mode, Config.AUTOPILOT_LEDGER_DIR or PROJECT_ROOT)
        if not lp.is_file():
            return ""
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        dst = lp.with_name(f"{lp.name}.cleared_{date}")
        n = 2
        while dst.exists():  # 同日多次清除：追加序号防覆盖
            dst = lp.with_name(f"{lp.name}.cleared_{date}_{n}")
            n += 1
        try:
            lp.rename(dst)
        except OSError:
            return ""
        return dst.name

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
            # β：终结态瞬间捕获 halt reason（唯一可靠源，供 watcher 决策）+ 清 PID 文件
            self._last_exit_reason = self._read_breaker_reason()
            self._clear_pid()
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

    # ── β 自动续命 ───────────────────────────────────────────────────────
    def relaunch_if_intended(self) -> None:
        """web 启动时调用：若 autopilot_intended_running=True，从持久化设置重拉 autopilot。
        镜像 realtime_manager.load_persisted：幂等 + 全程 try/except（绝不崩 app）。
        先无条件对账孤儿（关掉"stop 置 flag=False 后 web 死、孤儿还活"的洞）。"""
        if self._boot_relaunch_done:
            return
        self._boot_relaunch_done = True
        try:
            self._reconcile_orphan()
            settings = load_settings()
            if not settings.get("autopilot_intended_running"):
                return
            # advisory stay-down：上一轮 halt 是回撤/STOP_SIGNAL → 不自动复活
            # （注：breaker_reason 在健康运行期可能陈旧，故仅为建议；权威判断在 watcher）
            reason = self._last_exit_reason or self._read_breaker_reason()
            if reason.startswith("回撤熔断") or reason.startswith("STOP_SIGNAL"):
                save_settings({"autopilot_intended_running": False})
                _log(f"boot relaunch: 抑制（上次 halt={reason}），已清意图标志")
                return
            sf, mode, symbol, tf = self._relaunch_args()
            if not sf:
                return
            try:
                self.start(strategy_file=sf, mode=mode, symbol=symbol, timeframe=tf)
                _log(f"boot relaunch: 已重拉 {sf}")
            except (ValueError, RuntimeError) as exc:
                save_settings({"autopilot_intended_running": False})
                _log(f"boot relaunch: 失败（{exc}），已清意图标志")
        except Exception as exc:  # noqa: BLE001 - 绝不崩 app
            try:
                from web.server_log import log_error
                log_error("autopilot boot relaunch handler failed", exc)
            except Exception:  # noqa: BLE001
                _log(f"boot relaunch handler 异常: {exc}")

    def _relaunch_args(self) -> tuple[str, str, str | None, str | None]:
        """从 web_settings 读重拉所需四元组（strategy_file, mode, symbol, timeframe）。"""
        settings = load_settings()
        sf = str(settings.get("autopilot_last_strategy") or "").strip()
        mode = str(settings.get("autopilot_mode") or "paper")
        symbol = settings.get("autopilot_symbol") or None
        tf = settings.get("autopilot_timeframe") or None
        return sf, mode, symbol, tf

    def _ensure_watcher(self) -> None:
        """起 watcher 守护线程（幂等）。仅在 start() 内调用（持锁），无并发竞态。"""
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            return
        self._watcher_evt.clear()
        self._watcher_thread = threading.Thread(
            target=self._watcher_loop, name="autopilot-watcher", daemon=True
        )
        self._watcher_thread.start()

    def _watcher_loop(self) -> None:
        """退避重启引擎。2s 轮询 _refresh_state；检测到新终结态按 reason 决定
        stay_down/relaunch。锁内只做快照与 _refresh_state；退避睡眠与 start() 调用
        都在锁外（start 自己拿 self._lock，避免死锁）。"""
        while not self._watcher_evt.is_set():
            with self._lock:
                self._refresh_state()
                job = self._job
                finished_at = job.finished_at if job else None
                job_state = job.state if job else None
                exit_code = job.exit_code if job else None
                reason = self._last_exit_reason
                stopped_by_user = self._stopped_by_user
                # 稳定运行 >120s → 重置退避（瞬时 blip 后快速恢复）
                if (job is not None and job_state == JobState.RUNNING
                        and self._last_start_monotonic > 0
                        and time.monotonic() - self._last_start_monotonic > 120):
                    self._relaunch_attempt = 0

            # 无终结任务 / 这次退出已处理 → 睡 2s 再看
            if (job is None or finished_at is None
                    or finished_at == self._last_handled_finished_at):
                if self._watcher_evt.wait(timeout=2.0):
                    return
                continue

            self._last_handled_finished_at = finished_at
            action = self._classify_exit(stopped_by_user, exit_code, reason)
            _log(
                f"watcher: 终结 state={job_state} code={exit_code} "
                f"reason={reason!r} stopped_by_user={stopped_by_user} → {action}"
            )
            self._notify_watcher_event(job, job_state, exit_code, reason, action)

            if action == "stay_down":
                save_settings({"autopilot_intended_running": False})
                _log("watcher: stay-down，已清意图标志")
                if self._watcher_evt.wait(timeout=2.0):
                    return
                continue

            # action == "relaunch"：内层退避重试，直到成功 / 放弃 / 关停
            while not self._watcher_evt.is_set():
                delay = (10, 60, 300)[min(self._relaunch_attempt, 2)]
                _log(f"watcher: {delay}s 后重拉（attempt {self._relaunch_attempt + 1}/5）")
                if self._watcher_evt.wait(timeout=delay):
                    return
                sf, mode, symbol, tf = self._relaunch_args()
                if not sf:
                    save_settings({"autopilot_intended_running": False})
                    _log("watcher: 无 strategy_file，已清意图标志")
                    break
                try:
                    self.start(strategy_file=sf, mode=mode, symbol=symbol, timeframe=tf)
                    _log(f"watcher: 重拉成功 {sf}")
                    break  # start() 已重置 _relaunch_attempt=0；回外层等下一次终结
                except (ValueError, RuntimeError) as exc:
                    self._relaunch_attempt += 1
                    _log(f"watcher: 重拉失败（{exc}），attempt={self._relaunch_attempt}")
                    if self._relaunch_attempt >= 5:
                        save_settings({"autopilot_intended_running": False})
                        _log("watcher: 连续 5 次失败，放弃，已清意图标志")
                        break

    def _notify_watcher_event(self, job, job_state, exit_code, reason, action) -> None:
        """watcher 检测到进程终结 → 飞书知会（🟡 级，B4/ADR-0007）。失败静默。"""
        try:
            from web.feishu_notify import send_text

            sym = job.symbol if job else "?"
            mode = job.mode if job else "?"
            text = (
                f"[autopilot-watcher][{sym}/{mode}] 进程退出: state={job_state} "
                f"code={exit_code} reason={reason or '-'} → {action}"
            )
            send_text(text)
        except Exception:  # noqa: BLE001 - 知会失败绝不影响 watcher 主逻辑
            pass

    def _classify_exit(
        self, stopped_by_user: bool, exit_code: int | None, reason: str
    ) -> str:
        """终结态 → 'stay_down' | 'relaunch'（权威表见部署 plan + ADR-0007）。
        stay_down：用户 stop / 回撤熔断 / 执行熔断 / STOP_SIGNAL / 干净退出(exit 0)。
        relaunch：断网熔断 / 未知崩溃(code>0) / 信号死亡(code<0)。
        执行熔断 stay_down 理由：能穿透 bar 内 3 次重试的失败基本是持久的
        （凭据吊销/保证金不足），重拉无意义且刷屏——资金安全的终点是人。"""
        if stopped_by_user:
            return "stay_down"
        if (reason.startswith("回撤熔断") or reason.startswith("执行熔断")
                or reason.startswith("STOP_SIGNAL")):
            return "stay_down"
        if exit_code == 0:
            return "stay_down"
        return "relaunch"

    def shutdown(self) -> None:
        """web 关停时调用：唤醒 watcher 使其退出。"""
        self._watcher_evt.set()
        t = self._watcher_thread
        if t is not None:
            try:
                t.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass

    def _reconcile_orphan(self) -> None:
        """开机无条件执行：若发现一个活的 autopilot 孤儿子进程（上次 web 崩溃后残留），
        用引擎现成的 STOP_SIGNAL 契约让它优雅退出。保证 autopilot_state.json 唯一主人，
        新 web 拿到真实 _proc 句柄（stop()/status() 才能用）。"""
        try:
            if not _PID_PATH.exists():
                return
            pid_text = _PID_PATH.read_text(encoding="utf-8").strip()
            if not pid_text.isdigit():
                self._clear_pid()
                return
            pid = int(pid_text)
            if not _is_pid_alive(pid):
                self._clear_pid()
                return
            # PID 复用防御：state 不新鲜 → 大概率不是我们的孤儿
            if not self._state_is_fresh():
                _log(f"orphan reconcile: pid={pid} 活但 state 陈旧，视为 PID 复用，清 PID 文件")
                self._clear_pid()
                return
            _log(f"orphan reconcile: 发现活孤儿 pid={pid}，写 STOP_SIGNAL 优雅退出")
            self._stop_orphan_via_signal(pid)
        except Exception as exc:  # noqa: BLE001
            _log(f"orphan reconcile 异常: {exc}")

    def _state_is_fresh(self) -> bool:
        """autopilot_state.json 的 last_ts 在 2×bar_seconds 内 → 视为活进程在写。"""
        try:
            from config import Config
            from autopilot.state import AutopilotState
            st = AutopilotState.load(Config.AUTOPILOT_STATE_FILE)
            if not st or st.last_ts <= 0:
                return False
            bar_seconds = _TF_SECONDS.get(st.timeframe or "1h", 3600)
            age = int(time.time()) - int(st.last_ts)
            return age <= 2 * bar_seconds
        except Exception:  # noqa: BLE001
            return False

    def _stop_orphan_via_signal(self, pid: int, timeout_s: float = 15.0) -> None:
        """写 STOP_SIGNAL 让引擎优雅退出（run_forever 每轮检查），轮询 pid 至死；
        超时则硬杀（SIGKILL / taskkill）。退出后务必删 STOP_SIGNAL，免得新子进程一启动就被它停。"""
        try:
            from config import Config
            (PROJECT_ROOT / Config.AUTOPILOT_STOP_SIGNAL).touch()
        except OSError as exc:  # noqa: BLE001
            _log(f"orphan stop: 写 STOP_SIGNAL 失败: {exc}")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not _is_pid_alive(pid):
                self._remove_stop_signal()
                self._clear_pid()
                _log(f"orphan stop: pid={pid} 已优雅退出")
                return
            time.sleep(0.5)
        _log(f"orphan stop: pid={pid} {timeout_s:.0f}s 未退，硬杀")
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception as exc:  # noqa: BLE001
            _log(f"orphan stop: 硬杀失败: {exc}")
        self._remove_stop_signal()
        self._clear_pid()

    def _remove_stop_signal(self) -> None:
        try:
            from config import Config
            p = PROJECT_ROOT / Config.AUTOPILOT_STOP_SIGNAL
            if p.exists():
                p.unlink()
        except OSError:  # noqa: BLE001
            pass

    def _write_pid(self, pid: int) -> None:
        try:
            _PID_PATH.write_text(str(pid), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            _log(f"写 PID 文件失败: {exc}")

    def _clear_pid(self) -> None:
        try:
            _PID_PATH.unlink(missing_ok=True)
        except OSError:  # noqa: BLE001
            pass

    def _read_breaker_reason(self) -> str:
        """读 autopilot_state.json 的 breaker_reason（halt 瞬间写入）。失败/无文件 → ''。"""
        try:
            from config import Config
            from autopilot.state import AutopilotState
            st = AutopilotState.load(Config.AUTOPILOT_STATE_FILE)
            return (st.breaker_reason if st else "") or ""
        except Exception:  # noqa: BLE001
            return ""


autopilot_manager = AutopilotManager()
