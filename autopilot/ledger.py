"""冷账本（Cold Ledger，E1/ADR-0007）：append-only JSONL 流水，永不删头。

与 hot state（autopilot_state.json，cap 1000 的滚动窗口）分工：
  - hot  = "现在怎么样"——前端 4s 轮询的实时面板读它；小、快、原地覆写。
  - cold = "发生过什么"——日历/权益曲线/成交记录/事件时间线等审计视图读它。
    1h bar 下 hot 的 1000 根 ≈ 42 天，长跑后头部静默滑出窗口；cold 永不丢。

行格式：{"type": "bar"|"trade"|"event", "ts": <unix秒>, ...}（ts 对 bar/trade 用
bar 的时间戳而非写入时刻，保证与 hot state 的记录可对齐）。

写侧（engine 进程内单例，懒开句柄）：append 失败 swallow + log——账本写失败
不杀交易进程。读侧（web tail）：每次短开短关，不持锁，坏行跳过。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

TYPES = ("bar", "trade", "event")


def _safe_symbol(symbol: str) -> str:
    """symbol → 文件名安全片段（与 autopilot_manager._safe_log_symbol 同正则）。"""
    clean = re.sub(r"[^\w\-]", "_", symbol or "auto")
    return clean[:50] or "auto"


def ledger_path(symbol: str, mode: str, base_dir: str | Path | None = None) -> Path:
    """冷账本路径：{base_dir}/autopilot_ledger_{symbol}_{mode}.jsonl（base_dir 空=项目根）。"""
    base = Path(base_dir) if base_dir else Path.cwd()
    return base / f"autopilot_ledger_{_safe_symbol(symbol)}_{mode}.jsonl"


class Ledger:
    """单进程写侧 + 跨进程读侧。写侧懒开 "a" 句柄常驻；读侧 tail 短开短关。"""

    def __init__(self, path: str | Path, log: Callable[[str], None] = print) -> None:
        self.path = Path(path)
        self._log = log
        self._fp = None  # 懒开：首次 append 才建文件/目录

    # ── 写侧（engine 进程内）──────────────────────────────────────────
    def append(self, type_: str, data: dict[str, Any]) -> None:
        if type_ not in TYPES:
            raise ValueError(f"未知账本行类型: {type_}（可选 {list(TYPES)}）")
        try:
            if self._fp is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fp = self.path.open("a", encoding="utf-8")
            row = {"type": type_, **data}
            self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fp.flush()
        except OSError as exc:  # noqa: BLE001 - 账本写失败不杀交易进程
            self._log(f"[autopilot] 冷账本写入失败: {exc}")

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:  # noqa: BLE001
                pass
            self._fp = None

    # ── 读侧（web 进程）───────────────────────────────────────────────
    def tail(self, n: int = 200, types: set[str] | None = None) -> list[dict[str, Any]]:
        """最近 n 行（可按类型过滤）。文件不存在/坏行 → 尽力而为，不抛。"""
        if n <= 0:
            return []
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # 坏行（半写状态等）跳过，读侧永不抛
            if types and row.get("type") not in types:
                continue
            out.append(row)
            if len(out) > n * 3:  # 过滤后仍留裕量：粗截，最后再切尾巴
                out = out[-n * 2:]
        return out[-n:]

    def read_all(self, types: set[str] | None = None) -> list[dict[str, Any]]:
        """全量读（readiness 汇总用）。文件不存在 → []。"""
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if types and row.get("type") not in types:
                continue
            out.append(row)
        return out
