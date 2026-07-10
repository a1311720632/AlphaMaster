"""Persisted UI settings for the training web console."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "web_settings.json"

_DEFAULT = {
    "last_data_file": "",
    "last_strategy_file": "",
    "debug_mode": False,
}


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return dict(_DEFAULT)
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT)
    out = dict(_DEFAULT)
    out.update({k: v for k, v in data.items() if k in _DEFAULT})
    out["debug_mode"] = bool(out.get("debug_mode", False))
    out["last_strategy_file"] = str(out.get("last_strategy_file") or "").strip()
    return out


def save_settings(data: dict) -> dict:
    current = load_settings()
    if "last_data_file" in data:
        current["last_data_file"] = str(data["last_data_file"] or "").strip()
    if "last_strategy_file" in data:
        current["last_strategy_file"] = str(data["last_strategy_file"] or "").strip()
    if "debug_mode" in data:
        current["debug_mode"] = bool(data["debug_mode"])
    SETTINGS_PATH.write_text(
        json.dumps(current, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return current
