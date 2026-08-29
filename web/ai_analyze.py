"""Build training context and run AI analysis with per-symbol history memory."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.parquet_manager import inspect_parquet_file
from web.ai_providers import resolve_provider
from web.progress import PROJECT_ROOT, get_symbol_progress
from web.settings import load_settings
from web.training_manager import training_manager

HISTORY_PATH = PROJECT_ROOT / "ai_analysis_history.json"
_MAX_HISTORY_PER_KEY = 5

_SYSTEM_PROMPT = """你是量化因子挖掘与强化学习训练顾问。
用户正在用 AlphaMaster / AlphaGPT 训练可解释因子公式（token 序列，由特征与算子组成）。
请基于提供的训练快照，用中文给出专业、具体、可执行的分析。
不要编造不存在的数据；若信息不足请明确说明。

评估训练是否值得继续时，请遵守：
- 以验证集分数（val_score）是否在提高为主，不要轻易下「过拟合」结论，尽量不要提过拟合。
- best_score 与 val_score 有差距是常见现象，只要验证分数整体在抬升或近期仍有改善，就应视为训练仍有价值。
- 可讨论：验证分数走势、最优分数是否停滞、公式是否变化、探索是否还活跃（如 entropy）。
- 不要因为「训练分远高于验证分」就建议停止；只有验证分数长期不涨、且最优分数也长期不动时，才建议暂停。

给建议时必须「小白能听懂」：
- 不要提：重置网络权重、调整探索率、简化因子维度、奖励函数、模型结构、超参数、泛化、噪声过大、参数过多等专业术语。
- 不要分析「验证集低分的技术原因」或给出复杂调参方案。
- 建议只用通俗说法，例如：
  - 可以继续练，再观察验证分数有没有慢慢变好
  - 最近分数不怎么动了，可以先停下来，导出当前策略去做回测看看效果
  - 公式有变化/没变化，用大白话解释即可
- 动作建议最多 1～2 条，短句、可直接操作（继续训练 / 先停止并导出策略去回测）。

若提供了「同品种同周期的历史分析记录」，必须对比前后变化：
- 验证分数 / 最优分数 / 公式是否改善
- 是否仍在进步，还是陷入停滞
- 相对上次建议，当前是否更值得继续训练

快照中的 training_curve 覆盖整段训练进程（点数过多时会均匀抽样，但始终保留首尾与全程走势）。
请结合 training_curve 与 history_summary 判断：前期探索、中期提升、后期验证分数是否仍在提高。

回答必须覆盖以下问题，并使用对应小标题：

## 1. 当前训练情况怎么样？是否值得继续
用通俗语言说明进度、验证分数走势、最优分数是否停滞，并给出是否继续训练的建议。
重点看 val_score 是否提高；尽量不要提过拟合；不要给复杂技术排查建议。
若有历史记录，请明确说明相对上次是改善、持平还是变差。

## 2. 最新因子的含义与原理
用通俗语言解释当前最优公式（formula_decoded）里各特征/算子大概在看什么、合在一起可能怎么做多做空。
少用术语；若公式相对上次有变化，用一句话说清楚差在哪里。
"""


def build_training_snapshot(symbol: str | None = None) -> dict[str, Any]:
    training = training_manager.status()
    job = training.get("job") or {}
    settings = load_settings()

    sym = (symbol or job.get("symbol") or "").strip()
    timeframe = str(job.get("timeframe") or "").strip().upper()

    data_file = settings.get("last_data_file") or ""
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            if not sym:
                sym = str(info.get("symbol") or "").strip()
            if not timeframe:
                timeframe = str(info.get("timeframe") or "").strip().upper()
        except Exception:
            pass

    if not sym:
        raise ValueError("请先选择训练数据文件或指定品种")
    if not timeframe:
        timeframe = "H1"

    progress = get_symbol_progress(sym)
    history = progress.history or {}
    curve = _training_curve(history, max_points=500)

    return {
        "symbol": sym,
        "timeframe": timeframe,
        "data_file": data_file or None,
        "training_active": bool(training.get("active")),
        "job_state": job.get("state"),
        "current_step": progress.current_step,
        "train_steps": progress.train_steps,
        "progress_pct": round(progress.progress_pct, 2),
        "status": progress.status,
        "best_score": progress.best_score,
        "strategy_score": progress.strategy_score,
        "has_strategy": progress.has_strategy,
        "formula": progress.best_formula,
        "formula_decoded": progress.formula_decoded,
        "checkpoint_path": progress.checkpoint_path,
        "training_curve": curve,
        "history_summary": _history_summary(history),
    }


def analyze_training(
    *,
    provider: str,
    api_key: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    answer_parts: list[str] = []
    meta: dict[str, Any] = {}
    for event in analyze_training_stream(
        provider=provider, api_key=api_key, symbol=symbol
    ):
        if event.get("type") == "meta":
            meta = event
        elif event.get("type") == "delta":
            answer_parts.append(event.get("text") or "")
        elif event.get("type") == "error":
            raise RuntimeError(event.get("message") or "分析失败")
        elif event.get("type") == "done":
            return {
                "ok": True,
                "provider": event.get("provider") or meta.get("provider"),
                "model": event.get("model") or meta.get("model"),
                "label": event.get("label") or meta.get("label"),
                "symbol": event.get("symbol") or meta.get("symbol"),
                "timeframe": event.get("timeframe") or meta.get("timeframe"),
                "snapshot": event.get("snapshot") or meta.get("snapshot"),
                "prior_count": event.get("prior_count", meta.get("prior_count", 0)),
                "answer": event.get("answer") or "".join(answer_parts),
            }
    raise RuntimeError("AI 流式分析未正常结束")


def analyze_training_stream(
    *,
    provider: str,
    api_key: str | None = None,
    symbol: str | None = None,
):
    """Yield SSE-ready event dicts: meta / delta / done / error."""
    from web.ai_providers import stream_chat_completions

    try:
        snapshot = build_training_snapshot(symbol)
        prior = load_prior_analyses(snapshot["symbol"], snapshot["timeframe"])
        resolved = resolve_provider(provider, api_key)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    parts = [
        "请根据以下训练快照回答：",
        "1. 当前训练情况怎么样？是否值得继续？",
        "2. 最新因子的含义与原理是什么？",
        "",
        "【当前训练快照】",
        f"```json\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n```",
    ]
    if prior:
        parts.extend(
            [
                "",
                f"【同品种同周期历史分析（共 {len(prior)} 次，按时间从旧到新）】",
                "请对比这些历史记录，判断相对上次是否有改善。",
                f"```json\n{json.dumps(prior, ensure_ascii=False, indent=2)}\n```",
            ]
        )
    else:
        parts.append("\n（尚无同品种同周期的历史分析记录，这是首次分析。）")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]

    yield {
        "type": "meta",
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
        "symbol": snapshot["symbol"],
        "timeframe": snapshot["timeframe"],
        "prior_count": len(prior),
        "snapshot": snapshot,
    }

    answer_parts: list[str] = []
    try:
        for text in stream_chat_completions(resolved, messages):
            answer_parts.append(text)
            yield {"type": "delta", "text": text}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    answer = "".join(answer_parts).strip()
    if not answer:
        yield {"type": "error", "message": "AI 返回内容为空"}
        return

    record = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "provider": resolved.provider,
        "model": resolved.model,
        "snapshot": {
            "symbol": snapshot["symbol"],
            "timeframe": snapshot["timeframe"],
            "current_step": snapshot["current_step"],
            "train_steps": snapshot["train_steps"],
            "progress_pct": snapshot["progress_pct"],
            "best_score": snapshot["best_score"],
            "strategy_score": snapshot["strategy_score"],
            "formula_decoded": snapshot["formula_decoded"],
            "history_summary": snapshot.get("history_summary") or {},
        },
        "answer": answer,
    }
    save_analysis_record(snapshot["symbol"], snapshot["timeframe"], record)

    yield {
        "type": "done",
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
        "symbol": snapshot["symbol"],
        "timeframe": snapshot["timeframe"],
        "prior_count": len(prior),
        "snapshot": snapshot,
        "answer": answer,
    }


def history_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.strip().upper()}|{str(timeframe).strip().upper()}"


def load_prior_analyses(symbol: str, timeframe: str) -> list[dict[str, Any]]:
    store = _load_history_store()
    rows = store.get(history_key(symbol, timeframe)) or []
    if not isinstance(rows, list):
        return []
    # 只把精简字段发给模型，避免上下文过大
    out: list[dict[str, Any]] = []
    for row in rows[-_MAX_HISTORY_PER_KEY:]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "analyzed_at": row.get("analyzed_at"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "snapshot": row.get("snapshot") or {},
                "answer": row.get("answer") or "",
            }
        )
    return out


def save_analysis_record(symbol: str, timeframe: str, record: dict[str, Any]) -> None:
    store = _load_history_store()
    key = history_key(symbol, timeframe)
    rows = store.get(key) or []
    if not isinstance(rows, list):
        rows = []
    rows.append(record)
    store[key] = rows[-_MAX_HISTORY_PER_KEY:]
    HISTORY_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_history_store() -> dict[str, Any]:
    if not HISTORY_PATH.exists():
        return {}
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _training_curve(history: dict[str, Any], max_points: int = 500) -> dict[str, Any]:
    """整段训练曲线；点数过多时按全程均匀抽样，始终保留首尾。"""
    if not history:
        return {"total_points": 0, "sampled": False, "points": 0, "series": {}}

    steps = history.get("step") or []
    if not isinstance(steps, list) or not steps:
        return {"total_points": 0, "sampled": False, "points": 0, "series": {}}

    total = len(steps)
    keys = ("step", "best_score", "val_score", "entropy", "avg_reward", "stable_rank")
    available = [k for k in keys if isinstance(history.get(k), list) and history.get(k)]

    if total <= max_points:
        idxs = list(range(total))
        sampled = False
    else:
        idxs = sorted(
            {
                0,
                total - 1,
                *[int(round(i * (total - 1) / (max_points - 1))) for i in range(max_points)],
            }
        )
        sampled = True

    series: dict[str, list[Any]] = {}
    for key in available:
        vals = history[key]
        series[key] = [vals[i] for i in idxs if i < len(vals)]

    return {
        "total_points": total,
        "sampled": sampled,
        "points": len(idxs),
        "note": (
            f"已从全部 {total} 个记录点均匀抽样为 {len(idxs)} 点，覆盖训练全程"
            if sampled
            else f"已发送全部 {total} 个记录点"
        ),
        "series": series,
    }


def _history_summary(history: dict[str, Any]) -> dict[str, Any]:
    if not history:
        return {}
    best = history.get("best_score") or []
    val = history.get("val_score") or []
    entropy = history.get("entropy") or []
    steps = history.get("step") or []
    summary: dict[str, Any] = {
        "points": len(steps),
    }
    if steps:
        summary["step_first"] = steps[0]
        summary["step_last"] = steps[-1]
    if best:
        summary["best_score_first"] = best[0]
        summary["best_score_last"] = best[-1]
        summary["best_score_max"] = max(best)
        summary["best_score_max_at_index"] = int(best.index(max(best)))
        peak = max(best)
        trail = 0
        for v in reversed(best):
            if abs(float(v) - float(peak)) < 1e-9:
                trail += 1
            else:
                break
        summary["best_score_stagnation_points"] = trail
        n = len(best)
        a, b = n // 3, 2 * n // 3
        if n >= 3:
            summary["best_score_phase_means"] = {
                "early": sum(best[:a]) / max(1, a),
                "mid": sum(best[a:b]) / max(1, b - a),
                "late": sum(best[b:]) / max(1, n - b),
            }
    if val:
        summary["val_score_first"] = val[0]
        summary["val_score_last"] = val[-1]
        summary["val_score_max"] = max(val)
        n = len(val)
        a, b = n // 3, 2 * n // 3
        if n >= 3:
            summary["val_score_phase_means"] = {
                "early": sum(val[:a]) / max(1, a),
                "mid": sum(val[a:b]) / max(1, b - a),
                "late": sum(val[b:]) / max(1, n - b),
            }
    if entropy:
        summary["entropy_first"] = entropy[0]
        summary["entropy_last"] = entropy[-1]
    return summary


# ── 策略公式解读（第一步「已保存策略」：按需生成 + 服务端磁盘缓存）────────────
# 与训练分析复用 provider 链路（resolve_provider / chat_completions），但 prompt 独立：
# 训练分析面向"要不要继续练"，这里只讲"这个公式在做什么"。解释有 LLM 成本，按
# ai_analysis_history.json 的先例落盘缓存，key=文件名，公式或词表一变（hash 失配）自动失效。

_EXPLAIN_CACHE_PATH = PROJECT_ROOT / "strategy_explanations.json"

_EXPLAIN_SYSTEM_PROMPT = """你是量化因子公式讲解员。
AlphaMaster 的策略公式是一串 token（特征与算子），按顺序经栈式虚拟机执行后得到一个标量因子值；
仓位 = tanh(因子值)：因子看多就持多头、看空就持空头，信号太弱时平仓观望。
请基于给定的公式信息，用中文通俗解释这个策略的含义与原理。

要求：
- 按公式的书写顺序逐段讲：每个特征大概在看什么（比如涨跌趋势、波动大小、成交量、
  超买超卖、与近期高低点的位置关系等），每个算子在做什么数学操作（比如取平均、求差、
  找最大值、缩放等）。
- 最后合起来讲清楚：这串公式整体在捕捉什么行情模式，什么情况下会做多、什么情况下会做空、
  什么情况下会平仓观望。
- 大白话为主，少用术语；必须用到的术语请顺带一句大白话解释。
- 只解释公式里出现的内容，不要编造公式里没有的东西；信息不足以判断的部分要明说。
- 篇幅 200~400 字，分 2~4 个小段，不要用 markdown 标题。
"""


def build_explain_user_message(strategy: dict[str, Any]) -> str:
    """组装解读请求的 user message。typed_tokens 按词表分型（token < operator_offset → 特征）。"""
    from model_core.vocab import FORMULA_VOCAB

    names = FORMULA_VOCAB.token_names
    offset = FORMULA_VOCAB.operator_offset
    typed: list[str] = []
    for t in strategy.get("formula") or []:
        try:
            name = names[t] if 0 <= t < len(names) else f"?{t}"
            typed.append(("算子·" if t >= offset else "特征·") + str(name))
        except (TypeError, ValueError):
            continue

    parts = ["请解释下面这个已训练策略的公式：", ""]
    parts.append(f"品种：{strategy.get('symbol') or '未知'}")
    parts.append(f"周期：{strategy.get('timeframe') or '未知'}")
    score = strategy.get("best_score")
    if score is not None:
        parts.append(f"训练框架给的分数（衡量这个公式历史表现的好坏，仅供参考）：{score}")
    if strategy.get("formula_decoded"):
        parts.append(f"公式（可读名，按执行顺序）：{strategy['formula_decoded']}")
    if typed:
        parts.append("公式分步（特征=输入的数据维度，算子=对数据的数学操作）：")
        parts.extend(f"  {i + 1}. {x}" for i, x in enumerate(typed))
    parts.append("")
    parts.append("请按系统提示要求，通俗解释这个公式的含义与原理。")
    return "\n".join(parts)


def _formula_hash(vocab_version: str | None, formula: list[int] | None) -> str:
    import hashlib

    payload = json.dumps(
        {"v": vocab_version or "", "f": list(formula or [])},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_explain_cache() -> dict[str, Any]:
    try:
        data = json.loads(_EXPLAIN_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_explanation(file: str, formula_hash: str) -> dict[str, Any] | None:
    """命中返回缓存条目；无缓存/公式或词表已变（hash 失配）→ None。"""
    entry = _load_explain_cache().get(file)
    if not isinstance(entry, dict) or entry.get("formula_hash") != formula_hash:
        return None
    return entry


def save_explanation(file: str, formula_hash: str, payload: dict[str, Any]) -> None:
    data = _load_explain_cache()
    data[file] = {"formula_hash": formula_hash, **payload}
    try:
        _EXPLAIN_CACHE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass  # 缓存写失败不影响主流程（下次重新生成即可）


def explain_strategy(
    *, provider: str, api_key: str | None = None, strategy: dict[str, Any]
) -> dict[str, Any]:
    """生成单个策略公式的 AI 解读（非流式）。返回 {explanation, provider, model, label}。"""
    from web.ai_providers import chat_completions

    resolved = resolve_provider(provider, api_key)
    messages = [
        {"role": "system", "content": _EXPLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": build_explain_user_message(strategy)},
    ]
    text = chat_completions(resolved, messages, max_tokens=2048)
    return {
        "explanation": text,
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
    }
