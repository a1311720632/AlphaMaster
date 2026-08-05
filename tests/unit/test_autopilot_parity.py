"""自动驾驶 ↔ 回测 奇偶不变量（ADR-0001 的可执行形式）。

核心断言：autopilot 的信号计算（autopilot.engine.compute_target_from_bars）与回测
**逐字同路径**（compute_features → StackVM.execute → compute_target_positions_stateless，
见 model_core/backtest.py:424），不加任何额外变换——不做 sign 塌缩、不做额外缩放/截断。

任何额外变换都会让本测试变红。这是“连续仓位·对齐回测”的中心守护。
"""
from __future__ import annotations

import pytest

from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB
from strategy_manager.signal import compute_target_positions_stateless
from web.data_sources.base import Bar, bars_to_raw_dict

from autopilot.engine import compute_target_from_bars

# 单特征公式（token 0 = 第一个激活特征）。足以触发 VM 的 normalize_output + tanh +
# 地板全链路；本测试守护的是 autopilot 的输出变换，不是 VM 内部（后者由 tests/property 覆盖）。
_FORMULA = [0]
_MIN_EXPOSURE = 0.05  # Config.MIN_TRADE_EXPOSURE


def _make_bars(n: int = 300, seed: int = 0) -> list[Bar]:
    """确定性的合成 OHLCV（无网络/无文件）。LCG 伪随机游走，保证可复现。

    n 必须 ≥ ~220：特征归一化用滚动窗 _NORM_WINDOW=200，不足则特征全零、信号恒 0。
    默认 300 与 AUTOPILOT_LOOKBACK_BARS 一致，给足 warmup。
    """
    bars: list[Bar] = []
    state = seed + 1
    price = 100.0
    ts0 = 1_700_000_000
    for i in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        rnd = (state / 0x7FFFFFFF) - 0.5  # ∈ [-0.5, 0.5]
        op = price
        price = max(1.0, price * (1.0 + rnd * 0.02))
        bars.append(
            Bar(
                ts=ts0 + i * 3600,
                open=op,
                high=max(op, price) * 1.005,
                low=min(op, price) * 0.995,
                close=price,
                volume=1000.0 + abs(rnd) * 100,
            )
        )
    return bars


def _backtest_target(formula: list[int], bars: list[Bar]) -> float:
    """手工复现回测同一条计算链（backtest.py:424）。"""
    raw = bars_to_raw_dict(bars)
    feats = MT5FeatureEngineer.compute_features(raw)
    factor = StackVM().execute([int(t) for t in formula], feats)
    assert factor is not None
    return float(compute_target_positions_stateless(factor)[0, -1].item())


def test_parity_autopilot_equals_backtest_chain():
    """autopilot 目标仓位 == 手工复现回测同一条链。零额外变换。"""
    bars = _make_bars(300)
    ap_pos, _ = compute_target_from_bars(_FORMULA, bars)
    bt_pos = _backtest_target(_FORMULA, bars)
    assert ap_pos is not None
    assert ap_pos == pytest.approx(bt_pos, abs=1e-6)


def test_parity_floor_and_range():
    """目标仓位 ∈ [-1,1] 且满足 MIN_TRADE_EXPOSURE 地板（要么 0，要么 |pos|≥阈值）。"""
    bars = _make_bars(300, seed=7)
    pos, _ = compute_target_from_bars(_FORMULA, bars)
    assert pos is not None
    assert -1.0 <= pos <= 1.0
    assert pos == 0.0 or abs(pos) >= _MIN_EXPOSURE - 1e-9


def test_no_sign_collapse():
    """autopilot 不塌缩成离散 {-1,0,+1}——这是 ADR-0001 修复的旧 MT5 缺陷。

    扫描多组 bar，至少存在一个连续小数仓位 |pos|∈(阈值, 0.99)，证明 sizing 是
    连续的而非 sign。若全部塌缩到 0/±1 则断言失败。
    """
    seen_fractional = False
    for seed in range(60):
        bars = _make_bars(300, seed=seed)
        pos, _ = compute_target_from_bars(_FORMULA, bars)
        if pos is None:
            continue
        if _MIN_EXPOSURE < abs(pos) < 0.99:
            seen_fractional = True
            break
    assert seen_fractional, "60 组样本中未观察到连续小数仓位——疑似被 sign 塌缩"


def test_invalid_formula_returns_none():
    """越界 token → None（不抛、不误下单）。"""
    bars = _make_bars(300)
    pos, _ = compute_target_from_bars([FORMULA_VOCAB.size + 5], bars)
    assert pos is None


# ── Hypothesis property 变体（与 tests/property 同套件）──────────────────────
from hypothesis import given, settings, strategies as st  # noqa: E402


@settings(max_examples=12, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=9_999_999),
    n=st.integers(min_value=220, max_value=300),
)
def test_parity_property(seed: int, n: int):
    """性质：任意合法 bar 序列 → autopilot 目标 == 回测目标，且满足值域/地板。"""
    bars = _make_bars(n, seed=seed)
    pos, _ = compute_target_from_bars(_FORMULA, bars)
    assert pos is not None
    assert pos == pytest.approx(_backtest_target(_FORMULA, bars), abs=1e-6)
    assert -1.0 <= pos <= 1.0
    assert pos == 0.0 or abs(pos) >= _MIN_EXPOSURE - 1e-9
