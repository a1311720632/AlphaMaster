# 实时 3000 根门槛根因修复 — vm.py rolling z-score + live_signal.py 阈值解耦

## 日期
2026-08-03

## 目标
修复实时信号计算被错误要求 3000 根 K 线的问题。根因是 vm.py 的 `_normalize_output` 使用 expanding（累积）z-score，导致因子值依赖历史长度。改为 rolling z-score 后，实盘门槛降至 800。

## 问题分析

### 根因链
1. `model_core/vm.py` 的 `_normalize_output`（N=1 单品种模式）使用 expanding z-score：`ts_mean = cumsum / cnt`，最后一根 bar 的 mean/std 吃全部历史
2. 这导致同一数据点的因子值随历史长度漂移（−1.06@800 → −1.49@3000），甚至符号反转
3. `live_signal._min_bars()` 直接套用 `Config.MIN_BARS=3000`（训练侧阈值）作为实盘门槛
4. 结果：通达信 1h 数据服务器仅存约 2000 根，永远不够 3000 → 一直 insufficient

### 关键发现
- **指标本身只需 ~200 根**：raw factor 在 300→1988 范围内完全恒定（−0.11650）
- **发散 100% 来自归一化**：expanding z-score 的累积均值/标准差收敛缓慢
- 公式中的 VM 操作符 SCALE/JUMP 有全历史记忆，但实际公式未用到这些算子
- 真正的长记忆效应在 `_normalize_output` 的 expanding z-score

## 修改内容

### 改动 1：`model_core/vm.py` — `_normalize_output` N=1 分支
- **原逻辑**：expanding z-score（cumsum/cumsum_sq，最后一根 bar 依赖全部历史）
- **新逻辑**：rolling z-score（窗口 500，通过 pad+unfold 实现滑动窗口）
  - 最后一根 bar 只依赖最近 500 期，与历史长度无关
  - T < 500 时退化为 expanding（仍因果，无 look-ahead）
  - warm-up 期（前 499 根）输出 0（因子中性，不出信号）
- N>1 截面归一化不变，常数因子不变

### 改动 2：`strategy_manager/live_signal.py` — `_min_bars()` 解耦
- **原逻辑**：`max(Config.MIN_BARS, 500)` → 实盘门槛 3000
- **新逻辑**：独立阈值 800（特征 warm-up 200 + 滚动归一化 500），不引用 `Config.MIN_BARS`
- `_DEFAULT_MIN_BARS` 从 500 改为 800
- 支持 `Config.REALTIME_MIN_BARS` 覆盖

## 验证结果

### 因子稳定性测试（同一段数据，末尾对齐，不同历史长度）
| T | rolling z-score last_factor | expanding z-score last_factor |
|---|---|---|
| 500 | −1.488768 | −1.490259 |
| 800 | −1.488768 | −1.064148 |
| 1000 | −1.488768 | −1.259210 |
| 1200 | −1.488768 | −1.327126 |
| 1500 | −1.488768 | −1.175352 |
| 1988 | −1.488768 | −1.166829 |
| 3000 | −1.488768 | −1.274785 |

- Rolling：T≥500 后 **完全一致**（diff=0.00e+00）
- Expanding：随 T 漂移，最大差 0.42

### 边界测试
- T < 500（退化为 expanding）：正常，不报错
- 常数因子：原样返回
- N>1 截面 z-score：mean≈0, std≈1，不变
- warm-up 期：前 499 根全 0，第 500 根起有值
- `_min_bars()` = 800

## 注意事项
- `_normalize_output` 全局生效（训练/回测/实盘一致），旧策略因子刻度会变
- 建议重训策略以获得精确校准
- `Config.MIN_BARS=3000`（训练数据过滤）未动
