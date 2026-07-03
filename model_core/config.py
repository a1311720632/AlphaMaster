"""
model_core/config.py — 模型层配置

仅保留模型训练所需的参数。
品种、数据、风控等全局配置统一由根目录 config.py 的 Config 类管理。
"""
import torch
from .vocab import FORMULA_VOCAB


class ModelConfig:
    # ── 训练设备 ─────────────────────────────────────────────────────────
    # 注意：本任务 CPU 训练速度反而比 GPU 快（实测约 2.3 倍），故强制用 CPU。
    # 原因：
    #   1. 张量太小——forex 组仅 (2 品种 × 3508 × 20 特征)，单个算子的
    #      计算量小于 CUDA kernel 启动开销（数十微秒），GPU 算得快但启动慢。
    #   2. 训练循环是 Python 串行调度：每 step 逐条跑 128 条公式 × 8 个
    #      VM 步 × 4 个 walk-forward 折，GPU 被切成上万个碎片段，吃不满。
    #   3. host↔device 拷贝 + kernel 启动延迟主导总耗时，而非张量计算本身。
    #   4. 实测 GPU 利用率 ~51%，正是 GPU 一半时间在干等 Python 喂下一个
    #      kernel 的证据（不是“还能压榨”，而是“调度瓶颈”）。
    # 基准测试（forex 组, 50 步, RTX 4060, 2026-07-03）：
    #   cuda: 4.48 s/步  Best=4.875
    #   cpu : 1.91 s/步  Best=5.103
    #   加速比 = 0.43x（GPU 反而慢 2.3 倍）
    # 若后续改为批量并行公式评估（一次喂大批张量进 GPU），再切回 cuda。
    DEVICE = torch.device("cpu")

    # ── 训练参数（阶段 A：找简单稳定公式）────────────────────────────────
    # 阶段 A（当前）：MAX_FORMULA_LEN=8，TRAIN_STEPS=300
    #   目标：找简单、稳定、低换手的基础公式，防过拟合
    # 阶段 B（完成阶段A后切换）：
    #   MAX_FORMULA_LEN=14，TRAIN_STEPS=500，ELITE_REPLAY_FRAC=0.35
    #   目标：围绕好公式附近做组合增强
    BATCH_SIZE      = 128
    TRAIN_STEPS     = 3000  # 每品种训练步数（多因子模式下每品种独立跑）
    MAX_FORMULA_LEN = 8     # 阶段B改为 14

    # ── 特征维度（由 vocab.py 自动派生，无需手动修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 为主，IC 做门控而非线性加权 ────────────────────
    REWARD_ALPHA:      float = 1.0
    IC_GATE_THRESH:    float = 0.002  # 0.005→0.002：降低门控阈值，让IC更频繁参与调节
    IC_GATE_MULT:      float = 1.15
    IC_NEG_MULT:       float = 0.85

    # ── 熵保护 ─────────────────────────────────────────────────────────
    ENTROPY_COEFF_MAX:   float = 0.50
    ENTROPY_COEFF_POWER: float = 1.3
    ENTROPY_COLLAPSE_THRESH: float = 0.5
    ENTROPY_COLLAPSE_STEPS:  int   = 15

    # ── Elite Replay ──────────────────────────────────────────────────
    ELITE_REPLAY_FRAC:  float = 0.25   # 阶段A；阶段B改为 0.35
    ELITE_POOL_SIZE:    int   = 30
    ELITE_REWARD_SCALE: float = 1.2

    # ── 坍塌重启 ───────────────────────────────────────────────────────
    MAX_RESTARTS:   int   = 8
    RESTART_NOISE:  float = 0.05

    # ── 因子去相关参数 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 10
    CORR_THRESHOLD:   float = 0.7
    CORR_PENALTY:     float = 0.5

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    WF_GAP: int = 20
