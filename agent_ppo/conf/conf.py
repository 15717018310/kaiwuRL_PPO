#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Config:

    # =========================
    # 特征维度
    # hero(4) + monster0(5) + monster1(5) + treasure(3) + buff(3) + map(16) + legal_action(16) + progress(2) = 54
    # =========================
    FEATURES = [
        4,   # hero: x, z, flash_cd, buff_time
        5,   # monster0: exist, dir_x, dir_z, speed, dist
        5,   # monster1: exist, dir_x, dir_z, speed, dist
        3,   # treasure: dir_x, dir_z, dist
        3,   # buff: dir_x, dir_z, dist
        16,  # map: 4x4 local grid
        16,  # legal_action mask
        2,   # progress: step_norm, remaining_norm
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)  # 54
    DIM_OF_OBSERVATION = FEATURE_LEN

    # =========================
    # 动作空间
    # =========================
    ACTION_NUM = 16   # 8移动 + 8闪现

    # =========================
    # Value head
    # =========================
    VALUE_NUM = 1

    # =========================
    # PPO 超参数
    # =========================
    GAMMA  = 0.995   # 折扣因子（偏长期生存）
    LAMDA  = 0.97    # GAE lambda

    INIT_LEARNING_RATE_START = 0.0002

    # entropy 系数（动态衰减）
    BETA_START = 0.01
    BETA_MIN   = 0.0001
    BETA_DECAY = 0.995   # 每 train_step 衰减一次

    CLIP_PARAM     = 0.15  # PPO clip
    VF_COEF        = 0.5   # value loss 权重（从1.0降低，避免 critic 主导）
    GRAD_CLIP_RANGE = 0.5  # 梯度裁剪

    PPO_EPOCHS = 4         # 每批数据重复更新次数
