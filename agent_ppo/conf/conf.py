#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Config:

    # =========================
    # 特征维度（阶段2优化：54→80维）
    # hero(8) + monster(12) + treasure(8) + buff(3) + map(20) + legal_action(16) + progress(2) = 80
    # =========================
    FEATURES = [
        8,   # hero_enhanced: x, z, flash_cd, buff_time, flash_stage_x3, has_buff_x1 = 4+4
        12,  # monsters_enhanced: monster0(5) + monster1(5) + min_dist_level(1) + count(1)
        8,   # treasure_enhanced: closest(3) + visible_count(1) + cluster_degree(1) + priority_x3(3)
        3,   # buff: dir_x, dir_z, dist (unchanged)
        20,  # map_enhanced: local_grid(16) + horizontal_walkable(1) + vertical_walkable(1) + escape_space_x2(2)
        16,  # legal_action mask
        2,   # progress: step_norm, remaining_norm
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)  # 80
    DIM_OF_OBSERVATION = FEATURE_LEN

    # =========================
    # LSTM 时序配置（阶段6优化）
    # =========================
    USE_LSTM = True                # 启用LSTM时序建模
    LSTM_HIDDEN_DIM = 128          # LSTM隐藏维度
    LSTM_NUM_LAYERS = 1            # LSTM层数
    SEQUENCE_LEN = 5               # 历史轨迹长度（5步）
    HISTORY_FEATURE_DIM = 16       # 历史特征维度：5步×(hero_pos(2)+monster_pos(2)+velocity(4))

    # 加上历史特征后的总维度：80 + 16 = 96
    DIM_WITH_HISTORY = DIM_OF_OBSERVATION + HISTORY_FEATURE_DIM  # 96

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

    INIT_LEARNING_RATE_START = 0.0003  # 提升从0.0002→0.0003（稍快学习）

    # entropy 系数（动态衰减）
    BETA_START = 0.01
    BETA_MIN   = 0.0001
    BETA_DECAY = 0.9995   # 提升从0.995→0.9995（更慢衰减，保留探索）

    CLIP_PARAM     = 0.2   # 提升从0.15→0.2（PPO clip稍放松）
    VF_COEF        = 1.0   # 提升从0.5→1.0（加强value拟合）
    GRAD_CLIP_RANGE = 0.5  # 梯度裁剪

    PPO_EPOCHS = 8         # 提升从4→8（每批数据更多迭代）
