#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Config:
    # =========================
    # 特征维度配置（总计 127 维）
    # hero(4) + monster_0(5) + monster_1(5) + treasure(3) + buff(3) + map(81) + legal_action(16) + progress(2) + flash_radar(8) = 127
    # =========================
    FEATURES = [
        4,    # hero: x, z, flash_cd, buff_time
        5,    # monster_0: valid, dir_x, dir_z, speed, dist
        5,    # monster_1: valid, dir_x, dir_z, speed, dist
        3,    # treasure: dir_x, dir_z, dist
        3,    # buff: dir_x, dir_z, dist
        81,   # map: 9×9 local crop flattened
        16,   # legal_action mask
        2,    # progress: step_norm, remaining_norm
        8,    # flash_radar: 8个闪现方向的落点安全性探测
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)  # 127
    DIM_OF_OBSERVATION = FEATURE_LEN

    # =========================
    # LSTM 时序配置
    # =========================
    USE_LSTM = False               
    LSTM_HIDDEN_DIM = 128          
    LSTM_NUM_LAYERS = 1            
    SEQUENCE_LEN = 5               
    HISTORY_FEATURE_DIM = 16       
    DIM_WITH_HISTORY = DIM_OF_OBSERVATION + HISTORY_FEATURE_DIM # 500

    # =========================
    # 动作与价值空间
    # =========================
    ACTION_NUM = 16   # 8移动 + 8闪现
    VALUE_NUM = 1

    # =========================
    # PPO 超参数（调优平衡版）
    # =========================
    GAMMA  = 0.99
    LAMDA  = 0.95
    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.01
    BETA_MIN   = 0.005
    BETA_DECAY = 0.999    
    CLIP_PARAM     = 0.2   
    VF_COEF        = 0.25   
    GRAD_CLIP_RANGE = 0.5  
    PPO_EPOCHS = 4         # 减少榨取次数，防止过拟合到自杀策略