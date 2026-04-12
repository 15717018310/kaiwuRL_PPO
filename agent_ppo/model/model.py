#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class MultiHeadAttention(nn.Module):
    """轻量级多头注意力机制，用于处理多个对象（怪物、宝箱等）"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, hidden_dim)
        batch_size = x.shape[0]

        # Linear transformations
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention, V)  # (batch_size, num_heads, head_dim)
        context = context.view(batch_size, self.hidden_dim)

        # Final linear transformation
        output = self.fc_out(context)
        return output


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION  # 80
        action_num = Config.ACTION_NUM  # 16
        value_num = Config.VALUE_NUM  # 1
        hidden_dim = Config.LSTM_HIDDEN_DIM  # 128

        self.use_lstm = Config.USE_LSTM

        if self.use_lstm:
            # ===== LSTM时序处理 (阶段6) =====
            # 特征嵌入：80维直接输入
            self.feature_embed = make_fc_layer(input_dim, hidden_dim, gain=np.sqrt(2))

            # LSTM: 128 -> 128
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=Config.LSTM_NUM_LAYERS,
                batch_first=True,
                dropout=0.1 if Config.LSTM_NUM_LAYERS > 1 else 0.0,
            )
            self.lstm_norm = nn.LayerNorm(hidden_dim)

            # 多头注意力（处理多个对象关系）
            self.attention = MultiHeadAttention(hidden_dim, num_heads=4)
            self.attention_norm = nn.LayerNorm(hidden_dim)
        else:
            # ===== 原始MLP架构（向后兼容） =====
            self.shared_backbone = nn.Sequential(
                make_fc_layer(input_dim, 256, gain=np.sqrt(2)),
                nn.LayerNorm(256),
                nn.ReLU(),
                make_fc_layer(256, hidden_dim, gain=np.sqrt(2)),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.attention = MultiHeadAttention(hidden_dim, num_heads=4)
            self.attention_norm = nn.LayerNorm(hidden_dim)

        # ===== 双价值网络 (阶段7) =====
        # 期望价值
        self.critic_mean_head = make_fc_layer(hidden_dim, value_num, gain=1.0)
        # 风险（标准差）
        self.critic_std_head = make_fc_layer(hidden_dim, value_num, gain=0.01)

        # Actor 头
        self.actor_head = make_fc_layer(hidden_dim, action_num, gain=0.01)

    def forward(self, obs, history_feat=None, inference=False):
        """
        前向传播
        obs: [batch_size, 80] 或 [batch_size, seq_len, 80]
        history_feat: [batch_size, 16] 历史轨迹特征
        inference: 是否为推理模式
        """
        if self.use_lstm:
            # 连接历史特征
            if history_feat is not None:
                if obs.dim() == 2:  # [batch_size, 80]
                    obs = torch.cat([obs, history_feat], dim=-1)  # [batch_size, 96]
                else:  # [batch_size, seq_len, 80]
                    # 广播history到序列的每个位置
                    history_expanded = history_feat.unsqueeze(1).expand(-1, obs.size(1), -1)
                    obs = torch.cat([obs, history_expanded], dim=-1)  # [batch_size, seq_len, 96]

            # 特征嵌入
            if obs.dim() == 2:  # 单帧：[batch_size, 96]
                x = self.feature_embed(obs)  # [batch_size, 128]
                x_lstm = x.unsqueeze(1)  # [batch_size, 1, 128]
            else:  # 序列：[batch_size, seq_len, 96]
                x = self.feature_embed(obs)  # [batch_size, seq_len, 128]
                x_lstm = x

            # LSTM处理
            lstm_out, (h_n, c_n) = self.lstm(x_lstm)  # [batch_size, seq_len, 128]
            lstm_out = self.lstm_norm(lstm_out)

            # 取最后一步的输出
            x = lstm_out[:, -1, :] if lstm_out.dim() == 3 else lstm_out
        else:
            # 原始MLP路径
            x = self.shared_backbone(obs)  # [batch_size, 128]

        # 注意力处理
        attn_output = self.attention(x)  # [batch_size, 128]
        x = x + attn_output  # 残差连接
        x = self.attention_norm(x)

        # Actor 和 Critic 头
        logits = self.actor_head(x)  # [batch_size, 16]

        # 双价值网络
        value_mean = self.critic_mean_head(x)  # [batch_size, 1]
        value_std = torch.nn.functional.softplus(self.critic_std_head(x)) + 1e-6  # [batch_size, 1]

        # 返回：logits, value_mean, value_std
        return logits, value_mean, value_std

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
