#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

        input_dim  = Config.DIM_OF_OBSERVATION  # 54
        action_num = Config.ACTION_NUM           # 16
        value_num  = Config.VALUE_NUM            # 1

        # Actor: 54 -> 256 -> 128 -> 64
        self.actor_backbone = nn.Sequential(
            make_fc_layer(input_dim, 256, gain=np.sqrt(2)),
            nn.LayerNorm(256),
            nn.ReLU(),
            make_fc_layer(256, 128, gain=np.sqrt(2)),
            nn.LayerNorm(128),
            nn.ReLU(),
            make_fc_layer(128, 64, gain=np.sqrt(2)),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Critic: 独立参数，54 -> 256 -> 128 -> 64
        self.critic_backbone = nn.Sequential(
            make_fc_layer(input_dim, 256, gain=np.sqrt(2)),
            nn.LayerNorm(256),
            nn.ReLU(),
            make_fc_layer(256, 128, gain=np.sqrt(2)),
            nn.LayerNorm(128),
            nn.ReLU(),
            make_fc_layer(128, 64, gain=np.sqrt(2)),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # 小 gain 让初始策略接近均匀分布，有利于早期探索
        self.actor_head  = make_fc_layer(64, action_num, gain=0.01)
        self.critic_head = make_fc_layer(64, value_num,  gain=1.0)

    def forward(self, obs, inference=False):
        logits = self.actor_head(self.actor_backbone(obs))
        value  = self.critic_head(self.critic_backbone(obs))
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
