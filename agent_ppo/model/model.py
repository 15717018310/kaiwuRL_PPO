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


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION  # 124
        action_num = Config.ACTION_NUM  # 16
        value_num = Config.VALUE_NUM  # 1

        self.actor = nn.Sequential(
            make_fc_layer(input_dim, 128, gain=np.sqrt(2)),
            nn.LayerNorm(128),
            nn.ReLU(),
            make_fc_layer(128, 64, gain=np.sqrt(2)),
            nn.LayerNorm(64),
            nn.ReLU(),
            make_fc_layer(64, action_num, gain=0.01),
        )

        self.critic = nn.Sequential(
            make_fc_layer(input_dim, 128, gain=np.sqrt(2)),
            nn.LayerNorm(128),
            nn.ReLU(),
            make_fc_layer(128, 64, gain=np.sqrt(2)),
            nn.LayerNorm(64),
            nn.ReLU(),
            make_fc_layer(64, value_num, gain=1.0),
        )

    def forward(self, obs, **kwargs):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
