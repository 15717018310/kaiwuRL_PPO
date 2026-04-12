#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import torch
import torch.nn.functional as F
import numpy as np

from agent_ppo.conf.conf import Config


def _to_numpy(x):
    """将 tensor / list / ndarray 统一转为 numpy array。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Algorithm:
    def __init__(self, model, optimizer, device, logger, monitor):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.train_step = 0

        self.clip_param = Config.CLIP_PARAM
        self.vf_coef    = Config.VF_COEF
        self.var_beta   = Config.BETA_START

    def learn(self, list_sample_data):
        """框架调用入口：list_sample_data 是 SampleData 列表。"""
        from agent_ppo.feature.definition import sample_process

        list_sample_data = sample_process(list_sample_data)

        obs      = np.stack([_to_numpy(s.obs).flatten()       for s in list_sample_data]).astype(np.float32)
        act      = np.array([int(_to_numpy(s.act).flat[0])    for s in list_sample_data], dtype=np.int64)
        old_prob = np.array([float(_to_numpy(s.prob).flat[int(_to_numpy(s.act).flat[0])]) for s in list_sample_data], dtype=np.float32)
        adv      = np.array([float(_to_numpy(s.advantage).flat[0])  for s in list_sample_data], dtype=np.float32)
        ret      = np.array([float(_to_numpy(s.reward_sum).flat[0]) for s in list_sample_data], dtype=np.float32)

        # Advantage 归一化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.model.set_train_mode()

        result = None
        for _ in range(Config.PPO_EPOCHS):
            result = self._update(obs, act, old_prob, adv, ret)

        self.train_step += 1

        # 每批数据只上报一次，避免超过 monitor 频率限制
        if self.monitor is not None and result is not None:
            try:
                self.monitor.put_data({os.getpid(): {
                    "total_loss":   result["total_loss"],
                    "policy_loss":  result["policy_loss"],
                    "value_loss":   result["value_loss"],
                    "entropy_loss": -self.var_beta * result["entropy"],
                }})
            except Exception:
                pass

        return result

    def _update(self, obs_np, act_np, old_prob_np, adv_np, ret_np):
        obs      = torch.tensor(obs_np,      dtype=torch.float32).to(self.device)
        act      = torch.tensor(act_np,      dtype=torch.long).to(self.device)
        old_prob = torch.tensor(old_prob_np, dtype=torch.float32).to(self.device)
        adv      = torch.tensor(adv_np,      dtype=torch.float32).to(self.device)
        ret      = torch.tensor(ret_np,      dtype=torch.float32).to(self.device)

        # 模型前向传播
        logits, values = self.model(obs)

        # 如果模型返回3个值（LSTM模式），提取第二个作为value
        if isinstance(values, tuple):
            value_mean, value_std = values  # 双价值网络
            values = value_mean  # 用mean作为主价值
        else:
            value_mean = values
            value_std = None  # 单价值网络

        dist     = torch.distributions.Categorical(logits=logits)
        new_prob = dist.log_prob(act)
        entropy  = dist.entropy().mean()

        # PPO clipped surrogate
        ratio = torch.exp(new_prob - old_prob)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # Value loss（阶段7：双价值网络）
        if value_std is not None:
            # MSE for mean
            value_mean_loss = F.mse_loss(value_mean.squeeze(-1), ret)
            # MSE for std: target是prediction error的绝对值
            residual = torch.abs(value_mean.squeeze(-1) - ret)
            value_std_loss = F.mse_loss(value_std.squeeze(-1), residual.detach())
            value_loss = value_mean_loss + 0.1 * value_std_loss
        else:
            # 原始单价值损失
            value_loss = F.mse_loss(values.squeeze(-1), ret)

        # 动态 entropy 衰减
        self.var_beta = max(Config.BETA_MIN, Config.BETA_START * (Config.BETA_DECAY ** self.train_step))
        entropy_loss  = -self.var_beta * entropy

        total_loss = policy_loss + self.vf_coef * value_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_RANGE)
        self.optimizer.step()

        return {
            "total_loss":  total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "entropy":     entropy.item(),
        }
