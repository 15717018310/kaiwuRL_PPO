#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import torch
import torch.nn.functional as F
import numpy as np

from agent_ppo.conf.conf import Config


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Algorithm:
    def __init__(self, model, actor_optimizer, critic_optimizer, device, logger, monitor):
        self.model = model
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.train_step = 0
        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.var_beta = Config.BETA_START

        self.sample_buffer = []
        self.min_batch_size = 256

    def learn(self, list_sample_data):
        self.sample_buffer.extend(list_sample_data)
        if len(self.sample_buffer) < self.min_batch_size:
            return None

        buffered = self.sample_buffer
        self.sample_buffer = []

        obs      = np.stack([_to_numpy(s.obs).flatten() for s in buffered]).astype(np.float32)
        act      = np.array([int(_to_numpy(s.act).flat[0]) for s in buffered], dtype=np.int64)
        old_prob = np.array([np.log(float(_to_numpy(s.prob).flat[int(_to_numpy(s.act).flat[0])]) + 1e-8) for s in buffered], dtype=np.float32)
        adv      = np.array([float(_to_numpy(s.advantage).flat[0]) for s in buffered], dtype=np.float32)
        ret      = np.array([float(_to_numpy(s.reward_sum).flat[0]) for s in buffered], dtype=np.float32)
        legal    = np.stack([_to_numpy(s.legal_action).flatten() for s in buffered]).astype(np.float32)

        adv_raw_mean = adv.mean()
        adv_raw_std = adv.std()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.model.set_train_mode()

        result = None
        for _ in range(Config.PPO_EPOCHS):
            result = self._update(obs, act, old_prob, adv, ret, legal, adv_raw_mean, adv_raw_std)

        self.train_step += 1

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

    def _update(self, obs_np, act_np, old_prob_np, adv_np, ret_np, legal_np, adv_raw_mean, adv_raw_std):
        obs      = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        act      = torch.tensor(act_np, dtype=torch.long).to(self.device)
        old_prob = torch.tensor(old_prob_np, dtype=torch.float32).to(self.device)
        adv      = torch.tensor(adv_np, dtype=torch.float32).to(self.device)
        ret      = torch.tensor(ret_np, dtype=torch.float32).to(self.device)
        legal    = torch.tensor(legal_np, dtype=torch.float32).to(self.device)

        logits, value = self.model(obs)

        masked_logits = logits.clone()
        masked_logits[legal == 0] = -1e10
        dist = torch.distributions.Categorical(logits=masked_logits)
        new_prob = dist.log_prob(act)
        entropy = dist.entropy().mean()

        # PPO clipped surrogate
        ratio = torch.exp(new_prob - old_prob)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # Value loss
        value_loss = F.mse_loss(value.squeeze(-1), ret)

        # Debug
        if self.train_step % 100 == 0:
            self.logger.info(f"[DEBUG] adv_raw: mean={adv_raw_mean:.4f} std={adv_raw_std:.4f}")
            self.logger.info(f"[DEBUG] ratio: mean={ratio.mean().item():.4f} std={ratio.std().item():.4f} min={ratio.min().item():.4f} max={ratio.max().item():.4f}")
            self.logger.info(f"[DEBUG] policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}")

        # Entropy decay
        self.var_beta = max(Config.BETA_MIN, Config.BETA_START * (Config.BETA_DECAY ** self.train_step))
        entropy_loss = -self.var_beta * entropy

        # Actor update
        actor_loss = policy_loss + entropy_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), Config.GRAD_CLIP_RANGE)
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self.vf_coef * value_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), Config.GRAD_CLIP_RANGE)
        self.critic_optimizer.step()

        total_loss = actor_loss.item() + critic_loss.item()

        return {
            "total_loss":  total_loss,
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "entropy":     entropy.item(),
        }
