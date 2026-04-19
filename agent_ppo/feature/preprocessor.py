#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0

def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

def _calc_flash_radar(map_info):
    """探测8个闪现方向落点的安全性 (8维)"""
    radar = np.zeros(8, dtype=np.float32)
    if map_info is None or len(map_info) == 0: return radar
    rows, cols = 21, 21
    cr, cc = 10, 10 # 21x21的中心点
    offsets = [(0, 10), (-8, 8), (-10, 0), (-8, -8), (0, -10), (8, -8), (10, 0), (8, 8)]
    for i, (dr, dc) in enumerate(offsets):
        tr, tc = cr + dr, cc + dc
        if 0 <= tr < rows and 0 <= tc < cols:
            if map_info[tr][tc] != 0: radar[i] = 1.0 # 1=可通行
    return radar

class Preprocessor:
    def __init__(self): self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist = 1.0
        self.last_min_treasure_dist = 1.0
        self.last_min_buff_dist = 1.0
        self.last_treasure_count = 0
        self.last_buff_count = 0

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation.get("legal_action", observation.get("legal_act", [1]*16))

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        flash_cd = hero.get("flash_cooldown", 0)
        
        # 1. 英雄特征
        hero_feat = np.array([
            _norm(hero_pos["x"], MAP_SIZE), _norm(hero_pos["z"], MAP_SIZE),
            _norm(flash_cd, MAX_FLASH_CD), _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION),
        ], dtype=np.float32)

        # 2. 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_monster_dist = 1.0
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]; m_pos = m["pos"]
                dx = (m_pos["x"] - hero_pos["x"]) / MAP_SIZE; dz = (m_pos["z"] - hero_pos["z"]) / MAP_SIZE
                dist = np.sqrt(dx * dx + dz * dz) + 1e-6
                cur_min_monster_dist = min(cur_min_monster_dist, dist)
                monster_feats.append(np.array([1.0, dx/dist, dz/dist, _norm(m.get("speed", 1), MAX_MONSTER_SPEED), np.clip(dist, 0, 1.5)], dtype=np.float32))
            else: monster_feats.append(np.zeros(5, dtype=np.float32))

        # 3. 宝箱特征
        organs = frame_state.get("organs", [])
        treasure_list = [o for o in organs if o["sub_type"] == 1 and o["status"] == 1]
        cur_min_treasure_dist = 1.0; treasure_feat = np.zeros(3, dtype=np.float32)
        if treasure_list:
            dists = []
            for t in treasure_list:
                tp = t["pos"]; dx = (tp["x"] - hero_pos["x"]) / MAP_SIZE; dz = (tp["z"] - hero_pos["z"]) / MAP_SIZE
                dists.append((np.sqrt(dx*dx+dz*dz), dx, dz))
            dists.sort(); d, dx, dz = dists[0]; cur_min_treasure_dist = np.clip(d, 0, 1.5)
            treasure_feat = np.array([dx/(d+1e-6), dz/(d+1e-6), cur_min_treasure_dist], dtype=np.float32)

        # 3b. Buff 特征
        buff_list = [o for o in organs if o["sub_type"] == 2 and o["status"] == 1]
        cur_min_buff_dist = 1.0; buff_feat = np.zeros(3, dtype=np.float32)
        if buff_list:
            dists = []
            for b in buff_list:
                bp = b["pos"]; dx = (bp["x"] - hero_pos["x"]) / MAP_SIZE; dz = (bp["z"] - hero_pos["z"]) / MAP_SIZE
                dists.append((np.sqrt(dx*dx+dz*dz), dx, dz))
            dists.sort(); d, dx, dz = dists[0]; cur_min_buff_dist = np.clip(d, 0, 1.5)
            buff_feat = np.array([dx/(d+1e-6), dz/(d+1e-6), cur_min_buff_dist], dtype=np.float32)

        # 4. 地图特征 (9×9 局部裁剪，以英雄为中心)
        if map_info and len(map_info) >= 21:
            full_map = np.array(map_info, dtype=np.float32)
            local_map = full_map[6:15, 6:15]  # 中心9×9
            map_feat = (local_map != 0).astype(np.float32).flatten()
        else:
            map_feat = np.zeros(81, dtype=np.float32)

        # 5. 闪现雷达与严格掩码 (核心逻辑)
        flash_radar = _calc_flash_radar(map_info)
        legal_action = [1] * 16
        # 只有在怪物距离小于 0.4 (约50格) 且闪现就绪时，才允许闪现
        can_flash = (flash_cd <= 0) and (cur_min_monster_dist < 0.4)
        if not can_flash:
            for i in range(8, 16): legal_action[i] = 0
        else:
            for i in range(8):
                if flash_radar[i] < 0.5: legal_action[i + 8] = 0 # 撞墙不准闪
        for i in range(8): legal_action[i] = 1 # 移动始终合法

        # 6. 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, 1.0 - step_norm], dtype=np.float32)

        # 特征拼接 (127维)
        feature = np.concatenate([hero_feat, monster_feats[0], monster_feats[1], treasure_feat, buff_feat, map_feat, np.array(legal_action, dtype=np.float32), progress_feat, flash_radar])

        reward = self._calc_reward(hero, env_info, cur_min_monster_dist, cur_min_treasure_dist, cur_min_buff_dist, last_action)
        return feature, legal_action, [reward]

    def _calc_reward(self, hero, env_info, cur_min_monster_dist, cur_min_treasure_dist, cur_min_buff_dist, last_action):
        reward = 0.2
        dist_diff = cur_min_monster_dist - self.last_min_monster_dist

        if dist_diff > 0: reward += 3.0 * dist_diff
        if cur_min_monster_dist < 0.1: reward -= 0.2

        # 宝箱
        treasure_gain = self.last_min_treasure_dist - cur_min_treasure_dist
        reward += 2.0 * treasure_gain
        cur_tc = hero.get("treasure_collected_count", 0)
        if cur_tc > self.last_treasure_count: reward += 8.0

        # 闪现
        if last_action >= 8:
            if dist_diff > 0.1: reward += 4.0
            else: reward -= 1.0

        # Buff 持续奖励：有buff时每步额外奖励，让agent理解buff=更快=更安全
        if hero.get("buff_remaining_time", 0) > 0:
            reward += 0.5

        # Buff 收集
        cur_bc = env_info.get("collected_buff", 0)
        buff_just_collected = cur_bc > self.last_buff_count
        if buff_just_collected: reward += 5.0
        self.last_buff_count = cur_bc

        # Buff 靠近（近距离引力加强，远距离弱于逃跑避免送死）
        if not buff_just_collected:
            buff_gain = self.last_min_buff_dist - cur_min_buff_dist
            buff_gain = np.clip(buff_gain, -0.05, 1.0)
            coeff = 2.5 + 3.0 * max(0, 1.0 - cur_min_buff_dist / 0.15)
            reward += coeff * buff_gain

        self.last_min_monster_dist = cur_min_monster_dist
        self.last_min_treasure_dist = cur_min_treasure_dist
        self.last_min_buff_dist = cur_min_buff_dist
        self.last_treasure_count = cur_tc
        return float(np.clip(reward, -5.0, 15.0))