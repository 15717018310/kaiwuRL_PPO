#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
MAX_DIST_BUCKET = 5.0
MAX_DIRECTION = 8.0

# 文档方向枚举 -> 单位向量
# 0=重叠/无效, 1=东, 2=东北, 3=北, 4=西北, 5=西, 6=西南, 7=南, 8=东南
_DIR_VEC = {
    0: (0.0,  0.0),
    1: (1.0,  0.0),
    2: (1.0, -1.0),
    3: (0.0, -1.0),
    4: (-1.0,-1.0),
    5: (-1.0, 0.0),
    6: (-1.0, 1.0),
    7: (0.0,  1.0),
    8: (1.0,  1.0),
}


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _direction_to_vec(direction: int):
    dx, dz = _DIR_VEC.get(direction, (0.0, 0.0))
    norm = np.sqrt(dx * dx + dz * dz)
    if norm > 1e-6:
        return dx / norm, dz / norm
    return 0.0, 0.0


def _closest_organ(organ_list, hero_pos):
    """返回最近物件的 (dir_x, dir_z, dist_norm)，无则全零。"""
    if not organ_list:
        return np.zeros(3, dtype=np.float32)
    best = None
    for o in organ_list:
        p = o["pos"]
        dx = (p["x"] - hero_pos["x"]) / MAP_SIZE
        dz = (p["z"] - hero_pos["z"]) / MAP_SIZE
        dist = float(np.sqrt(dx * dx + dz * dz))
        if best is None or dist < best[0]:
            best = (dist, dx, dz)
    dist, dx, dz = best
    dir_x = dx / (dist + 1e-6)
    dir_z = dz / (dist + 1e-6)
    return np.array([dir_x, dir_z, float(np.clip(dist, 0, 1.5))], dtype=np.float32)


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist = 1.0
        self.last_min_treasure_dist = 1.0
        self.last_treasure_count = 0
        self.last_collected_buff = 0

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation.get("legal_act", observation.get("legal_action", None))

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        # ===== HERO (4维) =====
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]

        hero_feat = np.array([
            _norm(hero_pos["x"], MAP_SIZE),
            _norm(hero_pos["z"], MAP_SIZE),
            _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD),
            _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION),
        ], dtype=np.float32)

        # ===== MONSTER (5维 × 2 = 10维) =====
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_monster_dist = 1.0

        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                m_pos = m.get("pos", None)
                speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                if m_pos is not None:
                    dx = (m_pos["x"] - hero_pos["x"]) / MAP_SIZE
                    dz = (m_pos["z"] - hero_pos["z"]) / MAP_SIZE
                    dist_exact = float(np.sqrt(dx * dx + dz * dz)) + 1e-6
                    dir_x = dx / dist_exact
                    dir_z = dz / dist_exact
                    dist_norm = float(np.clip(dist_exact, 0, 1.5))
                else:
                    dir_x, dir_z = _direction_to_vec(int(m.get("hero_relative_direction", 0)))
                    dist_norm = _norm(m.get("hero_l2_distance", 5), MAX_DIST_BUCKET)

                cur_min_monster_dist = min(cur_min_monster_dist, dist_norm)
                monster_feats.append(np.array([1.0, dir_x, dir_z, speed_norm, dist_norm], dtype=np.float32))
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # ===== ORGANS =====
        organs = frame_state.get("organs", [])
        treasure_list = [o for o in organs if o["sub_type"] == 1 and o["status"] == 1]
        buff_list     = [o for o in organs if o["sub_type"] == 2 and o["status"] == 1]

        # ===== TREASURE (3维) =====
        treasure_feat = _closest_organ(treasure_list, hero_pos)
        cur_min_treasure_dist = float(treasure_feat[2]) if treasure_list else 1.0

        # ===== BUFF (3维) =====
        # 没有 buff 时才引导去拾取
        has_buff = hero.get("buff_remaining_time", 0) > 0
        buff_feat = _closest_organ(buff_list if not has_buff else [], hero_pos)

        # ===== MAP 4×4 (16维) =====
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None:
            rows = len(map_info)
            cols = len(map_info[0]) if rows > 0 else 0
            cr, cc = rows // 2, cols // 2
            idx = 0
            for r in range(cr - 2, cr + 2):
                for c in range(cc - 2, cc + 2):
                    if 0 <= r < rows and 0 <= c < cols:
                        map_feat[idx] = float(map_info[r][c] != 0)
                    idx += 1

        # ===== ACTION MASK (16维) =====
        legal_action = [1] * 16
        if legal_act_raw is not None and isinstance(legal_act_raw, (list, tuple)) and len(legal_act_raw) > 0:
            if isinstance(legal_act_raw[0], bool):
                for i in range(min(16, len(legal_act_raw))):
                    legal_action[i] = int(legal_act_raw[i])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if i in valid_set else 0 for i in range(16)]
        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # ===== PROGRESS (2维) =====
        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, 1.0 - step_norm], dtype=np.float32)

        # ===== 拼接 (4+5+5+3+3+16+16+2 = 54维) =====
        feature = np.concatenate([
            hero_feat,                                    # 4
            monster_feats[0],                             # 5
            monster_feats[1],                             # 5
            treasure_feat,                                # 3
            buff_feat,                                    # 3
            map_feat,                                     # 16
            np.array(legal_action, dtype=np.float32),    # 16
            progress_feat,                                # 2
        ])  # = 54

        # ===== REWARD =====
        reward = self._calc_reward(
            hero, env_info, cur_min_monster_dist, cur_min_treasure_dist,
            last_action
        )

        return feature, legal_action, [reward]

    def _calc_reward(self, hero, env_info, cur_min_monster_dist,
                     cur_min_treasure_dist, last_action):
        """
        奖励设计：
          1. 基础生存
          2. 怪物距离塑形（越危险越强调远离）
          3. 危险区域指数惩罚
          4. 宝箱距离塑形（安全时才鼓励靠近）
          5. 吃到宝箱
          6. 拾取 buff
          7. 闪现策略（危险时奖励，安全时惩罚）
          8. 怪物加速后额外生存奖励
        """
        reward = 0.02  # 基础生存

        # 安全权重 [0,1]，越近怪物越小
        safe_weight = float(np.clip(cur_min_monster_dist / 0.5, 0.0, 1.0))

        # === 1. 怪物距离塑形 ===1
        dist_diff = cur_min_monster_dist - self.last_min_monster_dist
        reward += (0.5 + 0.5 * (1.0 - safe_weight)) * dist_diff
        self.last_min_monster_dist = cur_min_monster_dist

        # === 2. 危险区域指数惩罚 ===
        danger_penalty = float(np.exp(-5.0 * cur_min_monster_dist))
        reward -= 0.6 * danger_penalty

        # === 3. 宝箱距离塑形（安全时才鼓励靠近） ===
        treasure_gain = self.last_min_treasure_dist - cur_min_treasure_dist
        reward += (0.1 + 0.4 * safe_weight) * treasure_gain
        self.last_min_treasure_dist = cur_min_treasure_dist

        # === 4. 吃到宝箱 ===
        cur_treasure_count = hero.get("treasure_collected_count", 0)
        if cur_treasure_count > self.last_treasure_count:
            reward += 1.5 * (cur_treasure_count - self.last_treasure_count)
        self.last_treasure_count = cur_treasure_count

        # === 5. 拾取 buff（用 env_info 的 collected_buff 更准确） ===
        cur_collected_buff = env_info.get("collected_buff", 0)
        if cur_collected_buff > self.last_collected_buff:
            reward += 0.5
        self.last_collected_buff = cur_collected_buff

        # === 6. 闪现策略 ===
        used_flash = (last_action is not None and last_action >= 8)
        if used_flash:
            reward += 0.6 * (1.0 - safe_weight)   # 危险时用 → 奖励
            reward -= 0.3 * safe_weight             # 安全时乱用 → 惩罚

        # === 7. 怪物加速后额外生存奖励（鼓励在高压下存活） ===
        monster_speed = env_info.get("monster_speed", 1)
        if monster_speed > 1:
            reward += 0.01 * monster_speed

        # === 8. 防抖裁剪 ===
        reward = float(np.clip(reward, -2.0, 2.0))
        return reward
