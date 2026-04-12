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


def _calc_flash_cooldown_stage(flash_cd, max_cd=2000.0):
    """计算闪现冷却分级 (3维)：[无冷却, 冷却中, 即将就绪]"""
    if flash_cd <= 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 无冷却
    elif flash_cd > max_cd * 0.5:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 冷却中
    else:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # 即将就绪


def _calc_monster_danger_level(min_dist_norm):
    """计算怪物危险等级 (1维)：根据最小怪物距离"""
    danger = float(np.clip((0.7 - min_dist_norm) / 0.7, -1.0, 1.0))
    return np.array([danger], dtype=np.float32)


def _calc_treasure_priority(treasure_list, hero_pos):
    """计算宝箱优先级特征 (4维)：可见宝箱数 + 聚集度 + 分类统计"""
    if not treasure_list:
        return np.zeros(4, dtype=np.float32)

    visible_count = float(min(len(treasure_list), 10))

    dists = []
    for t in treasure_list:
        p = t["pos"]
        dx = (p["x"] - hero_pos["x"]) / MAP_SIZE
        dz = (p["z"] - hero_pos["z"]) / MAP_SIZE
        dist = float(np.sqrt(dx * dx + dz * dz))
        dists.append(dist)

    if len(dists) > 1:
        cluster_degree = float(np.std(dists))
    else:
        cluster_degree = 0.0
    cluster_degree = float(np.clip(cluster_degree, 0, 1.5))

    near_count = sum(1 for d in dists if d < 0.3) / (len(dists) + 1e-6)
    mid_count = sum(1 for d in dists if 0.3 <= d < 0.6) / (len(dists) + 1e-6)

    return np.array([
        visible_count / 10.0,
        cluster_degree,
        near_count,
        mid_count,
    ], dtype=np.float32)


def _calc_escape_space(map_info, center_idx=10):
    """计算逃脱空间评估 (2维)：四周是否有足够逃跑空间"""
    if map_info is None:
        return np.zeros(2, dtype=np.float32)

    rows = len(map_info)
    cols = len(map_info[0]) if rows > 0 else 0

    top_walkable = 0
    bottom_walkable = 0
    left_walkable = 0
    right_walkable = 0

    for r in range(rows):
        for c in range(cols):
            if r < 3:
                top_walkable += int(map_info[r][c] != 0)
            if r >= rows - 3:
                bottom_walkable += int(map_info[r][c] != 0)
            if c < 3:
                left_walkable += int(map_info[r][c] != 0)
            if c >= cols - 3:
                right_walkable += int(map_info[r][c] != 0)

    horizontal_space = float((left_walkable + right_walkable) / (6 * cols + 1e-6))
    vertical_space = float((top_walkable + bottom_walkable) / (6 * rows + 1e-6))

    return np.array([
        horizontal_space,
        vertical_space,
    ], dtype=np.float32)


class HistoryBuffer:
    """维护过去5步的英雄和怪物位置，用于LSTM时序建模"""
    def __init__(self, max_len=5):
        self.max_len = max_len
        self.hero_positions = []
        self.monster_positions = []

    def push(self, hero_pos, monster_pos):
        """记录当前步的位置"""
        self.hero_positions.append((hero_pos["x"], hero_pos["z"]))
        self.monster_positions.append((monster_pos["x"], monster_pos["z"]))
        if len(self.hero_positions) > self.max_len:
            self.hero_positions.pop(0)
            self.monster_positions.pop(0)

    def get_history_features(self):
        """返回历史特征 (16维)"""
        if len(self.hero_positions) < 2:
            return np.zeros(16, dtype=np.float32)

        features = []

        if len(self.hero_positions) >= 2:
            dx = (self.hero_positions[-1][0] - self.hero_positions[-2][0]) / MAP_SIZE
            dz = (self.hero_positions[-1][1] - self.hero_positions[-2][1]) / MAP_SIZE
            features.extend([dx, dz])
        else:
            features.extend([0.0, 0.0])

        if len(self.monster_positions) >= 2:
            dx = (self.monster_positions[-1][0] - self.monster_positions[-2][0]) / MAP_SIZE
            dz = (self.monster_positions[-1][1] - self.monster_positions[-2][1]) / MAP_SIZE
            features.extend([dx, dz])
        else:
            features.extend([0.0, 0.0])

        if len(self.hero_positions) >= 2:
            v_x = (self.hero_positions[-1][0] - self.hero_positions[-2][0]) / MAP_SIZE
            v_z = (self.hero_positions[-1][1] - self.hero_positions[-2][1]) / MAP_SIZE
            features.extend([v_x, v_z])
        else:
            features.extend([0.0, 0.0])

        if len(self.monster_positions) >= 2:
            v_x = (self.monster_positions[-1][0] - self.monster_positions[-2][0]) / MAP_SIZE
            v_z = (self.monster_positions[-1][1] - self.monster_positions[-2][1]) / MAP_SIZE
            features.extend([v_x, v_z])
        else:
            features.extend([0.0, 0.0])

        if len(self.hero_positions) >= 3:
            v1_x = (self.hero_positions[-2][0] - self.hero_positions[-3][0]) / MAP_SIZE
            v1_z = (self.hero_positions[-2][1] - self.hero_positions[-3][1]) / MAP_SIZE
            v2_x = (self.hero_positions[-1][0] - self.hero_positions[-2][0]) / MAP_SIZE
            v2_z = (self.hero_positions[-1][1] - self.hero_positions[-2][1]) / MAP_SIZE
            a_x = v2_x - v1_x
            a_z = v2_z - v1_z
            features.extend([a_x, a_z])
        else:
            features.extend([0.0, 0.0])

        if len(self.monster_positions) >= 3:
            v1_x = (self.monster_positions[-2][0] - self.monster_positions[-3][0]) / MAP_SIZE
            v1_z = (self.monster_positions[-2][1] - self.monster_positions[-3][1]) / MAP_SIZE
            v2_x = (self.monster_positions[-1][0] - self.monster_positions[-2][0]) / MAP_SIZE
            v2_z = (self.monster_positions[-1][1] - self.monster_positions[-2][1]) / MAP_SIZE
            a_x = v2_x - v1_x
            a_z = v2_z - v1_z
            features.extend([a_x, a_z])
        else:
            features.extend([0.0, 0.0])

        return np.array(features[:16], dtype=np.float32)

    def reset(self):
        """重置缓冲区"""
        self.hero_positions.clear()
        self.monster_positions.clear()


class Preprocessor:
    """特征预处理器，包含特征提取和奖励计算"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist = 1.0
        self.last_min_treasure_dist = 1.0
        self.last_treasure_count = 0
        self.last_collected_buff = 0
        self.history_buffer = HistoryBuffer(max_len=5)

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation.get("legal_act", observation.get("legal_action", None))

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]

        flash_cd = hero.get("flash_cooldown", 0)
        buff_time = hero.get("buff_remaining_time", 0)
        has_buff = 1.0 if buff_time > 0 else 0.0

        hero_feat_base = np.array([
            _norm(hero_pos["x"], MAP_SIZE),
            _norm(hero_pos["z"], MAP_SIZE),
            _norm(flash_cd, MAX_FLASH_CD),
            _norm(buff_time, MAX_BUFF_DURATION),
        ], dtype=np.float32)

        flash_stage = _calc_flash_cooldown_stage(flash_cd, MAX_FLASH_CD)
        hero_feat = np.concatenate([hero_feat_base, flash_stage, np.array([has_buff], dtype=np.float32)])

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

        danger_level = _calc_monster_danger_level(cur_min_monster_dist)
        monster_count = np.array([min(len(monsters), 2.0) / 2.0], dtype=np.float32)

        organs = frame_state.get("organs", [])
        treasure_list = [o for o in organs if o["sub_type"] == 1 and o["status"] == 1]
        buff_list = [o for o in organs if o["sub_type"] == 2 and o["status"] == 1]

        treasure_feat_closest = _closest_organ(treasure_list, hero_pos)
        cur_min_treasure_dist = float(treasure_feat_closest[2]) if treasure_list else 1.0
        treasure_priority = _calc_treasure_priority(treasure_list, hero_pos)
        treasure_feat = np.concatenate([treasure_feat_closest, treasure_priority])

        has_buff_flag = hero.get("buff_remaining_time", 0) > 0
        buff_feat = _closest_organ(buff_list if not has_buff_flag else [], hero_pos)

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

        horizontal_walkable = 0.0
        vertical_walkable = 0.0
        if map_info is not None and len(map_info) > 0:
            rows = len(map_info)
            cols = len(map_info[0]) if rows > 0 else 0
            for r in range(rows):
                for c in range(3):
                    horizontal_walkable += float(map_info[r][c] != 0)
                    if c + (cols - 3) < cols:
                        horizontal_walkable += float(map_info[r][c + (cols - 3)] != 0)
            horizontal_walkable /= (6 * rows + 1e-6)

            for c in range(cols):
                for r in range(3):
                    vertical_walkable += float(map_info[r][c] != 0)
                    if r + (rows - 3) < rows:
                        vertical_walkable += float(map_info[r + (rows - 3)][c] != 0)
            vertical_walkable /= (6 * cols + 1e-6)

        escape_space = _calc_escape_space(map_info)
        map_feat_enhanced = np.concatenate([
            map_feat,
            np.array([horizontal_walkable, vertical_walkable], dtype=np.float32),
            escape_space,
        ])

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

        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, 1.0 - step_norm], dtype=np.float32)

        # 记录历史
        if len(monsters) > 0:
            monster_pos = monsters[0].get("pos", hero_pos)
        else:
            monster_pos = hero_pos
        self.history_buffer.push(hero_pos, monster_pos)

        feature = np.concatenate([
            hero_feat,                                    # 8
            monster_feats[0],                             # 5
            monster_feats[1],                             # 5
            danger_level,                                 # 1
            monster_count,                                # 1
            treasure_feat,                                # 8
            buff_feat,                                    # 3
            map_feat_enhanced,                            # 20
            np.array(legal_action, dtype=np.float32),    # 16
            progress_feat,                                # 2
        ])  # = 80

        reward = self._calc_reward(
            hero, env_info, cur_min_monster_dist, cur_min_treasure_dist,
            last_action
        )

        return feature, legal_action, [reward]

    def _calc_reward(self, hero, env_info, cur_min_monster_dist,
                     cur_min_treasure_dist, last_action):
        """计算奖励（10项平衡设计）"""
        reward = 0.05
        safe_weight = float(np.clip(cur_min_monster_dist / 0.5, 0.0, 1.0))
        danger_weight = 1.0 - safe_weight

        dist_diff = cur_min_monster_dist - self.last_min_monster_dist
        reward += (0.5 + 0.5 * danger_weight) * dist_diff
        self.last_min_monster_dist = cur_min_monster_dist

        danger_penalty = float(np.exp(-5.0 * cur_min_monster_dist))
        reward -= 0.8 * danger_penalty

        treasure_gain = self.last_min_treasure_dist - cur_min_treasure_dist
        reward += (0.2 + 0.8 * safe_weight) * treasure_gain
        self.last_min_treasure_dist = cur_min_treasure_dist

        cur_treasure_count = hero.get("treasure_collected_count", 0)
        if cur_treasure_count > self.last_treasure_count:
            reward += 3.0 * (cur_treasure_count - self.last_treasure_count)
        self.last_treasure_count = cur_treasure_count

        cur_collected_buff = env_info.get("collected_buff", 0)
        if cur_collected_buff > self.last_collected_buff:
            reward += 0.8
        self.last_collected_buff = cur_collected_buff

        used_flash = (last_action is not None and last_action >= 8)
        if used_flash:
            reward += 1.0 * danger_weight
            reward -= 0.5 * safe_weight
            if dist_diff > 0.1:
                reward += 0.3

        monster_speed = env_info.get("monster_speed", 1)
        if monster_speed > 1:
            reward += 0.02 * monster_speed

        if safe_weight > 0.7 and not used_flash:
            reward += 0.05

        reward = float(np.clip(reward, -2.5, 3.0))
        return reward
