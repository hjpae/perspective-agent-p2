# cear_pilot/envs/nzone_common.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym


@dataclass
class NZoneCommonConfig:
    # grid
    width: int = 23
    height: int = 7
    obs_dim: int = 8
    max_steps: int = 240

    # encounter control (FORMATION)
    use_encounter: bool = True
    encounter_columns: Tuple[int, ...] = (3, 9, 15, 21)
    encounter_profiles: Tuple[str, ...] = ("misleading", "misleading", "misleading", "misleading")

    # sigma gradient
    phase2_sigma_left: float = 0.50
    phase2_sigma_right: float = 0.03

    # start
    phase2_start_xy: Tuple[int, int] = (0, 3)

    # hidden state dynamics (simplified)
    reliability_init: float = 0.0
    fragility_init: float = 0.1
    conflict_load_init: float = 0.0

    reliability_decay: float = 0.01
    fragility_decay: float = 0.005
    conflict_decay: float = 0.02

    # effect magnitudes
    supportive_delta: float = 0.15
    misleading_delta: float = -0.18

    supportive_fragility_relief: float = 0.04
    misleading_fragility_boost: float = 0.06

    supportive_conflict_relief: float = 0.05
    misleading_conflict_boost: float = 0.07


class NZoneCommonEnv(gym.Env):

    def __init__(self, config: NZoneCommonConfig):
        super().__init__()
        self.cfg = config

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x, self.y = self.cfg.phase2_start_xy
        self.t = 0

        # hidden state
        self.reliability = self.cfg.reliability_init
        self.fragility = self.cfg.fragility_init
        self.conflict_load = self.cfg.conflict_load_init

        obs = self._get_obs()
        info = self._get_info(encounter=False, outcome=1)

        return obs, info

    def step(self, action: int):
        self.t += 1

        # movement
        if action == 0:   # up
            self.y = max(0, self.y - 1)
        elif action == 1: # down
            self.y = min(self.cfg.height - 1, self.y + 1)
        elif action == 2: # left
            self.x = max(0, self.x - 1)
        elif action == 3: # right
            self.x = min(self.cfg.width - 1, self.x + 1)
        elif action == 4: # stay
            pass

        # encounter check
        on_encounter = False
        encounter_idx = -1
        encounter_profile = "none"
        outcome = 1  # 0=supportive, 1=neutral, 2=misleading

        if self.cfg.use_encounter and self.x in self.cfg.encounter_columns:
            on_encounter = True
            encounter_idx = self.cfg.encounter_columns.index(self.x)
            encounter_profile = self.cfg.encounter_profiles[encounter_idx]

            if encounter_profile == "misleading":
                outcome = 2
                self._apply_misleading()
            elif encounter_profile == "supportive":
                outcome = 0
                self._apply_supportive()
            else:
                outcome = 1

        # passive decay
        self._decay_states()

        obs = self._get_obs()
        info = self._get_info(
            encounter=on_encounter,
            outcome=outcome,
            encounter_idx=encounter_idx,
            encounter_profile=encounter_profile
        )

        terminated = False
        truncated = self.t >= self.cfg.max_steps

        return obs, 0.0, terminated, truncated, info

    def _sigma(self):
        # linear gradient (left → right)
        ratio = self.x / (self.cfg.width - 1)
        return (1 - ratio) * self.cfg.phase2_sigma_left + ratio * self.cfg.phase2_sigma_right

    def _get_obs(self):
        sigma = self._sigma()

        # simple noisy observation (8-dim)
        noise = np.random.randn(self.cfg.obs_dim) * sigma
        return noise.astype(np.float32)

    def _apply_supportive(self):
        self.reliability += self.cfg.supportive_delta
        self.fragility -= self.cfg.supportive_fragility_relief
        self.conflict_load -= self.cfg.supportive_conflict_relief

    def _apply_misleading(self):
        self.reliability += self.cfg.misleading_delta
        self.fragility += self.cfg.misleading_fragility_boost
        self.conflict_load += self.cfg.misleading_conflict_boost

    def _decay_states(self):
        self.reliability *= (1 - self.cfg.reliability_decay)
        self.fragility *= (1 - self.cfg.fragility_decay)
        self.conflict_load *= (1 - self.cfg.conflict_decay)

    def _get_info(
        self,
        encounter: bool,
        outcome: int,
        encounter_idx: int = -1,
        encounter_profile: str = "none",
    ) -> Dict[str, Any]:

        return {
            "x": self.x,
            "y": self.y,
            "zone_id": self._zone_id(),

            "on_encounter": encounter,
            "encounter_idx": encounter_idx,
            "encounter_profile": encounter_profile,
            "encounter_outcome": outcome,

            "reliability_estimate": float(self.reliability),
            "fragility": float(self.fragility),
            "conflict_load": float(self.conflict_load),

            "current_sigma": float(self._sigma()),
        }

    def _zone_id(self):
        # simple 5 buckets
        boundaries = [4, 9, 14, 19]
        for i, b in enumerate(boundaries):
            if self.x < b:
                return i
        return 4