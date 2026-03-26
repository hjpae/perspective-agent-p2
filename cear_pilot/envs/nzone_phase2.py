# cear_pilot/envs/nzone_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("This environment requires gymnasium. Install with: pip install gymnasium") from e


@dataclass
class NZonePhase2Config:
    # -------------------------
    # grid / observation
    # -------------------------
    width: int = 23
    height: int = 7
    obs_dim: int = 8
    max_steps: int = 240
    include_xy: bool = False
    reward_scale: float = 0.0

    # -------------------------
    # formation-stage geometry
    # -------------------------
    start_xy: Tuple[int, int] = (0, 3)

    # symmetric reporting buckets: 4 / 5 / 5 / 5 / 4
    report_zone_boundaries: Tuple[int, ...] = (4, 9, 14, 19)

    # encounter schedule
    use_encounter: bool = True
    encounter_columns: Tuple[int, ...] = (3, 9, 15, 21)
    encounter_profiles: Tuple[str, ...] = (
        "misleading",
        "misleading",
        "misleading",
        "misleading",
    )

    # optional weak marker injected into observation when standing on encounter column
    encounter_signal: float = 1.0
    encounter_dims: Tuple[int, int] = (0, 1)

    # -------------------------
    # global observation field
    # -------------------------
    zone_mu_scale: float = 0.45
    row_mu_scale: float = 0.10
    use_reflection_padding: bool = True

    # formation-stage sigma gradient: left noisy -> right predictable
    sigma_left: float = 0.50
    sigma_right: float = 0.03

    # -------------------------
    # hidden formation dynamics
    # -------------------------
    reliability_init: float = 0.0
    fragility_init: float = 0.08
    rupture_memory_init: float = 0.0
    conflict_load_init: float = 0.0

    reliability_decay: float = 0.02
    fragility_decay: float = 0.004
    rupture_memory_decay: float = 0.96
    conflict_load_decay: float = 0.95

    supportive_reliability_delta: float = 0.18
    misleading_reliability_delta: float = -0.22

    supportive_fragility_relief: float = 0.05
    misleading_fragility_boost: float = 0.10

    supportive_conflict_relief: float = 0.05
    misleading_conflict_boost: float = 0.08

    supportive_rupture_relief: float = 0.03
    misleading_rupture_increment: float = 0.12

    # keep rupture output key for compatibility, but default to formation-clean/off
    use_rupture_flag: bool = False
    rupture_prob_scale: float = 0.0

    # -------------------------
    # movement / embodiment
    # -------------------------
    mirror_x: bool = False
    mirror_actions: bool = False

    # local patch ordering: NW, N, NE, W, E, SW, S, SE
    patch_order: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    )


class NZonePhase2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    OUTCOME_SUPPORTIVE = 0
    OUTCOME_NEUTRAL = 1
    OUTCOME_MISLEADING = 2

    def __init__(self, config: Optional[NZonePhase2Config] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or NZonePhase2Config()
        self.render_mode = render_mode

        if self.cfg.obs_dim != 8:
            raise ValueError(f"This env expects obs_dim=8 for the 8-neighbor patch, got {self.cfg.obs_dim}")

        self.W = int(self.cfg.width)
        self.H = int(self.cfg.height)
        self.max_steps = int(self.cfg.max_steps)
        self.base_obs_dim = int(self.cfg.obs_dim)
        self.obs_dim = self.base_obs_dim + (2 if self.cfg.include_xy else 0)

        self.action_space = spaces.Discrete(5)
        high = np.ones((self.obs_dim,), dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._rng = np.random.default_rng(0)

        self._mu_map = np.zeros((self.H, self.W), dtype=np.float32)
        self._sigma_map = np.zeros((self.H, self.W), dtype=np.float32)
        self._build_static_maps(seed=0)

        self.x = 0
        self.y = 0
        self.t = 0
        self.visited: set[Tuple[int, int]] = set()

        self.reliability_estimate = 0.0
        self.fragility = 0.0
        self.rupture_memory = 0.0
        self.conflict_load = 0.0
        self._rupture_fired_this_step = False

    # -------------------------
    # map construction
    # -------------------------
    def _build_static_maps(self, seed: int) -> None:
        self._mu_map = self._build_mu_map(seed)
        self._sigma_map = self._build_sigma_map()

    def _build_mu_map(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.linspace(-1.0, 1.0, self.W, dtype=np.float32)
        row_center = (self.H - 1) / 2.0
        y = (np.arange(self.H, dtype=np.float32) - row_center) / max(1.0, row_center)

        X = np.tile(x[None, :], (self.H, 1))
        Y = np.tile(y[:, None], (1, self.W))
        base = self.cfg.zone_mu_scale * np.tanh(1.6 * X) + self.cfg.row_mu_scale * Y
        jitter = 0.015 * rng.normal(size=(self.H, self.W)).astype(np.float32)
        return (base + jitter).astype(np.float32)

    def _build_sigma_map(self) -> np.ndarray:
        col_sigmas = np.linspace(
            float(self.cfg.sigma_left),
            float(self.cfg.sigma_right),
            self.W,
            dtype=np.float32,
        )
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for y in range(self.H):
            m[y, :] = np.clip(col_sigmas, 0.005, None)
        return m

    # -------------------------
    # helpers
    # -------------------------
    def report_zone_id_of_x(self, x: int) -> int:
        x = int(np.clip(x, 0, self.W - 1))
        b0, b1, b2, b3 = [int(v) for v in self.cfg.report_zone_boundaries]
        if x < b0:
            return 0
        if x < b1:
            return 1
        if x < b2:
            return 2
        if x < b3:
            return 3
        return 4

    def zone_id(self) -> int:
        return self.report_zone_id_of_x(self.x)

    def is_encounter_column(self, x: int) -> bool:
        return int(x) in set(int(v) for v in self.cfg.encounter_columns)

    def encounter_index_of_x(self, x: int) -> int:
        cols = list(self.cfg.encounter_columns)
        try:
            return cols.index(int(x))
        except ValueError:
            return -1

    def _reflect_index(self, idx: int, size: int) -> int:
        if size <= 1:
            return 0
        i = int(idx)
        while i < 0 or i >= size:
            if i < 0:
                i = -i
            else:
                i = 2 * size - 2 - i
        return i

    def _patch_coord(self, x: int, y: int) -> Tuple[int, int]:
        if self.cfg.use_reflection_padding:
            return self._reflect_index(x, self.W), self._reflect_index(y, self.H)
        return int(np.clip(x, 0, self.W - 1)), int(np.clip(y, 0, self.H - 1))

    def _mx(self, x: int) -> int:
        return (self.W - 1 - int(x)) if self.cfg.mirror_x else int(x)

    def _swap_lr(self, action: int) -> int:
        if not (self.cfg.mirror_x and self.cfg.mirror_actions):
            return int(action)
        if action == self.ACTION_LEFT:
            return self.ACTION_RIGHT
        if action == self.ACTION_RIGHT:
            return self.ACTION_LEFT
        return int(action)

    def _clip_xy(self, x: int, y: int) -> Tuple[int, int]:
        return int(np.clip(x, 0, self.W - 1)), int(np.clip(y, 0, self.H - 1))

    def _effective_sigma(self, x: int, y: int) -> float:
        return float(self._sigma_map[int(y), int(x)])

    def _sample_cell_signal(self, x: int, y: int) -> float:
        px, py = self._patch_coord(x, y)
        mu = float(self._mu_map[py, px])
        sigma = self._effective_sigma(px, py)
        return float(mu + self._rng.normal(0.0, sigma))

    def _observe(self) -> np.ndarray:
        vals = np.zeros((self.base_obs_dim,), dtype=np.float32)
        for i, (dx, dy) in enumerate(self.cfg.patch_order):
            vals[i] = self._sample_cell_signal(self.x + dx, self.y + dy)

        if self.cfg.use_encounter and self.is_encounter_column(self.x):
            for d in self.cfg.encounter_dims:
                if 0 <= int(d) < self.base_obs_dim:
                    vals[int(d)] += float(self.cfg.encounter_signal)

        if self.cfg.include_xy:
            xy = np.array(
                [
                    self._mx(self.x) / max(1, self.W - 1),
                    self.y / max(1, self.H - 1),
                ],
                dtype=np.float32,
            )
            vals = np.concatenate([vals, xy], axis=0)

        return vals.astype(np.float32)

    def _encounter_profile(self, encounter_idx: int) -> str:
        profs = list(self.cfg.encounter_profiles)
        if len(profs) == 0 or encounter_idx < 0:
            return "none"
        if encounter_idx >= len(profs):
            return str(profs[-1])
        return str(profs[encounter_idx]).lower().strip()

    def _apply_supportive(self) -> None:
        self.reliability_estimate = float(np.clip(
            self.reliability_estimate + self.cfg.supportive_reliability_delta, -1.0, 1.0
        ))
        self.fragility = float(np.clip(
            self.fragility - self.cfg.supportive_fragility_relief, 0.0, 1.0
        ))
        self.conflict_load = float(np.clip(
            self.conflict_load - self.cfg.supportive_conflict_relief, 0.0, 1.0
        ))
        self.rupture_memory = float(np.clip(
            self.rupture_memory - self.cfg.supportive_rupture_relief, 0.0, 1.0
        ))

    def _apply_misleading(self) -> None:
        self.reliability_estimate = float(np.clip(
            self.reliability_estimate + self.cfg.misleading_reliability_delta, -1.0, 1.0
        ))
        self.fragility = float(np.clip(
            self.fragility + self.cfg.misleading_fragility_boost, 0.0, 1.0
        ))
        self.conflict_load = float(np.clip(
            self.conflict_load + self.cfg.misleading_conflict_boost, 0.0, 1.0
        ))
        self.rupture_memory = float(np.clip(
            self.rupture_memory + self.cfg.misleading_rupture_increment, 0.0, 1.0
        ))

    def _decay_hidden_state(self) -> None:
        if self.reliability_estimate > 0.0:
            self.reliability_estimate = max(0.0, self.reliability_estimate - float(self.cfg.reliability_decay))
        elif self.reliability_estimate < 0.0:
            self.reliability_estimate = min(0.0, self.reliability_estimate + float(self.cfg.reliability_decay))

        self.fragility = float(np.clip(self.fragility - self.cfg.fragility_decay, 0.0, 1.0))
        self.rupture_memory = float(np.clip(self.rupture_memory * self.cfg.rupture_memory_decay, 0.0, 1.0))
        self.conflict_load = float(np.clip(self.conflict_load * self.cfg.conflict_load_decay, 0.0, 1.0))

    def _maybe_fire_rupture(self) -> bool:
        if not self.cfg.use_rupture_flag or self.cfg.rupture_prob_scale <= 0.0:
            return False
        p = float(np.clip(
            self.cfg.rupture_prob_scale * (0.50 * self.fragility + 0.50 * self.rupture_memory),
            0.0,
            1.0,
        ))
        return bool(self._rng.random() < p)

    def _info_dict(
        self,
        on_encounter: bool,
        encounter_idx: int,
        encounter_profile: str,
        encounter_outcome: int,
    ) -> Dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "t": int(self.t),
            "zone_id": int(self.zone_id()),

            "on_encounter": int(on_encounter),
            "encounter_idx": int(encounter_idx),
            "encounter_profile": str(encounter_profile),
            "encounter_outcome": int(encounter_outcome),

            "current_sigma": float(self._effective_sigma(self.x, self.y)),
            "reliability_estimate": float(self.reliability_estimate),
            "fragility": float(self.fragility),
            "rupture_memory": float(self.rupture_memory),
            "conflict_load": float(self.conflict_load),

            "rupture": int(self._rupture_fired_this_step),
            "phase": "phase2",
        }

    # -------------------------
    # gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.x, self.y = self._clip_xy(*self.cfg.start_xy)
        self.t = 0
        self.visited = {(self.x, self.y)}

        self.reliability_estimate = float(self.cfg.reliability_init)
        self.fragility = float(self.cfg.fragility_init)
        self.rupture_memory = float(self.cfg.rupture_memory_init)
        self.conflict_load = float(self.cfg.conflict_load_init)
        self._rupture_fired_this_step = False

        obs = self._observe()
        info = self._info_dict(
            on_encounter=False,
            encounter_idx=-1,
            encounter_profile="none",
            encounter_outcome=self.OUTCOME_NEUTRAL,
        )
        return obs, info

    def step(self, action: int):
        self._decay_hidden_state()
        self._rupture_fired_this_step = False

        action = self._swap_lr(int(action))

        dx, dy = 0, 0
        if action == self.ACTION_UP:
            dy = -1
        elif action == self.ACTION_DOWN:
            dy = +1
        elif action == self.ACTION_LEFT:
            dx = -1
        elif action == self.ACTION_RIGHT:
            dx = +1

        self.x, self.y = self._clip_xy(self.x + dx, self.y + dy)
        self.t += 1
        self.visited.add((self.x, self.y))

        on_enc = False
        enc_idx = -1
        enc_profile = "none"
        enc_outcome = self.OUTCOME_NEUTRAL

        if self.cfg.use_encounter and self.is_encounter_column(self.x):
            on_enc = True
            enc_idx = self.encounter_index_of_x(self.x)
            enc_profile = self._encounter_profile(enc_idx)

            if enc_profile == "supportive":
                enc_outcome = self.OUTCOME_SUPPORTIVE
                self._apply_supportive()
            elif enc_profile == "misleading":
                enc_outcome = self.OUTCOME_MISLEADING
                self._apply_misleading()
            else:
                enc_outcome = self.OUTCOME_NEUTRAL

            self._rupture_fired_this_step = self._maybe_fire_rupture()

        obs = self._observe()
        reward = float(self.cfg.reward_scale)
        terminated = False
        truncated = bool(self.t >= self.max_steps)

        info = self._info_dict(
            on_encounter=on_enc,
            encounter_idx=enc_idx,
            encounter_profile=enc_profile,
            encounter_outcome=enc_outcome,
        )
        return obs, reward, terminated, truncated, info


def make_env(**kwargs) -> NZonePhase2Env:
    cfg = NZonePhase2Config(**kwargs)
    return NZonePhase2Env(config=cfg)