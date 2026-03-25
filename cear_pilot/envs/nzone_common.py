# cear_pilot/envs/nzone_common.py
# -*- coding: utf-8 -*-
"""
Common grid environment for both Phase 1 and Phase 2.

- Shared scaffold for Phase 1 and Phase 2: 7 rows x 23 columns by default
- Exteroceptive observation is the 8-neighbor local patch
- Reflection padding is used ONLY when constructing the 8-neighbor patch
- Phase 1: stronger nonlinear predictability gradient, center start, encounter columns inert
- Phase 2: milder linear predictability gradient, left-biased start, encounter columns active
- Encounter columns are part of the same predictability field as ordinary columns
- For logging / figures, columns are also grouped into 5 reporting zones
"""

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
class NZoneCommonConfig:
    # -------------------------
    # shared scaffold
    # -------------------------
    phase: str = "phase1"  # "phase1" or "phase2"
    width: int = 23
    height: int = 7
    obs_dim: int = 8  # must stay 8 for 8-neighbor patch
    max_steps: int = 240
    include_xy: bool = False
    reward_scale: float = 0.0  # reward-free by default

    # global observation field
    zone_mu_scale: float = 0.45
    row_mu_scale: float = 0.10
    use_reflection_padding: bool = True

    # encounter geometry: [4, 8, 12, 16, 20] in 1-indexed discussion => 0-based below
    encounter_columns: Tuple[int, ...] = (3, 7, 11, 15, 19)
    encounter_signal: float = 1.00
    encounter_dims: Tuple[int, int] = (0, 1)

    # reporting buckets only (for figures / summaries)
    report_zone_boundaries: Tuple[int, ...] = (5, 10, 14, 18)  # => 5 buckets

    # movement / embodiment
    mirror_x: bool = False
    mirror_actions: bool = False

    # -------------------------
    # phase 1 shaping
    # -------------------------
    phase1_start_xy: Tuple[int, int] = (11, 3)  # exact center of 7x23
    phase1_sigma_left: float = 0.60
    phase1_sigma_center: float = 0.30
    phase1_sigma_right: float = 0.03
    phase1_left_power: float = 0.90
    phase1_right_power: float = 1.85

    # -------------------------
    # phase 2 assay pressure
    # -------------------------
    phase2_start_xy: Tuple[int, int] = (0, 3)
    phase2_sigma_left: float = 0.50
    phase2_sigma_right: float = 0.05

    # row-wise stance modulation (applied to sigma in phase2)
    row_sigma_offsets: Tuple[float, ...] = (-0.03, -0.015, 0.0, 0.0, 0.0, 0.015, 0.03)
    row_exposure_mults: Tuple[float, ...] = (0.60, 0.75, 1.00, 1.00, 1.00, 1.25, 1.40)

    # mild ecological modulation (phase2 only)
    use_slip: bool = True
    base_action_slip: float = 0.00
    misleading_action_slip_boost: float = 0.10
    supportive_action_slip_relief: float = 0.03

    use_obs_corruption: bool = True
    misleading_obs_sigma_boost: float = 1.50
    supportive_obs_sigma_relief: float = 0.04

    # hidden encounter dynamics (phase2 only)
    use_encounter: bool = True
    supportive_reliability_delta: float = 0.18
    misleading_reliability_delta: float = -0.22
    neutral_reliability_decay: float = 0.02

    fragility_init: float = 0.08
    fragility_decay: float = 0.004
    rupture_memory_init: float = 0.00
    rupture_memory_decay: float = 0.96
    conflict_load_init: float = 0.00
    conflict_load_decay: float = 0.95

    # delayed supportive / misleading windows
    supportive_window_min: int = 3
    supportive_window_max: int = 6
    misleading_window_min: int = 2
    misleading_window_max: int = 5

    # per-outcome magnitude multipliers before row exposure scaling
    supportive_sigma_relief: float = 0.05
    supportive_fragility_relief: float = 0.05
    misleading_sigma_boost: float = 0.08
    misleading_fragility_boost: float = 0.10
    misleading_rupture_prob: float = 0.30

    # station semantics used only in phase2
    encounter_profiles: Tuple[str, ...] = (
        "ambiguous",
        "confirm",
        "perturb",
        "accumulate",
        "recovery",
    )

    # local patch ordering: NW, N, NE, W, E, SW, S, SE
    patch_order: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    )


class NZoneCommonEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    OUTCOME_SUPPORTIVE = 0
    OUTCOME_NEUTRAL = 1
    OUTCOME_MISLEADING = 2

    def __init__(self, config: Optional[NZoneCommonConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or NZoneCommonConfig()
        self.render_mode = render_mode

        if self.cfg.obs_dim != 8:
            raise ValueError(f"This env expects obs_dim=8 for the 8-neighbor patch, got {self.cfg.obs_dim}")
        if self.cfg.phase not in ("phase1", "phase2"):
            raise ValueError(f"phase must be 'phase1' or 'phase2', got {self.cfg.phase!r}")

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

        self.fragility = float(self.cfg.fragility_init)
        self.rupture_memory = float(self.cfg.rupture_memory_init)
        self.conflict_load = float(self.cfg.conflict_load_init)
        self.reliability_estimate = 0.0

        self._supportive_timer = 0
        self._misleading_timer = 0
        self._last_outcome_code = 1
        self._rupture_fired_this_step = False
        self._obs_sigma_delta = 0.0
        self._action_slip_delta = 0.0

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
        if self.cfg.phase == "phase1":
            col_sigmas = self._phase1_sigma_vector()
        else:
            col_sigmas = self._phase2_sigma_vector()

        row_offsets = np.array(self.cfg.row_sigma_offsets, dtype=np.float32)
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for y in range(self.H):
            m[y, :] = np.clip(col_sigmas + row_offsets[y], 0.005, None)
        return m

    def _phase1_sigma_vector(self) -> np.ndarray:
        left = float(self.cfg.phase1_sigma_left)
        center = float(self.cfg.phase1_sigma_center)
        right = float(self.cfg.phase1_sigma_right)
        c = self.W // 2

        sig = np.zeros((self.W,), dtype=np.float32)
        for x in range(0, c + 1):
            u = (c - x) / max(1, c)
            sig[x] = center + (left - center) * (u ** float(self.cfg.phase1_left_power))
        for x in range(c, self.W):
            u = (x - c) / max(1, self.W - 1 - c)
            sig[x] = center + (right - center) * (u ** float(self.cfg.phase1_right_power))

        sig[0] = left
        sig[c] = center
        sig[-1] = right
        return sig.astype(np.float32)

    def _phase2_sigma_vector(self) -> np.ndarray:
        return np.linspace(
            float(self.cfg.phase2_sigma_left),
            float(self.cfg.phase2_sigma_right),
            self.W,
            dtype=np.float32,
        )

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

    def row_band(self, y: int) -> str:
        y = int(np.clip(y, 0, self.H - 1))
        if y <= 1:
            return "cautious"
        if y >= 5:
            return "exposed"
        return "balanced"

    def row_exposure_mult(self, y: int) -> float:
        return float(self.cfg.row_exposure_mults[int(np.clip(y, 0, self.H - 1))])

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

    def _apply_action(self, action: int) -> Tuple[int, int]:
        action = self._swap_lr(int(action))

        if self.cfg.phase == "phase2" and self.cfg.use_slip:
            p_slip = float(np.clip(self.cfg.base_action_slip + self._action_slip_delta, 0.0, 0.95))
            if self._rng.random() < p_slip:
                action = int(self._rng.choice([0, 1, 2, 3, 4]))

        dx, dy = 0, 0
        if action == self.ACTION_UP:
            dy = -1
        elif action == self.ACTION_DOWN:
            dy = +1
        elif action == self.ACTION_LEFT:
            dx = -1
        elif action == self.ACTION_RIGHT:
            dx = +1

        return self._clip_xy(self.x + dx, self.y + dy)

    def _effective_sigma(self, x: int, y: int) -> float:
        s = float(self._sigma_map[int(y), int(x)])
        if self.cfg.phase == "phase2":
            s = max(0.005, s + float(self._obs_sigma_delta))
        return float(s)

    def _sample_cell_signal(self, x: int, y: int) -> float:
        px, py = self._patch_coord(x, y)
        mu = float(self._mu_map[py, px])
        sigma = self._effective_sigma(px, py)
        return float(mu + self._rng.normal(0.0, sigma))

    def _observe(self) -> np.ndarray:
        vals = np.zeros((self.base_obs_dim,), dtype=np.float32)
        for i, (dx, dy) in enumerate(self.cfg.patch_order):
            vals[i] = self._sample_cell_signal(self.x + dx, self.y + dy)

        if self.cfg.phase == "phase2" and self.cfg.use_encounter and self.is_encounter_column(self.x):
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

    def _decay_hidden_state(self) -> None:
        self.fragility = float(np.clip(self.fragility - self.cfg.fragility_decay, 0.0, 1.0))
        self.rupture_memory = float(np.clip(self.rupture_memory * self.cfg.rupture_memory_decay, 0.0, 1.0))
        self.conflict_load = float(np.clip(self.conflict_load * self.cfg.conflict_load_decay, 0.0, 1.0))

        if self.reliability_estimate > 0.0:
            self.reliability_estimate = max(0.0, self.reliability_estimate - float(self.cfg.neutral_reliability_decay))
        elif self.reliability_estimate < 0.0:
            self.reliability_estimate = min(0.0, self.reliability_estimate + float(self.cfg.neutral_reliability_decay))

    def _encounter_profile(self, encounter_idx: int) -> str:
        profs = list(self.cfg.encounter_profiles)
        if len(profs) == 0:
            return "ambiguous"
        if encounter_idx < 0:
            return "ambiguous"
        if encounter_idx >= len(profs):
            return profs[-1]
        return str(profs[encounter_idx])

    def _sample_encounter_outcome(self, encounter_idx: int, y: int) -> int:
        profile = self._encounter_profile(encounter_idx)

        if profile == "ambiguous":
            p_sup, p_neu, p_mis = 0.25, 0.50, 0.25
        elif profile == "confirm":
            if self.reliability_estimate >= 0.0:
                p_sup, p_neu, p_mis = 0.55, 0.25, 0.20
            else:
                p_sup, p_neu, p_mis = 0.20, 0.25, 0.55
        elif profile == "perturb":
            p_sup, p_neu, p_mis = 0.10, 0.25, 0.65
        elif profile == "accumulate":
            p_sup, p_neu, p_mis = 0.15, 0.20, 0.65
        elif profile == "recovery":
            p_sup, p_neu, p_mis = 0.65, 0.20, 0.15
        else:
            p_sup, p_neu, p_mis = 0.30, 0.40, 0.30

        p_sup += 0.10 * max(0.0, self.reliability_estimate)
        p_mis += 0.10 * max(0.0, self.conflict_load)
        z = max(1e-8, p_sup + p_neu + p_mis)
        p_sup, p_neu, p_mis = p_sup / z, p_neu / z, p_mis / z

        r = self._rng.random()
        if r < p_sup:
            return self.OUTCOME_SUPPORTIVE
        if r < p_sup + p_neu:
            return self.OUTCOME_NEUTRAL
        return self.OUTCOME_MISLEADING

    def _apply_supportive_window(self, exposure: float) -> None:
        self._supportive_timer = int(
            self._rng.integers(self.cfg.supportive_window_min, self.cfg.supportive_window_max + 1)
        )
        self._misleading_timer = max(0, self._misleading_timer - 1)

        relief = float(self.cfg.supportive_sigma_relief) * float(exposure)
        self._obs_sigma_delta = -relief
        self._action_slip_delta = -float(self.cfg.supportive_action_slip_relief) * float(exposure)

        self.fragility = float(np.clip(self.fragility - self.cfg.supportive_fragility_relief * exposure, 0.0, 1.0))
        self.reliability_estimate = float(
            np.clip(self.reliability_estimate + self.cfg.supportive_reliability_delta * exposure, -1.0, 1.0)
        )
        self._last_outcome_code = 0

    def _apply_misleading_window(self, exposure: float) -> None:
        self._misleading_timer = int(
            self._rng.integers(self.cfg.misleading_window_min, self.cfg.misleading_window_max + 1)
        )
        self._supportive_timer = max(0, self._supportive_timer - 1)

        boost = float(self.cfg.misleading_sigma_boost) * float(exposure)
        self._obs_sigma_delta = boost
        self._action_slip_delta = float(self.cfg.misleading_action_slip_boost) * float(exposure)

        self.fragility = float(np.clip(self.fragility + self.cfg.misleading_fragility_boost * exposure, 0.0, 1.0))
        self.rupture_memory = float(np.clip(self.rupture_memory + 0.25 * exposure, 0.0, 1.0))
        self.conflict_load = float(np.clip(self.conflict_load + 0.20 * exposure, 0.0, 1.0))
        self.reliability_estimate = float(
            np.clip(self.reliability_estimate + self.cfg.misleading_reliability_delta * exposure, -1.0, 1.0)
        )
        self._last_outcome_code = 2

        p_rup = np.clip(
            self.cfg.misleading_rupture_prob * exposure + 0.25 * self.fragility + 0.15 * self.rupture_memory,
            0.0,
            1.0,
        )
        self._rupture_fired_this_step = bool(self._rng.random() < p_rup)

    def _apply_neutral_window(self) -> None:
        self._last_outcome_code = 1

    def _update_windows(self) -> None:
        self._obs_sigma_delta = 0.0
        self._action_slip_delta = 0.0
        self._rupture_fired_this_step = False

        if self._supportive_timer > 0:
            self._supportive_timer -= 1
            self._obs_sigma_delta = -float(self.cfg.supportive_obs_sigma_relief)
            self._action_slip_delta = -float(self.cfg.supportive_action_slip_relief)

        if self._misleading_timer > 0:
            self._misleading_timer -= 1
            self._obs_sigma_delta += float(self.cfg.misleading_obs_sigma_boost)
            self._action_slip_delta += float(self.cfg.misleading_action_slip_boost)

            p_rup = np.clip(
                0.10 + 0.25 * self.fragility + 0.15 * self.rupture_memory + 0.10 * self.conflict_load,
                0.0,
                1.0,
            )
            self._rupture_fired_this_step = bool(self._rng.random() < p_rup)

        self._action_slip_delta = float(np.clip(self._action_slip_delta, -0.20, 0.50))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.cfg.phase == "phase1":
            self.x, self.y = self._clip_xy(*self.cfg.phase1_start_xy)
        else:
            self.x, self.y = self._clip_xy(*self.cfg.phase2_start_xy)

        self.t = 0
        self.visited = {(self.x, self.y)}

        self.fragility = float(self.cfg.fragility_init)
        self.rupture_memory = float(self.cfg.rupture_memory_init)
        self.conflict_load = float(self.cfg.conflict_load_init)
        self.reliability_estimate = 0.0

        self._supportive_timer = 0
        self._misleading_timer = 0
        self._last_outcome_code = 1
        self._rupture_fired_this_step = False
        self._obs_sigma_delta = 0.0
        self._action_slip_delta = 0.0

        obs = self._observe()
        info = self._info_dict(
            on_encounter=False,
            encounter_idx=-1,
            encounter_profile="none",
            encounter_outcome=self.OUTCOME_NEUTRAL,
        )
        return obs, info

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
            "row_band": str(self.row_band(self.y)),
            "row_exposure_mult": float(self.row_exposure_mult(self.y)),
            "current_sigma": float(self._effective_sigma(self.x, self.y)),
            "reliability_estimate": float(self.reliability_estimate),
            "fragility": float(self.fragility),
            "rupture_memory": float(self.rupture_memory),
            "conflict_load": float(self.conflict_load),
            "supportive_timer": int(self._supportive_timer),
            "misleading_timer": int(self._misleading_timer),
            "rupture": int(self._rupture_fired_this_step),
            "phase": str(self.cfg.phase),
        }

    def step(self, action: int):
        if self.cfg.phase == "phase2":
            self._decay_hidden_state()
            self._update_windows()

        self.x, self.y = self._apply_action(int(action))
        self.t += 1
        self.visited.add((self.x, self.y))

        on_enc = False
        enc_idx = -1
        enc_profile = "none"
        enc_outcome = self.OUTCOME_NEUTRAL

        if self.cfg.phase == "phase2" and self.cfg.use_encounter and self.is_encounter_column(self.x):
            on_enc = True
            enc_idx = self.encounter_index_of_x(self.x)
            enc_profile = self._encounter_profile(enc_idx)
            enc_outcome = self._sample_encounter_outcome(enc_idx, self.y)
            exposure = float(self.row_exposure_mult(self.y))

            if enc_outcome == self.OUTCOME_SUPPORTIVE:
                self._apply_supportive_window(exposure)
            elif enc_outcome == self.OUTCOME_MISLEADING:
                self._apply_misleading_window(exposure)
            else:
                self._apply_neutral_window()

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


def make_phase1_env(**kwargs) -> NZoneCommonEnv:
    cfg = NZoneCommonConfig(phase="phase1", use_encounter=False, **kwargs)
    return NZoneCommonEnv(config=cfg)


def make_phase2_env(**kwargs) -> NZoneCommonEnv:
    cfg = NZoneCommonConfig(phase="phase2", use_encounter=True, **kwargs)
    return NZoneCommonEnv(config=cfg)