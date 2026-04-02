# cear_pilot/envs/nzone_phase2.py
# -*- coding: utf-8 -*-
"""
Phase 2 environment: sigma gradient + temporal perturbation.

Environment itself NEVER changes. Sigma gradient is fixed (left=noisy, right=clean).
Perturbation = transient obs distortion at scheduled timesteps.
The distortion makes left side appear temporarily cleaner and right side noisier
(inverting the natural gradient for a brief window).

The idea: if g learns from these perturbations, it should eventually gate perception
such that "left=clean" persists even after the perturbation fades (hysteresis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("gymnasium required") from e


@dataclass
class NZonePhase2Config:
    width: int = 23
    height: int = 7
    obs_dim: int = 8
    max_steps: int = 300
    include_xy: bool = False
    reward_scale: float = 0.0
    start_xy: Tuple[int, int] = (11, 3)

    report_zone_boundaries: Tuple[int, ...] = (4, 9, 14, 19)

    zone_mu_scale: float = 0.45
    row_mu_scale: float = 0.10
    use_reflection_padding: bool = True
    sigma_left: float = 0.20
    sigma_right: float = 0.10

    n_perturbations: int = 4
    perturbation_duration: int = 15
    perturbation_scale: float = 0.12
    perturbation_jitter_std: float = 5.0

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

    def __init__(self, config: Optional[NZonePhase2Config] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or NZonePhase2Config()
        self.render_mode = render_mode

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

        self._inversion_pattern = np.linspace(1.0, -1.0, self.W, dtype=np.float32)

        self.x = 0
        self.y = 0
        self.t = 0
        self.visited: set = set()

        self.perturbation_steps: List[int] = []
        self._perturbation_active = False
        self._perturbation_remaining = 0
        self._perturbation_trace = 0.0

    def _build_static_maps(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = np.linspace(-1.0, 1.0, self.W, dtype=np.float32)
        row_center = (self.H - 1) / 2.0
        y = (np.arange(self.H, dtype=np.float32) - row_center) / max(1.0, row_center)
        X = np.tile(x[None, :], (self.H, 1))
        Y = np.tile(y[:, None], (1, self.W))
        base = self.cfg.zone_mu_scale * np.tanh(1.6 * X) + self.cfg.row_mu_scale * Y
        jitter = 0.015 * rng.normal(size=(self.H, self.W)).astype(np.float32)
        self._mu_map = (base + jitter).astype(np.float32)

        col_sigmas = np.linspace(
            float(self.cfg.sigma_left), float(self.cfg.sigma_right),
            self.W, dtype=np.float32,
        )
        self._sigma_map = np.tile(col_sigmas[None, :], (self.H, 1)).astype(np.float32)

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

    def _clip_xy(self, x: int, y: int) -> Tuple[int, int]:
        return int(np.clip(x, 0, self.W - 1)), int(np.clip(y, 0, self.H - 1))

    def report_zone_id(self, x: int) -> int:
        x = int(np.clip(x, 0, self.W - 1))
        for zi, b in enumerate(self.cfg.report_zone_boundaries):
            if x < int(b):
                return zi
        return len(self.cfg.report_zone_boundaries)

    def _sample_cell(self, x: int, y: int) -> float:
        px, py = self._patch_coord(x, y)
        mu = float(self._mu_map[py, px])
        sigma = float(self._sigma_map[py, px])
        return float(self._rng.normal(mu, sigma))

    def _perturbation_distortion(self) -> np.ndarray:
        if not self._perturbation_active:
            return np.zeros(self.base_obs_dim, dtype=np.float32)
        scale = float(self.cfg.perturbation_scale)
        distortion = np.zeros(self.base_obs_dim, dtype=np.float32)
        for i, (dx, dy) in enumerate(self.cfg.patch_order):
            px, _ = self._patch_coord(self.x + dx, self.y + dy)
            distortion[i] = scale * self._inversion_pattern[px]
        return distortion

    def _observe(self) -> np.ndarray:
        vals = np.zeros(self.base_obs_dim, dtype=np.float32)
        for i, (dx, dy) in enumerate(self.cfg.patch_order):
            vals[i] = self._sample_cell(self.x + dx, self.y + dy)
        vals = vals + self._perturbation_distortion()
        if self.cfg.include_xy:
            xy = np.array([
                self.x / max(1, self.W - 1),
                self.y / max(1, self.H - 1),
            ], dtype=np.float32)
            vals = np.concatenate([vals, xy])
        return vals.astype(np.float32)

    def _schedule_perturbations(self) -> List[int]:
        n = int(self.cfg.n_perturbations)
        if n <= 0:
            return []
        upper = self.max_steps - int(self.cfg.perturbation_duration) - 10
        centers = np.linspace(40, upper, n, dtype=np.float32)
        jitter = self._rng.normal(0.0, float(self.cfg.perturbation_jitter_std), size=n)
        steps = np.clip(np.round(centers + jitter), 10, upper).astype(int)
        steps.sort()
        for _ in range(5):
            for i in range(1, len(steps)):
                if steps[i] - steps[i-1] < 20:
                    steps[i] = steps[i-1] + 20
            steps = np.clip(steps, 10, upper)
        return [int(s) for s in steps]

    def _update_perturbation(self) -> None:
        if self.t in self.perturbation_steps:
            self._perturbation_active = True
            self._perturbation_remaining = int(self.cfg.perturbation_duration)
            self._perturbation_trace = 1.0
        if self._perturbation_active:
            self._perturbation_remaining -= 1
            if self._perturbation_remaining <= 0:
                self._perturbation_active = False
        self._perturbation_trace *= 0.95

    def _info_dict(self) -> Dict[str, Any]:
        return {
            "x": int(self.x), "y": int(self.y), "t": int(self.t),
            "zone_id": self.report_zone_id(self.x),
            "current_sigma": float(self._sigma_map[self.y, self.x]),
            "perturbation_active": int(self._perturbation_active),
            "perturbation_trace": float(self._perturbation_trace),
            "n_perturbations": int(self.cfg.n_perturbations),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.x, self.y = self._clip_xy(*self.cfg.start_xy)
        self.t = 0
        self.visited = {(self.x, self.y)}
        self._perturbation_active = False
        self._perturbation_remaining = 0
        self._perturbation_trace = 0.0
        self.perturbation_steps = self._schedule_perturbations()
        return self._observe(), self._info_dict()

    def step(self, action: int):
        dx, dy = 0, 0
        if action == self.ACTION_UP:    dy = -1
        elif action == self.ACTION_DOWN:  dy = 1
        elif action == self.ACTION_LEFT:  dx = -1
        elif action == self.ACTION_RIGHT: dx = 1
        self.x, self.y = self._clip_xy(self.x + dx, self.y + dy)
        self.t += 1
        self.visited.add((self.x, self.y))
        self._update_perturbation()
        obs = self._observe()
        return obs, float(self.cfg.reward_scale), False, self.t >= self.max_steps, self._info_dict()


def make_env(**kwargs) -> NZonePhase2Env:
    return NZonePhase2Env(config=NZonePhase2Config(**kwargs))