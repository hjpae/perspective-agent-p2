# cear_pilot/envs/nzone_phase1.py
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
class NZonePhase1Config:
    """
    Phase 1:
    - AAAI-style orientation shaping
    - reward-free
    - slow g should learn directional prior / predictability preference
    - no encounter / no hidden ecology
    """
    width: int = 23
    height: int = 7
    obs_dim: int = 8
    max_steps: int = 240

    include_xy: bool = False
    reward_scale: float = 0.0

    # observation field
    zone_mu_scale: float = 0.45
    row_mu_scale: float = 0.10
    use_reflection_padding: bool = True

    # reporting buckets
    report_zone_boundaries: Tuple[int, ...] = (4, 9, 14, 19)

    # embodiment / symmetry tests
    mirror_x: bool = False
    mirror_actions: bool = False

    # phase-1 start and sigma geometry
    phase1_start_xy: Tuple[int, int] = (11, 3)
    phase1_sigma_left: float = 0.60
    phase1_sigma_center: float = 0.30
    phase1_sigma_right: float = 0.03
    phase1_left_power: float = 0.90
    phase1_right_power: float = 1.85

    # static world control
    static_map_seed: int = 0
    resample_static_maps_on_reset: bool = False

    # local patch ordering: NW, N, NE, W, E, SW, S, SE
    patch_order: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    )


class NZonePhase1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    def __init__(self, config: Optional[NZonePhase1Config] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or NZonePhase1Config()
        self.render_mode = render_mode

        if self.cfg.obs_dim != 8:
            raise ValueError(f"Phase1 env expects obs_dim=8, got {self.cfg.obs_dim}")

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

        # Build one fixed static world at init.
        self._static_map_seed = int(self.cfg.static_map_seed)
        self._build_static_maps(seed=self._static_map_seed)

        self.x = 0
        self.y = 0
        self.t = 0
        self.visited: set[Tuple[int, int]] = set()

    # -------------------------
    # static maps
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

    def _build_sigma_map(self) -> np.ndarray:
        # column-only sigma, identical across rows
        col_sigmas = self._phase1_sigma_vector()
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for y in range(self.H):
            m[y, :] = np.clip(col_sigmas, 0.005, None)
        return m

    # -------------------------
    # geometry helpers
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

    # -------------------------
    # dynamics
    # -------------------------
    def _apply_action(self, action: int) -> Tuple[int, int]:
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

        return self._clip_xy(self.x + dx, self.y + dy)

    def _effective_sigma(self, x: int, y: int) -> float:
        return float(self._sigma_map[int(y), int(x)])

    def _sample_cell_signal(self, x: int, y: int) -> float:
        px, py = self._patch_coord(x, y)
        mu = float(self._mu_map[py, px])
        sigma = self._effective_sigma(px, py)
        return float(self._rng.normal(mu, sigma))

    def _get_obs(self) -> np.ndarray:
        vals = []
        for dx, dy in self.cfg.patch_order:
            px, py = self._patch_coord(self.x + dx, self.y + dy)
            vals.append(self._sample_cell_signal(px, py))

        obs = np.asarray(vals, dtype=np.float32)

        if self.cfg.include_xy:
            xy = np.asarray(
                [
                    (2.0 * self._mx(self.x) / max(1, self.W - 1)) - 1.0,
                    (2.0 * self.y / max(1, self.H - 1)) - 1.0,
                ],
                dtype=np.float32,
            )
            obs = np.concatenate([obs, xy], axis=0)

        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "t": int(self.t),
            "x": int(self.x),
            "y": int(self.y),
            "zone_id": int(self.zone_id()),
            "current_sigma": float(self._effective_sigma(self.x, self.y)),
            "reward": float(self.cfg.reward_scale),
            "static_map_seed": int(self._static_map_seed),
            "resample_static_maps_on_reset": bool(self.cfg.resample_static_maps_on_reset),
        }

    # -------------------------
    # gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Always reset rollout RNG if seed is provided.
        if seed is not None:
            self._rng = np.random.default_rng(seed)

            # Rebuild static maps only when explicitly requested.
            if bool(self.cfg.resample_static_maps_on_reset):
                self._static_map_seed = int(seed)
                self._build_static_maps(seed=int(seed))

        self.x, self.y = self.cfg.phase1_start_xy
        self.t = 0
        self.visited = {(self.x, self.y)}

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        self.t += 1
        self.x, self.y = self._apply_action(action)
        self.visited.add((self.x, self.y))

        obs = self._get_obs()
        info = self._get_info()

        terminated = False
        truncated = self.t >= self.max_steps
        reward = float(self.cfg.reward_scale)

        return obs, reward, terminated, truncated, info


def make_env(**kwargs) -> NZonePhase1Env:
    cfg = NZonePhase1Config(**kwargs)
    return NZonePhase1Env(config=cfg)