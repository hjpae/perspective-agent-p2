# cear_pilot/envs/nzone_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("This environment requires gymnasium. Install with: pip install gymnasium") from e


@dataclass
class NZonePhase2Config:
    # -------------------------
    # Grid / observation
    # -------------------------
    width: int = 23
    height: int = 7
    obs_dim: int = 8
    max_steps: int = 300
    include_xy: bool = False
    reward_scale: float = 0.0
    start_xy: Tuple[int, int] = (0, 3)

    # Reporting buckets: 4 / 5 / 5 / 5 / 4
    report_zone_boundaries: Tuple[int, ...] = (4, 9, 14, 19)

    # -------------------------
    # Static world field
    # -------------------------
    zone_mu_scale: float = 0.45
    row_mu_scale: float = 0.10
    use_reflection_padding: bool = True

    # Weak background sigma gradient
    sigma_left: float = 0.20
    sigma_right: float = 0.10

    # -------------------------
    # Scheduled hidden events
    # -------------------------
    schedule_pattern: str = "1-1-1-1"   # one of: "1-1-1-1", "2-2", "3-1", "1-3"
    valence_sequence: str = "SSSS"      # exactly 4 chars over {"S", "M"}
    schedule_jitter_std: float = 5.0
    min_event_gap: int = 6
    event_delay_steps: int = 3

    # Weak ambiguous event cue
    use_event_marker: bool = True
    event_marker_signal: float = 0.25
    event_marker_dims: Tuple[int, int] = (0, 1)

    # -------------------------
    # Hidden consequence state c_t
    # -------------------------
    c_init: float = 0.0
    c_decay: float = 0.91
    c_max: float = 1.0
    supportive_impulse: float = 0.45
    misleading_impulse: float = -0.45
    distortion_scale: float = 0.30

    # structured distortion over the 8-neighbor patch.
    # Patch order: NW, N, NE, W, E, SW, S, SE
    supportive_basis: Tuple[float, ...] = (
        -1.0, 0.0,  1.0,
        -1.0,       1.0,
        -1.0, 0.0,  1.0,
    )
    misleading_basis: Tuple[float, ...] = (
         1.0, 0.0, -1.0,
         1.0,      -1.0,
         1.0, 0.0, -1.0,
    )

    # -------------------------
    # Embodiment
    # -------------------------
    mirror_x: bool = False
    mirror_actions: bool = False

    # Local patch order: NW, N, NE, W, E, SW, S, SE
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

    VALENCE_SUPPORTIVE = "supportive"
    VALENCE_MISLEADING = "misleading"

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

        self._supportive_basis = self._normalize_basis(np.asarray(self.cfg.supportive_basis, dtype=np.float32))
        self._misleading_basis = self._normalize_basis(np.asarray(self.cfg.misleading_basis, dtype=np.float32))

        self.x = 0
        self.y = 0
        self.t = 0
        self.visited: set[Tuple[int, int]] = set()

        self.c_t = float(self.cfg.c_init)

        self.event_steps: List[int] = []
        self.event_valences: List[str] = []
        self.consequence_steps: List[int] = []
        self._pending_events: List[Dict[str, Any]] = []

        self._event_now = False
        self._event_id_now = -1
        self._event_valence_hidden_now = "none"

        self._consequence_now = False
        self._consequence_id_now = -1
        self._consequence_valence_hidden_now = "none"

        self._steps_since_last_event = -1
        self._steps_since_last_consequence = -1

    # -------------------------
    # Static maps
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
    # Helpers
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

    def _effective_sigma(self, x: int, y: int) -> float:
        return float(self._sigma_map[int(y), int(x)])

    def _sample_cell_signal(self, x: int, y: int) -> float:
        px, py = self._patch_coord(x, y)
        mu = float(self._mu_map[py, px])
        sigma = self._effective_sigma(px, py)
        return float(mu + self._rng.normal(0.0, sigma))

    @staticmethod
    def _normalize_basis(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return v.astype(np.float32)
        return (v / n).astype(np.float32)

    def _parse_valence_sequence(self, seq: str) -> List[str]:
        s = str(seq).strip().upper()
        if len(s) != 4 or any(ch not in ("S", "M") for ch in s):
            raise ValueError("valence_sequence must be a length-4 string over {'S','M'}")
        return [
            self.VALENCE_SUPPORTIVE if ch == "S" else self.VALENCE_MISLEADING
            for ch in s
        ]

    def _sample_event_steps(self) -> List[int]:
        upper = self.max_steps - int(self.cfg.event_delay_steps) - 1
        pattern = str(self.cfg.schedule_pattern).strip()

        if pattern == "1-1-1-1":
            centers = [45, 105, 165, 225]

        elif pattern == "2-2":
            centers = [80, 88, 212, 220]

        elif pattern == "3-1":
            centers = [64, 70, 76, 210]

        elif pattern == "1-3":
            centers = [90, 204, 210, 216]

        else:
            raise ValueError(
                f"Unknown schedule_pattern='{pattern}'. "
                "Use one of: '1-1-1-1', '2-2', '3-1', '1-3'."
            )

        xs = np.asarray(centers, dtype=np.float32)
        xs = xs + self._rng.normal(0.0, float(self.cfg.schedule_jitter_std), size=4).astype(np.float32)
        xs = np.clip(np.round(xs), 1, upper).astype(np.int32)
        xs.sort()

        # Min-gap repair
        min_gap = int(self.cfg.min_event_gap)
        for _ in range(8):
            for i in range(1, 4):
                if int(xs[i] - xs[i - 1]) < min_gap:
                    xs[i] = xs[i - 1] + min_gap
            xs = np.clip(xs, 1, upper)

            for i in range(2, -1, -1):
                if int(xs[i + 1] - xs[i]) < min_gap:
                    xs[i] = xs[i + 1] - min_gap
            xs = np.clip(xs, 1, upper)
            xs.sort()

        return [int(v) for v in xs.tolist()]

    def _queue_event_if_needed(self) -> None:
        self._event_now = False
        self._event_id_now = -1
        self._event_valence_hidden_now = "none"

        for idx, step in enumerate(self.event_steps):
            if int(step) == int(self.t):
                self._event_now = True
                self._event_id_now = int(idx)
                self._event_valence_hidden_now = str(self.event_valences[idx])

                self._pending_events.append(
                    {
                        "event_id": int(idx),
                        "fire_step": int(self.t + self.cfg.event_delay_steps),
                        "valence": str(self.event_valences[idx]),
                    }
                )
                self._steps_since_last_event = 0
                break

    def _apply_pending_consequences(self) -> None:
        self._consequence_now = False
        self._consequence_id_now = -1
        self._consequence_valence_hidden_now = "none"

        if len(self._pending_events) == 0:
            return

        still_pending: List[Dict[str, Any]] = []
        impulses: List[float] = []

        fired_any = False
        last_fired_id = -1
        last_fired_valence = "none"

        for item in self._pending_events:
            if int(item["fire_step"]) == int(self.t):
                fired_any = True
                last_fired_id = int(item["event_id"])
                last_fired_valence = str(item["valence"])
                if str(item["valence"]) == self.VALENCE_SUPPORTIVE:
                    impulses.append(float(self.cfg.supportive_impulse))
                else:
                    impulses.append(float(self.cfg.misleading_impulse))
            else:
                still_pending.append(item)

        self._pending_events = still_pending

        if not fired_any:
            return

        total_impulse = float(np.sum(np.asarray(impulses, dtype=np.float32)))
        pre = float(self.c_t)
        raw = float(self.cfg.c_decay) * pre + total_impulse
        self.c_t = float(self.cfg.c_max * np.tanh(raw / max(1e-6, float(self.cfg.c_max))))

        self._consequence_now = True
        self._consequence_id_now = int(last_fired_id)
        self._consequence_valence_hidden_now = str(last_fired_valence)
        self._steps_since_last_consequence = 0

    def _decay_c_state(self) -> None:
        self.c_t = float(self.cfg.c_decay) * float(self.c_t)

    def _distortion_vector(self) -> np.ndarray:
        pos = max(float(self.c_t), 0.0)
        neg = max(-float(self.c_t), 0.0)
        distortion = (pos * self._supportive_basis) + (neg * self._misleading_basis)
        distortion = float(self.cfg.distortion_scale) * distortion
        return distortion.astype(np.float32)

    def _observe(self) -> np.ndarray:
        vals = np.zeros((self.base_obs_dim,), dtype=np.float32)
        for i, (dx, dy) in enumerate(self.cfg.patch_order):
            vals[i] = self._sample_cell_signal(self.x + dx, self.y + dy)

        vals = vals + self._distortion_vector()

        if self.cfg.use_event_marker and self._event_now:
            for d in self.cfg.event_marker_dims:
                if 0 <= int(d) < self.base_obs_dim:
                    vals[int(d)] += float(self.cfg.event_marker_signal)

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

    def _info_dict(self) -> Dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "t": int(self.t),
            "zone_id": int(self.zone_id()),
            "current_sigma": float(self._effective_sigma(self.x, self.y)),

            "schedule_pattern": str(self.cfg.schedule_pattern),
            "valence_sequence": str(self.cfg.valence_sequence),

            "event_now": int(self._event_now),
            "event_id": int(self._event_id_now),
            "event_valence_hidden": str(self._event_valence_hidden_now),

            "consequence_now": int(self._consequence_now),
            "consequence_id": int(self._consequence_id_now),
            "consequence_valence_hidden": str(self._consequence_valence_hidden_now),

            "event_steps": list(self.event_steps),
            "event_valences": list(self.event_valences),
            "consequence_steps": list(self.consequence_steps),

            "steps_since_last_event": int(self._steps_since_last_event),
            "steps_since_last_consequence": int(self._steps_since_last_consequence),
            "pending_consequences": int(len(self._pending_events)),

            "c_state": float(self.c_t),
        }

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.x, self.y = self._clip_xy(*self.cfg.start_xy)
        self.t = 0
        self.visited = {(self.x, self.y)}

        self.c_t = float(self.cfg.c_init)

        self._event_now = False
        self._event_id_now = -1
        self._event_valence_hidden_now = "none"

        self._consequence_now = False
        self._consequence_id_now = -1
        self._consequence_valence_hidden_now = "none"

        self._steps_since_last_event = -1
        self._steps_since_last_consequence = -1

        self.event_steps = self._sample_event_steps()
        self.event_valences = self._parse_valence_sequence(self.cfg.valence_sequence)
        self.consequence_steps = [int(s + self.cfg.event_delay_steps) for s in self.event_steps]
        self._pending_events = []

        obs = self._observe()
        info = self._info_dict()
        return obs, info

    def step(self, action: int):
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

        if self._steps_since_last_event >= 0:
            self._steps_since_last_event += 1
        if self._steps_since_last_consequence >= 0:
            self._steps_since_last_consequence += 1

        self._decay_c_state()
        self._queue_event_if_needed()
        self._apply_pending_consequences()

        obs = self._observe()
        reward = float(self.cfg.reward_scale)
        terminated = False
        truncated = bool(self.t >= self.max_steps)

        info = self._info_dict()
        return obs, reward, terminated, truncated, info


def make_env(**kwargs) -> NZonePhase2Env:
    cfg = NZonePhase2Config(**kwargs)
    return NZonePhase2Env(config=cfg)