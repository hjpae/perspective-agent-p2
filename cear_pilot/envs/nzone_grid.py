# cear_pilot/envs/nzone_grid.py
# -*- coding: utf-8 -*-
"""
N-zone Gridworld (Gymnasium Env)

Phase 2 update:
- Keeps 3-zone structure and reward-free dynamics
- Adds ambiguous encounter tiles E
- Hidden context variables:
    * fragility
    * rupture_memory
- Delayed rupture after encounter:
    same percept, different downstream consequence depending on hidden context/history
- Rupture aftermath:
    * temporary observation corruption
    * temporary extra action-slip

Existing Phase-2 ecology hooks (slip/drift/volatility/hazard) are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError(
        "This environment requires gymnasium. Install with: pip install gymnasium"
    ) from e


@dataclass
class NZoneConfig:
    width: int = 15
    height: int = 9
    obs_dim: int = 8
    max_steps: int = 240

    # observation mean separation scale
    zone_mu_scale: float = 0.5
    zone_sigma: Tuple[float, float, float] = (0.60, 0.30, 0.05)

    include_xy: bool = False

    # Existing ecology toggles
    use_slip: bool = False
    use_drift: bool = False
    use_volatility: bool = False
    use_hazard: bool = False

    p_slip: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    slip_mode: str = "random_action"  # "stay" / "random_action" / "reverse"

    p_drift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    drift_vec: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (0, 0), (0, 0), (0, 0)
    )

    volatile_zone: int = 0
    volatile_period: int = 40
    volatile_strength: float = 0.0

    hazard_mode: str = "teleport"  # "teleport" / "sensor_blackout" / "reset"
    p_hazard: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hazard_teleport_to: Tuple[int, int] = (0, 0)
    hazard_blackout_steps: int = 6

    phase2_obs_mu_scale: float = 0.5
    phase2_obs_equal_sigma: bool = True

    mirror_x: bool = False
    mirror_actions: bool = True

    # -------------------------
    # New Phase 2 encounter logic
    # -------------------------
    use_encounter: bool = True

    # Explicit encounter cells. If empty, defaults will be created automatically.
    encounter_coords: Tuple[Tuple[int, int], ...] = ()

    # Which dims get a fixed "encounter percept" overlay
    encounter_dims: Tuple[int, int] = (0, 1)
    encounter_signal: float = 1.25

    # Hidden context dynamics
    fragility_init: float = 0.10
    rupture_memory_init: float = 0.00

    fragility_decay: float = 0.01
    rupture_memory_decay: float = 0.95

    zone_fragility_delta: Tuple[float, float, float] = (-0.02, 0.00, +0.02)

    # Delayed rupture scheduler
    encounter_delay_min: int = 2
    encounter_delay_max: int = 5

    rupture_base_prob: float = 0.15
    rupture_fragility_weight: float = 0.60
    rupture_memory_weight: float = 0.20

    # Rupture aftermath
    rupture_obs_corrupt_steps: int = 3
    rupture_obs_sigma: float = 3.0
    rupture_action_slip_prob: float = 0.30
    rupture_memory_increment: float = 0.30
    no_rupture_memory_delta: float = -0.10


class NZoneGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    def __init__(self, config: Optional[NZoneConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or NZoneConfig()
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

        self._zone_mu = np.zeros((3, self.base_obs_dim), dtype=np.float32)
        self._zone_sigma = np.array(self.cfg.zone_sigma, dtype=np.float32)

        self._p_slip_rt = np.array(self.cfg.p_slip, dtype=np.float32)
        self._p_drift_rt = np.array(self.cfg.p_drift, dtype=np.float32)
        self._drift_vec_rt = [tuple(v) for v in self.cfg.drift_vec]

        self._blackout_timer = 0

        # rupture aftermath
        self._rupture_obs_timer = 0
        self._rupture_action_timer = 0
        self._rupture_fired_this_step = False

        # hidden context
        self.fragility = float(self.cfg.fragility_init)
        self.rupture_memory = float(self.cfg.rupture_memory_init)

        # encounter schedule
        self.pending_ruptures: List[int] = []

        # state
        self.x = 0
        self.y = 0
        self.t = 0
        self.visited = set()

        self.encounter_tiles = self._build_encounter_tiles()
        self._init_zone_prototypes(seed=0)

    # -----------------
    # Helpers
    # -----------------
    def _phase2_active(self) -> bool:
        return bool(
            self.cfg.use_slip
            or self.cfg.use_drift
            or self.cfg.use_volatility
            or self.cfg.use_hazard
            or self.cfg.use_encounter
        )

    def _init_zone_prototypes(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, size=(3, self.base_obs_dim)).astype(np.float32)
        base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9)

        mu_scale = float(self.cfg.phase2_obs_mu_scale) if self._phase2_active() else float(self.cfg.zone_mu_scale)
        self._zone_mu = base * mu_scale

        if self._phase2_active() and self.cfg.phase2_obs_equal_sigma:
            s = float(np.mean(np.array(self.cfg.zone_sigma, dtype=np.float32)))
            self._zone_sigma = np.array([s, s, s], dtype=np.float32)
        else:
            self._zone_sigma = np.array(self.cfg.zone_sigma, dtype=np.float32)

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

    def zone_id(self) -> int:
        x_eff = self._mx(self.x)
        if x_eff < self.W / 3:
            return 0
        elif x_eff < 2 * self.W / 3:
            return 1
        else:
            return 2

    def _clip_xy(self, x: int, y: int) -> Tuple[int, int]:
        x = int(np.clip(x, 0, self.W - 1))
        y = int(np.clip(y, 0, self.H - 1))
        return x, y

    def _reverse_action(self, action: int) -> int:
        if action == self.ACTION_UP:
            return self.ACTION_DOWN
        if action == self.ACTION_DOWN:
            return self.ACTION_UP
        if action == self.ACTION_LEFT:
            return self.ACTION_RIGHT
        if action == self.ACTION_RIGHT:
            return self.ACTION_LEFT
        return self.ACTION_STAY

    def _build_encounter_tiles(self) -> set[Tuple[int, int]]:
        if len(self.cfg.encounter_coords) > 0:
            return {self._clip_xy(int(x), int(y)) for x, y in self.cfg.encounter_coords}

        # default: one encounter tile in each zone, near middle row
        y = self.H // 2
        xs = [max(1, self.W // 6), self.W // 2, min(self.W - 2, (5 * self.W) // 6)]
        return {self._clip_xy(x, y) for x in xs}

    def _is_encounter_tile(self, x: int, y: int) -> bool:
        return (int(x), int(y)) in self.encounter_tiles

    def _observe(self) -> np.ndarray:
        zid = self.zone_id()
        mu = self._zone_mu[zid]
        sigma = float(self._zone_sigma[zid])

        if self._blackout_timer > 0:
            sigma = max(sigma, 3.0)

        if self._rupture_obs_timer > 0:
            sigma = max(sigma, float(self.cfg.rupture_obs_sigma))

        obs = mu + self._rng.normal(0, sigma, size=(self.base_obs_dim,)).astype(np.float32)

        # Fixed encounter percept: same visible cue, hidden context remains unobserved
        if self.cfg.use_encounter and self._is_encounter_tile(self.x, self.y):
            dims = [d for d in self.cfg.encounter_dims if 0 <= int(d) < self.base_obs_dim]
            for d in dims:
                obs[int(d)] += float(self.cfg.encounter_signal)

        if self.cfg.include_xy:
            x_rep = self._mx(self.x)
            obs_xy = np.array(
                [x_rep / max(1, self.W - 1), self.y / max(1, self.H - 1)],
                dtype=np.float32,
            )
            obs = np.concatenate([obs, obs_xy], axis=0)

        return obs.astype(np.float32)

    def set_zone_sigma(self, zone_sigma):
        self._zone_sigma = np.array([float(x) for x in zone_sigma], dtype=np.float32)

    # -----------------
    # Encounter / hidden-context dynamics
    # -----------------
    def _update_hidden_context(self, zid: int) -> None:
        df = float(self.cfg.zone_fragility_delta[zid])
        self.fragility += df

        # small relaxation toward lower tension
        self.fragility -= float(self.cfg.fragility_decay)

        self.fragility = float(np.clip(self.fragility, 0.0, 1.0))
        self.rupture_memory = float(np.clip(self.rupture_memory * self.cfg.rupture_memory_decay, 0.0, 1.0))

    def _schedule_encounter_if_needed(self) -> bool:
        if not self.cfg.use_encounter:
            return False
        if not self._is_encounter_tile(self.x, self.y):
            return False

        dmin = int(self.cfg.encounter_delay_min)
        dmax = int(self.cfg.encounter_delay_max)
        delay = int(self._rng.integers(dmin, dmax + 1))
        self.pending_ruptures.append(delay)
        return True

    def _advance_pending_ruptures(self) -> None:
        if not self.cfg.use_encounter or len(self.pending_ruptures) == 0:
            self._rupture_fired_this_step = False
            return

        self._rupture_fired_this_step = False
        new_queue = []
        for k in self.pending_ruptures:
            kk = int(k) - 1
            if kk <= 0:
                self._evaluate_rupture_event()
            else:
                new_queue.append(kk)
        self.pending_ruptures = new_queue

    def _evaluate_rupture_event(self) -> None:
        p = (
            float(self.cfg.rupture_base_prob)
            + float(self.cfg.rupture_fragility_weight) * float(self.fragility)
            + float(self.cfg.rupture_memory_weight) * float(self.rupture_memory)
        )
        p = float(np.clip(p, 0.0, 1.0))

        if self._rng.random() < p:
            self._rupture_fired_this_step = True
            self._rupture_obs_timer = int(max(self._rupture_obs_timer, self.cfg.rupture_obs_corrupt_steps))
            self._rupture_action_timer = int(max(self._rupture_action_timer, self.cfg.rupture_obs_corrupt_steps))
            self.rupture_memory = float(np.clip(
                self.rupture_memory + float(self.cfg.rupture_memory_increment), 0.0, 1.0
            ))
        else:
            self.rupture_memory = float(np.clip(
                self.rupture_memory + float(self.cfg.no_rupture_memory_delta), 0.0, 1.0
            ))

    # -----------------
    # Existing ecology hooks
    # -----------------
    def _apply_slip(self, action: int, zid: int) -> Tuple[int, bool]:
        p = 0.0
        if self.cfg.use_slip:
            p += float(np.clip(self._p_slip_rt[zid], 0.0, 1.0))
        if self._rupture_action_timer > 0:
            p += float(np.clip(self.cfg.rupture_action_slip_prob, 0.0, 1.0))
        p = float(np.clip(p, 0.0, 1.0))

        if p <= 0.0 or self._rng.random() >= p:
            return action, False

        mode = str(self.cfg.slip_mode).lower().strip()
        if mode == "stay":
            return self.ACTION_STAY, True
        if mode == "reverse":
            return self._reverse_action(action), True
        return int(self._rng.integers(0, 5)), True

    def _apply_drift(self, x: int, y: int, zid: int) -> Tuple[int, int, bool]:
        if not self.cfg.use_drift:
            return x, y, False

        p = float(np.clip(self._p_drift_rt[zid], 0.0, 1.0))
        if self._rng.random() >= p:
            return x, y, False

        dx, dy = self._drift_vec_rt[zid]
        x2, y2 = self._clip_xy(x + int(dx), y + int(dy))
        return x2, y2, True

    def _update_volatility(self, zid: int) -> bool:
        if not self.cfg.use_volatility:
            return False
        if zid != int(self.cfg.volatile_zone):
            return False
        if self.cfg.volatile_period <= 0:
            return False
        if (self.t % int(self.cfg.volatile_period)) != 0:
            return False

        strength = float(max(0.0, self.cfg.volatile_strength))

        if self.cfg.use_slip and strength > 0.0:
            delta = (self._rng.random() * 2.0 - 1.0) * strength
            self._p_slip_rt[zid] = float(np.clip(self._p_slip_rt[zid] + delta, 0.0, 1.0))

        if self.cfg.use_drift and strength > 0.0:
            if self._rng.random() < min(1.0, strength):
                dx, dy = self._drift_vec_rt[zid]
                ndx, ndy = -int(dy), int(dx)
                if self._rng.random() < 0.5:
                    ndx, ndy = -ndx, -ndy
                self._drift_vec_rt[zid] = (ndx, ndy)

        return True

    def _apply_hazard(self, x: int, y: int, zid: int) -> Tuple[int, int, bool]:
        if not self.cfg.use_hazard:
            return x, y, False

        p = float(np.clip(self.cfg.p_hazard[zid], 0.0, 1.0))
        if self._rng.random() >= p:
            return x, y, False

        mode = str(self.cfg.hazard_mode).lower().strip()
        if mode == "teleport":
            tx, ty = self.cfg.hazard_teleport_to
            tx, ty = self._clip_xy(int(tx), int(ty))
            return tx, ty, True

        if mode == "sensor_blackout":
            self._blackout_timer = int(max(1, self.cfg.hazard_blackout_steps))
            return x, y, True

        if mode == "reset":
            cx, cy = self.W // 2, self.H // 2
            cx, cy = self._clip_xy(cx, cy)
            return cx, cy, True

        return x, y, False

    # -----------------
    # Gym API
    # -----------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._init_zone_prototypes(seed=seed)

        self._p_slip_rt = np.array(self.cfg.p_slip, dtype=np.float32)
        self._p_drift_rt = np.array(self.cfg.p_drift, dtype=np.float32)
        self._drift_vec_rt = [tuple(v) for v in self.cfg.drift_vec]

        self._blackout_timer = 0
        self._rupture_obs_timer = 0
        self._rupture_action_timer = 0
        self._rupture_fired_this_step = False

        self.fragility = float(self.cfg.fragility_init)
        self.rupture_memory = float(self.cfg.rupture_memory_init)
        self.pending_ruptures = []

        self.x = self.W // 2
        self.y = self.H // 2
        self.t = 0

        self.visited = set()
        self.visited.add((self.x, self.y))

        obs = self._observe()
        info = {
            "zone_id": self.zone_id(),
            "x": self._mx(self.x),
            "y": self.y,
            "t": self.t,
            "phase2_active": self._phase2_active(),
            "fragility": float(self.fragility),
            "rupture_memory": float(self.rupture_memory),
            "on_encounter": bool(self._is_encounter_tile(self.x, self.y)),
            "rupture": False,
            "pending_ruptures": int(len(self.pending_ruptures)),
        }
        return obs, info

    def step(self, action: int):
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be int, got {type(action)}")

        action = int(action)
        action = self._swap_lr(action)
        if action < 0 or action > 4:
            raise ValueError(f"Invalid action: {action}")

        old_pos = (self.x, self.y)
        zid_before = self.zone_id()

        volatility_event = self._update_volatility(zid_before)

        a_eff, slipped = self._apply_slip(action, zid_before)

        x, y = self.x, self.y
        if a_eff == self.ACTION_UP:
            y -= 1
        elif a_eff == self.ACTION_DOWN:
            y += 1
        elif a_eff == self.ACTION_LEFT:
            x -= 1
        elif a_eff == self.ACTION_RIGHT:
            x += 1

        x, y = self._clip_xy(x, y)

        x, y, drifted = self._apply_drift(x, y, zid_before)

        self.x, self.y = x, y
        zid_after_move = self.zone_id()

        x, y, hazard_event = self._apply_hazard(self.x, self.y, zid_after_move)
        self.x, self.y = x, y

        # hidden context update from current zone
        self._update_hidden_context(self.zone_id())

        # entering/occupying encounter tile schedules delayed consequence
        encounter_event = self._schedule_encounter_if_needed()

        # time update
        self.t += 1

        # advance rupture timers after time advances
        self._advance_pending_ruptures()

        if self._blackout_timer > 0:
            self._blackout_timer -= 1
        if self._rupture_obs_timer > 0:
            self._rupture_obs_timer -= 1
        if self._rupture_action_timer > 0:
            self._rupture_action_timer -= 1

        new_pos = (self.x, self.y)
        moved = new_pos != old_pos

        obs = self._observe()

        reward = 0.0
        terminated = False
        truncated = self.t >= self.max_steps

        info = {
            "zone_id": self.zone_id(),
            "x": self._mx(self.x),
            "y": self.y,
            "t": self.t,
            "a_in": int(action),
            "a_eff": int(a_eff),
            "moved": bool(moved),
            "slip": bool(slipped),
            "drift": bool(drifted),
            "hazard": bool(hazard_event),
            "blackout_timer": int(self._blackout_timer),
            "volatility_update": bool(volatility_event),

            "p_slip_rt": float(self._p_slip_rt[self.zone_id()]),
            "p_drift_rt": float(self._p_drift_rt[self.zone_id()]),
            "drift_vec_rt": tuple(self._drift_vec_rt[self.zone_id()]),

            # new Phase 2 diagnostics
            "fragility": float(self.fragility),
            "rupture_memory": float(self.rupture_memory),
            "on_encounter": bool(self._is_encounter_tile(self.x, self.y)),
            "encounter_event": bool(encounter_event),
            "pending_ruptures": int(len(self.pending_ruptures)),
            "rupture": bool(self._rupture_fired_this_step),
            "rupture_obs_timer": int(self._rupture_obs_timer),
            "rupture_action_timer": int(self._rupture_action_timer),
        }

        return obs, reward, terminated, truncated, info

    # -----------------
    # Rendering
    # -----------------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        else:
            self._render_ascii()

    def _render_ascii(self):
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        for ex, ey in self.encounter_tiles:
            grid[ey][ex] = "E"
        grid[self.y][self.x] = "A"
        s = "\n".join("".join(row) for row in grid)
        print(s)
        print(
            f"t={self.t} zone={self.zone_id()} pos=({self._mx(self.x)},{self.y}) "
            f"fragility={self.fragility:.2f} rupture_mem={self.rupture_memory:.2f} "
            f"pending={len(self.pending_ruptures)} rupture={self._rupture_fired_this_step}"
        )

    def _render_rgb(self):
        cell = 24
        img = np.zeros((self.H * cell, self.W * cell, 3), dtype=np.uint8)

        zone_colors = np.array(
            [
                [255, 210, 210],  # zone 0
                [200, 230, 255],  # zone 1
                [225, 220, 220],  # zone 2
            ],
            dtype=np.uint8,
        )

        for y in range(self.H):
            for x in range(self.W):
                x_eff = self._mx(x)
                if x_eff < self.W / 3:
                    zid = 0
                elif x_eff < 2 * self.W / 3:
                    zid = 1
                else:
                    zid = 2

                y0, y1 = y * cell, (y + 1) * cell
                x0, x1 = x * cell, (x + 1) * cell
                img[y0:y1, x0:x1] = zone_colors[zid]

        # encounter tiles
        for ex, ey in self.encounter_tiles:
            y0, y1 = ey * cell, (ey + 1) * cell
            x0, x1 = ex * cell, (ex + 1) * cell
            img[y0:y1, x0:x1] = np.array([255, 215, 0], dtype=np.uint8)

        # agent
        ay, ax = self.y, self.x
        y0, y1 = ay * cell, (ay + 1) * cell
        x0, x1 = ax * cell, (ax + 1) * cell
        img[y0:y1, x0:x1] = np.array([0, 0, 0], dtype=np.uint8)

        img[::cell, :, :] = 0
        img[:, ::cell, :] = 0
        return img

    def close(self):
        pass


def make_env(**kwargs) -> NZoneGridEnv:
    cfg = NZoneConfig(**kwargs)
    return NZoneGridEnv(config=cfg)