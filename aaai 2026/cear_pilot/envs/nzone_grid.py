# cear_pilot/envs/nzone_grid.py
# -*- coding: utf-8 -*-
"""
N-zone Gridworld (Gymnasium Env)

Patched for Phase 2:
- No intrinsic reward shaping (reward is always 0.0)
- Zone-wise ecology via:
  1) slip (action failure)
  2) drift (wind)
  3) volatility (time-varying parameters in a volatile zone)
  4) hazard (teleport / sensor_blackout / reset) without reward penalties

Zones:
  3 vertical zones (0 / 1 / 2)

Actions:
  0: up, 1: down, 2: left, 3: right, 4: stay
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError(
        "This environment requires gymnasium. Install with: pip install gymnasium"
    ) from e


# -------------------------
# Config
# -------------------------
@dataclass
class NZoneConfig:
    width: int = 15
    height: int = 9
    obs_dim: int = 8
    max_steps: int = 240

    # observation mean separation scale (Phase 2 default)
    zone_mu_scale: float = 0.5  # Phase 1: was 2.5

    # per-zone observation noise
    zone_sigma: Tuple[float, float, float] = (0.60, 0.30, 0.05)  # z0 volatile

    # include normalized (x,y) in obs tail
    include_xy: bool = False

    ## ---- Phase 2 (environmental tweak) toggles ----
    use_slip: bool = False
    use_drift: bool = False
    use_volatility: bool = False
    use_hazard: bool = False

    # 1. Slip (action failure due to low controllability)
    p_slip: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    slip_mode: str = "random_action"  # "stay" / "random_action" / "reverse"

    # 2. Drift (external "wind" that pushes agent)
    p_drift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    drift_vec: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0), (0, 0))

    # 3. Volatility/Nonstability: slip/drift parameters can change over time
    volatile_zone: int = 0
    volatile_period: int = 40
    volatile_strength: float = 0.0  # additional slip probability / drift randomness

    # 4. Hazard: ext. penalty-less version
    hazard_mode: str = "teleport"  # "teleport" / "sensor_blackout" / "reset"
    p_hazard: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hazard_teleport_to: Tuple[int, int] = (0, 0)
    hazard_blackout_steps: int = 6  # only used for sensor_blackout

    # Observation regime control
    phase2_obs_mu_scale: float = 0.5
    phase2_obs_equal_sigma: bool = True

    # --- Mirror control (left-right reflection) ---
    mirror_x: bool = False          # if True, left-right mirror the world
    mirror_actions: bool = True     # if True, swap LEFT/RIGHT action semantics


# -------------------------
# Env
# -------------------------
class NZoneGridEnv(gym.Env):
    """
    Gridworld:
    - no extrinsic task reward
    - Phase 2 adds ecology via transition/observation perturbations
    - reward is always 0.0 (to avoid reward shaping)
    """

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

        # zone prototype means / noise
        self._zone_mu = np.zeros((3, self.base_obs_dim), dtype=np.float32)
        self._zone_sigma = np.array(self.cfg.zone_sigma, dtype=np.float32)

        # runtime copies (so volatility can modify without mutating cfg)
        self._p_slip_rt = np.array(self.cfg.p_slip, dtype=np.float32)
        self._p_drift_rt = np.array(self.cfg.p_drift, dtype=np.float32)
        self._drift_vec_rt = [tuple(v) for v in self.cfg.drift_vec]  # list of (dx,dy)

        # hazard state
        self._blackout_timer = 0

        # state
        self.x = 0
        self.y = 0
        self.t = 0

        # visited cells (legacy; not used for reward shaping anymore)
        self.visited = set()

        self._init_zone_prototypes(seed=0)

    # -----------------
    # Helpers
    # -----------------
    def _phase2_active(self) -> bool:
        return bool(self.cfg.use_slip or self.cfg.use_drift or self.cfg.use_volatility or self.cfg.use_hazard)

    def _init_zone_prototypes(self, seed: int) -> None:
        rng = np.random.default_rng(seed)

        base = rng.normal(0, 1, size=(3, self.base_obs_dim)).astype(np.float32)
        base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9)

        mu_scale = float(self.cfg.phase2_obs_mu_scale) if self._phase2_active() else float(self.cfg.zone_mu_scale)
        self._zone_mu = base * mu_scale

        # sigma equalization (optional) to avoid g becoming a sensory-cluster label
        if self._phase2_active() and self.cfg.phase2_obs_equal_sigma:
            s = float(np.mean(np.array(self.cfg.zone_sigma, dtype=np.float32)))
            self._zone_sigma = np.array([s, s, s], dtype=np.float32)
        else:
            self._zone_sigma = np.array(self.cfg.zone_sigma, dtype=np.float32)

    # -----------------
    # Mirror helpers
    # -----------------
    def _mx(self, x: int) -> int:
        """Mirror x coordinate (for reporting + zone assignment)."""
        return (self.W - 1 - int(x)) if self.cfg.mirror_x else int(x)

    def _swap_lr(self, action: int) -> int:
        """Swap LEFT/RIGHT actions if mirror_actions is enabled."""
        if not (self.cfg.mirror_x and self.cfg.mirror_actions):
            return int(action)
        if action == self.ACTION_LEFT:
            return self.ACTION_RIGHT
        if action == self.ACTION_RIGHT:
            return self.ACTION_LEFT
        return int(action)

    # -----------------
    # Core zone logic (MIRROR-AWARE)
    # -----------------
    def zone_id(self) -> int:
        # IMPORTANT: zone should be determined in mirrored coordinates when mirror_x=True
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
        # Note: reverse is defined in "effective action space" after mirroring.
        if action == self.ACTION_UP:
            return self.ACTION_DOWN
        if action == self.ACTION_DOWN:
            return self.ACTION_UP
        if action == self.ACTION_LEFT:
            return self.ACTION_RIGHT
        if action == self.ACTION_RIGHT:
            return self.ACTION_LEFT
        return self.ACTION_STAY

    def _observe(self) -> np.ndarray:
        zid = self.zone_id()
        mu = self._zone_mu[zid]
        sigma = float(self._zone_sigma[zid])

        # hazard: sensor blackout -> temporarily huge observation noise
        if self._blackout_timer > 0:
            sigma = max(sigma, 3.0)

        obs = mu + self._rng.normal(0, sigma, size=(self.base_obs_dim,)).astype(np.float32)

        if self.cfg.include_xy:
            # IMPORTANT: report x in mirrored coordinates for consistency with zone_id/logging
            x_rep = self._mx(self.x)
            obs_xy = np.array(
                [x_rep / max(1, self.W - 1), self.y / max(1, self.H - 1)],
                dtype=np.float32,
            )
            obs = np.concatenate([obs, obs_xy], axis=0)

        return obs.astype(np.float32)

    # -----------------
    # Demo helper: runtime sigma switch
    # -----------------
    def set_zone_sigma(self, zone_sigma):
        """Change observation noise online (for regime-switch demos)."""
        self._zone_sigma = np.array([float(x) for x in zone_sigma], dtype=np.float32)

    # -----------------
    # Ecology hooks (Phase 2)
    # -----------------
    def _apply_slip(self, action: int, zid: int) -> Tuple[int, bool]:
        """1) Slip: with probability p_slip[zid], action is corrupted."""
        if not self.cfg.use_slip:
            return action, False

        p = float(np.clip(self._p_slip_rt[zid], 0.0, 1.0))
        if self._rng.random() >= p:
            return action, False

        mode = str(self.cfg.slip_mode).lower().strip()
        if mode == "stay":
            return self.ACTION_STAY, True
        if mode == "reverse":
            return self._reverse_action(action), True

        return int(self._rng.integers(0, 5)), True

    def _apply_drift(self, x: int, y: int, zid: int) -> Tuple[int, int, bool]:
        """2) Drift: with probability p_drift[zid], apply drift vector after movement."""
        if not self.cfg.use_drift:
            return x, y, False

        p = float(np.clip(self._p_drift_rt[zid], 0.0, 1.0))
        if self._rng.random() >= p:
            return x, y, False

        dx, dy = self._drift_vec_rt[zid]
        x2, y2 = self._clip_xy(x + int(dx), y + int(dy))
        return x2, y2, True

    def _update_volatility(self, zid: int) -> bool:
        """
        3) Volatility:
        In volatile_zone, every volatile_period steps, randomly perturbs slip/drift parameters
        (stationarity-breaking without using reward penalties).
        """
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
        """4) Hazard: teleport / sensor_blackout / reset (no reward penalties)."""
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

        # runtime params reset
        self._p_slip_rt = np.array(self.cfg.p_slip, dtype=np.float32)
        self._p_drift_rt = np.array(self.cfg.p_drift, dtype=np.float32)
        self._drift_vec_rt = [tuple(v) for v in self.cfg.drift_vec]
        self._blackout_timer = 0

        self.x = self.W // 2
        self.y = self.H // 2
        self.t = 0

        self.visited = set()
        self.visited.add((self.x, self.y))

        obs = self._observe()
        info = {
            "zone_id": self.zone_id(),
            "x": self._mx(self.x),  # IMPORTANT: report mirrored x
            "y": self.y,
            "t": self.t,
            "phase2_active": self._phase2_active(),
        }
        return obs, info

    def step(self, action: int):
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be int, got {type(action)}")

        action = int(action)
        # IMPORTANT: mirror action semantics early (for mirror control)
        action = self._swap_lr(action)

        if action < 0 or action > 4:
            raise ValueError(f"Invalid action: {action}")

        old_pos = (self.x, self.y)
        zid_before = self.zone_id()

        # Volatility update at start (affects this step)
        volatility_event = self._update_volatility(zid_before)

        # 1) Slip: possibly corrupt action (based on zone before movement)
        a_eff, slipped = self._apply_slip(action, zid_before)

        # Apply effective action (deterministic movement)
        x, y = self.x, self.y
        if a_eff == self.ACTION_UP:
            y -= 1
        elif a_eff == self.ACTION_DOWN:
            y += 1
        elif a_eff == self.ACTION_LEFT:
            x -= 1
        elif a_eff == self.ACTION_RIGHT:
            x += 1
        elif a_eff == self.ACTION_STAY:
            pass

        x, y = self._clip_xy(x, y)

        # 2) Drift
        x, y, drifted = self._apply_drift(x, y, zid_before)

        # 4) Hazard (based on zone after movement)
        self.x, self.y = x, y
        zid_after_move = self.zone_id()
        x, y, hazard_event = self._apply_hazard(self.x, self.y, zid_after_move)
        self.x, self.y = x, y

        # time update
        self.t += 1

        # blackout countdown
        if self._blackout_timer > 0:
            self._blackout_timer -= 1

        new_pos = (self.x, self.y)
        moved = new_pos != old_pos

        obs = self._observe()

        reward = 0.0
        terminated = False
        truncated = self.t >= self.max_steps

        info = {
            "zone_id": self.zone_id(),
            "x": self._mx(self.x),  # IMPORTANT: report mirrored x
            "y": self.y,
            "t": self.t,

            # diagnostics
            "a_in": int(action),
            "a_eff": int(a_eff),
            "moved": bool(moved),

            "slip": bool(slipped),
            "drift": bool(drifted),
            "hazard": bool(hazard_event),
            "blackout_timer": int(self._blackout_timer),
            "volatility_update": bool(volatility_event),

            # expose runtime params (for analysis/debug)
            "p_slip_rt": float(self._p_slip_rt[self.zone_id()]),
            "p_drift_rt": float(self._p_drift_rt[self.zone_id()]),
            "drift_vec_rt": tuple(self._drift_vec_rt[self.zone_id()]),
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
        grid[self.y][self.x] = "A"
        s = "\n".join("".join(row) for row in grid)
        print(s)
        print(f"t={self.t} zone={self.zone_id()} pos=({self._mx(self.x)},{self.y}) blackout={self._blackout_timer}")

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

        # IMPORTANT: color zones using mirrored-x ecology so the visualization matches zone_id()
        for y in range(self.H):
            for x in range(self.W):
                x_eff = self._mx(x)  # mirror for ecology visualization
                if x_eff < self.W / 3:
                    zid = 0
                elif x_eff < 2 * self.W / 3:
                    zid = 1
                else:
                    zid = 2

                y0, y1 = y * cell, (y + 1) * cell
                x0, x1 = x * cell, (x + 1) * cell
                img[y0:y1, x0:x1] = zone_colors[zid]

        # agent (black square at TRUE coordinates)
        ay, ax = self.y, self.x
        y0, y1 = ay * cell, (ay + 1) * cell
        x0, x1 = ax * cell, (ax + 1) * cell
        img[y0:y1, x0:x1] = np.array([0, 0, 0], dtype=np.uint8)

        # grid lines
        img[::cell, :, :] = 0
        img[:, ::cell, :] = 0

        return img

    def close(self):
        pass


def make_env(**kwargs) -> NZoneGridEnv:
    cfg = NZoneConfig(**kwargs)
    return NZoneGridEnv(config=cfg)
