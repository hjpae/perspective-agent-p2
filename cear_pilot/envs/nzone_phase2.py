# cear_pilot/envs/nzone_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("This environment requires gymnasium. Install with: pip install gymnasium") from e


@dataclass
class NZonePhase2Config:
    """
    Phase 2:
    - 3 zones, 6 columns each => width=18 
    - encounter columns every 3rd column: [.., E, .., E, .., E, ..]
    - start at rightmost edge of Z0, center row
    - weaker sigma gradient than Phase 1
    - encounter columns expose hidden reliability structure
    - maturity target: flexible but stable recalibration of engagement threshold
    """
    width: int = 18
    height: int = 9
    obs_dim: int = 8
    max_steps: int = 240

    zone_sigma: Tuple[float, float, float] = (0.42, 0.30, 0.20)
    zone_mu_scale: float = 0.5
    include_xy: bool = False

    # fixed encounter columns every 3rd column
    encounter_signal: float = 1.20
    encounter_dims: Tuple[int, int] = (0, 1)

    # weaker ecology than the previous collapse-prone version
    p_slip: Tuple[float, float, float] = (0.00, 0.03, 0.08)
    p_drift: Tuple[float, float, float] = (0.00, 0.04, 0.08)
    drift_vec: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (0, 0), (1, 0), (-1, 0)
    )
    use_slip: bool = True
    use_drift: bool = True
    use_volatility: bool = True

    volatile_zone: int = 2
    volatile_period: int = 20
    volatile_strength: float = 0.10

    # hidden context
    fragility_init: float = 0.10
    fragility_decay: float = 0.005
    rupture_memory_init: float = 0.00
    rupture_memory_decay: float = 0.96
    conflict_load_init: float = 0.00
    conflict_load_decay: float = 0.95

    zone_fragility_delta: Tuple[float, float, float] = (0.00, 0.02, 0.04)

    # hidden reliability mode
    mode_period: int = 30
    mode_switch_prob: float = 0.60  # chance to switch mode at mode boundary
    # 0=open, 1=mixed, 2=guarded

    # rupture mechanics
    rupture_base_prob: float = 0.30
    rupture_fragility_weight: float = 0.80
    rupture_memory_weight: float = 0.30
    rupture_load_weight: float = 0.25
    rupture_reliability_relief: float = 0.20

    rupture_obs_corrupt_steps: int = 4
    rupture_obs_sigma: float = 3.0
    rupture_action_slip_prob: float = 0.25
    rupture_memory_increment: float = 0.30
    conflict_load_increment: float = 0.25

    supportive_reliability_delta: float = 0.18
    misleading_reliability_delta: float = -0.20
    neutral_reliability_decay: float = 0.02

    encounter_delay_min: int = 1
    encounter_delay_max: int = 4

    mirror_x: bool = False
    mirror_actions: bool = False


class NZonePhase2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    MODE_OPEN = 0
    MODE_MIXED = 1
    MODE_GUARDED = 2

    OUTCOME_SUPPORTIVE = 0
    OUTCOME_NEUTRAL = 1
    OUTCOME_MISLEADING = 2

    def __init__(self, config: Optional[NZonePhase2Config] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or NZonePhase2Config()
        self.render_mode = render_mode

        self.W = int(self.cfg.width)
        self.H = int(self.cfg.height)
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

        self.x = 0
        self.y = 0
        self.t = 0

        self.fragility = float(self.cfg.fragility_init)
        self.rupture_memory = float(self.cfg.rupture_memory_init)
        self.conflict_load = float(self.cfg.conflict_load_init)

        self.reliability_estimate = 0.0
        self.mode = self.MODE_MIXED

        self.pending_ruptures: List[int] = []
        self._rupture_obs_timer = 0
        self._rupture_action_timer = 0
        self._rupture_fired_this_step = False

        self._last_outcome = self.OUTCOME_NEUTRAL
        self._last_outcome_code = 1

        self._init_zone_prototypes(seed=0)

    # -------------------------
    # helpers
    # -------------------------
    def _init_zone_prototypes(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, size=(3, self.base_obs_dim)).astype(np.float32)
        base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9)
        self._zone_mu = base * float(self.cfg.zone_mu_scale)

    def _zone_width(self) -> int:
        return self.W // 3

    def zone_id_of_x(self, x: int) -> int:
        zw = self._zone_width()
        if x < zw:
            return 0
        elif x < 2 * zw:
            return 1
        return 2

    def zone_id(self) -> int:
        return self.zone_id_of_x(self.x)

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

    def _encounter_columns(self) -> List[int]:
        # 0-based: 2,5,8,11,14,17 for width 18
        return [i for i in range(self.W) if (i + 1) % 3 == 0]

    def _is_encounter_column(self, x: int) -> bool:
        return int(x) in self._encounter_columns()

    def _observe(self) -> np.ndarray:
        zid = self.zone_id()
        mu = self._zone_mu[zid]
        sigma = float(self._zone_sigma[zid])

        if self._rupture_obs_timer > 0:
            sigma = max(sigma, float(self.cfg.rupture_obs_sigma))

        obs = mu + self._rng.normal(0, sigma, size=(self.base_obs_dim,)).astype(np.float32)

        # fixed encounter percept overlay: same percept, hidden meaning varies by history/mode
        if self._is_encounter_column(self.x):
            dims = [d for d in self.cfg.encounter_dims if 0 <= int(d) < self.base_obs_dim]
            for d in dims:
                obs[int(d)] += float(self.cfg.encounter_signal)

        if self.cfg.include_xy:
            obs_xy = np.array(
                [self.x / max(1, self.W - 1), self.y / max(1, self.H - 1)],
                dtype=np.float32,
            )
            obs = np.concatenate([obs, obs_xy], axis=0)

        return obs.astype(np.float32)

    def _maybe_switch_mode(self) -> None:
        if self.cfg.mode_period <= 0:
            return
        if self.t > 0 and (self.t % int(self.cfg.mode_period) == 0):
            if self._rng.random() < float(self.cfg.mode_switch_prob):
                choices = [self.MODE_OPEN, self.MODE_MIXED, self.MODE_GUARDED]
                choices.remove(self.mode)
                self.mode = int(self._rng.choice(choices))

    def _mode_probs(self) -> Tuple[float, float, float]:
        if self.mode == self.MODE_OPEN:
            return (0.65, 0.20, 0.15)
        if self.mode == self.MODE_GUARDED:
            return (0.15, 0.20, 0.65)
        return (0.33, 0.34, 0.33)

    def _sample_encounter_outcome(self) -> int:
        p_sup, p_neu, p_mis = self._mode_probs()

        # small history-sensitive modulation:
        # if reliability estimate is already high, supportive evidence becomes slightly more likely;
        # if conflict load is high, misleading evidence becomes slightly more likely.
        p_sup = p_sup + 0.10 * max(0.0, self.reliability_estimate)
        p_mis = p_mis + 0.10 * max(0.0, self.conflict_load)
        z = p_sup + p_neu + p_mis
        p_sup, p_neu, p_mis = p_sup / z, p_neu / z, p_mis / z

        r = self._rng.random()
        if r < p_sup:
            return self.OUTCOME_SUPPORTIVE
        if r < p_sup + p_neu:
            return self.OUTCOME_NEUTRAL
        return self.OUTCOME_MISLEADING

    def _apply_encounter_outcome(self, outcome: int) -> None:
        if outcome == self.OUTCOME_SUPPORTIVE:
            self.reliability_estimate = float(np.clip(
                self.reliability_estimate + float(self.cfg.supportive_reliability_delta), -1.0, 1.0
            ))
            self.conflict_load = float(np.clip(self.conflict_load - 0.08, 0.0, 1.0))
            self.fragility = float(np.clip(self.fragility - 0.05, 0.0, 1.0))
            self._last_outcome_code = 0

        elif outcome == self.OUTCOME_MISLEADING:
            self.reliability_estimate = float(np.clip(
                self.reliability_estimate + float(self.cfg.misleading_reliability_delta), -1.0, 1.0
            ))
            self.conflict_load = float(np.clip(
                self.conflict_load + float(self.cfg.conflict_load_increment), 0.0, 1.0
            ))
            self.fragility = float(np.clip(self.fragility + 0.12, 0.0, 1.0))
            delay = int(self._rng.integers(self.cfg.encounter_delay_min, self.cfg.encounter_delay_max + 1))
            self.pending_ruptures.append(delay)
            self._last_outcome_code = 2

        else:
            # neutral
            if self.reliability_estimate > 0:
                self.reliability_estimate = max(0.0, self.reliability_estimate - self.cfg.neutral_reliability_decay)
            elif self.reliability_estimate < 0:
                self.reliability_estimate = min(0.0, self.reliability_estimate + self.cfg.neutral_reliability_decay)
            self._last_outcome_code = 1

    def _advance_pending_ruptures(self) -> None:
        self._rupture_fired_this_step = False
        if len(self.pending_ruptures) == 0:
            return

        new_queue = []
        for k in self.pending_ruptures:
            kk = int(k) - 1
            if kk <= 0:
                self._evaluate_rupture()
            else:
                new_queue.append(kk)
        self.pending_ruptures = new_queue

    def _evaluate_rupture(self) -> None:
        p = (
            float(self.cfg.rupture_base_prob)
            + float(self.cfg.rupture_fragility_weight) * float(self.fragility)
            + float(self.cfg.rupture_memory_weight) * float(self.rupture_memory)
            + float(self.cfg.rupture_load_weight) * float(self.conflict_load)
            - float(self.cfg.rupture_reliability_relief) * max(0.0, float(self.reliability_estimate))
        )
        p = float(np.clip(p, 0.0, 1.0))

        if self._rng.random() < p:
            self._rupture_fired_this_step = True
            self._rupture_obs_timer = int(max(self._rupture_obs_timer, self.cfg.rupture_obs_corrupt_steps))
            self._rupture_action_timer = int(max(self._rupture_action_timer, self.cfg.rupture_obs_corrupt_steps))

            self.rupture_memory = float(np.clip(
                self.rupture_memory + float(self.cfg.rupture_memory_increment), 0.0, 1.0
            ))
            self.conflict_load = float(np.clip(
                self.conflict_load + float(self.cfg.conflict_load_increment), 0.0, 1.0
            ))
            self.reliability_estimate = float(np.clip(self.reliability_estimate - 0.08, -1.0, 1.0))

    def _update_hidden_context(self, zid: int) -> None:
        self.fragility += float(self.cfg.zone_fragility_delta[zid])
        self.fragility -= float(self.cfg.fragility_decay)
        self.fragility = float(np.clip(self.fragility, 0.0, 1.0))

        self.rupture_memory = float(np.clip(
            self.rupture_memory * float(self.cfg.rupture_memory_decay), 0.0, 1.0
        ))
        self.conflict_load = float(np.clip(
            self.conflict_load * float(self.cfg.conflict_load_decay), 0.0, 1.0
        ))

    def _apply_slip(self, action: int, zid: int) -> Tuple[int, bool]:
        p = float(np.clip(self._p_slip_rt[zid], 0.0, 1.0))
        if self._rupture_action_timer > 0:
            p = float(np.clip(p + self.cfg.rupture_action_slip_prob, 0.0, 1.0))
        if self._rng.random() >= p:
            return action, False

        # use milder corruption than previous version: mostly stay or reverse
        if self._rng.random() < 0.6:
            return self.ACTION_STAY, True
        return self._reverse_action(action), True

    def _apply_drift(self, x: int, y: int, zid: int) -> Tuple[int, int, bool]:
        p = float(np.clip(self._p_drift_rt[zid], 0.0, 1.0))
        if self._rng.random() >= p:
            return x, y, False
        dx, dy = self._drift_vec_rt[zid]
        return self._clip_xy(x + int(dx), y + int(dy))[0], self._clip_xy(x + int(dx), y + int(dy))[1], True

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

        if strength > 0.0:
            if self.cfg.use_slip:
                delta = (self._rng.random() * 2.0 - 1.0) * strength
                self._p_slip_rt[zid] = float(np.clip(self._p_slip_rt[zid] + delta, 0.0, 1.0))
            if self.cfg.use_drift:
                delta = (self._rng.random() * 2.0 - 1.0) * strength
                self._p_drift_rt[zid] = float(np.clip(self._p_drift_rt[zid] + delta, 0.0, 1.0))
        return True

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._init_zone_prototypes(seed=seed)

        self._p_slip_rt = np.array(self.cfg.p_slip, dtype=np.float32)
        self._p_drift_rt = np.array(self.cfg.p_drift, dtype=np.float32)
        self._drift_vec_rt = [tuple(v) for v in self.cfg.drift_vec]

        # Start at right edge of Z0, center row
        self.x = (self.W // 3) - 1  # for width=18 => 5
        self.y = self.H // 2
        self.t = 0

        self.fragility = float(self.cfg.fragility_init)
        self.rupture_memory = float(self.cfg.rupture_memory_init)
        self.conflict_load = float(self.cfg.conflict_load_init)
        self.reliability_estimate = 0.0
        self.mode = self.MODE_MIXED

        self.pending_ruptures = []
        self._rupture_obs_timer = 0
        self._rupture_action_timer = 0
        self._rupture_fired_this_step = False
        self._last_outcome_code = 1

        obs = self._observe()
        info = {
            "zone_id": self.zone_id(),
            "x": int(self.x),
            "y": int(self.y),
            "t": int(self.t),
            "on_encounter": bool(self._is_encounter_column(self.x)),
            "encounter_event": False,
            "encounter_type": int(self._last_outcome_code),
            "recent_reliability": float(self.reliability_estimate),
            "reliability_mode": int(self.mode),
            "fragility": float(self.fragility),
            "rupture_memory": float(self.rupture_memory),
            "conflict_load": float(self.conflict_load),
            "rupture": False,
            "pending_ruptures": 0,
            "slip": False,
            "drift": False,
            "hazard": False,
        }
        return obs, info

    def step(self, action: int):
        action = int(action)
        old_pos = (self.x, self.y)
        zid_before = self.zone_id()

        self._maybe_switch_mode()
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

        encounter_event = False
        if self._is_encounter_column(self.x):
            encounter_event = True
            outcome = self._sample_encounter_outcome()
            self._apply_encounter_outcome(outcome)

        self._update_hidden_context(self.zone_id())

        self.t += 1
        self._advance_pending_ruptures()

        if self._rupture_obs_timer > 0:
            self._rupture_obs_timer -= 1
        if self._rupture_action_timer > 0:
            self._rupture_action_timer -= 1

        moved = (self.x, self.y) != old_pos
        obs = self._observe()

        reward = 0.0
        terminated = False
        truncated = self.t >= int(self.cfg.max_steps)

        info = {
            "zone_id": int(self.zone_id()),
            "x": int(self.x),
            "y": int(self.y),
            "t": int(self.t),
            "a_in": int(action),
            "a_eff": int(a_eff),
            "moved": bool(moved),
            "slip": bool(slipped),
            "drift": bool(drifted),
            "hazard": False,
            "volatility_update": bool(volatility_event),

            "on_encounter": bool(self._is_encounter_column(self.x)),
            "encounter_event": bool(encounter_event),
            "encounter_type": int(self._last_outcome_code),  # 0 supportive, 1 neutral, 2 misleading
            "recent_reliability": float(self.reliability_estimate),
            "reliability_mode": int(self.mode),

            "fragility": float(self.fragility),
            "rupture_memory": float(self.rupture_memory),
            "conflict_load": float(self.conflict_load),

            "rupture": bool(self._rupture_fired_this_step),
            "pending_ruptures": int(len(self.pending_ruptures)),
            "rupture_obs_timer": int(self._rupture_obs_timer),
            "rupture_action_timer": int(self._rupture_action_timer),
        }

        return obs, reward, terminated, truncated, info

    # -------------------------
    # rendering
    # -------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        self._render_ascii()

    def _render_ascii(self):
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        for ex in self._encounter_columns():
            for yy in range(self.H):
                grid[yy][ex] = "E"
        grid[self.y][self.x] = "A"
        print("\n".join("".join(row) for row in grid))
        print(
            f"t={self.t} zone={self.zone_id()} pos=({self.x},{self.y}) "
            f"mode={self.mode} rel={self.reliability_estimate:.2f} "
            f"frag={self.fragility:.2f} rmem={self.rupture_memory:.2f} "
            f"cload={self.conflict_load:.2f} rup={self._rupture_fired_this_step}"
        )

    def _render_rgb(self):
        cell = 24
        img = np.zeros((self.H * cell, self.W * cell, 3), dtype=np.uint8)

        zone_colors = np.array(
            [
                [245, 215, 215],
                [215, 235, 245],
                [220, 235, 220],
            ],
            dtype=np.uint8,
        )

        for y in range(self.H):
            for x in range(self.W):
                zid = self.zone_id_of_x(x)
                y0, y1 = y * cell, (y + 1) * cell
                x0, x1 = x * cell, (x + 1) * cell
                img[y0:y1, x0:x1] = zone_colors[zid]

        for ex in self._encounter_columns():
            x0, x1 = ex * cell, (ex + 1) * cell
            img[:, x0:x1] = np.array([255, 225, 120], dtype=np.uint8)

        y0, y1 = self.y * cell, (self.y + 1) * cell
        x0, x1 = self.x * cell, (self.x + 1) * cell
        img[y0:y1, x0:x1] = np.array([0, 0, 0], dtype=np.uint8)

        img[::cell, :, :] = 0
        img[:, ::cell, :] = 0
        return img

    def close(self):
        pass


def make_env(**kwargs) -> NZonePhase2Env:
    cfg = NZonePhase2Config(**kwargs)
    return NZonePhase2Env(config=cfg)