# cear_pilot/envs/nzone_phase1.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass

from cear_pilot.envs.nzone_grid import NZoneGridEnv, NZoneConfig


@dataclass
class NZonePhase1Config(NZoneConfig):
    """
    Phase 1:
    - AAAI-like orientation shaping
    - 3 zones, 5 columns each => width=15
    - start at center of Z1 (inherited default reset from NZoneGridEnv)
    - no encounter / no ecology complications
    """
    width: int = 15
    height: int = 9
    obs_dim: int = 8
    max_steps: int = 240

    zone_sigma: tuple[float, float, float] = (0.60, 0.30, 0.05)
    zone_mu_scale: float = 0.5
    include_xy: bool = False

    use_slip: bool = False
    use_drift: bool = False
    use_volatility: bool = False
    use_hazard: bool = False
    use_encounter: bool = False

    phase2_obs_mu_scale: float = 0.5
    phase2_obs_equal_sigma: bool = False


class NZonePhase1Env(NZoneGridEnv):
    def __init__(self, config: NZonePhase1Config | None = None, render_mode: str | None = None):
        super().__init__(config=config or NZonePhase1Config(), render_mode=render_mode)


def make_env(**kwargs) -> NZonePhase1Env:
    cfg = NZonePhase1Config(**kwargs)
    return NZonePhase1Env(config=cfg)