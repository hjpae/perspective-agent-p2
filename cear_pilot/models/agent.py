# cear_pilot/models/agent.py
# -*- coding: utf-8 -*-
"""
CEAR Agent - Phase 2 with salience gating.

Key change from Phase 1:
  g_prev is passed to encoder, enabling the FiLM feedback loop:
  g_prev → salience gate → z_t → world_latent → g_t
  (g organizes perception, not just reads it)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from cear_pilot.models.encoder import EncoderBundle, EncoderConfig
from cear_pilot.models.world_latent import WorldLatent, WorldLatentConfig
from cear_pilot.models.state_head import StateHead, StateHeadConfig
from cear_pilot.models.policy import PolicyNet, PolicyConfig


@dataclass
class AgentConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    world: WorldLatentConfig = field(default_factory=WorldLatentConfig)
    state: StateHeadConfig = field(default_factory=StateHeadConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    device: str = "cpu"


class CEARAgent(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.encoder.z_dim == cfg.world.z_dim
        assert cfg.encoder.p_dim == cfg.world.p_dim
        assert cfg.world.g_dim == cfg.state.g_dim
        assert cfg.encoder.z_dim == cfg.state.z_dim
        assert cfg.encoder.p_dim == cfg.state.p_dim
        assert cfg.state.s_dim == cfg.policy.s_dim

        self.enc = EncoderBundle(cfg.encoder)
        self.world = WorldLatent(cfg.world)
        self.state = StateHead(cfg.state)
        self.policy = PolicyNet(cfg.policy)

        self.device_ = torch.device(cfg.device)
        self.to(self.device_)

        self._g: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None

    def reset(self, batch_size: int = 1) -> None:
        gd = self.cfg.world.g_dim
        self._g = torch.zeros((batch_size, gd), device=self.device_, dtype=torch.float32)
        self._alpha = torch.zeros((batch_size, 1), device=self.device_, dtype=torch.float32)

    def get_latents(self) -> Dict[str, torch.Tensor]:
        if self._g is None:
            raise RuntimeError("Call reset() first.")
        return {"g": self._g, "alpha": self._alpha}

    @torch.no_grad()
    def set_g(self, g_new: torch.Tensor) -> None:
        if g_new.ndim != 2 or g_new.shape[-1] != self.cfg.world.g_dim:
            raise ValueError(f"Expected (B, {self.cfg.world.g_dim}), got {tuple(g_new.shape)}")
        self._g = g_new.to(self.device_).detach().clone()

    def forward_step(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        ablate_g: bool = False,
        err_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self._g is None:
            self.reset(batch_size=x_t.shape[0])

        # ★ Key change: g_prev gates perception
        z_t, p_emb = self.enc(x_t, p_t, g_t=self._g)

        if ablate_g:
            g_t = torch.zeros_like(self._g)
            alpha_t = torch.zeros((x_t.shape[0], 1), device=x_t.device)
        else:
            out_world = self.world(self._g, z_t, p_emb, err_t=err_t)
            g_t = out_world["g"]
            alpha_t = out_world["alpha"]

        s_t = self.state(z_t, p_emb, g_t)
        logits = self.policy(s_t)

        self._g = g_t.detach()
        self._alpha = alpha_t.detach()

        return {
            "z": z_t,
            "p_emb": p_emb,
            "g": g_t,
            "alpha": alpha_t,
            "s": s_t,
            "logits": logits,
        }

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        greedy: bool = False,
        ablate_g: bool = False,
        err_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward_step(x_t, p_t, ablate_g=ablate_g, err_t=err_t)
        action = self.policy.sample_action(out["logits"], greedy=greedy)
        return action, out