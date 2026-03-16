# cear_pilot/models/agent.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import EncoderBundle, EncoderConfig
from .world_latent import WorldLatent, WorldLatentConfig
from .state_head import StateHead, StateHeadConfig
from .policy import PolicyNet, PolicyConfig


@dataclass
class AgentConfig:
    encoder: EncoderConfig = EncoderConfig()
    world: WorldLatentConfig = WorldLatentConfig()
    state: StateHeadConfig = StateHeadConfig()
    policy: PolicyConfig = PolicyConfig()
    device: str = "cpu"


class CEARAgent(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg

        # Dim consistency
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

        self._g: Optional[torch.Tensor] = None  # (B, g_dim)

    def reset(self, batch_size: int = 1) -> None:
        self._g = torch.zeros((batch_size, self.cfg.world.g_dim), device=self.device_, dtype=torch.float32)

    def get_latents(self) -> Dict[str, torch.Tensor]:
        if self._g is None:
            raise RuntimeError("Call reset() first.")
        return {"g": self._g}

    def forward_step(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        ablate_g: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict: z, p_emb, g, s, logits
        """
        if self._g is None:
            self.reset(batch_size=x_t.shape[0])

        z_t, p_emb = self.enc(x_t, p_t)

        if ablate_g:
            g_t = torch.zeros_like(self._g)
        else:
            g_t = self.world(self._g, z_t, p_emb)

        # fast manifold shaped by g
        s_t = self.state(z_t, p_emb, g_t)

        # policy uses ONLY s (no direct g input)
        logits = self.policy(s_t)

        # update internal g
        self._g = g_t.detach()

        return {"z": z_t, "p_emb": p_emb, "g": g_t, "s": s_t, "logits": logits}

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        greedy: bool = False,
        ablate_g: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward_step(x_t, p_t, ablate_g=ablate_g)
        action = self.policy.sample_action(out["logits"], greedy=greedy)
        return action, out

    @torch.no_grad()
    def apply_perturbation(self, kind: str = "shock", scale: float = 1.0) -> None:
        if self._g is None:
            raise RuntimeError("Call reset() first.")
        B, D = self._g.shape
        if kind == "shock":
            self._g = torch.randn((B, D), device=self._g.device) * float(scale)
        elif kind == "swap":
            v = torch.randn((B, D), device=self._g.device)
            self._g = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-9)
        elif kind == "zero":
            self._g = torch.zeros_like(self._g)
        else:
            raise ValueError(f"Unknown perturbation kind: {kind}")
