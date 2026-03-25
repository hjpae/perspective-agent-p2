# cear_pilot/models/agent.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import EncoderBundle, EncoderConfig
from .world_latent import WorldLatent, WorldLatentConfig
from .state_head import StateHead, StateHeadConfig
from .policy import PolicyNet, PolicyConfig


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
        self._g_candidate: Optional[torch.Tensor] = None

    def reset(self, batch_size: int = 1) -> None:
        self._g = torch.zeros((batch_size, self.cfg.world.g_dim), device=self.device_, dtype=torch.float32)
        self._alpha = torch.zeros((batch_size, 1), device=self.device_, dtype=torch.float32)
        self._g_candidate = torch.zeros((batch_size, self.cfg.world.g_dim), device=self.device_, dtype=torch.float32)

    def get_latents(self) -> Dict[str, torch.Tensor]:
        if self._g is None:
            raise RuntimeError("Call reset() first.")
        out = {"g": self._g}
        if self._alpha is not None:
            out["alpha"] = self._alpha
        if self._g_candidate is not None:
            out["g_candidate"] = self._g_candidate
        return out

    @torch.no_grad()
    def set_g(self, g_new: torch.Tensor) -> None:
        if g_new.ndim != 2 or g_new.shape[-1] != self.cfg.world.g_dim:
            raise ValueError(f"Expected g_new shape (B, {self.cfg.world.g_dim}), got {tuple(g_new.shape)}")
        self._g = g_new.to(self.device_).detach().clone()

    def forward_step(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        ablate_g: bool = False,
        err_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        err_t:
            Optional scalar summary per batch item, shape (B,1) or (B,).
            Intended for adaptive-alpha update laws.
        """
        if self._g is None:
            self.reset(batch_size=x_t.shape[0])

        z_t, p_emb = self.enc(x_t, p_t)

        if ablate_g:
            g_t = torch.zeros_like(self._g)
            alpha_t = torch.zeros((x_t.shape[0], 1), device=x_t.device, dtype=x_t.dtype)
            g_candidate = torch.zeros_like(self._g)
        else:
            g_t, alpha_t, g_candidate = self.world(self._g, z_t, p_emb, err_t=err_t)

        s_t = self.state(z_t, p_emb, g_t)
        logits = self.policy(s_t)

        self._g = g_t.detach()
        self._alpha = alpha_t.detach()
        self._g_candidate = g_candidate.detach()

        return {
            "z": z_t,
            "p_emb": p_emb,
            "g": g_t,
            "alpha": alpha_t,
            "g_candidate": g_candidate,
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