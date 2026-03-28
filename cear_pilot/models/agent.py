# cear_pilot/models/agent.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

_THIS_FILE = Path(__file__).resolve()
for _p in (_THIS_FILE.parents[2], _THIS_FILE.parents[1], _THIS_FILE.parent):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

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
        self._g_candidate: Optional[torch.Tensor] = None
        self._g_drift: Optional[torch.Tensor] = None
        self._energy: Optional[torch.Tensor] = None
        self._grad_norm: Optional[torch.Tensor] = None
        self._basin_id: Optional[torch.Tensor] = None
        self._basin_probs: Optional[torch.Tensor] = None

    def reset(self, batch_size: int = 1) -> None:
        gd = self.cfg.world.g_dim
        self._g = torch.zeros((batch_size, gd), device=self.device_, dtype=torch.float32)
        self._alpha = torch.zeros((batch_size, 1), device=self.device_, dtype=torch.float32)
        self._g_candidate = torch.zeros((batch_size, gd), device=self.device_, dtype=torch.float32)
        self._g_drift = torch.zeros((batch_size, gd), device=self.device_, dtype=torch.float32)
        self._energy = torch.zeros((batch_size, 1), device=self.device_, dtype=torch.float32)
        self._grad_norm = torch.zeros((batch_size, 1), device=self.device_, dtype=torch.float32)
        self._basin_id = torch.zeros((batch_size,), device=self.device_, dtype=torch.long)
        self._basin_probs = torch.zeros((batch_size, self.cfg.world.n_prototypes), device=self.device_, dtype=torch.float32)

    def get_latents(self) -> Dict[str, torch.Tensor]:
        if self._g is None:
            raise RuntimeError("Call reset() first.")
        out = {
            "g": self._g,
            "alpha": self._alpha,
            "g_candidate": self._g_candidate,
            "g_drift": self._g_drift,
            "energy": self._energy,
            "grad_norm": self._grad_norm,
            "basin_id": self._basin_id,
            "basin_probs": self._basin_probs,
        }
        return {k: v for k, v in out.items() if v is not None}

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
        if self._g is None:
            self.reset(batch_size=x_t.shape[0])

        z_t, p_emb = self.enc(x_t, p_t)

        if ablate_g:
            g_t = torch.zeros_like(self._g)
            out_world = {
                "g": g_t,
                "alpha": torch.zeros((x_t.shape[0], 1), device=x_t.device, dtype=x_t.dtype),
                "g_candidate": torch.zeros_like(self._g),
                "g_drift": torch.zeros_like(self._g),
                "energy": torch.zeros((x_t.shape[0], 1), device=x_t.device, dtype=x_t.dtype),
                "grad_norm": torch.zeros((x_t.shape[0], 1), device=x_t.device, dtype=x_t.dtype),
                "basin_id": torch.zeros((x_t.shape[0],), device=x_t.device, dtype=torch.long),
                "basin_probs": torch.zeros((x_t.shape[0], self.cfg.world.n_prototypes), device=x_t.device, dtype=x_t.dtype),
                "prototype_centers": self.world.prototype_centers,
            }
        else:
            out_world = self.world(self._g, z_t, p_emb, err_t=err_t)
            g_t = out_world["g"]

        s_t = self.state(z_t, p_emb, g_t)
        logits = self.policy(s_t)

        self._g = out_world["g"].detach()
        self._alpha = out_world["alpha"].detach()
        self._g_candidate = out_world["g_candidate"].detach()
        self._g_drift = out_world["g_drift"].detach()
        self._energy = out_world["energy"].detach()
        self._grad_norm = out_world["grad_norm"].detach()
        self._basin_id = out_world["basin_id"].detach()
        self._basin_probs = out_world["basin_probs"].detach()

        out = {
            "z": z_t,
            "p_emb": p_emb,
            "s": s_t,
            "logits": logits,
        }
        out.update(out_world)
        return out

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
            self._g = self._g + torch.randn((B, D), device=self._g.device) * float(scale)
        elif kind == "swap":
            proto = self.world.prototype_centers.detach()
            if proto.shape[0] >= 2:
                self._g = proto[torch.randint(low=0, high=proto.shape[0], size=(B,), device=self._g.device)].clone()
            else:
                v = torch.randn((B, D), device=self._g.device)
                self._g = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-9)
        elif kind == "zero":
            self._g = torch.zeros_like(self._g)
        elif kind == "flip":
            self._g = -self._g * float(scale)
        else:
            raise ValueError(f"Unknown perturbation kind: {kind}")
