# cear_pilot/models/encoder.py
# -*- coding: utf-8 -*-
"""
Observation encoder with g-conditioned salience gating (FiLM).

Key addition: ObservationEncoder.forward(x_t, g_t=None)
  - When g_t is None (Phase 1 compat): z_t = tanh(MLP(x_t))
  - When g_t is provided: z_raw = tanh(MLP(x_t)), then FiLM modulates z_raw.
    z_t = (1 + gamma) * z_raw + beta
    where (gamma, beta) = split(Linear(g_t))

(1 + gamma) initialization: gamma starts near 0, so gating starts as identity.
Phase 1 encoder weights load cleanly — FiLM layer is the only new parameter.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    obs_dim: int = 8
    proprio_dim: int = 5
    z_dim: int = 16
    p_dim: int = 8
    g_dim: int = 12      # needed for FiLM layer
    hidden: int = 64
    dropout: float = 0.0
    use_salience_gate: bool = True


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = MLP(cfg.obs_dim, cfg.z_dim, cfg.hidden, cfg.dropout)

        # FiLM: g → (gamma, beta) for salience gating
        if cfg.use_salience_gate:
            self.film = nn.Linear(cfg.g_dim, cfg.z_dim * 2)
            # Init so gamma≈0, beta≈0 → gating starts as identity
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x_t: torch.Tensor, g_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_raw = torch.tanh(self.mlp(x_t))

        if g_t is not None and self.cfg.use_salience_gate and hasattr(self, "film"):
            gamma, beta = self.film(g_t).chunk(2, dim=-1)
            z_t = (1.0 + gamma) * z_raw + beta
        else:
            z_t = z_raw

        return z_t


class ProprioEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.mlp = MLP(cfg.proprio_dim, cfg.p_dim, cfg.hidden, cfg.dropout)

    def forward(self, p_t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(p_t))


class EncoderBundle(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_enc = ObservationEncoder(cfg)
        self.prop_enc = ProprioEncoder(cfg)

    def forward(
        self,
        x_t: torch.Tensor,
        p_t: Optional[torch.Tensor] = None,
        g_t: Optional[torch.Tensor] = None,
    ):
        z_t = self.obs_enc(x_t, g_t=g_t)
        if p_t is None:
            B = x_t.shape[0]
            p_emb = torch.zeros((B, self.cfg.p_dim), device=x_t.device, dtype=x_t.dtype)
        else:
            p_emb = self.prop_enc(p_t)
        return z_t, p_emb