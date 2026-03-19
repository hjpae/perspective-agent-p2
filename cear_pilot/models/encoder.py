# cear_pilot/models/encoder.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    obs_dim: int = 5
    proprio_dim: int = 5          # e.g., one-hot last action
    z_dim: int = 16               # extero latent
    p_dim: int = 8                # encoded proprio
    hidden: int = 64
    dropout: float = 0.0


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
        self.mlp = MLP(cfg.obs_dim, cfg.z_dim, cfg.hidden, cfg.dropout)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(x_t))


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

    def forward(self, x_t: torch.Tensor, p_t: Optional[torch.Tensor] = None):
        z_t = self.obs_enc(x_t)
        if p_t is None:
            B = x_t.shape[0]
            p_emb = torch.zeros((B, self.cfg.p_dim), device=x_t.device, dtype=x_t.dtype)
        else:
            p_emb = self.prop_enc(p_t)
        return z_t, p_emb
