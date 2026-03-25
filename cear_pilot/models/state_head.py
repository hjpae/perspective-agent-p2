# cear_pilot/models/state_head.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StateHeadConfig:
    z_dim: int = 16
    p_dim: int = 8
    g_dim: int = 12
    s_dim: int = 16
    hidden: int = 64
    dropout: float = 0.0
    g_influence: float = 1.0


class StateHead(nn.Module):
    def __init__(self, cfg: StateHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.z_dim + cfg.p_dim + cfg.g_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.s_dim),
        )
        self.ln = nn.LayerNorm(cfg.s_dim)

    def forward(self, z_t: torch.Tensor, p_emb: torch.Tensor, g_t: torch.Tensor) -> torch.Tensor:
        g_scaled = g_t * float(self.cfg.g_influence)
        x = torch.cat([z_t, p_emb, g_scaled], dim=-1)
        s = torch.tanh(self.net(x))
        s = self.ln(s)
        return s