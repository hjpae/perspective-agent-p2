# cear_pilot/models/world_latent.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class WorldLatentConfig:
    z_dim: int = 16
    p_dim: int = 8
    g_dim: int = 12
    g_damping: float = 0.1       # slow update
    layernorm: bool = True


class WorldLatent(nn.Module):
    def __init__(self, cfg: WorldLatentConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRUCell(input_size=cfg.z_dim + cfg.p_dim, hidden_size=cfg.g_dim)
        self.ln = nn.LayerNorm(cfg.g_dim) if cfg.layernorm else nn.Identity()

        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, g_prev: torch.Tensor, z_t: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, p_emb], dim=-1)
        h = self.gru(x, g_prev)
        h = self.ln(h)
        d = float(self.cfg.g_damping)
        g_t = (1.0 - d) * g_prev + d * h
        return g_t
