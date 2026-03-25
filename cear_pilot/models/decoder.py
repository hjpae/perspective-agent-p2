# cear_pilot/models/decoder.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DecoderConfig:
    g_dim: int = 12
    n_actions: int = 5
    obs_dim: int = 8
    hidden: int = 64
    dropout: float = 0.0


class ObsDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.g_dim + cfg.n_actions
        self.g_ln = nn.LayerNorm(cfg.g_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.obs_dim),
        )

    def forward(self, g_t: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        g = self.g_ln(g_t)
        x = torch.cat([g, a_onehot], dim=-1)
        return self.net(x)

    def predict_all_actions(self, g_t: torch.Tensor) -> torch.Tensor:
        B = g_t.shape[0]
        A = self.cfg.n_actions
        device = g_t.device
        dtype = g_t.dtype

        g = self.g_ln(g_t)

        eye = torch.eye(A, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        g_rep = g.unsqueeze(1).repeat(1, A, 1)

        x = torch.cat([g_rep, eye], dim=-1)
        out = self.net(x.view(B * A, -1)).view(B, A, -1)
        return out