# cear_pilot/models/world_latent.py
# -*- coding: utf-8 -*-
"""
Perspective latent with self-modulating plasticity.

Key property: g_prev determines its own update rate (alpha).
This is NOT standard RNN/POMDP belief update where input drives the gate.
Here, the current stance decides how much new evidence can change it.

g_t = (1 - alpha_t) * g_prev + alpha_t * GRU(z_t, p_emb)
alpha_t = sigmoid(alpha_net(z_t, p_emb, g_prev, err_t))
                                        ^^^^^^^
                              self-modulating: g controls its own plasticity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class WorldLatentConfig:
    z_dim: int = 16
    p_dim: int = 8
    g_dim: int = 12

    layernorm: bool = True
    update_mode: str = "adaptive"  # "fixed" or "adaptive"

    alpha_fixed: float = 0.10
    alpha_min: float = 0.03
    alpha_max: float = 0.30
    alpha_hidden: int = 32

    use_error_feedback: bool = True
    err_dim: int = 6  # simplified: PE features + perturbation signal

    g_damping: float = 0.10  # legacy fallback


class WorldLatent(nn.Module):
    def __init__(self, cfg: WorldLatentConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRUCell(
            input_size=cfg.z_dim + cfg.p_dim,
            hidden_size=cfg.g_dim,
        )
        self.ln = nn.LayerNorm(cfg.g_dim) if cfg.layernorm else nn.Identity()

        # Alpha net input: z_t + p_emb + g_prev [+ err_t]
        alpha_in_dim = cfg.z_dim + cfg.p_dim + cfg.g_dim
        if cfg.use_error_feedback:
            alpha_in_dim += int(cfg.err_dim)

        self.alpha_net = nn.Sequential(
            nn.Linear(alpha_in_dim, cfg.alpha_hidden),
            nn.Tanh(),
            nn.Linear(cfg.alpha_hidden, 1),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Start alpha near the low end so g is initially stable
        last_linear = self.alpha_net[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.constant_(last_linear.bias, -0.75)

    def _compute_alpha(
        self,
        g_prev: torch.Tensor,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        err_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = str(self.cfg.update_mode).lower().strip()

        if mode == "fixed":
            a = float(self.cfg.alpha_fixed)
            return torch.full(
                (g_prev.shape[0], 1), a,
                dtype=g_prev.dtype, device=g_prev.device,
            )

        # adaptive: g_prev is an input → self-modulating
        parts = [z_t, p_emb, g_prev]

        if self.cfg.use_error_feedback and err_t is not None:
            err_t = err_t.to(device=g_prev.device, dtype=g_prev.dtype)
            if err_t.ndim == 1:
                err_t = err_t.unsqueeze(0)
            if err_t.shape[0] == 1 and g_prev.shape[0] > 1:
                err_t = err_t.expand(g_prev.shape[0], -1)
            parts.append(err_t)

        raw = self.alpha_net(torch.cat(parts, dim=-1))

        center = 0.5 * (float(self.cfg.alpha_min) + float(self.cfg.alpha_max))
        half = 0.5 * (float(self.cfg.alpha_max) - float(self.cfg.alpha_min))
        alpha = center + half * torch.tanh(raw)
        return alpha.clamp(min=float(self.cfg.alpha_min), max=float(self.cfg.alpha_max))

    def forward(
        self,
        g_prev: torch.Tensor,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        err_t: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([z_t, p_emb], dim=-1)
        h_t = self.ln(self.gru(x, g_prev))

        alpha_t = self._compute_alpha(g_prev, z_t, p_emb, err_t)

        # Core update: EMA with self-modulating rate
        g_t = (1.0 - alpha_t) * g_prev + alpha_t * h_t

        return {
            "g": g_t,
            "alpha": alpha_t,
        }