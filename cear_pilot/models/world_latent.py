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

    # legacy fixed update
    g_damping: float = 0.10
    layernorm: bool = True

    # new update-law controls
    update_mode: str = "fixed"   # "fixed" or "adaptive"

    # fixed mode
    alpha_fixed: float = 0.10

    # adaptive mode
    alpha_min: float = 0.03
    alpha_max: float = 0.30
    alpha_hidden: int = 32

    # if True, alpha depends partly on prediction error summary supplied by caller
    use_error_feedback: bool = True


class WorldLatent(nn.Module):
    """
    Slow perspective latent g.

    Backward compatibility:
    - If update_mode == "fixed", behaves like the original damped update.
    - If update_mode == "adaptive", alpha_t is inferred from current latent input
      (and optional error summary), then used as:
          g_t = (1 - alpha_t) * g_prev + alpha_t * h_t

    forward(...) returns:
        g_t, alpha_t, h_t
    """

    def __init__(self, cfg: WorldLatentConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRUCell(input_size=cfg.z_dim + cfg.p_dim, hidden_size=cfg.g_dim)
        self.ln = nn.LayerNorm(cfg.g_dim) if cfg.layernorm else nn.Identity()

        alpha_in_dim = cfg.z_dim + cfg.p_dim + cfg.g_dim + (1 if cfg.use_error_feedback else 0)
        self.alpha_net = nn.Sequential(
            nn.Linear(alpha_in_dim, cfg.alpha_hidden),
            nn.Tanh(),
            nn.Linear(cfg.alpha_hidden, 1),
        )

        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _compute_alpha(
        self,
        g_prev: torch.Tensor,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        err_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns alpha_t with shape (B, 1).
        """
        mode = str(self.cfg.update_mode).lower().strip()

        if mode == "fixed":
            alpha = torch.full(
                (g_prev.shape[0], 1),
                float(self.cfg.alpha_fixed if self.cfg.alpha_fixed is not None else self.cfg.g_damping),
                dtype=g_prev.dtype,
                device=g_prev.device,
            )
            return alpha

        if mode != "adaptive":
            raise ValueError(f"Unknown update_mode: {self.cfg.update_mode}")

        parts = [z_t, p_emb, g_prev]
        if self.cfg.use_error_feedback:
            if err_t is None:
                err_t = torch.zeros((g_prev.shape[0], 1), dtype=g_prev.dtype, device=g_prev.device)
            elif err_t.ndim == 1:
                err_t = err_t.unsqueeze(-1)
            parts.append(err_t)

        a_in = torch.cat(parts, dim=-1)
        raw = self.alpha_net(a_in)  # (B,1)
        gate = torch.sigmoid(raw)

        a_min = float(self.cfg.alpha_min)
        a_max = float(self.cfg.alpha_max)
        alpha = a_min + (a_max - a_min) * gate
        return alpha

    def forward(
        self,
        g_prev: torch.Tensor,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        err_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            g_prev: (B, g_dim)
            z_t:    (B, z_dim)
            p_emb:  (B, p_dim)
            err_t:  optional (B,1) error / conflict summary

        Returns:
            g_t:       (B, g_dim)
            alpha_t:   (B, 1)
            h_t:       (B, g_dim) candidate updated state before slow mixing
        """
        x = torch.cat([z_t, p_emb], dim=-1)
        h = self.gru(x, g_prev)
        h = self.ln(h)

        alpha_t = self._compute_alpha(g_prev=g_prev, z_t=z_t, p_emb=p_emb, err_t=err_t)
        g_t = (1.0 - alpha_t) * g_prev + alpha_t * h
        return g_t, alpha_t, h