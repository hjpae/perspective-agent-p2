# cear_pilot/models/world_latent.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class WorldLatentConfig:
    z_dim: int = 16
    p_dim: int = 8
    g_dim: int = 12

    layernorm: bool = True
    update_mode: str = "fixed"  # fixed or adaptive

    alpha_fixed: float = 0.10
    alpha_min: float = 0.03
    alpha_max: float = 0.30
    alpha_hidden: int = 32

    # always used in phase2
    use_error_feedback: bool = True
    err_dim: int = 10

    g_damping: float = 0.10  # legacy compatibility

    energy_mode: str = "prototype_wells"  # prototype_wells or none
    dyn_eta: float = 0.10
    confine_lambda: float = 0.01
    n_prototypes: int = 2
    well_depth: float = 0.60
    well_width: float = 1.25
    learnable_prototypes: bool = True
    learnable_well_depth: bool = True
    learnable_well_width: bool = False
    prototype_init_scale: float = 0.35
    temperature: float = 1.0


class WorldLatent(nn.Module):
    """
    Slow world / perspective latent with basin-shaped energy dynamics.

    err_t is now a vector-valued evidence package, not a single scalar.
    The trainer packages evidence; alpha_net learns how to combine it.
    """

    def __init__(self, cfg: WorldLatentConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRUCell(input_size=cfg.z_dim + cfg.p_dim, hidden_size=cfg.g_dim)
        self.ln = nn.LayerNorm(cfg.g_dim) if cfg.layernorm else nn.Identity()

        alpha_in_dim = cfg.z_dim + cfg.p_dim + cfg.g_dim
        if cfg.use_error_feedback:
            alpha_in_dim += int(cfg.err_dim)

        self.alpha_net = nn.Sequential(
            nn.Linear(alpha_in_dim, cfg.alpha_hidden),
            nn.Tanh(),
            nn.Linear(cfg.alpha_hidden, 1),
        )

        proto = torch.randn(cfg.n_prototypes, cfg.g_dim) * float(cfg.prototype_init_scale)
        if cfg.learnable_prototypes:
            self.prototype_centers = nn.Parameter(proto)
        else:
            self.register_buffer("prototype_centers", proto)

        raw_depth = torch.tensor(float(cfg.well_depth)).log().view(1)
        if cfg.learnable_well_depth:
            self.raw_well_depth = nn.Parameter(raw_depth)
        else:
            self.register_buffer("raw_well_depth", raw_depth)

        raw_width = torch.tensor(float(cfg.well_width)).log().repeat(cfg.n_prototypes)
        if cfg.learnable_well_width:
            self.raw_well_width = nn.Parameter(raw_width)
        else:
            self.register_buffer("raw_well_width", raw_width)

        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p is self.prototype_centers:
                continue
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Start adaptive alpha slightly below the center so it does not immediately saturate high before learning context structure.
        last_linear = self.alpha_net[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.constant_(last_linear.bias, -0.75)

    @property
    def well_depth(self) -> torch.Tensor:
        return torch.exp(self.raw_well_depth).clamp(min=1e-6)

    @property
    def well_width(self) -> torch.Tensor:
        return torch.exp(self.raw_well_width).clamp(min=1e-4)

    def _prepare_err_t(
        self,
        err_t: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Expected err_t layout (B, err_dim):
            0: pred_err_now_ratio
            1: pred_err_ema_short_ratio
            2: pred_err_ema_long_log
            3: pred_err_rise_ratio (signed)
            4: recent_event_trace
            5: recent_consequence_trace
            6: current_event_flag
            7: c_state_signed
            8: supportive_flag
            9: misleading_flag
        """
        if err_t is None:
            return torch.zeros((batch_size, int(self.cfg.err_dim)), device=device, dtype=dtype)

        err_t = err_t.to(device=device, dtype=dtype)

        if err_t.ndim == 1:
            err_t = err_t.unsqueeze(0)

        if err_t.shape[0] != batch_size:
            if err_t.shape[0] == 1:
                err_t = err_t.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"err_t batch mismatch: expected {batch_size}, got {tuple(err_t.shape)}"
                )

        if err_t.shape[-1] != int(self.cfg.err_dim):
            raise ValueError(
                f"err_t feature dim mismatch: expected {self.cfg.err_dim}, got {err_t.shape[-1]}"
            )

        return err_t

    def _normalize_err_t(self, err_t: torch.Tensor) -> torch.Tensor:
        x = err_t.clone()
        out = torch.zeros_like(x)

        # 0: pred_err_now_ratio
        # 1: pred_err_ema_short_ratio
        # 2: pred_err_ema_long_log
        # 3: pred_err_rise_ratio (signed)
        # 4: recent_event_trace
        # 5: recent_consequence_trace
        # 6: current_event_flag
        # 7: c_state_signed
        # 8: supportive_flag
        # 9: misleading_flag

        # Ratios can easily blow up or stay persistently positive.
        # Compress them aggressively.
        out[:, 0:1] = torch.tanh(torch.log1p(torch.relu(x[:, 0:1])) / 2.5)
        out[:, 1:2] = torch.tanh(torch.log1p(torch.relu(x[:, 1:2])) / 2.5)
        out[:, 2:3] = torch.tanh(torch.relu(x[:, 2:3]) / 2.0)

        # Signed trend should stay signed and bounded.
        out[:, 3:4] = torch.tanh(x[:, 3:4] / 2.0)

        # Traces / flags / valence channels are already bounded-ish; keep them gentle.
        out[:, 4:5] = torch.tanh(torch.relu(x[:, 4:5]))
        out[:, 5:6] = torch.tanh(torch.relu(x[:, 5:6]))
        out[:, 6:7] = torch.clamp(x[:, 6:7], 0.0, 1.0)

        # Signed consequence load
        out[:, 7:8] = torch.tanh(x[:, 7:8] / 1.5)

        # Supportive / misleading magnitudes
        out[:, 8:9] = torch.tanh(torch.relu(x[:, 8:9]))
        out[:, 9:10] = torch.tanh(torch.relu(x[:, 9:10]))

        return out

    def _compute_alpha(
        self,
        g_prev: torch.Tensor,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        err_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = str(self.cfg.update_mode).lower().strip()
        if mode == "fixed":
            a = float(self.cfg.alpha_fixed if self.cfg.alpha_fixed is not None else self.cfg.g_damping)
            return torch.full((g_prev.shape[0], 1), a, dtype=g_prev.dtype, device=g_prev.device)

        if mode != "adaptive":
            raise ValueError(f"Unknown update_mode: {self.cfg.update_mode}")

        parts = [z_t, p_emb, g_prev]

        if self.cfg.use_error_feedback:
            err_t = self._prepare_err_t(
                err_t=err_t,
                batch_size=g_prev.shape[0],
                device=g_prev.device,
                dtype=g_prev.dtype,
            )
            err_t = self._normalize_err_t(err_t)
            parts.append(err_t)

        a_in = torch.cat(parts, dim=-1)
        raw = self.alpha_net(a_in)

        # Centered parameterization: alpha moves around a midpoint instead of only pushing toward one bound.
        alpha_center = 0.5 * (float(self.cfg.alpha_min) + float(self.cfg.alpha_max))
        alpha_half_range = 0.5 * (float(self.cfg.alpha_max) - float(self.cfg.alpha_min))

        alpha = alpha_center + alpha_half_range * torch.tanh(raw)
        alpha = torch.clamp(alpha, min=float(self.cfg.alpha_min), max=float(self.cfg.alpha_max))
        return alpha

    def _energy_terms(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if str(self.cfg.energy_mode).lower().strip() == "none":
            B = g.shape[0]
            zeros_b = torch.zeros((B,), dtype=g.dtype, device=g.device)
            zeros_bg = torch.zeros_like(g)
            zeros_k = torch.zeros((B, self.cfg.n_prototypes), dtype=g.dtype, device=g.device)
            return zeros_b, zeros_bg, zeros_k, zeros_k

        proto = self.prototype_centers.unsqueeze(0)
        diff = g.unsqueeze(1) - proto
        widths = self.well_width.view(1, -1, 1)
        sqdist = torch.sum(diff * diff, dim=-1)
        logits = -sqdist / (2.0 * (widths.squeeze(-1) ** 2) * float(self.cfg.temperature))
        probs = torch.softmax(logits, dim=-1)

        depth = self.well_depth.view(1)
        confine = 0.5 * float(self.cfg.confine_lambda) * torch.sum(g * g, dim=-1)
        well = -depth * float(self.cfg.temperature) * torch.logsumexp(logits, dim=-1)
        energy = confine + well

        grad_confine = float(self.cfg.confine_lambda) * g
        grad_well = depth * torch.sum(probs.unsqueeze(-1) * (diff / (widths ** 2)), dim=1)
        grad = grad_confine + grad_well
        return energy, grad, probs, logits

    def forward(
        self,
        g_prev: torch.Tensor,
        z_t: torch.Tensor,
        p_emb: torch.Tensor,
        err_t: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([z_t, p_emb], dim=-1)
        h_t = self.ln(self.gru(x, g_prev))
        alpha_t = self._compute_alpha(g_prev=g_prev, z_t=z_t, p_emb=p_emb, err_t=err_t)

        energy_prev, grad_prev, basin_probs, basin_logits = self._energy_terms(g_prev)

        g_drift = g_prev - float(self.cfg.dyn_eta) * grad_prev
        g_t = g_drift + alpha_t * (h_t - g_prev)

        energy_t, grad_t, basin_probs_t, basin_logits_t = self._energy_terms(g_t)
        basin_id = torch.argmax(basin_probs_t, dim=-1)
        grad_norm = torch.norm(grad_t, dim=-1, keepdim=True)

        return {
            "g": g_t,
            "alpha": alpha_t,
            "g_candidate": h_t,
            "g_drift": g_drift,
            "energy_prev": energy_prev.unsqueeze(-1),
            "energy": energy_t.unsqueeze(-1),
            "grad_prev": grad_prev,
            "grad": grad_t,
            "grad_norm": grad_norm,
            "basin_probs_prev": basin_probs,
            "basin_logits_prev": basin_logits,
            "basin_probs": basin_probs_t,
            "basin_logits": basin_logits_t,
            "basin_id": basin_id,
            "prototype_centers": self.prototype_centers,
            "well_depth": self.well_depth.view(1, 1).expand(g_t.shape[0], 1),
            "well_width_mean": self.well_width.mean().view(1, 1).expand(g_t.shape[0], 1),
        }