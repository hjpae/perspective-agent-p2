# cear_pilot/models/policy.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PolicyConfig:
    s_dim: int = 16
    hidden: int = 64
    n_actions: int = 5
    dropout: float = 0.0


class PolicyNet(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.s_dim, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.n_actions),
        )

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        return self.net(s_t)

    @torch.no_grad()
    def sample_action(self, logits: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        if greedy:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)