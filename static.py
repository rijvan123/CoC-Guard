"""CoC-Guard static-evidence configuration for BODMAS and EMBER 2018.

On structured PE tracks the temporal event learning, graph propagation,
custody assessment and temporal-graph fusion are inactive (Section IV-C). The
applicable configuration consists of a static encoder over the attribute
vector and its availability mask, the calibrated malware head (Eq. 18) and the
dataset-supported screening components (Eq. 19 with the cross-view term masked
and the remaining coefficients renormalised). Investigation-level reliability
for replay admission is the availability-aware rule reliability of Eq. (6).
"""
from __future__ import annotations

import torch
from torch import nn

from .heads import CalibratedHead


class StaticCoCGuard(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=(512, 256), d: int = 128, dropout: float = 0.2,
                 use_mask: bool = True):
        super().__init__()
        self.use_mask = use_mask
        dims = [in_dim * (2 if use_mask else 1)] + list(hidden)
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], d), nn.LayerNorm(d)]
        self.encoder = nn.Sequential(*layers)
        self.head_m = CalibratedHead(d, n_classes)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, m: torch.Tensor | None = None) -> dict:
        if self.use_mask:
            m = torch.ones_like(x) if m is None else m
            inp = torch.cat([x * m, m], dim=-1)
        else:
            inp = x
        z = self.encoder(inp)
        logits = self.head_m(z)
        rho = (m.float().mean(dim=-1) if m is not None else torch.ones(x.shape[0], device=x.device))
        return {"z_m": z, "logits_m": logits, "probs_m": self.head_m.probs(logits), "rho_bar": rho}
