"""Integrity-conditioned event reliability (Eqs. 6-7), the core mechanism.

    rho^r_{i,t} = (1 - omega^T delta / (omega^T 1 + eps)) [lambda_b |m^b|_1/d_b + lambda_c |m^c|_1/d_c]   (6)
    rho_{i,t}   = rho^r_{i,t} [alpha_rho + (1 - alpha_rho) sigma(w_rho^T [x~^b || x~^c || v || m] + b_rho)]   (7)

Because the bracket in Eq. (7) lies in [alpha_rho, 1], the learned term can
lower reliability but can never raise it above the rule-based bound rho^r:
a learned estimate cannot restore reliability after an explicit verification
failure. ``rho`` is the single quantity that regulates temporal attention
(Eq. 9), temporal pooling (Eq. 10), graph pooling (Eq. 14), fusion quality
(Eq. 15) and replay admission (Eq. 23).
"""
from __future__ import annotations

import torch
from torch import nn

from ..custody.policy import VERIFICATION_COMPONENTS


def rule_reliability(delta: torch.Tensor, mb: torch.Tensor | None, mc: torch.Tensor | None,
                     omega: torch.Tensor, lambda_b: float = 0.5, lambda_c: float = 0.5,
                     eps: float = 1e-6) -> torch.Tensor:
    """Eq. (6). ``delta``: [..., 5]; masks: [..., d] or ``None`` for a view absent from the track."""
    violation = 1.0 - (delta * omega).sum(-1) / (omega.sum() + eps)
    parts, weights = [], []
    if mb is not None and mb.shape[-1] > 0:
        parts.append(mb.sum(-1) / mb.shape[-1])
        weights.append(lambda_b)
    if mc is not None and mc.shape[-1] > 0:
        parts.append(mc.sum(-1) / mc.shape[-1])
        weights.append(lambda_c)
    if not parts:
        completeness = torch.ones_like(violation)
    else:
        total = sum(weights)
        completeness = sum((w / total) * p for w, p in zip(weights, parts))
    return violation * completeness


class LearnedReliability(nn.Module):
    """Eq. (7): bounded learned consistency term on top of rho^r."""

    def __init__(self, in_dim: int, alpha_rho: float = 0.6, severity: dict | None = None,
                 lambda_b: float = 0.5, lambda_c: float = 0.5, eps: float = 1e-6):
        super().__init__()
        if not 0.0 <= alpha_rho <= 1.0:
            raise ValueError("alpha_rho must lie in [0, 1]")
        self.alpha = float(alpha_rho)
        self.lin = nn.Linear(in_dim, 1)
        sev = severity or {"hash": 1.0, "time": 0.6, "auth": 0.8, "transfer": 0.8, "provenance": 0.5}
        self.register_buffer("omega", torch.tensor([float(sev[k]) for k in VERIFICATION_COMPONENTS]))
        self.lambda_b, self.lambda_c, self.eps = lambda_b, lambda_c, eps

    def forward(self, features: torch.Tensor, delta: torch.Tensor, mb: torch.Tensor | None,
                mc: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        rho_r = rule_reliability(delta, mb, mc, self.omega, self.lambda_b, self.lambda_c, self.eps)
        learned = torch.sigmoid(self.lin(features)).squeeze(-1)
        rho = rho_r * (self.alpha + (1.0 - self.alpha) * learned)
        return rho, rho_r
