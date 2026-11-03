"""Calibrated dual-head inference (Eq. 18).

    p^m_i = softmax(W_m z^m_i / T_m),   p^c_i = softmax(W_c z^c_i / T_c)

The temperatures are estimated after training on the development partition
only (:mod:`cocguard.openworld.calibration`); during training they are 1.
Malware and custody heads have separate parameters, temperatures and
decision criteria, so a custody failure never changes the malware label.
"""
from __future__ import annotations

import torch
from torch import nn


class CalibratedHead(nn.Module):
    def __init__(self, d: int, n_classes: int):
        super().__init__()
        self.lin = nn.Linear(d, n_classes)
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Uncalibrated logits W z."""
        return self.lin(z)

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature.clamp_min(1e-3), dim=-1)

    def set_temperature(self, value: float) -> None:
        self.temperature.fill_(float(value))
