"""Reliability-Controlled Fusion (Eqs. 15-17) and the six comparators of Table IX.

    q^k_i = [a^k_i, 1 - u^k_i, rho-bar^k_i, 1 - d^TG_i]^T,  k in {T, G}                               (15)
    s^{y,k}_i = w_{y,k}^T [q^T || q^G] + b_{y,k},
    g^{y,k}_i = a^k exp(s^{y,k}) / (sum_j a^j exp(s^{y,j}) + eps),   y in {m, c}                        (16)
    z^m_i = g^{m,T} W^m_T h^T + g^{m,G} W^m_G h^G,   z^c_i = g^{c,T} W^c_T h^T + g^{c,G} W^c_G h^G      (17)

Fusion modes:

``rcf``           complete Reliability-Controlled Fusion (all four signals, availability masking)
``concat``        direct concatenation without RCF: z^y = W^y [h^T || h^G]
``equal``         fixed 0.5 / 0.5 weighting
``attention``     conventional content attention over the two branch representations
``uncertainty``   gates driven only by 1 - u^k
``availability``  gates driven only by a^k (with masking)
``integrity``     gates driven only by rho-bar^k

The quality vectors are evidence descriptors and enter the gates with a
stop-gradient, so the gates cannot be optimised by manipulating the auxiliary
predictions that define uncertainty and disagreement.
"""
from __future__ import annotations

import torch
from torch import nn

FUSION_MODES = ("rcf", "concat", "equal", "attention", "uncertainty", "availability", "integrity")


class ReliabilityControlledFusion(nn.Module):
    def __init__(self, d_T: int, d_G: int, d: int = 128, mode: str = "rcf", use_disagreement: bool = True,
                 tasks: tuple = ("m", "c"), eps: float = 1e-6):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion mode {mode!r}; choose from {FUSION_MODES}")
        self.mode, self.use_disagreement, self.tasks, self.eps = mode, use_disagreement, tuple(tasks), eps
        self.q_dim = {"rcf": 4 if use_disagreement else 3, "uncertainty": 1, "availability": 1, "integrity": 1}.get(mode, 0)
        self.proj = nn.ModuleDict()
        self.gate = nn.ModuleDict()
        self.att = nn.ModuleDict()
        for y in self.tasks:
            if mode == "concat":
                self.proj[y] = nn.Linear(d_T + d_G, d)
            else:
                self.proj[y + "T"] = nn.Linear(d_T, d)
                self.proj[y + "G"] = nn.Linear(d_G, d)
            if self.q_dim:
                self.gate[y] = nn.Linear(2 * self.q_dim, 2)       # [w_{y,T}; w_{y,G}], b_{y,k}
            if mode == "attention":
                self.att[y] = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1, bias=False))

    def _quality(self, a, u, rho_bar, d_tg) -> torch.Tensor:
        if self.mode == "rcf":
            parts = [a, 1.0 - u, rho_bar] + ([1.0 - d_tg] if self.use_disagreement else [])
        elif self.mode == "uncertainty":
            parts = [1.0 - u]
        elif self.mode == "availability":
            parts = [a]
        else:  # integrity
            parts = [rho_bar]
        return torch.stack(parts, dim=-1)

    def forward(self, h_T: torch.Tensor, h_G: torch.Tensor, a_T: torch.Tensor, a_G: torch.Tensor,
                u_T: torch.Tensor, u_G: torch.Tensor, rho_T: torch.Tensor, rho_G: torch.Tensor,
                d_tg: torch.Tensor) -> dict:
        B = h_T.shape[0]
        out: dict = {}
        q_T = q_G = None
        if self.q_dim:
            q_T = self._quality(a_T, u_T, rho_T, d_tg).detach()
            q_G = self._quality(a_G, u_G, rho_G, d_tg).detach()
            out["q_T"], out["q_G"] = q_T, q_G
        avail = torch.stack([a_T, a_G], dim=-1)
        for y in self.tasks:
            if self.mode == "concat":
                out["z_" + y] = self.proj[y](torch.cat([h_T, h_G], dim=-1))
                out["g_" + y] = torch.full((B, 2), float("nan"), device=h_T.device)
                continue
            zT, zG = self.proj[y + "T"](h_T), self.proj[y + "G"](h_G)
            if self.mode == "equal":
                g = torch.full((B, 2), 0.5, device=h_T.device)
            elif self.mode == "attention":
                s = torch.cat([self.att[y](zT), self.att[y](zG)], dim=-1)
                g = torch.softmax(s, dim=-1)
            else:
                s = self.gate[y](torch.cat([q_T, q_G], dim=-1))
                s = s - s.max(dim=-1, keepdim=True).values
                ex = torch.exp(s)
                if self.mode in ("rcf", "availability"):
                    ex = ex * avail                                   # Eq. (16): unavailable branch -> 0
                g = ex / (ex.sum(dim=-1, keepdim=True) + self.eps)
            out["g_" + y] = g
            out["z_" + y] = g[:, :1] * zT + g[:, 1:] * zG
        return out
