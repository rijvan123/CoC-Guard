"""Transaction-Consistency Learning (Eqs. 8-10).

    u^0_{i,t} = W_u [x~^b || x~^c || v || m] + p_t + phi(dtau_{i,t})                                    (8)
    alpha^{l,r}_{i,tj} = softmax_j[ (W_Q u_t)^T (W_K u_j) / sqrt(d_h) + b^tau_tj + b^c_tj + log(rho_j + eps) ]  (9)
    h^T_i = sum_t pi^T_t u^{L_T}_t,   pi^T_t = exp(w_T^T u_t) rho_t / (sum_j exp(w_T^T u_j) rho_j + eps)   (10)

``b^tau`` is a learned per-head bias over bucketised, signed elapsed time
between two events; ``b^c`` is a learned per-head custody-compatibility bias
over evidence identity, recorded predecessor links (both directions), actor
and device continuity, chronological feasibility and operation pairs. A
low-reliability source event is down-weighted by ``log(rho_j + eps)`` but is
never removed, so it stays available as an analysis target.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ElapsedTimeEncoding(nn.Module):
    """phi(dtau): Time2Vec on the signed log elapsed time plus an availability flag."""

    def __init__(self, d: int, k: int = 16):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.per = nn.Linear(1, k - 1)
        self.out = nn.Linear(k + 2, d)

    def forward(self, dtau: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        x = (torch.sign(dtau) * torch.log1p(dtau.abs() / 1.0)).unsqueeze(-1) * avail.unsqueeze(-1)
        feats = torch.cat([self.lin(x), torch.sin(self.per(x)), torch.sign(dtau).unsqueeze(-1) * avail.unsqueeze(-1),
                           avail.unsqueeze(-1)], dim=-1)
        return self.out(feats)


def sinusoidal_positions(T: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class PairwiseBiases(nn.Module):
    """b^tau_tj and b^c_tj of Eq. (9), shared by all temporal layers."""

    N_CUSTODY_FEATURES = 7

    def __init__(self, heads: int, n_ops: int, time_buckets: int = 16):
        super().__init__()
        self.heads = heads
        self.time_buckets = time_buckets
        self.time_table = nn.Embedding(2 * time_buckets + 1, heads)
        self.custody_w = nn.Parameter(torch.zeros(self.N_CUSTODY_FEATURES, heads))
        self.n_ops = n_ops + 1
        self.op_table = nn.Embedding(self.n_ops * self.n_ops, heads)
        nn.init.zeros_(self.time_table.weight)
        nn.init.zeros_(self.op_table.weight)

    def forward(self, dtau: torch.Tensor, tau_avail: torch.Tensor, pair: torch.Tensor,
                prev: torch.Tensor) -> torch.Tensor:
        B, T = dtau.shape
        # absolute (relative-to-start) times from elapsed times
        tau = torch.cumsum(dtau * tau_avail, dim=1)
        diff = tau.unsqueeze(2) - tau.unsqueeze(1)               # tau_t - tau_j  [B,T,T]
        mag = torch.log1p(diff.abs()) / math.log1p(3.0e7) * (self.time_buckets - 1)
        bucket = mag.clamp(0, self.time_buckets - 1).long()
        bucket = torch.where(diff >= 0, bucket, bucket + self.time_buckets)
        both = (tau_avail.unsqueeze(2) * tau_avail.unsqueeze(1)) > 0
        bucket = torch.where(both, bucket, torch.full_like(bucket, 2 * self.time_buckets))
        b_tau = self.time_table(bucket)                         # [B,T,T,H]

        idx = torch.arange(T, device=dtau.device)
        def same(col):
            a = pair[:, :, col]
            return ((a.unsqueeze(2) == a.unsqueeze(1)) & (a.unsqueeze(2) >= 0)).float()
        is_pred = (prev.unsqueeze(2) == idx.view(1, 1, T)).float()            # j is predecessor of t
        is_succ = is_pred.transpose(1, 2)                                     # t is predecessor of j
        order = (idx.view(1, T, 1) - idx.view(1, 1, T)).float()               # t - j
        feasible = ((torch.sign(diff) * torch.sign(order)) >= 0).float() * both.float()
        feats = torch.stack([same(0), same(1), same(2), is_pred, is_succ, feasible, 1.0 - both.float()], dim=-1)
        b_c = feats @ self.custody_w                                          # [B,T,T,H]
        ops = pair[:, :, 3].clamp(-1, self.n_ops - 2) + 1                    # 0 = unknown
        op_idx = ops.unsqueeze(2) * self.n_ops + ops.unsqueeze(1)
        b_c = b_c + self.op_table(op_idx)
        return (b_tau + b_c).permute(0, 3, 1, 2)                              # [B,H,T,T]


class ReliabilityBiasedAttention(nn.Module):
    """Multi-head attention with the additive biases of Eq. (9)."""

    def __init__(self, d: int, heads: int, dropout: float, eps: float = 1e-6):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh, self.eps = heads, d // heads, eps
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pair_bias: torch.Tensor, rho: torch.Tensor, key_mask: torch.Tensor,
                use_rho: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, d = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + pair_bias
        if use_rho:
            scores = scores + torch.log(rho + self.eps).view(B, 1, 1, T)
        scores = scores.masked_fill(~key_mask.view(B, 1, 1, T), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, T, d)
        return self.o(out), attn


class TemporalLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float, eps: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = ReliabilityBiasedAttention(d, heads, dropout, eps)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pair_bias, rho, key_mask, use_rho=True):
        a, attn = self.attn(self.ln1(x), pair_bias, rho, key_mask, use_rho)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, attn


class TemporalEncoder(nn.Module):
    """Transaction-Consistency Learning branch."""

    def __init__(self, in_dim: int, d: int = 128, layers: int = 3, heads: int = 4, ffn: int = 512,
                 dropout: float = 0.2, n_ops: int = 16, time_buckets: int = 16, eps: float = 1e-6):
        super().__init__()
        self.d, self.eps = d, eps
        self.proj = nn.Linear(in_dim, d)               # W_u
        self.phi = ElapsedTimeEncoding(d)              # phi(dtau)
        self.biases = PairwiseBiases(heads, n_ops, time_buckets)
        self.layers = nn.ModuleList([TemporalLayer(d, heads, ffn, dropout, eps) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.w_T = nn.Linear(d, 1)                     # pooling vector w_T

    def forward(self, feats: torch.Tensor, dtau: torch.Tensor, tau_avail: torch.Tensor, pair: torch.Tensor,
                prev: torch.Tensor, rho: torch.Tensor, mask: torch.Tensor, use_rho: bool = True) -> dict:
        B, T, _ = feats.shape
        u = self.proj(feats) + sinusoidal_positions(T, self.d, feats.device).unsqueeze(0) + self.phi(dtau, tau_avail)
        bias = self.biases(dtau, tau_avail, pair, prev)
        attn = None
        rho_eff = rho if use_rho else torch.ones_like(rho)
        for layer in self.layers:
            u, attn = layer(u, bias, rho_eff, mask, use_rho)
        u = self.ln(u)
        # Eq. (10): reliability-aware pooling, computed in a numerically stable form
        s = self.w_T(u).squeeze(-1).masked_fill(~mask, float("-inf"))
        s = s - s.max(dim=1, keepdim=True).values.clamp_min(-1e4)
        num = torch.exp(s) * rho_eff * mask.float()
        pi = num / (num.sum(dim=1, keepdim=True) + self.eps)
        h = (pi.unsqueeze(-1) * u).sum(dim=1)
        received = attn.mean(dim=1).sum(dim=1) if attn is not None else torch.zeros(B, T, device=feats.device)
        return {"h": h, "u": u, "pi": pi, "attention_received": received, "attn": attn}
