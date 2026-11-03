"""GCN [18] and GAT [19] baselines over the evidence graph (Table A4).

The graph is treated as homogeneous (relation types and edge support are not
used), which is exactly what distinguishes these comparators from the
support-weighted relational propagation of Eq. (13). Layers follow the
standard formulations (symmetric normalisation with self-loops for GCN;
multi-head attention with LeakyReLU(0.2) for GAT), implemented with plain
PyTorch scatter operations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...data.schema import NODE_TYPES, TrackSpec
from ..scatter import scatter_mean, scatter_sum, segment_softmax
from .sequence import _EventInput, _Heads


def _with_self_loops(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    loops = torch.arange(n, device=edge_index.device)
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.cat([ei, torch.stack([loops, loops])], dim=1)


class GCNLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, h, ei):
        n = h.shape[0]
        src, dst = ei
        deg = scatter_sum(torch.ones_like(dst, dtype=h.dtype), dst, n)
        norm = deg[src].clamp_min(1).rsqrt() * deg[dst].clamp_min(1).rsqrt()
        return scatter_sum(self.lin(h)[src] * norm.unsqueeze(-1), dst, n)


class GATLayer(nn.Module):
    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.lin = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1)

    def forward(self, h, ei):
        n = h.shape[0]
        src, dst = ei
        x = self.lin(h).view(n, self.h, self.dh)
        e = F.leaky_relu((x[src] * self.a_src).sum(-1) + (x[dst] * self.a_dst).sum(-1), 0.2)
        alpha = segment_softmax(e, dst, n)
        return scatter_sum((alpha.unsqueeze(-1) * x[src]).reshape(-1, self.h * self.dh), dst, n)


class GNNBaseline(nn.Module):
    def __init__(self, spec: TrackSpec, kind: str = "gcn", hidden: int = 128, layers: int = 3, heads: int = 4,
                 dropout: float = 0.25):
        super().__init__()
        self.kind = kind
        self.inp = _EventInput(spec)
        self.event_proj = nn.Linear(self.inp.out_dim, hidden)
        self.node_proj = nn.Linear(spec.d_node, hidden)
        self.layers = nn.ModuleList([(GCNLayer(hidden) if kind == "gcn" else GATLayer(hidden, heads)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.heads = _Heads(spec, hidden, hidden)
        self.tasks = self.heads.tasks
        self.hidden = hidden

    def forward(self, batch) -> dict:
        B, T = batch.ev_mask.shape
        n = batch.node_type.shape[0]
        h = self.node_proj(batch.node_x)
        feats = self.inp(batch)
        valid = (batch.event_node >= 0) & batch.ev_mask
        h = h.index_put((batch.event_node[valid],), self.event_proj(feats[valid]))
        ei = _with_self_loops(batch.edge_index, n)
        for layer, norm in zip(self.layers, self.norms):
            h = self.drop(F.gelu(norm(layer(h, ei) + h)))
        pooled = scatter_mean(h, batch.node_batch, B) if n else feats.new_zeros(B, self.hidden)
        ev_h = feats.new_zeros(B, T, self.hidden)
        ev_h[valid] = h[batch.event_node[valid]]
        out = self.heads(pooled, ev_h, 1.0 - batch.v, batch.ev_mask)
        zeros = pooled.new_zeros(B)
        out.update({"rho": batch.ev_mask.float(), "a_T": zeros, "a_G": batch.a_G, "rho_T": zeros, "rho_G": zeros + 1,
                    "pool_T": batch.ev_mask.float() / batch.ev_mask.float().sum(1, keepdim=True).clamp_min(1),
                    "attention_received": batch.ev_mask.float()})
        return out
