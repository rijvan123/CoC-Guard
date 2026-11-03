"""Confidence-aware heterogeneous evidence-graph learning (Eqs. 11-14).

    h^l_v = LN[ W^l_0 h^{l-1}_v + sum_{r in T_E} sum_{u in N_r(v)} beta-bar^{l,r}_uv eta_uv W^l_r h^{l-1}_u ]   (13)
    h^G_i = ||_{k in T*_V} ( sum_{v: phi(v)=k} pi^G_v h^{(L_G)}_v )                                         (14)

``beta-bar`` is a multi-head, relation-specific attention coefficient
normalised over the neighbours of ``v`` under relation ``r``; every relation is
propagated in both directions with its own transformation (r and r^-1). The
edge support ``eta_uv`` (Eq. 12) scales each message, so explicitly invalid
relations remain in the graph (for custody assessment and localisation) but
contribute less as trusted behavioural evidence. Type-aware pooling weighs
nodes by their reliability and returns a node-type availability mask, so an
unavailable entity type is never read as an observed but weak representation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..data.schema import NODE_TYPES, RELATION_TYPES
from .scatter import scatter_max, scatter_mean, scatter_sum, segment_softmax

N_NODE_TYPES = len(NODE_TYPES)
N_REL = len(RELATION_TYPES)


class RelationalSupportLayer(nn.Module):
    def __init__(self, d: int, heads: int = 4, n_rel: int = 2 * N_REL, dropout: float = 0.2):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.dh, self.n_rel = d, heads, d // heads, n_rel
        self.W0 = nn.Linear(d, d)
        self.Wr = nn.Parameter(torch.empty(n_rel, d, d))
        self.a_src = nn.Parameter(torch.empty(n_rel, heads, self.dh))
        self.a_dst = nn.Parameter(torch.empty(n_rel, heads, self.dh))
        nn.init.xavier_uniform_(self.Wr.view(n_rel * d, d))
        nn.init.xavier_uniform_(self.a_src.view(n_rel, -1))
        nn.init.xavier_uniform_(self.a_dst.view(n_rel, -1))
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = h.shape[0]
        out = self.W0(h)
        E = edge_index.shape[1]
        beta_all = h.new_zeros((E, self.h))
        if E:
            src, dst = edge_index
            msg = h.new_zeros((E, self.d))
            score = h.new_zeros((E, self.h))
            for r in torch.unique(edge_type).tolist():
                sel = (edge_type == r).nonzero(as_tuple=True)[0]
                hs = h[src[sel]] @ self.Wr[r]
                hd = h[dst[sel]] @ self.Wr[r]
                msg[sel] = hs
                score[sel] = ((hs.view(-1, self.h, self.dh) * self.a_src[r]).sum(-1)
                              + (hd.view(-1, self.h, self.dh) * self.a_dst[r]).sum(-1))
            score = F.leaky_relu(score, 0.2)
            group = dst * self.n_rel + edge_type             # neighbours of v under relation r
            beta = segment_softmax(score, group, N * self.n_rel)          # beta-bar [E,H]
            weighted = (beta * eta.unsqueeze(-1)).unsqueeze(-1) * msg.view(E, self.h, self.dh)
            out = out + scatter_sum(weighted.reshape(E, self.d), dst, N)
            beta_all = beta
        return self.drop(F.gelu(self.ln(out))), beta_all


class EvidenceGraphEncoder(nn.Module):
    def __init__(self, event_in_dim: int, d_node: int, d: int = 128, layers: int = 3, heads: int = 4,
                 dropout: float = 0.2, eps: float = 1e-6):
        super().__init__()
        self.d, self.eps = d, eps
        self.event_proj = nn.Linear(event_in_dim, d)
        self.type_proj = nn.ModuleList([nn.Linear(d_node, d) for _ in range(N_NODE_TYPES)])
        self.layers = nn.ModuleList([RelationalSupportLayer(d, heads, 2 * N_REL, dropout) for _ in range(layers)])
        self.w_G = nn.Linear(d, 1)
        self.type_bias = nn.Parameter(torch.zeros(N_NODE_TYPES))
        self.out_dim = N_NODE_TYPES * d + N_NODE_TYPES

    def forward(self, node_type, node_x, node_batch, edge_index, edge_type, eta, event_node, event_feats,
                event_rho, ev_mask, batch_size: int) -> dict:
        N = node_type.shape[0]
        device = node_x.device
        B = batch_size
        if N == 0:
            return {"h": torch.zeros(B, self.out_dim, device=device), "node_h": torch.zeros(0, self.d, device=device),
                    "type_mask": torch.zeros(B, N_NODE_TYPES, device=device), "rho_bar": torch.zeros(B, device=device),
                    "event_h": torch.zeros(*event_node.shape, self.d, device=device), "beta": None}
        h = torch.zeros(N, self.d, device=device, dtype=node_x.dtype)
        for k in range(N_NODE_TYPES):
            sel = (node_type == k).nonzero(as_tuple=True)[0]
            if len(sel):
                h[sel] = self.type_proj[k](node_x[sel])
        valid = (event_node >= 0) & ev_mask
        ev_idx = event_node[valid]
        h[ev_idx] = self.event_proj(event_feats[valid])
        # both directions of every relation (r and r^-1)
        ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        et = torch.cat([edge_type, edge_type + N_REL])
        eta2 = torch.cat([eta, eta])
        beta = None
        for layer in self.layers:
            h, beta = layer(h, ei, et, eta2)
        # node reliability: events use rho_t; entities inherit (rho_event * eta) of incident relations
        node_rho = torch.zeros(N, device=device, dtype=h.dtype)
        is_event = torch.zeros(N, dtype=torch.bool, device=device)
        is_event[ev_idx] = True
        node_rho[ev_idx] = event_rho[valid].to(h.dtype)
        if ei.shape[1]:
            src, dst = ei
            contrib = torch.where(is_event[src], node_rho[src] * eta2, eta2)
            ent_rho = scatter_mean(contrib, dst, N)
            node_rho = torch.where(is_event, node_rho, ent_rho)
        else:
            node_rho = torch.where(is_event, node_rho, torch.ones_like(node_rho))
        # Eq. (14): type-aware reliability pooling with a node-type availability mask
        group = node_batch * N_NODE_TYPES + node_type
        s = self.w_G(h).squeeze(-1) + self.type_bias[node_type]
        mx = scatter_max(s, group, B * N_NODE_TYPES)[group]
        num = torch.exp(s - mx) * node_rho
        den = scatter_sum(num, group, B * N_NODE_TYPES)[group]
        pi = num / (den + self.eps)
        pooled = scatter_sum(pi.unsqueeze(-1) * h, group, B * N_NODE_TYPES).view(B, N_NODE_TYPES, self.d)
        present = scatter_sum(torch.ones_like(s), group, B * N_NODE_TYPES).view(B, N_NODE_TYPES) > 0
        pooled = pooled * present.unsqueeze(-1)
        h_G = torch.cat([pooled.reshape(B, -1), present.float()], dim=-1)
        # integrity support of the graph branch: mean edge support per investigation
        if edge_index.shape[1]:
            eb = node_batch[edge_index[1]]
            rho_bar = scatter_mean(eta, eb, B)
            has_edges = scatter_sum(torch.ones_like(eta), eb, B) > 0
            rho_bar = torch.where(has_edges, rho_bar, torch.zeros_like(rho_bar))
        else:
            rho_bar = torch.zeros(B, device=device)
        event_h = torch.zeros(*event_node.shape, self.d, device=device, dtype=h.dtype)
        event_h[valid] = h[ev_idx]
        return {"h": h_G, "node_h": h, "type_mask": present.float(), "rho_bar": rho_bar, "event_h": event_h,
                "beta": beta, "node_rho": node_rho}
