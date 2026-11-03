"""The complete CoC-Guard model, f_Theta(S_i) = (p^m_i, p^c_i, s^u_i, r_i) (Eq. 3).

This module assembles the learned stages of Algorithm 1:

1. deterministic verification outcomes v (pre-computed, frozen) and the
   integrity-conditioned reliability rho (Eqs. 6-7);
2. Transaction-Consistency Learning (Eqs. 8-10) and the heterogeneous evidence
   graph (Eqs. 11-14);
3. evidence-quality vectors, Reliability-Controlled Fusion and separate
   calibrated malware and custody heads (Eqs. 15-18);
4. event-level transaction localisation (the target of L_loc in Eq. 29).

Open-World Screening (Eqs. 19-21) is applied on top of the outputs by
:class:`cocguard.openworld.screening.OpenWorldScreener`, replay by
:mod:`cocguard.replay`, and counterfactual verification by
:mod:`cocguard.counterfactual` with frozen parameters.

The ``variant`` switches reproduce the ablations of Table X:
``use_temporal``, ``use_graph``, ``use_rules`` (deterministic verification),
``fusion`` (RCF or a comparator) and ``use_disagreement``.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from ..data.schema import TrackSpec
from .fusion import ReliabilityControlledFusion
from .graph import EvidenceGraphEncoder
from .heads import CalibratedHead
from .reliability import LearnedReliability
from .temporal import TemporalEncoder

DEFAULT_VARIANT = {"fusion": "rcf", "use_temporal": True, "use_graph": True, "use_rules": True,
                   "use_disagreement": True}


def normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    k = p.shape[-1]
    if k <= 1:
        return torch.zeros(p.shape[:-1], device=p.device)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / math.log(k)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence with base-2 logarithm (bounded in [0, 1])."""
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a.clamp_min(1e-12)) - torch.log(b.clamp_min(1e-12)))).sum(-1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2.0)


class CoCGuard(nn.Module):
    def __init__(self, spec: TrackSpec, model_cfg: dict, variant: dict | None = None):
        super().__init__()
        self.spec = spec
        self.variant = {**DEFAULT_VARIANT, **(variant or {})}
        cfg = dict(model_cfg)
        d = int(cfg.get("hidden_dim", 128))
        self.d = d
        self.eps = float(cfg.get("eps", 1e-6))
        emb = int(cfg.get("categorical_embedding_dim", 16))
        self.malware_active = spec.num_malware_classes > 0
        self.custody_active = spec.num_custody_classes > 0
        self.cat_emb = nn.ModuleList([nn.Embedding(max(2, int(n)), emb, padding_idx=0) for n in spec.cat_vocab_sizes])
        in_dim = spec.d_b + spec.d_c + emb * len(spec.cat_vocab_sizes) + 5 + spec.d_b + spec.d_c + 1
        self.in_dim = in_dim
        self.reliability = LearnedReliability(in_dim, alpha_rho=float(cfg.get("alpha_rho", 0.6)),
                                              severity=cfg.get("severity"), lambda_b=float(cfg.get("lambda_b", 0.5)),
                                              lambda_c=float(cfg.get("lambda_c", 0.5)), eps=self.eps)
        dropout = float(cfg.get("dropout", 0.2))
        self.temporal = TemporalEncoder(in_dim, d, int(cfg.get("temporal_layers", 3)), int(cfg.get("temporal_heads", 4)),
                                        int(cfg.get("ffn_dim", 512)), dropout, n_ops=max(16, spec.n_operations),
                                        time_buckets=int(cfg.get("time_buckets", 16)), eps=self.eps)
        self.graph = EvidenceGraphEncoder(in_dim, spec.d_node, d, int(cfg.get("graph_layers", 3)),
                                          int(cfg.get("graph_heads", 4)), dropout, self.eps)
        d_G = self.graph.out_dim
        tasks = tuple(t for t, on in (("m", self.malware_active), ("c", self.custody_active)) if on)
        self.tasks = tasks
        self.fusion = ReliabilityControlledFusion(d, d_G, d, mode=self.variant["fusion"],
                                                  use_disagreement=self.variant["use_disagreement"], tasks=tasks,
                                                  eps=self.eps)
        self.aux = nn.ModuleDict()
        if self.malware_active:
            self.head_m = CalibratedHead(d, spec.num_malware_classes)
            self.aux["mT"] = nn.Linear(d, spec.num_malware_classes)
            self.aux["mG"] = nn.Linear(d_G, spec.num_malware_classes)
        if self.custody_active:
            self.head_c = CalibratedHead(d, spec.num_custody_classes)
            self.aux["cT"] = nn.Linear(d, spec.num_custody_classes)
            self.aux["cG"] = nn.Linear(d_G, spec.num_custody_classes)
            self.loc_head = nn.Sequential(nn.Linear(2 * d + 5, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform weights and zero biases (Table A6)."""
        for name, p in self.named_parameters():
            if p.dim() == 2 and "cat_emb" not in name and "table" not in name and not name.endswith("Wr"):
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and name.endswith("bias"):
                nn.init.zeros_(p)

    # ------------------------------------------------------------------
    def event_features(self, batch) -> torch.Tensor:
        parts = [batch.xb, batch.xc]
        if len(self.cat_emb):
            parts.append(torch.cat([emb(batch.cat[..., k].clamp(0, emb.num_embeddings - 1))
                                    for k, emb in enumerate(self.cat_emb)], dim=-1))
        v = batch.v if self.variant["use_rules"] else torch.ones_like(batch.v)
        parts += [v, batch.mb, batch.mc, batch.tau_avail.unsqueeze(-1)]
        return torch.cat(parts, dim=-1)

    def forward(self, batch) -> dict:
        mask = batch.ev_mask
        B, T = mask.shape
        feats = self.event_features(batch)
        v = batch.v if self.variant["use_rules"] else torch.ones_like(batch.v)
        delta = (1.0 - v) * mask.unsqueeze(-1).float()
        mb = batch.mb if self.spec.d_b else None
        mc = batch.mc if self.spec.d_c else None
        rho, rho_r = self.reliability(feats, delta, mb, mc)
        rho = rho * mask.float()
        rho_r = rho_r * mask.float()
        out: dict = {"rho": rho, "rho_rule": rho_r}

        # ---- temporal branch (Eqs. 8-10) ----------------------------------
        a_T = batch.a_T * float(self.variant["use_temporal"] and self.spec.has_temporal)
        if self.variant["use_temporal"] and self.spec.has_temporal:
            t_out = self.temporal(feats, batch.dtau, batch.tau_avail, batch.pair, batch.prev, rho, mask)
            h_T, u_ev = t_out["h"] * a_T.unsqueeze(-1), t_out["u"]
            out.update(pool_T=t_out["pi"], attention_received=t_out["attention_received"], attn=t_out["attn"])
        else:
            h_T = feats.new_zeros(B, self.d)
            u_ev = feats.new_zeros(B, T, self.d)
            out.update(pool_T=feats.new_zeros(B, T), attention_received=feats.new_zeros(B, T), attn=None)

        # ---- graph branch (Eqs. 11-14) ------------------------------------
        a_G = batch.a_G * float(self.variant["use_graph"] and self.spec.has_graph)
        sup = batch.edge_support
        eta = sup.prod(dim=-1) if self.variant["use_rules"] else sup[:, 3]
        if self.variant["use_graph"] and self.spec.has_graph and batch.node_type.numel():
            g_out = self.graph(batch.node_type, batch.node_x, batch.node_batch, batch.edge_index, batch.edge_type,
                               eta, batch.event_node, feats, rho, mask, B)
            h_G = g_out["h"] * a_G.unsqueeze(-1)
            ev_h = g_out["event_h"]
            rho_G = g_out["rho_bar"] * a_G
            out.update(graph_beta=g_out["beta"], node_rho=g_out.get("node_rho"))
        else:
            h_G = feats.new_zeros(B, self.graph.out_dim)
            ev_h = feats.new_zeros(B, T, self.d)
            rho_G = feats.new_zeros(B)
            a_G = a_G * 0.0
        out.update(h_T=h_T, h_G=h_G, a_T=a_T, a_G=a_G, eta=eta)

        # ---- branch-level auxiliary predictions: uncertainty and disagreement (Eq. 15)
        u_T = feats.new_zeros(B)
        u_G = feats.new_zeros(B)
        d_tg = feats.new_zeros(B)
        for y in self.tasks:
            lt, lg = self.aux[y + "T"](h_T), self.aux[y + "G"](h_G)
            out["aux_" + y + "T"], out["aux_" + y + "G"] = lt, lg
            pt, pg = torch.softmax(lt, -1), torch.softmax(lg, -1)
            u_T = u_T + normalized_entropy(pt) / len(self.tasks)
            u_G = u_G + normalized_entropy(pg) / len(self.tasks)
            d_tg = d_tg + js_divergence(pt, pg) / len(self.tasks)
        u_T = torch.where(a_T > 0, u_T, torch.ones_like(u_T))
        u_G = torch.where(a_G > 0, u_G, torch.ones_like(u_G))
        both = (a_T > 0) & (a_G > 0)
        d_tg = torch.where(both, d_tg, torch.zeros_like(d_tg))
        rho_T = (rho.sum(1) / mask.float().sum(1).clamp_min(1.0)) * a_T
        out.update(u_T=u_T, u_G=u_G, d_tg=d_tg, rho_T=rho_T, rho_G=rho_G,
                   rho_bar=(rho.sum(1) / mask.float().sum(1).clamp_min(1.0)))

        # ---- fusion and calibrated heads (Eqs. 16-18) ------------------------
        fused = self.fusion(h_T, h_G, a_T, a_G, u_T, u_G, rho_T, rho_G, d_tg)
        out.update({k: v for k, v in fused.items()})
        if self.malware_active:
            out["logits_m"] = self.head_m(fused["z_m"])
            out["probs_m"] = self.head_m.probs(out["logits_m"])
        if self.custody_active:
            out["logits_c"] = self.head_c(fused["z_c"])
            out["probs_c"] = self.head_c.probs(out["logits_c"])
            loc_in = torch.cat([u_ev, ev_h, delta], dim=-1)
            out["loc_logits"] = self.loc_head(loc_in).squeeze(-1).masked_fill(~mask, -1e4)
        return out


@contextmanager
def mc_dropout(model: nn.Module):
    """Enable dropout only (seeded stochastic realisations of a frozen model)."""
    was_training = model.training
    model.eval()
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in drops:
        m.train()
    try:
        yield model
    finally:
        model.train(was_training)
        if not was_training:
            model.eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
