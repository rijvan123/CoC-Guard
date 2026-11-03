"""BiLSTM [16] and temporal CNN [17] baselines over ordered event sequences (Table A4).

Both consume the same event attributes as CoC-Guard ([x~^b || x~^c || embeddings || v || m])
and expose the same output keys (``logits_m``/``logits_c``/``loc_logits``), so they can be
trained with :class:`cocguard.training.trainer.Trainer` without temporal-graph fusion.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...data.schema import TrackSpec


class _EventInput(nn.Module):
    def __init__(self, spec: TrackSpec, emb: int = 16):
        super().__init__()
        self.cat_emb = nn.ModuleList([nn.Embedding(max(2, int(n)), emb, padding_idx=0) for n in spec.cat_vocab_sizes])
        self.out_dim = spec.d_b + spec.d_c + emb * len(spec.cat_vocab_sizes) + 5 + spec.d_b + spec.d_c + 1

    def forward(self, batch) -> torch.Tensor:
        parts = [batch.xb, batch.xc]
        if len(self.cat_emb):
            parts.append(torch.cat([e(batch.cat[..., k].clamp(0, e.num_embeddings - 1)) for k, e in enumerate(self.cat_emb)], -1))
        parts += [batch.v, batch.mb, batch.mc, batch.tau_avail.unsqueeze(-1)]
        return torch.cat(parts, -1)


class _Heads(nn.Module):
    def __init__(self, spec: TrackSpec, d: int, d_event: int):
        super().__init__()
        self.tasks = tuple(t for t, n in (("m", spec.num_malware_classes), ("c", spec.num_custody_classes)) if n > 0)
        if spec.num_malware_classes:
            self.m = nn.Linear(d, spec.num_malware_classes)
        if spec.num_custody_classes:
            self.c = nn.Linear(d, spec.num_custody_classes)
            self.loc = nn.Linear(d_event + 5, 1)

    def forward(self, z, events, delta, mask) -> dict:
        out = {"z_m": z, "z_c": z}
        if hasattr(self, "m"):
            out["logits_m"] = self.m(z)
            out["probs_m"] = torch.softmax(out["logits_m"], -1)
        if hasattr(self, "c"):
            out["logits_c"] = self.c(z)
            out["probs_c"] = torch.softmax(out["logits_c"], -1)
            out["loc_logits"] = self.loc(torch.cat([events, delta], -1)).squeeze(-1).masked_fill(~mask, -1e4)
        return out


def _common(batch, z_events: torch.Tensor, pooled: torch.Tensor) -> dict:
    B, T = batch.ev_mask.shape
    zeros = pooled.new_zeros(B)
    return {"rho": batch.ev_mask.float(), "a_T": batch.a_T, "a_G": zeros, "rho_T": zeros + 1, "rho_G": zeros,
            "pool_T": batch.ev_mask.float() / batch.ev_mask.float().sum(1, keepdim=True).clamp_min(1),
            "attention_received": batch.ev_mask.float()}


class BiLSTMBaseline(nn.Module):
    def __init__(self, spec: TrackSpec, hidden: int = 128, layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.inp = _EventInput(spec)
        self.lstm = nn.LSTM(self.inp.out_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.heads = _Heads(spec, 2 * hidden, 2 * hidden)
        self.tasks = self.heads.tasks

    def forward(self, batch) -> dict:
        x = self.inp(batch)
        lengths = batch.ev_mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        h = self.drop(h)
        m = batch.ev_mask.float().unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        out = self.heads(pooled, h, 1.0 - batch.v, batch.ev_mask)
        out.update(_common(batch, h, pooled))
        return out


class _TCNBlock(nn.Module):
    def __init__(self, d: int, dilation: int, kernels=(3, 5, 7), dropout: float = 0.25):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, dilation=dilation, padding=(k - 1) * dilation) for k in kernels])
        self.proj = nn.Conv1d(d * len(kernels), d, 1)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):  # x: [B, d, T]
        T = x.shape[-1]
        ys = [F.gelu(c(x)[..., :T]) for c in self.convs]     # causal (left-padded) dilated convolutions
        y = self.proj(torch.cat(ys, 1))
        return self.norm((x + self.drop(y)).transpose(1, 2)).transpose(1, 2)


class TCNBaseline(nn.Module):
    def __init__(self, spec: TrackSpec, hidden: int = 128, layers: int = 4, dropout: float = 0.25, kernels=(3, 5, 7)):
        super().__init__()
        self.inp = _EventInput(spec)
        self.proj = nn.Linear(self.inp.out_dim, hidden)
        self.blocks = nn.ModuleList([_TCNBlock(hidden, 2 ** l, kernels, dropout) for l in range(layers)])
        self.heads = _Heads(spec, hidden, hidden)
        self.tasks = self.heads.tasks

    def forward(self, batch) -> dict:
        x = self.proj(self.inp(batch)) * batch.ev_mask.unsqueeze(-1).float()
        h = x.transpose(1, 2)
        for b in self.blocks:
            h = b(h) * batch.ev_mask.unsqueeze(1).float()
        h = h.transpose(1, 2)
        m = batch.ev_mask.float().unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        out = self.heads(pooled, h, 1.0 - batch.v, batch.ev_mask)
        out.update(_common(batch, h, pooled))
        return out
