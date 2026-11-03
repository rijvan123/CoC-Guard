"""Scatter / segment operations in plain PyTorch (no compiled extensions)."""
from __future__ import annotations

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    if src.numel():
        out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    total = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones_like(index, dtype=src.dtype), index, dim_size).clamp_min(1.0)
    return total / count.view(-1, *([1] * (src.dim() - 1)))


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_full((dim_size,) + tuple(src.shape[1:]), float("-inf"))
    if src.numel():
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    return out


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax of ``scores`` within groups given by ``index`` (dim 0)."""
    if scores.numel() == 0:
        return scores
    mx = scatter_max(scores, index, dim_size)
    ex = torch.exp(scores - mx[index])
    den = scatter_sum(ex, index, dim_size)
    return ex / den[index].clamp_min(1e-16)
