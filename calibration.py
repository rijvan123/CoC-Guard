"""Temperature scaling on the development partition (T_m, T_c in Eq. 18), ECE and Brier score."""
from __future__ import annotations

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Minimise the development negative log-likelihood over a single temperature."""
    labels = np.asarray(labels)
    sel = labels >= 0
    if sel.sum() < 2:
        return 1.0
    lg = torch.tensor(np.asarray(logits)[sel], dtype=torch.float64)
    y = torch.tensor(labels[sel], dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(log_t).item())
    return float(np.clip(t, 0.05, 20.0))


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(acc[sel].mean() - conf[sel].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), np.asarray(labels)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())
