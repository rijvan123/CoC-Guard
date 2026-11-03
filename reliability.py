"""Rule-based event reliability rho^r (Eq. 6), NumPy reference implementation.

    rho^r_{i,t} = (1 - omega^T delta_{i,t} / (omega^T 1 + eps))
                  * [ lambda_b ||m^b_{i,t}||_1 / d_b + lambda_c ||m^c_{i,t}||_1 / d_c ],
    lambda_b + lambda_c = 1.

When an entire evidence view is not native to an experimental track (for
example custody records on BODMAS), its completeness term is removed and the
remaining coefficient is renormalised to one ("availability-aware" reliability,
the same convention used for Eq. 19). The differentiable PyTorch version used
inside the model is :func:`cocguard.models.reliability.rule_reliability`.
"""
from __future__ import annotations

import numpy as np

from .policy import VERIFICATION_COMPONENTS

DEFAULT_SEVERITY = {"hash": 1.0, "time": 0.6, "auth": 0.8, "transfer": 0.8, "provenance": 0.5}


def severity_vector(severity: dict | None = None) -> np.ndarray:
    severity = severity or DEFAULT_SEVERITY
    return np.array([float(severity[k]) for k in VERIFICATION_COMPONENTS], dtype=np.float32)


def rule_reliability(delta: np.ndarray, mb: np.ndarray | None, mc: np.ndarray | None,
                     severity: dict | None = None, lambda_b: float = 0.5, lambda_c: float = 0.5,
                     eps: float = 1e-6) -> np.ndarray:
    """Vectorised Eq. (6) for one history. ``delta`` has shape [T, 5].

    ``mb``/``mc`` are availability masks of shape [T, d_b]/[T, d_c] or ``None``
    when the corresponding view is not native to the track.
    """
    omega = severity_vector(severity)
    violation = 1.0 - (delta @ omega) / (omega.sum() + eps)
    parts, weights = [], []
    if mb is not None and mb.shape[-1] > 0:
        parts.append(mb.sum(axis=-1) / mb.shape[-1])
        weights.append(lambda_b)
    if mc is not None and mc.shape[-1] > 0:
        parts.append(mc.sum(axis=-1) / mc.shape[-1])
        weights.append(lambda_c)
    if not parts:
        completeness = np.ones(delta.shape[0], dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        w = w / w.sum()
        completeness = sum(wi * p for wi, p in zip(w, parts))
    return (violation * completeness).astype(np.float32)
