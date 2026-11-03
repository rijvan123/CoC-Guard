"""Candidate identification and prioritisation (Section III-I, Table VIII).

Candidates are event factors and entity-centred relation factors
(:func:`cocguard.counterfactual.interventions.candidate_factors`). Event-level
scores are aggregated over the events of a factor.

``random``          uniformly random order;
``attention``       attention received in the last temporal layer (aggregation weights);
``shap``            Shapley values of events for the predicted-class probability, estimated
                    by antithetic permutation sampling with the masking intervention (the
                    computation of SHAP's permutation explainer with an all-masked baseline);
                    a factor's value is the sum of its events' values (additivity);
``responsibility``  CoC-Guard responsibility: deterministic violations, localisation
                    scores, model sensitivity (gradient x input) and reliability-aware
                    pooling weights, prioritised by intervention validity.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.collate import collate_investigations
from .interventions import apply_intervention, as_factor, validity

METHODS = ("random", "attention", "shap", "responsibility")


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min() if len(x) else 0.0
    return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)


@torch.no_grad()
def _attention(model, sample, device) -> np.ndarray:
    model.eval()
    out = model(collate_investigations([sample]).to(device))
    return out["attention_received"][0, : len(sample["v"])].cpu().numpy()


def _sensitivity(model, sample, device) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    batch = collate_investigations([sample]).to(device)
    batch["xb"].requires_grad_(True)
    if batch["xc"].numel():
        batch["xc"].requires_grad_(True)
    out = model(batch)
    T = len(sample["v"])
    key = "probs_m" if "probs_m" in out else "probs_c"
    torch.log(out[key][0].max().clamp_min(1e-12)).backward()
    sal = (batch["xb"].grad * batch["xb"]).abs().sum(-1)[0, :T]
    if batch["xc"].numel() and batch["xc"].grad is not None:
        sal = sal + (batch["xc"].grad * batch["xc"]).abs().sum(-1)[0, :T]
    loc = out["loc_logits"][0, :T].detach().cpu().numpy() if "loc_logits" in out else None
    return sal.detach().cpu().numpy(), out["pool_T"][0, :T].detach().cpu().numpy(), loc


@torch.no_grad()
def _shap(model, sample, device, permutations: int, rng: np.random.Generator, kind: str = "mask",
          batch_size: int = 64) -> np.ndarray:
    """Permutation-sampling Shapley values of events (antithetic permutations)."""
    model.eval()
    T = len(sample["v"])
    out0 = model(collate_investigations([sample]).to(device))
    key = "probs_m" if "probs_m" in out0 else "probs_c"
    yhat = int(out0[key][0].argmax())
    phi = np.zeros(T)
    perms = []
    for _ in range(max(1, permutations // 2)):
        p = rng.permutation(T)
        perms += [p, p[::-1]]
    masked_all = apply_intervention(sample, {"id": "all", "events": list(range(T)), "node": None}, kind, rng,
                                    jitter_seconds=0.0)
    for perm in perms:
        states, cur = [masked_all], masked_all
        for j in perm:  # reveal events one by one
            cur = {**cur, "xb": cur["xb"].copy(), "mb": cur["mb"].copy(), "cat": cur["cat"].copy()}
            cur["xb"][j], cur["mb"][j], cur["cat"][j] = sample["xb"][j], sample["mb"][j], sample["cat"][j]
            states.append(cur)
        vals = []
        for k in range(0, len(states), batch_size):
            out = model(collate_investigations(states[k:k + batch_size]).to(device))
            vals.append(out[key][:, yhat].cpu().numpy())
        phi[perm] += np.diff(np.concatenate(vals))
    return phi / len(perms)


def _aggregate(event_scores: np.ndarray, factors: list[dict]) -> np.ndarray:
    return np.array([float(np.sum(event_scores[f["events"]])) for f in factors])


def rank_candidates(method: str, model, sample: dict, device, rng: np.random.Generator,
                    shap_permutations: int = 10, kind: str = "mask", factors: list[dict] | None = None) -> np.ndarray:
    """Return an ordering (indices into ``factors``; events if ``factors`` is None)."""
    T = len(sample["v"])
    factors = [as_factor(t) for t in range(T)] if factors is None else factors
    if method == "random":
        return rng.permutation(len(factors))
    if method == "attention":
        return np.argsort(-_aggregate(_attention(model, sample, device), factors), kind="stable")
    if method == "shap":
        return np.argsort(-_aggregate(_shap(model, sample, device, shap_permutations, rng, kind), factors), kind="stable")
    if method == "responsibility":
        sal, pool, loc = _sensitivity(model, sample, device)
        violated = (sample["v"].min(axis=1) < 1).astype(float)
        parts = [_norm(sal), _norm(pool), violated]
        if loc is not None:
            parts.append(_norm(loc))
        score = _aggregate(np.mean(parts, axis=0), factors)
        prior = np.array([1.0 if validity(sample, apply_intervention(sample, f, kind, rng, jitter_seconds=0.0), f, kind)[0]
                          else 0.05 for f in factors])
        return np.argsort(-(score * prior), kind="stable")
    raise ValueError(f"unknown selection method {method!r}")
