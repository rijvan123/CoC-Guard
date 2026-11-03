"""Statistical procedures of Section IV-E and Table XII.

* prediction-level **paired stratified bootstrap** (10,000 replicates) of a
  metric difference between two methods evaluated on the same observations,
  with 95% percentile intervals and two-sided bootstrap p-values;
* **McNemar's test** for paired binary decisions (exact binomial version);
* **exact Wilcoxon signed-rank test** over matched seeds (with five pairs the
  smallest attainable two-sided p-value is 0.0625, so seed-level results are
  exploratory only);
* **Holm** step-down correction within a family of comparisons.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats


def _strata_indices(strata: np.ndarray) -> list[np.ndarray]:
    return [np.where(strata == s)[0] for s in np.unique(strata)]


def stratified_resample(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = [rng.choice(idx, size=len(idx), replace=True) for idx in _strata_indices(strata)]
    return np.concatenate(parts)


def paired_bootstrap(metric: Callable[[np.ndarray], float], metric_b: Callable[[np.ndarray], float],
                     strata: Sequence, n_boot: int = 10000, seed: int = 11, alpha: float = 0.05) -> dict:
    """Bootstrap the difference metric(A) - metric(B) over shared observations.

    ``metric`` and ``metric_b`` map an index array (a resample of the shared
    observations) to a scalar for method A and method B respectively.
    """
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    full = np.arange(len(strata))
    observed = float(metric(full) - metric_b(full))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = stratified_resample(strata, rng)
        diffs[b] = metric(idx) - metric_b(idx)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2]) if len(diffs) else (np.nan, np.nan)
    p_le = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)
    p_ge = (np.sum(diffs >= 0) + 1) / (len(diffs) + 1)
    p = float(min(1.0, 2 * min(p_le, p_ge)))
    return {"difference": observed, "ci_low": float(lo), "ci_high": float(hi), "p_value": p, "replicates": int(len(diffs))}


def stratified_bootstrap_ci(metric: Callable[[np.ndarray], float], strata: Sequence, n_boot: int = 10000,
                            seed: int = 11, alpha: float = 0.05) -> tuple[float, float]:
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    vals = np.array([metric(stratified_resample(strata, rng)) for _ in range(n_boot)], dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}


def wilcoxon_exact(a: Sequence[float], b: Sequence[float]) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": int(len(d))}
    try:
        res = stats.wilcoxon(a, b, method="exact", alternative="two-sided")
    except TypeError:  # SciPy < 1.9
        res = stats.wilcoxon(a, b, mode="exact", alternative="two-sided")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n_pairs": int(len(d))}


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
