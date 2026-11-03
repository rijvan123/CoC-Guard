"""Evaluation metrics (Section IV-D).

Open-world conventions
----------------------
* ``UNKNOWN = -1`` denotes both the true label of a withheld-family sample and
  the "unknown/review" decision of Eq. (21).
* Known-family macro F1 is averaged over the known classes only, computed on all
  test observations (an unknown sample assigned to a known class is a false
  positive of that class; a rejected known sample is a false negative).
* The known-unknown H-score is the harmonic mean of known-family macro F1 and
  unknown recall (it reproduces the H-scores of Table V from its other columns).
* ECE (15 bins) and Brier score are computed from the calibrated known-class
  distribution on known-family test observations.

Custody conventions
-------------------
* Integrity accuracy: the predicted technical custody state equals the true
  state (intact, a violation type, or multiple).
* Violation recall: compromised histories predicted as not intact.
* Localisation accuracy: the top-ranked event is an affected event, over all
  compromised histories.
* Intact FPR: intact histories predicted as compromised.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..openworld.calibration import brier_score, expected_calibration_error

UNKNOWN = -1


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]):
    labels = np.asarray(list(labels))
    tp = np.array([np.sum((y_pred == c) & (y_true == c)) for c in labels], dtype=float)
    fp = np.array([np.sum((y_pred == c) & (y_true != c)) for c in labels], dtype=float)
    fn = np.array([np.sum((y_pred != c) & (y_true == c)) for c in labels], dtype=float)
    return tp, fp, fn


def macro_prf(y_true, y_pred, labels: Sequence[int]) -> tuple[float, float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp, fp, fn = _confusion_counts(y_true, y_pred, labels)
    present = (tp + fn) > 0
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    if not present.any():
        return 0.0, 0.0, 0.0
    return float(prec[present].mean()), float(rec[present].mean()), float(f1[present].mean())


def auroc(y_true, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    try:
        if probs.ndim == 1 or probs.shape[1] == 2:
            score = probs if probs.ndim == 1 else probs[:, 1]
            return float(roc_auc_score(y_true, score))
        classes = np.unique(y_true)
        if len(classes) < 2:
            return float("nan")
        # macro one-vs-rest over the classes present in the evaluation set
        vals = [roc_auc_score((y_true == c).astype(int), probs[:, c]) for c in classes]
        return float(np.mean(vals))
    except ValueError:
        return float("nan")


def classification_metrics(y_true, y_pred, probs: np.ndarray | None = None,
                           labels: Sequence[int] | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(np.unique(y_true)) if labels is None else list(labels)
    p, r, f = macro_prf(y_true, y_pred, labels)
    out = {"accuracy": float(np.mean(y_true == y_pred)), "macro_precision": p, "macro_recall": r, "macro_f1": f}
    if probs is not None:
        out["auroc"] = auroc(y_true, probs)
        valid = (y_true >= 0) & (y_true < probs.shape[1])
        out["ece"] = expected_calibration_error(probs[valid], y_true[valid])
        out["brier"] = brier_score(probs[valid], y_true[valid])
    return out


def h_score(known_f1: float, unknown_recall: float) -> float:
    if known_f1 + unknown_recall <= 0:
        return 0.0
    return 2.0 * known_f1 * unknown_recall / (known_f1 + unknown_recall)


def open_world_metrics(y_true, y_pred, known_classes: Sequence[int], probs: np.ndarray | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _, _, kf1 = macro_prf(y_true, y_pred, known_classes)
    unk = y_true == UNKNOWN
    rej = y_pred == UNKNOWN
    tp_u = float(np.sum(unk & rej))
    out = {
        "known_macro_f1": kf1,
        "unknown_precision": tp_u / max(1.0, float(rej.sum())),
        "unknown_recall": tp_u / max(1.0, float(unk.sum())),
        "review_rate": float(rej.mean()),
        "false_review_known": float(np.mean(rej[~unk])) if (~unk).any() else 0.0,
    }
    out["h_score"] = h_score(out["known_macro_f1"], out["unknown_recall"])
    if probs is not None and (~unk).any():
        out["ece"] = expected_calibration_error(probs[~unk], y_true[~unk])
        out["brier"] = brier_score(probs[~unk], y_true[~unk])
    return out


def custody_metrics(state_true, state_pred, compromised_score, loc_scores: Sequence[np.ndarray] | None,
                    affected: Sequence[Sequence[int]] | None, intact_index: int = 0) -> dict:
    st, sp = np.asarray(state_true), np.asarray(state_pred)
    comp_t = st != intact_index
    comp_p = sp != intact_index
    out = {
        "integrity_accuracy": float(np.mean(st == sp)),
        "binary_integrity_accuracy": float(np.mean(comp_t == comp_p)),
        "violation_recall": float(np.mean(comp_p[comp_t])) if comp_t.any() else float("nan"),
        "intact_fpr": float(np.mean(comp_p[~comp_t])) if (~comp_t).any() else float("nan"),
    }
    score = np.asarray(compromised_score, dtype=float)
    if comp_t.any() and (~comp_t).any():
        out["auroc"] = float(roc_auc_score(comp_t.astype(int), score))
        out["auprc"] = float(average_precision_score(comp_t.astype(int), score))
    if loc_scores is not None and affected is not None:
        hits = []
        for i in np.where(comp_t)[0]:
            s = np.asarray(loc_scores[i])
            if len(s) == 0 or not len(affected[i]):
                hits.append(0.0)
                continue
            hits.append(float(int(np.argmax(s)) in set(affected[i])))
        out["localization_accuracy"] = float(np.mean(hits)) if hits else float("nan")
    return out


def localization_hits(loc_scores: Sequence[np.ndarray], affected: Sequence[Sequence[int]]) -> np.ndarray:
    return np.array([float(len(s) > 0 and len(a) > 0 and int(np.argmax(s)) in set(a))
                     for s, a in zip(loc_scores, affected)])
