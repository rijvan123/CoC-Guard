"""Open-world comparators (Section IV-C, Table A5).

* Confidence-rejection MLP: reject when the maximum known-class probability is
  below a development-selected threshold (0.88 in the paper's configuration).
* OpenMax-MLP [21]: SciPy re-implementation of OSDN's OpenMax (eucos distance to
  per-class mean activation vectors, Weibull fit on the ``tail_size`` largest
  distances, recalibration of the top-``alpha`` activations); tail size 20, alpha 10.
* DOMR-style Episodic OVR [22]: matched-protocol re-implementation (no official
  code is available): 5-way / 5-shot / 15-query episodes with pseudo-unknown
  queries, prototype classification plus aggregated one-versus-rest scores with
  a 0.20 margin.
* Mecha-style Metric ANN [24]: matched-protocol re-implementation (no official
  code is available): triplet metric learning with semi-hard mining
  (margin 0.30, 128-dimensional embedding) and open-set k-nearest-neighbour
  matching (k = 10, cosine distance); the reinforcement-based refinement of the
  original system is approximated by development-selected radius calibration.

DOMR-style and Mecha-style replace their original Android / raw-byte encoders
with the same track-compatible encoder used by the other methods, so they are
methodological adaptations rather than reproductions of the original systems.
All rejection scores follow the convention "higher = less supported".
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import weibull_min
from torch import nn

from ...data.collate import collate_investigations


# ---------------------------------------------------------------------------
# OpenMax
# ---------------------------------------------------------------------------
class OpenMax:
    def __init__(self, tail_size: int = 20, alpha: int = 10):
        self.tail, self.alpha = tail_size, alpha
        self.mav = None
        self.weibull: list = []

    @staticmethod
    def eucos(av: np.ndarray, mav: np.ndarray) -> np.ndarray:
        eu = np.linalg.norm(av - mav, axis=-1) / 200.0
        cos = 1.0 - (av @ mav) / (np.linalg.norm(av, axis=-1) * np.linalg.norm(mav) + 1e-12)
        return eu + cos

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "OpenMax":
        C = logits.shape[1]
        pred = logits.argmax(1)
        self.mav = np.zeros((C, C))
        self.weibull = []
        for c in range(C):
            sel = (y == c) & (pred == c)
            if not sel.any():
                sel = y == c
            av = logits[sel]
            self.mav[c] = av.mean(0)
            d = np.sort(self.eucos(av, self.mav[c]))[-self.tail:]
            if len(d) >= 3 and np.ptp(d) > 1e-9:
                shape, loc, scale = weibull_min.fit(d, floc=0)
            else:
                shape, loc, scale = 1.0, 0.0, float(d.mean() + 1e-6)
            self.weibull.append((shape, loc, scale))
        return self

    def probabilities(self, logits: np.ndarray, exclude: int | None = None) -> np.ndarray:
        """Recalibrated probabilities over [known classes..., unknown]."""
        C = logits.shape[1]
        classes = [c for c in range(C) if c != exclude]
        av = logits[:, classes]
        alpha = min(self.alpha, len(classes))
        out = np.zeros((len(av), len(classes) + 1))
        for i in range(len(av)):
            v = av[i].copy()
            ranked = np.argsort(-v)
            w = np.zeros(len(classes))
            for r in range(alpha):
                k = ranked[r]
                c = classes[k]
                dist = self.eucos(logits[i:i + 1], self.mav[c])[0]
                wscore = weibull_min.cdf(dist, *self.weibull[c])
                w[k] = ((alpha - r) / alpha) * wscore
            revised = v * (1.0 - w)
            unknown = np.sum(v - revised)
            scores = np.concatenate([revised, [unknown]])
            scores = scores - scores.max()
            e = np.exp(scores)
            out[i] = e / e.sum()
        return out

    def score_and_pred(self, logits: np.ndarray, exclude: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        p = self.probabilities(logits, exclude)
        classes = np.array([c for c in range(logits.shape[1]) if c != exclude])
        known = p[:, :-1]
        score = np.maximum(p[:, -1], 1.0 - known.max(1))
        return score, classes[known.argmax(1)]


def confidence_score_and_pred(probs: np.ndarray, exclude: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    classes = np.array([c for c in range(probs.shape[1]) if c != exclude])
    p = probs[:, classes]
    p = p / p.sum(1, keepdims=True)
    return 1.0 - p.max(1), classes[p.argmax(1)]


# ---------------------------------------------------------------------------
# Metric heads on top of a track-compatible encoder
# ---------------------------------------------------------------------------
def _embed(backbone, samples, device, batch_size=128) -> np.ndarray:
    backbone.eval()
    zs = []
    with torch.no_grad():
        for k in range(0, len(samples), batch_size):
            out = backbone(collate_investigations(samples[k:k + batch_size]).to(device))
            zs.append(F.normalize(out["z_m"], dim=-1).cpu().numpy())
    return np.concatenate(zs) if zs else np.zeros((0, 1))


class EpisodicOVR:
    """DOMR-style episodic one-versus-rest classifier."""

    def __init__(self, backbone: nn.Module, n_classes: int, device, ways: int = 5, shots: int = 5, queries: int = 15,
                 margin: float = 0.20, lr: float = 3e-4, episodes: int = 200, seed: int = 11):
        self.backbone, self.C, self.device = backbone.to(device), n_classes, device
        self.ways, self.shots, self.queries, self.margin = ways, shots, queries, margin
        self.lr, self.episodes, self.seed = lr, episodes, seed
        self.scale = nn.Parameter(torch.tensor(10.0, device=device))
        self.bias = nn.Parameter(torch.full((n_classes,), 1.0 - margin, device=device))
        self.protos = None

    def _episode(self, by_class, rng):
        classes = [c for c, idx in by_class.items() if len(idx) >= 2]
        ways = min(self.ways, max(2, len(classes) - 1))
        chosen = rng.choice(classes, size=ways, replace=False)
        others = [c for c in classes if c not in chosen]
        sup, qry, qlab, unk = [], [], [], []
        for k, c in enumerate(chosen):
            idx = rng.choice(by_class[c], size=min(len(by_class[c]), self.shots + self.queries), replace=False)
            ns = max(1, min(self.shots, len(idx) - 1))
            sup.append(idx[:ns])
            qry += list(idx[ns:])
            qlab += [k] * (len(idx) - ns)
        if others:
            pool = np.concatenate([by_class[c] for c in others])
            unk = list(rng.choice(pool, size=min(len(pool), self.queries), replace=False))
        return chosen, sup, qry, qlab, unk

    def fit(self, train, dev=None):
        rng = np.random.default_rng(self.seed)
        y = np.array([s["y_m"] for s in train])
        by_class = {c: np.where(y == c)[0] for c in np.unique(y[y >= 0])}
        params = list(self.backbone.parameters()) + [self.scale, self.bias]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-5)
        self.backbone.train()
        self.history, block, t0 = [], [], time.time()
        every = max(1, self.episodes // 50)
        for ep in range(self.episodes):
            chosen, sup, qry, qlab, unk = self._episode(by_class, rng)
            flat_sup = np.concatenate(sup)
            batch_idx = list(flat_sup) + list(qry) + list(unk)
            out = self.backbone(collate_investigations([train[i] for i in batch_idx]).to(self.device))
            z = F.normalize(out["z_m"], dim=-1)
            ns = [len(s) for s in sup]
            zs, zq = z[: len(flat_sup)], z[len(flat_sup):]
            protos = torch.stack([F.normalize(t.mean(0), dim=-1) for t in torch.split(zs, ns)])
            cos = zq @ protos.T
            nq = len(qry)
            lab = torch.tensor(qlab, device=self.device)
            loss = F.cross_entropy(self.scale * cos[:nq], lab)
            target = torch.zeros_like(cos)
            target[torch.arange(nq), lab] = 1.0
            b = self.bias[torch.as_tensor(chosen, device=self.device)]
            loss = loss + F.binary_cross_entropy_with_logits(self.scale * (cos - b), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            block.append(float(loss.detach()))
            if (ep + 1) % every == 0 or ep + 1 == self.episodes:
                self.history.append({"epoch": ep, "stage": "episodic", "lr": self.lr,
                                     "train_loss": float(np.mean(block)), "elapsed_s": time.time() - t0})
                block = []
        z = _embed(self.backbone, train, self.device)
        self.protos = np.stack([z[y == c].mean(0) / max(1e-12, np.linalg.norm(z[y == c].mean(0))) for c in range(self.C)])
        return self

    def ovr_scores(self, samples) -> tuple[np.ndarray, np.ndarray]:
        z = _embed(self.backbone, samples, self.device)
        cos = z @ self.protos.T
        s = 1.0 / (1.0 + np.exp(-float(self.scale.detach()) * (cos - self.bias.detach().cpu().numpy())))
        return s, z

    @staticmethod
    def score_and_pred(ovr: np.ndarray, exclude: int | None = None):
        classes = np.array([c for c in range(ovr.shape[1]) if c != exclude])
        s = ovr[:, classes]
        return 1.0 - s.max(1), classes[s.argmax(1)]


class MetricANN:
    """Mecha-style triplet metric learning with open-set kNN matching."""

    def __init__(self, backbone: nn.Module, device, margin: float = 0.30, k: int = 10, lr: float = 3e-4,
                 epochs: int = 20, classes_per_batch: int = 8, per_class: int = 8, seed: int = 11):
        self.backbone, self.device = backbone.to(device), device
        self.margin, self.k, self.lr, self.epochs = margin, k, lr, epochs
        self.P, self.K, self.seed = classes_per_batch, per_class, seed
        self.bank_z = self.bank_y = None

    @staticmethod
    def semi_hard_triplet(z: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
        d = torch.cdist(z, z)
        same = y.unsqueeze(0) == y.unsqueeze(1)
        eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
        losses = []
        for a in range(len(y)):
            pos = (same[a] & ~eye[a]).nonzero(as_tuple=True)[0]
            neg = (~same[a]).nonzero(as_tuple=True)[0]
            if not len(pos) or not len(neg):
                continue
            for p in pos:
                dap = d[a, p]
                dn = d[a, neg]
                semi = dn[(dn > dap) & (dn < dap + margin)]
                dan = semi.min() if len(semi) else dn.min()
                losses.append(F.relu(dap - dan + margin))
        return torch.stack(losses).mean() if losses else z.sum() * 0.0

    def fit(self, train, dev=None):
        rng = np.random.default_rng(self.seed)
        y = np.array([s["y_m"] for s in train])
        classes = np.unique(y[y >= 0])
        by_class = {c: np.where(y == c)[0] for c in classes}
        opt = torch.optim.AdamW(self.backbone.parameters(), lr=self.lr, weight_decay=1e-5)
        steps = max(1, len(train) // (self.P * self.K))
        self.history, t0 = [], time.time()
        for ep in range(self.epochs):
            self.backbone.train()
            losses = []
            for _ in range(steps):
                cs = rng.choice(classes, size=min(self.P, len(classes)), replace=False)
                idx = np.concatenate([rng.choice(by_class[c], size=min(self.K, len(by_class[c])), replace=False) for c in cs])
                out = self.backbone(collate_investigations([train[i] for i in idx]).to(self.device))
                z = F.normalize(out["z_m"], dim=-1)
                loss = self.semi_hard_triplet(z, torch.as_tensor(y[idx], device=self.device), self.margin)
                if not loss.requires_grad:
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            self.history.append({"epoch": ep, "stage": "metric", "lr": self.lr,
                                 "train_loss": float(np.mean(losses)) if losses else float("nan"),
                                 "elapsed_s": time.time() - t0})
        self.bank_z = _embed(self.backbone, train, self.device)
        self.bank_y = y
        return self

    def neighbours(self, samples) -> tuple[np.ndarray, np.ndarray]:
        z = _embed(self.backbone, samples, self.device)
        sim = z @ self.bank_z.T
        return sim, z

    def score_and_pred(self, sim: np.ndarray, exclude: int | None = None, n_classes: int | None = None):
        keep = self.bank_y != exclude if exclude is not None else np.ones(len(self.bank_y), bool)
        s = sim[:, keep]
        yb = self.bank_y[keep]
        k = min(self.k, s.shape[1])
        nn_idx = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
        nn_sim = np.take_along_axis(s, nn_idx, axis=1)
        dist = 1.0 - nn_sim.mean(1)
        C = int(n_classes or (self.bank_y.max() + 1))
        votes = np.zeros((len(s), C))
        for j in range(k):
            np.add.at(votes, (np.arange(len(s)), yb[nn_idx[:, j]]), nn_sim[:, j])
        if exclude is not None:
            votes[:, exclude] = -np.inf
        return dist, votes.argmax(1)


def select_threshold_holdout(score_pred_fn, y_dev: np.ndarray, classes, tau_grid) -> dict:
    """Development threshold for a rejection score by pseudo-unknown hold-out.

    Each known class takes a turn as the pseudo-unknown family (it is removed
    from the decision space via ``score_pred_fn(exclude=f)``); the threshold
    maximising the mean known-unknown H-score over turns is returned.
    """
    from ...evaluation.metrics import UNKNOWN, h_score, macro_prf
    taus = np.asarray(tau_grid)
    totals = np.zeros(len(taus))
    rounds = 0
    for f in classes:
        unk = y_dev == f
        if not unk.any() or unk.all():
            continue
        score, pred = score_pred_fn(f)
        known_classes = [c for c in classes if c != f]
        y_true = np.where(unk, UNKNOWN, y_dev)
        for t, tau in enumerate(taus):
            yp = np.where(score > tau, UNKNOWN, pred)
            kf1 = macro_prf(y_true, yp, known_classes)[2]
            ur = float(np.mean(score[unk] > tau))
            totals[t] += h_score(kf1, ur)
        rounds += 1
    best = int(np.argmax(totals))
    return {"tau": float(taus[best]), "mean_h_score": float(totals[best] / max(1, rounds)), "rounds": rounds}
