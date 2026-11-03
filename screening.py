"""Open-World Screening and decision calibration (Eqs. 19-21).

    s^u_i   = lambda_p (1 - max_c p^m_{i,c}) + lambda_d d^rep_i + lambda_x d^TG_i,   sum lambda = 1    (19)
    d^rep_i = min_c (z^m_i - mu_c)^T Sigma_c^{-1} (z^m_i - mu_c) / (kappa_c + eps)                       (20)
    y^m_i   = unknown/review if s^u_i > tau_u, else argmax_c p^m_{i,c}                                   (21)

Prototypes ``mu_c``, regularised covariances ``Sigma_c`` (Ledoit-Wolf, shrunk
towards the pooled covariance) and normalisers ``kappa_c`` (a quantile of the
development distances of class c) are derived from the development partition.
When the cross-view term is unavailable (a static track, or an investigation
with a single available branch) it is masked and lambda_p, lambda_d are
renormalised to sum to one. The withheld target family never contributes to
any of these quantities; ``tau_u`` and the coefficients are selected on
pseudo-unknown development families by maximising the known-unknown H-score.
An unknown/review outcome means insufficient support for a known-family
assignment; it does not establish that a new family exists.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.covariance import LedoitWolf

from ..evaluation.metrics import UNKNOWN, h_score, macro_prf


class OpenWorldScreener:
    def __init__(self, lambda_p: float = 0.40, lambda_d: float = 0.35, lambda_x: float = 0.25, tau_u: float = 0.81,
                 kappa_quantile: float = 0.95, shrinkage: float = 0.10, eps: float = 1e-6):
        self.lambdas = np.array([lambda_p, lambda_d, lambda_x], dtype=float)
        if abs(self.lambdas.sum() - 1.0) > 1e-6 or (self.lambdas < 0).any():
            raise ValueError("lambda_p + lambda_d + lambda_x must equal 1 and be non-negative")
        self.tau_u = float(tau_u)
        self.kappa_quantile = kappa_quantile
        self.shrinkage = shrinkage
        self.eps = eps
        self.mu = self.prec = self.kappa = None
        self.classes: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _class_models(self, z: np.ndarray, y: np.ndarray, classes) -> tuple[np.ndarray, np.ndarray]:
        """Prototype and regularised precision of every class (pooled fallback for tiny classes)."""
        d = z.shape[1]
        centred = [z[y == c] - z[y == c].mean(0) for c in classes if (y == c).sum() >= 2]
        centred = np.concatenate(centred) if centred else np.zeros((0, d))
        pooled = LedoitWolf().fit(centred).covariance_ if len(centred) > 2 else np.eye(d)
        mus, precs = [], []
        for c in classes:
            zc = z[y == c]
            mu = zc.mean(0) if len(zc) else np.zeros(d)
            if len(zc) >= 5:
                cov = (1 - self.shrinkage) * LedoitWolf().fit(zc).covariance_ + self.shrinkage * pooled
            else:
                cov = pooled.copy()
            cov = cov + 1e-4 * max(np.trace(cov) / d, 1e-8) * np.eye(d)
            mus.append(mu)
            precs.append(np.linalg.pinv(cov))
        return np.stack(mus), np.stack(precs)

    @staticmethod
    def _mahalanobis(z, mu, prec) -> np.ndarray:
        diff = z - mu
        return np.einsum("nd,de,ne->n", diff, prec, diff)

    def fit_prototypes(self, z: np.ndarray, y: np.ndarray, classes=None, folds: int = 2,
                       seed: int = 11) -> "OpenWorldScreener":
        """Development-derived prototypes with cross-fitted normalisers.

        mu_c and Sigma_c use all development observations of class c. kappa_c is
        the ``kappa_quantile`` of *out-of-fold* distances (each development
        observation is measured against prototypes fitted on the other folds),
        which removes the in-sample optimism of measuring observations against
        a prototype they helped to build. The out-of-fold distance matrix is
        kept for development-only threshold selection (:meth:`oof_d_rep`).
        """
        z = np.asarray(z, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y[y >= 0]) if classes is None else np.asarray(classes)
        for c in classes:
            if not (y == c).any():
                raise ValueError(f"class {c} has no development samples for its prototype")
        self.mu, self.prec = self._class_models(z, y, classes)
        rng = np.random.default_rng(seed)
        fold = np.full(len(y), -1)
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            fold[idx] = np.arange(len(idx)) % folds
        oof = np.full((len(y), len(classes)), np.nan)
        for f in range(folds):
            tr, te = (fold != f) & (y >= 0), fold == f
            if not te.any() or not tr.any():
                continue
            mu_f, prec_f = self._class_models(z[tr], y[tr], classes)
            for k, c in enumerate(classes):
                if (y[tr] == c).any():
                    oof[te, k] = self._mahalanobis(z[te], mu_f[k], prec_f[k])
        for k, c in enumerate(classes):        # classes too small to cross-fit fall back to in-sample distances
            miss = np.isnan(oof[:, k])
            if miss.any():
                oof[miss, k] = self._mahalanobis(z[miss], self.mu[k], self.prec[k])
        own = np.array([oof[i, list(classes).index(c)] if c in classes else np.nan for i, c in enumerate(y)])
        pooled_q = float(np.nanquantile(own, self.kappa_quantile)) if np.isfinite(own).any() else 1.0
        kappas = []
        for k, c in enumerate(classes):
            dc = own[y == c]
            kappas.append(float(np.quantile(dc, self.kappa_quantile)) if len(dc) >= 5 else max(pooled_q, float(np.max(dc))))
        self.kappa = np.maximum(np.array(kappas), 1e-6)
        self.classes = classes
        self._oof = oof
        return self

    def oof_d_rep(self, exclude: int | None = None) -> np.ndarray:
        """Eq. (20) for the development observations, using out-of-fold distances."""
        keep = [k for k, c in enumerate(self.classes) if exclude is None or c != exclude]
        return (self._oof[:, keep] / (self.kappa[keep] + self.eps)).min(axis=1)

    def d_rep(self, z: np.ndarray, exclude: int | None = None) -> np.ndarray:
        """Eq. (20); ``exclude`` drops one prototype (pseudo-unknown simulation)."""
        z = np.asarray(z, dtype=np.float64)
        out = np.full(len(z), np.inf)
        for k, c in enumerate(self.classes):
            if exclude is not None and c == exclude:
                continue
            diff = z - self.mu[k]
            m = np.einsum("nd,de,ne->n", diff, self.prec[k], diff) / (self.kappa[k] + self.eps)
            out = np.minimum(out, m)
        return out

    @staticmethod
    def components(probs: np.ndarray, d_rep: np.ndarray, d_tg: np.ndarray | None) -> np.ndarray:
        conf_def = 1.0 - np.asarray(probs).max(axis=1)
        dtg = np.zeros_like(conf_def) if d_tg is None else np.asarray(d_tg, dtype=float)
        return np.stack([conf_def, np.asarray(d_rep, dtype=float), dtg], axis=1)

    def score_components(self, comps: np.ndarray, cross_view: np.ndarray | None = None,
                         lambdas: np.ndarray | None = None) -> np.ndarray:
        """Eq. (19) with masking and renormalisation of the cross-view term."""
        lam = self.lambdas if lambdas is None else np.asarray(lambdas, dtype=float)
        n = len(comps)
        cv = np.ones(n, bool) if cross_view is None else np.asarray(cross_view, bool)
        w = np.tile(lam, (n, 1))
        w[~cv, 2] = 0.0
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        return (w * comps).sum(axis=1)

    def score(self, probs: np.ndarray, z: np.ndarray, d_tg: np.ndarray | None = None,
              cross_view: np.ndarray | None = None) -> np.ndarray:
        if cross_view is None and d_tg is None:
            cross_view = np.zeros(len(probs), bool)
        return self.score_components(self.components(probs, self.d_rep(z), d_tg), cross_view)

    def decide(self, probs: np.ndarray, s_u: np.ndarray, tau_u: float | None = None) -> np.ndarray:
        tau = self.tau_u if tau_u is None else tau_u
        pred = np.asarray(probs).argmax(axis=1)
        if self.classes is not None:
            pred = self.classes[pred] if len(self.classes) == np.asarray(probs).shape[1] else pred
        return np.where(np.asarray(s_u) > tau, UNKNOWN, pred)

    # ------------------------------------------------------------------
    def select(self, comps_known: np.ndarray, y_known: np.ndarray, pred_known: np.ndarray,
               comps_unknown: np.ndarray, pred_unknown: np.ndarray, cv_known=None, cv_unknown=None,
               tau_grid=(0.40, 0.99, 0.01), lambda_step: float = 0.05, search_lambdas: bool = True) -> dict:
        """Select (lambda, tau_u) on development data by maximising the H-score.

        ``*_known`` describe known-family development observations and
        ``*_unknown`` the pseudo-unknown development families (never the
        withheld target family). Accepted pseudo-unknown observations count as
        false positives of the known class they were assigned to.
        """
        taus = np.arange(tau_grid[0], tau_grid[1] + 1e-9, tau_grid[2])
        if search_lambdas:
            steps = np.round(np.arange(0.0, 1.0 + 1e-9, lambda_step), 6)
            grid = [np.array([a, b, round(1 - a - b, 6)]) for a, b in itertools.product(steps, steps) if a + b <= 1 + 1e-9]
        else:
            grid = [self.lambdas]
        classes = np.unique(y_known)
        y_all = np.concatenate([y_known, np.full(len(comps_unknown), UNKNOWN)])
        best = {"h_score": -1.0}
        for lam in grid:
            sk = self.score_components(comps_known, cv_known, lam)
            su = self.score_components(comps_unknown, cv_unknown, lam)
            for tau in taus:
                pk = np.where(sk > tau, UNKNOWN, pred_known)
                pu = np.where(su > tau, UNKNOWN, pred_unknown)
                kf1 = macro_prf(y_all, np.concatenate([pk, pu]), classes)[2]
                ur = float(np.mean(su > tau)) if len(su) else 0.0
                h = h_score(kf1, ur)
                if h > best["h_score"] + 1e-12:
                    best = {"h_score": h, "known_macro_f1": kf1, "unknown_recall": ur, "tau_u": float(tau),
                            "lambda_p": float(lam[0]), "lambda_d": float(lam[1]), "lambda_x": float(lam[2])}
        return best

    def apply_selection(self, sel: dict) -> None:
        self.lambdas = np.array([sel["lambda_p"], sel["lambda_d"], sel["lambda_x"]], dtype=float)
        self.tau_u = float(sel["tau_u"])

    def state_dict(self) -> dict:
        return {"lambdas": self.lambdas.tolist(), "tau_u": self.tau_u, "kappa_quantile": self.kappa_quantile,
                "shrinkage": self.shrinkage, "classes": None if self.classes is None else self.classes.tolist(),
                "kappa": None if self.kappa is None else self.kappa.tolist()}
