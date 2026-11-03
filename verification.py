"""Repeated, seeded counterfactual verification (Eqs. 26-28).

    Delta_{i,j} = 1/R sum_r [ D(p^c_i, p~^c_{i,j,r}) + beta_m D(p^m_i, p~^m_{i,j,r}) ]                 (26)
    S_{i,j}     = clip_[0,1][ 1 - SD(Delta^(1..R)) / (|Mean(Delta^(1..R))| + eps) ]                      (27)
    r_{i,j}     = V_{i,j} Delta_{i,j} S_{i,j} Q_{i,j}                                                     (28)

D is the Jensen-Shannon divergence. Realisations are *paired*: in realisation r
the original and the intervened copy are evaluated under the same seeded
stochastic state of the frozen model (MC dropout) and the same report-level
realisation noise, so Delta^(r) isolates the effect of the intervention and
S measures how consistently that effect reproduces. Q is the fraction of realisations whose
change has the expected direction (removing supporting behaviour lowers the
probability of the predicted family; correcting a suspected custody
transaction raises the probability of the intact state). A candidate is
*counterfactually verified* only when it is valid, Delta >= 0.10,
S >= 0.80 and Q >= the directional threshold. Invalid interventions are
reported as unverifiable, never as zero-effect explanations. The verified
claim is restricted to stable influence under the defined report-level
intervention on the frozen model; it is not a causal, behavioural or legal
attribution.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.collate import collate_investigations
from ..models.cocguard import js_divergence, mc_dropout
import zlib

from .interventions import apply_intervention, as_factor, validity


class CounterfactualVerifier:
    def __init__(self, model, device, repetitions: int = 5, min_effect: float = 0.10, min_stability: float = 0.80,
                 min_direction: float = 0.80, beta_m: float = 0.30, use_mc_dropout: bool = True,
                 jitter_seconds: float = 1.0, constrained: bool = True, templates: np.ndarray | None = None,
                 eps: float = 1e-6, realization_noise: float = 0.0):
        self.model, self.device = model, device
        self.R = int(repetitions)
        self.min_effect, self.min_stability, self.min_direction = min_effect, min_stability, min_direction
        self.beta_m, self.use_mc_dropout, self.jitter = beta_m, use_mc_dropout, jitter_seconds
        self.constrained, self.templates, self.eps = constrained, templates, eps
        self.realization_noise = realization_noise
        self.evaluations = 0

    @torch.no_grad()
    def _forward(self, samples: list[dict], seed: int | None = None, stochastic: bool = False) -> dict:
        batch = collate_investigations(samples).to(self.device)
        if stochastic and self.use_mc_dropout:
            torch.manual_seed(int(seed or 0))
            with mc_dropout(self.model):
                out = self.model(batch)
        else:
            self.model.eval()
            out = self.model(batch)
        return {k: out[k].detach().cpu() for k in ("probs_m", "probs_c") if k in out}

    def original(self, sample: dict) -> dict:
        return {k: v[0] for k, v in self._forward([sample]).items()}

    def _realise(self, sample: dict, factor: dict, kind: str, seed: int, r: int) -> dict:
        rng = np.random.default_rng([int(seed), zlib.crc32(str(factor["id"]).encode()), int(r)])
        c = apply_intervention(sample, factor, kind, rng, self.templates, self.jitter)
        if self.realization_noise > 0 and len(c["v"]) > 2:
            # nondeterministic realisation: neighbouring events may be recorded in swapped order
            for t in range(1, len(c["v"]) - 1):
                if rng.random() < self.realization_noise:
                    for k in ("xb", "mb", "cat"):
                        c[k][[t, t + 1]] = c[k][[t + 1, t]]
        return c

    def verify(self, sample: dict, target, kind: str = "mask", seed: int = 0, original: dict | None = None) -> dict:
        return self.verify_many(sample, [target], kind, seed, original)[0]

    def verify_many(self, sample: dict, candidates, kind: str = "mask", seed: int = 0, original: dict | None = None,
                    chunk: int = 12) -> list[dict]:
        """Verify several candidate factors (events or relations) of one investigation."""
        original = original or self.original(sample)
        results: dict[str, dict] = {}
        pending = []
        factors = [as_factor(c) for c in candidates]
        for f in factors:
            copies = [self._realise(sample, f, kind, seed, r) for r in range(self.R)]
            valid, reason = validity(sample, copies[0], f, kind, self.constrained)
            rec = {"candidate": f["id"], "events": list(f["events"]), "level": f.get("level", "event"),
                   "intervention": kind, "valid": bool(valid), "reason": reason,
                   "repetitions": self.R, "effects": [], "delta": 0.0, "stability": 0.0, "direction": 0.0,
                   "responsibility": 0.0, "verified": False, "status": "unverifiable" if not valid else "not_verified"}
            results[f["id"]] = rec
            if valid:
                pending.append((f["id"], copies))
        for k in range(0, len(pending), chunk):
            block = pending[k:k + chunk]
            flat = [c for _, cs in block for c in cs]
            s_seed = int(np.random.default_rng([int(seed), k]).integers(0, 2**31 - 1))
            out = self._forward(flat, seed=s_seed, stochastic=True)
            if self.use_mc_dropout and all(c["xb"].shape == sample["xb"].shape for c in flat):
                # paired reference: the unmodified investigation under identical stochastic states
                ref = self._forward([sample] * len(flat), seed=s_seed, stochastic=True)
            else:
                ref = {key: v.unsqueeze(0).expand(len(flat), *v.shape) for key, v in original.items()}
            for b, (j, _) in enumerate(block):
                sl = slice(b * self.R, (b + 1) * self.R)
                self.evaluations += 1
                results[j].update(self._score({key: v[sl] for key, v in out.items()},
                                              {key: v[sl] for key, v in ref.items()}, kind, original))
        return [results[f["id"]] for f in factors]

    def _score(self, out: dict, ref: dict, kind: str, original: dict) -> dict:
        """Eqs. (26)-(28) from paired realisations (``ref`` row r pairs with ``out`` row r)."""
        effects = torch.zeros(self.R)
        direction = torch.zeros(self.R)
        if "probs_c" in out and "probs_c" in ref:
            pc0 = ref["probs_c"]
            effects += js_divergence(pc0, out["probs_c"])
            if kind == "correct":
                direction += (out["probs_c"][:, 0] > pc0[:, 0]).float()
        if "probs_m" in out and "probs_m" in ref:
            pm0 = ref["probs_m"]
            w = self.beta_m if "probs_c" in out else 1.0
            effects += w * js_divergence(pm0, out["probs_m"])
            if kind != "correct":
                yhat = int(original["probs_m"].argmax())
                direction += (out["probs_m"][:, yhat] < pm0[:, yhat]).float()
        eff = effects.numpy().astype(float)
        mean = float(eff.mean())
        sd = float(eff.std(ddof=1)) if self.R > 1 else 0.0
        stability = float(np.clip(1.0 - sd / (abs(mean) + self.eps), 0.0, 1.0))
        q = float(direction.mean())
        verified = mean >= self.min_effect and stability >= self.min_stability and q >= self.min_direction
        return {"effects": eff.tolist(), "delta": mean, "stability": stability, "direction": q,
                "responsibility": mean * stability * q, "verified": bool(verified),
                "status": "verified" if verified else "not_verified"}
