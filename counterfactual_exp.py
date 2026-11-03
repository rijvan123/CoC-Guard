"""Table VIII / Figure 9: report-level intervention verification on the CAPE track.

Protocol (Section IV-D-6):

1. A frozen, calibrated CoC-Guard model (re-used from the closed-world run when a
   checkpoint exists) and isolated analytical copies of CAPE event sequences and
   evidence graphs; executables are never re-executed.
2. Reference positives: for each evaluated test investigation, exhaustive
   constrained intervention over all events; an event is a reference positive
   when its valid intervention has a mean effect >= 0.10 and stability >= 0.80.
   The reference set is frozen before any selection method is evaluated.
3. Each selection method (random, attention-ranked, SHAP-ranked,
   responsibility-guided) examines candidates in its own order with a budget of
   |reference positives| + ``budget`` candidates. Reported: intervention-recovery
   recall, validity rate, mean stability, unverifiable share and mean number of
   intervention evaluations per verified explanation.
4. Responsibility-guided selection is repeated under five challenging
   conditions, and without validity constraints (unconstrained deletion) for
   the ablation of Table X.

The verified claim is restricted to intervention-supported influence on the
frozen model under the defined report-level intervention.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..counterfactual.interventions import candidate_factors, neutral_templates
from ..counterfactual.selection import METHODS, rank_candidates
from ..counterfactual.verification import CounterfactualVerifier
from ..data.schema import TrackSpec
from ..models.cocguard import CoCGuard
from .common import aggregate, base_parser, record, save_predictions, save_rows, seed_everything, setup, write_table
from .conditions import INTERVENTION_CONDITIONS
from .runners import calibrate, train_model
from .tracks import InvestigationTrack, split

PCT = ("recovery_recall", "validity", "stability", "unverifiable")


def make_verifier(model, device, cfg, templates, constrained=True, noise=0.0):
    c = cfg.counterfactual
    return CounterfactualVerifier(model, device, int(c.repetitions), float(c.min_effect), float(c.min_stability),
                                  float(c.min_direction), float(c.beta_m), bool(c.mc_dropout),
                                  float(c.timestamp_jitter_seconds), constrained, templates, realization_noise=noise)


def reference_positives(verifier, sample, seed) -> tuple[set, dict]:
    """Exhaustive constrained intervention over every event and relation factor (frozen reference set)."""
    original = verifier.original(sample)
    res = verifier.verify_many(sample, candidate_factors(sample), "mask", seed, original)
    ref = {r["candidate"] for r in res if r["valid"] and r["delta"] >= verifier.min_effect and r["stability"] >= verifier.min_stability}
    return ref, original


def evaluate_selection(model, device, cfg, samples, refs, method, seed, templates, kind="mask", constrained=True,
                       noise=0.0) -> dict:
    budget_extra = int(cfg.counterfactual.budget)
    verifier = make_verifier(model, device, cfg, templates, constrained, noise)
    rng = np.random.default_rng(seed)
    tot_ref = tot_rec = evaluated = valid = verified = 0
    stab, per_case = [], []
    t0 = time.perf_counter()
    for s, (ref, original) in zip(samples, refs):
        factors = candidate_factors(s)
        order = rank_candidates(method, model, s, device, rng, int(cfg.counterfactual.shap_permutations), "mask", factors)
        k = len(ref) + budget_extra
        res = verifier.verify_many(s, [factors[i] for i in order[:k]], kind, seed, original)
        for r in res:
            evaluated += 1
            if r["valid"] or not constrained:
                valid += int(r["valid"] or not constrained)
                stab.append(r["stability"])
            if r["verified"]:
                verified += 1
        rec = {r["candidate"] for r in res if r["verified"]}
        tot_ref += len(ref)
        tot_rec += len(rec & ref)
        per_case.append((len(ref), len(rec & ref), len(res), sum(r["valid"] for r in res)))
    return {"recovery_recall": tot_rec / max(1, tot_ref), "validity": valid / max(1, evaluated),
            "stability": float(np.mean(stab)) if stab else 0.0, "unverifiable": 1.0 - valid / max(1, evaluated),
            "mean_evaluations": evaluated / max(1, verified), "verified": verified, "evaluated": evaluated,
            "reference_positives": tot_ref, "seconds_per_case": (time.perf_counter() - t0) / max(1, len(samples)),
            "per_case": np.array(per_case, dtype=float).reshape(-1, 4)}


def load_or_train(track, cfg, device, seed, P, spec, ckpt_dir):
    path = ckpt_dir / f"cape_cocguard_seed{seed}.pt" if ckpt_dir is not None else None
    model = CoCGuard(spec, cfg.model)
    if path is not None and path.exists():
        state = torch.load(path, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state)
            return model.to(device)
        except RuntimeError:
            pass  # architecture differs (e.g. other configuration): retrain
    train_model("cocguard", model, P["train"], P["dev"], cfg, device, seed)
    calibrate(model, P["dev"], device)
    return model


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--conditions", nargs="*", default=list(INTERVENTION_CONDITIONS))
    p.add_argument("--checkpoints", default=None, help="directory with closed-world CAPE checkpoints")
    p.add_argument("--no-ablation", action="store_true")
    args = p.parse_args(argv)
    cfg, out, device, seeds = setup(args, "counterfactual")
    from pathlib import Path
    track = InvestigationTrack(args.data or f"{cfg.paths.processed}/cape")
    part = track.column("partition")
    samples, spec, _ = track.build(part == "train")
    P = split(samples, part)
    ckpt = Path(args.checkpoints) if args.checkpoints else Path(cfg.paths.results) / "closed_world" / "checkpoints"
    templates = neutral_templates(P["train"])
    rows: list = []
    for seed in seeds:
        seed_everything(cfg, seed)
        model = load_or_train(track, cfg, device, seed, P, spec, ckpt)
        rng = np.random.default_rng(seed)
        n = min(int(cfg.counterfactual.max_investigations), len(P["test"]))
        chosen = [P["test"][i] for i in sorted(rng.choice(len(P["test"]), size=n, replace=False))]
        ref_verifier = make_verifier(model, device, cfg, templates)
        refs = [reference_positives(ref_verifier, s, seed) for s in chosen]
        for m in args.methods:
            met = evaluate_selection(model, device, cfg, chosen, refs, m, seed, templates)
            save_predictions(out, f"cf_{m}", seed, per_case=met.pop("per_case"))
            record(rows, {"random": "Random", "attention": "Attention-ranked", "shap": "SHAP-ranked",
                          "responsibility": "Responsibility-guided"}[m], seed, met, condition="standard")
        for cond in args.conditions:
            fn = INTERVENTION_CONDITIONS[cond]
            noise = 0.15 if cond == "nondeterministic_realization" else 0.0
            crng = np.random.default_rng([seed, 99])
            cs = chosen if fn is None else [fn(s, crng) for s in chosen]
            cref_verifier = make_verifier(model, device, cfg, templates, noise=noise)
            crefs = [reference_positives(cref_verifier, s, seed) for s in cs]
            met = evaluate_selection(model, device, cfg, cs, crefs, "responsibility", seed, templates, noise=noise)
            met.pop("per_case", None)
            record(rows, "Responsibility-guided", seed, met, condition=cond)
        if not args.no_ablation:
            met = evaluate_selection(model, device, cfg, chosen, refs, "responsibility", seed, templates, kind="delete",
                                     constrained=False)
            met.pop("per_case", None)
            record(rows, "Responsibility-guided without validity constraints", seed, met, condition="ablation")
        save_rows(rows, out)
    df = aggregate(rows, ("condition", "method"), PCT)
    write_table(df, out, "table_VIII", "Table VIII. Report-level intervention verification (%)",
                ["condition", "method", "recovery_recall", "validity", "stability", "unverifiable", "mean_evaluations"])
    return df


if __name__ == "__main__":
    main()
