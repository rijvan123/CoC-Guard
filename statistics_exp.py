"""Table XII: paired statistical comparisons with prediction-level bootstrap inference.

Reads the per-observation predictions saved by the other experiments and
computes, for each comparison, the difference in points, the 95% percentile
interval from a paired stratified bootstrap (10,000 replicates; malware
comparisons stratified by family, custody comparisons by intact or violation
class, intervention comparisons by reference intervention type), McNemar's test
where decisions are paired binary outcomes, and Holm-adjusted p-values within
each family of comparisons. The replay comparison uses the exploratory exact
Wilcoxon signed-rank test over matched seeds.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.metrics import UNKNOWN, expected_calibration_error, h_score, macro_prf
from ..evaluation.statistics import holm, mcnemar, paired_bootstrap, wilcoxon_exact
from .common import base_parser, setup


def _load(pattern: str) -> dict:
    out = {}
    for f in sorted(glob.glob(pattern)):
        seed = int(Path(f).stem.split("seed")[-1])
        out[seed] = dict(np.load(f, allow_pickle=True))
    return out


def _pool(a: dict, b: dict, keys):
    seeds = sorted(set(a) & set(b))
    A = {k: np.concatenate([a[s][k] for s in seeds]) for k in keys if all(k in a[s] for s in seeds)}
    Bd = {k: np.concatenate([b[s][k] for s in seeds]) for k in keys if all(k in b[s] for s in seeds)}
    return A, Bd, seeds


def comparison(name, family, test, res):
    return {"comparison": name, "family": family, "test": test, "difference_points": 100 * res["difference"],
            "ci_low": 100 * res["ci_low"], "ci_high": 100 * res["ci_high"], "p_value": res["p_value"]}


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--results", default=None, help="root of the results directories")
    args = p.parse_args(argv)
    cfg, out, device, seeds = setup(args, "statistics")
    root = Path(args.results or cfg.paths.results)
    B = int(cfg.evaluation.bootstrap_replicates)
    rows, extra = [], []

    # 1) complete vs direct concatenation: CAPE macro F1 (family-stratified)
    a, b = _load(str(root / "closed_world/predictions/cape_cocguard_seed*.npz")), _load(str(root / "closed_world/predictions/cape_concat_seed*.npz"))
    if a and b:
        A, Bd, _ = _pool(a, b, ["y_true", "y_pred"])
        labels = sorted(np.unique(A["y_true"]))
        f = lambda P: (lambda idx: macro_prf(P["y_true"][idx], P["y_pred"][idx], labels)[2])
        res = paired_bootstrap(f(A), f(Bd), A["y_true"], B)
        rows.append(comparison("Complete vs direct concatenation - CAPE macro F1", "principal",
                               "Family-stratified paired bootstrap", res))
        extra.append({"comparison": "CAPE correctness (McNemar)", **mcnemar(A["y_true"] == A["y_pred"], Bd["y_true"] == Bd["y_pred"])})

    # 2) complete vs without RCF: unknown recall; 6) strong open-world baselines
    ow = root / "open_world/predictions"
    def ow_load(method):
        out_ = {}
        for f in sorted(glob.glob(str(ow / f"ordinary_*_{method}_seed*.npz"))):
            seed = int(Path(f).stem.split("seed")[-1])
            d = dict(np.load(f, allow_pickle=True))
            out_.setdefault(seed, []).append(d)
        return {s: {k: np.concatenate([d[k] for d in ds]) for k in ("y_true", "y_pred", "probs", "family") if k in ds[0]}
                for s, ds in out_.items()}
    cg = ow_load("cocguard")
    def unknown_recall(P):
        return lambda idx: float(np.mean(P["y_pred"][idx][P["y_true"][idx] == UNKNOWN] == UNKNOWN)) if np.any(P["y_true"][idx] == UNKNOWN) else np.nan
    def hs(P):
        def fn(idx):
            yt, yp = P["y_true"][idx], P["y_pred"][idx]
            known = sorted(set(np.unique(yt[yt >= 0]).tolist()))
            ur = float(np.mean(yp[yt == UNKNOWN] == UNKNOWN)) if np.any(yt == UNKNOWN) else np.nan
            return h_score(macro_prf(yt, yp, known)[2], ur)
        return fn
    def ece(P):
        return lambda idx: expected_calibration_error(P["probs"][idx][P["y_true"][idx] >= 0], P["y_true"][idx][P["y_true"][idx] >= 0])
    for other, label, family, metrics in (("concat", "without RCF", "principal", ("unknown recall",)),
                                          ("domr", "DOMR-style", "strong open-world baselines", ("unknown recall", "H-score")),
                                          ("mecha", "Mecha-style", "strong open-world baselines", ("unknown recall", "H-score", "ECE"))):
        o = ow_load(other)
        if not cg or not o:
            continue
        A, Bd, _ = _pool(cg, o, ["y_true", "y_pred", "probs", "family"])
        strata = A["family"]
        for metric in metrics:
            fn = {"unknown recall": unknown_recall, "H-score": hs, "ECE": ece}[metric]
            res = paired_bootstrap(fn(A), fn(Bd), strata, B)
            rows.append(comparison(f"CoC-Guard vs {label} - {metric}", family, "Family-stratified paired bootstrap", res))

    # 3) complete vs rule-only: custody accuracy (class-stratified)
    a, b = _load(str(root / "custody/predictions/custody_cocguard_seed*.npz")), _load(str(root / "custody/predictions/custody_rule_only_seed*.npz"))
    if a and b:
        A, Bd, _ = _pool(a, b, ["y_true", "y_pred"])
        acc = lambda P: (lambda idx: float(np.mean(P["y_true"][idx] == P["y_pred"][idx])))
        res = paired_bootstrap(acc(A), acc(Bd), (A["y_true"] > 0).astype(int), B)
        rows.append(comparison("Complete vs rule-only - custody accuracy", "principal", "History-level paired bootstrap", res))
        extra.append({"comparison": "custody correctness (McNemar)", **mcnemar(A["y_true"] == A["y_pred"], Bd["y_true"] == Bd["y_pred"])})

    # 4) responsibility vs SHAP: recovery recall (intervention-stratified)
    a, b = _load(str(root / "counterfactual/predictions/cf_responsibility_seed*.npz")), _load(str(root / "counterfactual/predictions/cf_shap_seed*.npz"))
    if a and b:
        A, Bd, _ = _pool(a, b, ["per_case"])
        ra, rb = A["per_case"], Bd["per_case"]
        rec = lambda R: (lambda idx: float(R[idx, 1].sum() / max(1.0, R[idx, 0].sum())))
        res = paired_bootstrap(rec(ra), rec(rb), np.zeros(len(ra), int), B)
        rows.append(comparison("Responsibility vs SHAP - recovery recall", "principal", "Intervention-stratified bootstrap", res))

    # 5) replay vs no replay: exploratory exact Wilcoxon over matched seeds
    runs = root / "chronological/runs.csv"
    if runs.exists():
        df = pd.read_csv(runs)
        a_ = df[df.method == "Complete CoC-Guard"].sort_values("seed")["total_reduction"].to_numpy()
        b_ = df[df.method == "Without Replay Memory"].sort_values("seed")["total_reduction"].to_numpy()
        if len(a_) and len(a_) == len(b_):
            w = wilcoxon_exact(a_, b_)
            rows.append({"comparison": "Replay vs no replay - chronological reduction", "family": "exploratory",
                         "test": "Exact Wilcoxon over matched seeds (exploratory)", "difference_points": 100 * float(np.mean(a_ - b_)),
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": w["p_value"]})
    table = pd.DataFrame(rows)
    if len(table):
        table["holm_p"] = np.nan
        for fam in table["family"].unique():
            if fam == "exploratory":
                table.loc[table.family == fam, "holm_p"] = table.loc[table.family == fam, "p_value"]
                continue
            sel = table.family == fam
            table.loc[sel, "holm_p"] = holm(table.loc[sel, "p_value"].to_numpy())
    table.to_csv(out / "table_XII.csv", index=False)
    pd.DataFrame(extra).to_csv(out / "mcnemar.csv", index=False)
    print(table.to_string(index=False) if len(table) else "no saved predictions found")


if __name__ == "__main__":
    main()
