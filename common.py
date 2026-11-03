"""Shared helpers for experiment scripts: CLI, result storage and aggregation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from ..config import Config, default_config_path, load_config, repo_root, save_config
from ..utils.io import environment_record, write_json
from ..utils.logging import get_logger
from ..utils.seed import set_seed

log = get_logger("experiments")


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", nargs="+", default=None, help="YAML file(s); later files override earlier ones")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override configuration values")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="subset of seeds (default: config seeds)")
    p.add_argument("--out", default=None, help="results directory (default: <paths.results>/<experiment>)")
    p.add_argument("--data", default=None, help="processed track directory")
    p.add_argument("--device", default=None)
    return p


def setup(args, experiment: str) -> tuple[Config, Path, torch.device, list[int]]:
    """Load the configuration and create the run directory with its provenance files.

    Every run directory receives ``config_used.yaml``, ``environment.json``
    (software versions, GPU, git commit), ``run.log`` (the complete console
    log) and ``training_logs/`` (one per-epoch CSV and one JSON summary per
    trained model).
    """
    cfg_paths = args.config or [default_config_path()]
    cfg = load_config(cfg_paths, args.set)
    if args.device:
        cfg.device = args.device
    out = Path(args.out) if args.out else Path(cfg.paths.results) / experiment
    out.mkdir(parents=True, exist_ok=True)
    attach_file_log(out / "run.log")
    from . import runners
    runners.set_training_log_dir(out / "training_logs")
    save_config(cfg, out / "config_used.yaml")
    write_json({**environment_record(), "experiment": experiment, "command_line": __import__("sys").argv,
                "config_files": [str(p) for p in cfg_paths], "overrides": list(args.set or [])}, out / "environment.json")
    dev = cfg.get("device", "auto")
    device = torch.device("cuda" if (dev == "auto" and torch.cuda.is_available()) else ("cpu" if dev == "auto" else dev))
    seeds = list(args.seeds) if args.seeds else list(cfg.seeds)
    log.info(f"{experiment}: device={device} seeds={seeds} out={out}")
    return cfg, out, device, seeds


def attach_file_log(path: Path) -> None:
    """Mirror the console log of the ``cocguard`` logger into ``path``."""
    import logging
    root = logging.getLogger("cocguard")
    path = Path(path).resolve()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path:
            return
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    root.addHandler(handler)


def seed_everything(cfg: Config, seed: int) -> None:
    set_seed(seed, bool(cfg.get("deterministic", False)))


def record(rows: list, method: str, seed: int, metrics: dict, **extra) -> None:
    rows.append({"method": method, "seed": seed, **extra, **{k: float(v) for k, v in metrics.items()
                                                            if isinstance(v, (int, float, np.floating, np.integer))}})


def save_predictions(out: Path, method: str, seed: int, **arrays) -> None:
    """Per-observation test outputs: ``.npz`` (all arrays) and ``.csv`` (one row per observation).

    The CSV keeps every one-dimensional field (true label, decision, scores,
    family, ...) and, for probability matrices, the predicted class and its
    calibrated probability.
    """
    stem = f"{method.replace(' ', '_').replace('/', '-')}_seed{seed}"
    path = out / "predictions" / f"{stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in arrays.items():
        if v is None:
            continue
        if isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            clean[k] = np.array(v, dtype=object)
        else:
            clean[k] = np.asarray(v)
    np.savez_compressed(path, **clean)
    n = max((len(v) for v in clean.values() if v.ndim >= 1 and v.dtype != object), default=0)
    cols = {}
    for k, v in clean.items():
        if v.dtype == object and v.ndim == 1 and len(v) == n and not isinstance(v[0], np.ndarray):
            cols[k] = v
        elif v.dtype != object and v.ndim == 1 and len(v) == n:
            cols[k] = v
        elif v.dtype != object and v.ndim == 2 and len(v) == n and k in ("probs", "probs_m", "probs_c"):
            cols[f"{k}_argmax"] = v.argmax(1)
            cols[f"{k}_max"] = v.max(1)
    if cols and n:
        pd.DataFrame(cols).to_csv(out / "predictions" / f"{stem}.csv", index_label="observation")


def aggregate(rows: list, group: Sequence[str] = ("method",), percent: Iterable[str] = ()) -> pd.DataFrame:
    """Mean and standard deviation across seeds (values in %, as in the paper, for ``percent`` columns)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metrics = [c for c in df.columns if c not in set(group) | {"seed"} and pd.api.types.is_numeric_dtype(df[c])]
    pct = set(percent)
    for c in metrics:
        if c in pct:
            df[c] = df[c] * 100.0
    g = df.groupby(list(group), sort=False)[metrics]
    mean, sd = g.mean(), g.std(ddof=1).fillna(0.0)
    out = mean.copy()
    for c in metrics:
        out[c + "_sd"] = sd[c]
    out["n_seeds"] = g.size()
    return out.reset_index()


def write_table(df: pd.DataFrame, out: Path, name: str, title: str, columns: Sequence[str] | None = None,
                decimals: int = 2) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    cols = list(columns) if columns else [c for c in df.columns if not c.endswith("_sd") and c != "n_seeds"]
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, (float, np.floating)):
                sd = r.get(c + "_sd")
                cells.append(f"{v:.{decimals}f} ± {sd:.{decimals}f}" if sd is not None and not pd.isna(sd) and r.get("n_seeds", 1) > 1
                             else f"{v:.{decimals}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n" + "\n".join(lines))


def save_rows(rows: list, out: Path, name: str = "runs") -> None:
    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
