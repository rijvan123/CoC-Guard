"""Preparation and loading of the four modality-compatible experimental tracks.

=========  ==============================  ==============================================
track      evidence                         partitions
=========  ==============================  ==============================================
custody    controlled custody histories     70/10/20 by base history (Table A3)
cape       CAPE events + evidence graphs    70/10/20 SHA-256 groups, family-stratified
bodmas     structured static PE (2,381)     chronological 70/10/20 with duplicate groups
ember      structured static PE (2,381)     official train/test; dev drawn from train
=========  ==============================  ==============================================

``prepare_*`` functions write *raw* (unnormalised) artefacts to
``data/processed/<track>``. Normalisation statistics and vocabularies are fitted
later, per protocol, on that protocol's training partition only
(:meth:`InvestigationTrack.build`), so a withheld family or held-out
violation type never influences preprocessing.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..custody.policy import CUSTODY_STATES, OPERATIONS
from ..data import cape as cape_mod
from ..data.custody_features import (CAT_FIELDS, D_B as CUSTODY_D_B, D_C as CUSTODY_D_C, history_to_sample)
from ..data.custody_generator import generate_controlled_histories, load_histories, manifest, save_histories
from ..data.graph import D_NODE
from ..data.partition import (chronological_partition, duplicate_groups, grouped_stratified_partition)
from ..data.preprocessing import (Standardizer, StaticScaler, Vocabulary, apply_event_standardizers,
                                  fit_event_standardizers)
from ..data.schema import TrackSpec
from ..utils.io import load_pickle, save_pickle, write_json
from ..utils.logging import get_logger

log = get_logger("tracks")


# ===========================================================================
# Custody track
# ===========================================================================
def prepare_custody(cfg, out_dir: str | Path, sources_csv: str | Path | None = None, progress: bool = True,
                    histories_dir: str | Path | None = None) -> Path:
    """Generate the controlled custody set, or load a released one (``histories_dir``), and build the track."""
    out = Path(out_dir)
    if histories_dir is not None:
        histories, registry, meta = load_histories(histories_dir)
        log.info(f"custody track: {len(histories)} histories loaded from {histories_dir}")
    else:
        sources = None
        if sources_csv is not None and Path(sources_csv).exists():
            sources = pd.read_csv(sources_csv)
            if "family" not in sources.columns and "classification_family" in sources.columns:
                sources = sources.rename(columns={"classification_family": "family"})
        histories, registry, meta = generate_controlled_histories(dict(cfg.custody), sources=sources, progress=progress)
    save_histories(histories, meta, out)
    write_split_files(out, "custody")
    samples = [history_to_sample(h, registry) for h in histories]
    save_pickle(samples, out / "samples_raw.pkl")
    write_json({"track": "custody", "d_b": CUSTODY_D_B, "d_c": CUSTODY_D_C, "n": len(samples)}, out / "track.json")
    log.info(f"custody track: {len(samples)} histories written to {out}")
    return out


# ===========================================================================
# CAPE track
# ===========================================================================
def _parse_one(args):
    path, family_index, meta, max_events, min_events, max_nodes, max_edges, api_freq = args
    try:
        report = cape_mod.load_report(path)
        return cape_mod.report_to_sample(report, family_index, meta, min_events=min_events, max_events=max_events,
                                         max_nodes=max_nodes, max_edges=max_edges, api_freq=api_freq)
    except Exception as exc:  # pragma: no cover - malformed report
        return {"error": f"{path}: {exc}", "meta": meta}


def prepare_cape(cfg, raw_dir: str | Path, out_dir: str | Path, report_subdirs: Sequence[str] = ("public_full_reports",
                 "public_small_reports", "public_reduced_reports", "reports", "."), workers: int = 1,
                 limit: int | None = None, freq_reports: int = 5000, seed: int = 11) -> Path:
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = cape_mod.load_labels(raw / "public_labels.csv")
    if limit:
        labels = labels.sample(n=min(limit, len(labels)), random_state=seed).reset_index(drop=True)
    dirs = [raw / d for d in report_subdirs if (raw / d).exists()]
    families = sorted(labels["family"].unique())
    fam_index = {f: i for i, f in enumerate(families)}
    # outer partition first (SHA-256 groups, family-stratified 70/10/20)
    groups = pd.factorize(labels["sha256"])[0]
    labels["partition"] = grouped_stratified_partition(groups, labels["family"].map(fam_index).to_numpy(),
                                                       tuple(cfg.custody.get("partition", (0.7, 0.1, 0.2))), seed)
    labels["group"] = groups
    m = cfg.model
    # frozen API document frequencies from (a sample of) training reports only
    api_freq = None
    if freq_reports:
        tr = labels[labels.partition == "train"]
        tr = tr.sample(n=min(freq_reports, len(tr)), random_state=seed)
        reports = []
        for sha in tr["sha256"]:
            p = cape_mod.find_report(dirs, sha)
            if p is not None:
                reports.append(cape_mod.load_report(p))
        api_freq = cape_mod.api_frequencies(reports)
        del reports
    jobs = []
    missing = 0
    for row in labels.itertuples(index=False):
        p = cape_mod.find_report(dirs, row.sha256)
        if p is None:
            missing += 1
            continue
        meta = {"id": row.sha256, "family": row.family, "type": row.classification_type,
                "date": None if pd.isna(row.date) else row.date.isoformat(), "partition": row.partition,
                "group": int(row.group)}
        jobs.append((str(p), fam_index[row.family], meta, int(m.max_events), int(m.min_events), int(m.max_nodes),
                     int(m.max_edges), api_freq))
    log.info(f"CAPE: {len(jobs)} reports to parse ({missing} listed reports not found)")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            samples = list(ex.map(_parse_one, jobs, chunksize=16))
    else:
        samples = [_parse_one(j) for j in jobs]
    errors = [s["error"] for s in samples if "error" in s]
    samples = [s for s in samples if "error" not in s]
    save_pickle(samples, out / "samples_raw.pkl")
    labels.to_csv(out / "manifest.csv", index=False)
    parsed = {sm["meta"]["id"] for sm in samples}
    labels.assign(parsed=labels["sha256"].isin(parsed).astype(int)).pipe(
        lambda df: write_split_files(out, "cape", df[["sha256", "family", "classification_type", "date", "group",
                                                      "partition", "parsed"]]))
    write_json({"track": "cape", "families": families, "n": len(samples), "errors": errors[:50],
                "n_errors": len(errors), "missing_reports": missing, "d_b": cape_mod.D_EVENT, "d_c": 0}, out / "track.json")
    log.info(f"CAPE track: {len(samples)} samples written to {out} ({len(errors)} parse errors)")
    return out


# ===========================================================================
# Investigation tracks: per-protocol preprocessing
# ===========================================================================
class InvestigationTrack:
    """Raw investigation samples plus per-protocol normalisation."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.info = json.loads((self.dir / "track.json").read_text())
        self.name = self.info["track"]
        self.raw: list[dict] = load_pickle(self.dir / "samples_raw.pkl")
        self.ids = [s["meta"]["id"] for s in self.raw]

    def __len__(self) -> int:
        return len(self.raw)

    def column(self, key: str) -> np.ndarray:
        return np.array([s["meta"].get(key) for s in self.raw], dtype=object)

    def build(self, train_mask: np.ndarray, malware_classes: Sequence | None = None,
              custody_classes: Sequence | None = None, label_map: dict | None = None) -> tuple[list[dict], TrackSpec, dict]:
        """Fit vocabularies and scalers on the masked training samples and
        return normalised shallow copies of *all* samples plus the TrackSpec."""
        train = [s for s, t in zip(self.raw, train_mask) if t]
        vocabs = []
        for k in range(len(train[0]["raw_cat"][0]) if train and train[0]["raw_cat"] else 0):
            vocabs.append(Vocabulary(min_count=1).fit(row[k] for s in train for row in s["raw_cat"]))
        log_b = None
        scalers = fit_event_standardizers(train, log_b=log_b)
        out = []
        for s in self.raw:
            c = dict(s)
            c["cat"] = np.array([[vocabs[k](row[k]) for k in range(len(vocabs))] for row in s["raw_cat"]],
                                dtype=np.int64).reshape(len(s["raw_cat"]), len(vocabs))
            out.append(c)
        apply_event_standardizers(out, scalers)
        for c in out:
            c.pop("raw_cat", None)
        if label_map is not None:
            for c in out:
                c["y_m"] = int(label_map.get(c["y_m"], -1)) if c["y_m"] >= 0 else -1
        if self.name == "custody":
            spec = TrackSpec("custody", d_b=CUSTODY_D_B, d_c=CUSTODY_D_C, d_node=D_NODE,
                             cat_vocab_sizes=[len(v) for v in vocabs], n_operations=len(OPERATIONS),
                             num_malware_classes=0, num_custody_classes=len(custody_classes or CUSTODY_STATES),
                             custody_classes=list(custody_classes or CUSTODY_STATES), has_custody_view=True)
        else:
            classes = list(malware_classes if malware_classes is not None else self.info.get("families", []))
            spec = TrackSpec("cape", d_b=cape_mod.D_EVENT, d_c=0, d_node=D_NODE, cat_vocab_sizes=[len(v) for v in vocabs],
                             n_operations=len(cape_mod.OP_CATS), num_malware_classes=len(classes),
                             malware_classes=classes, has_custody_view=False)
        return out, spec, {"vocabs": [v.state_dict() for v in vocabs],
                           "scalers": {k: v.state_dict() for k, v in scalers.items()}}


def custody_label_map(states: Sequence[str]) -> dict:
    """Map global custody-state indices onto a (possibly reduced) state list."""
    return {CUSTODY_STATES.index(s): i for i, s in enumerate(states)}


def remap_custody(samples: Sequence[dict], states: Sequence[str]) -> None:
    mapping = custody_label_map(states)
    for s in samples:
        s["y_c"] = int(mapping.get(s["y_c"], -1))


# ===========================================================================
# Static tracks
# ===========================================================================
def prepare_bodmas(cfg, raw_dir: str | Path, out_dir: str | Path, seed: int = 11) -> Path:
    from ..data.bodmas import load_bodmas
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y, meta = load_bodmas(raw_dir)
    groups = duplicate_groups(meta["sha"].tolist(), X, seed=seed)
    meta["group"] = groups
    meta["partition"] = chronological_partition(meta["timestamp"], groups, (0.7, 0.1, 0.2))
    np.save(out / "X.npy", X)
    meta.to_csv(out / "meta.csv", index=False)
    tab = meta.assign(row=np.arange(len(meta)))
    write_split_files(out, "bodmas", tab[[c for c in ("row", "sha", "timestamp", "family", "label", "group", "partition")
                                          if c in tab.columns]])
    write_json({"track": "bodmas", "n": int(len(y)), "d": int(X.shape[1]),
                "partition_counts": meta["partition"].value_counts().to_dict(),
                "n_families": int(meta.loc[meta.label == 1, "family"].nunique())}, out / "track.json")
    log.info(f"BODMAS track: {len(y)} rows, partitions {meta['partition'].value_counts().to_dict()}")
    return out


def prepare_ember(cfg, raw_dir: str | Path, out_dir: str | Path, dev_size: int = 60000, seed: int = 11) -> Path:
    from ..data.ember import load_ember
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = load_ember(raw_dir)
    rng = np.random.default_rng(seed)
    tr = d["train_idx"]
    dev_n = min(dev_size, max(1, int(round(0.1 * len(tr)))))
    dev = np.sort(rng.choice(tr, size=dev_n, replace=False))
    train = np.setdiff1d(tr, dev)
    np.save(out / "train_idx.npy", train)
    np.save(out / "dev_idx.npy", dev)
    np.save(out / "test_idx.npy", d["test_idx"])
    write_split_files(out, "ember", _ember_split_table(d, train, dev))
    write_json({"track": "ember", "raw_dir": str(Path(raw_dir).resolve()), "n_train": int(len(train)),
                "n_dev": int(len(dev)), "n_test": int(len(d["test_idx"]))}, out / "track.json")
    log.info(f"EMBER track: train {len(train)} / dev {len(dev)} / official test {len(d['test_idx'])}")
    return out


def _ember_split_table(d: dict, train: np.ndarray, dev: np.ndarray) -> pd.DataFrame:
    """Rows of the EMBER memmaps with their partition (and metadata.csv identifiers when present)."""
    n_tr, n_te = len(d["y_train"]), len(d["y_test"])
    df = pd.DataFrame({"subset": ["train"] * n_tr + ["test"] * n_te,
                       "row": np.concatenate([np.arange(n_tr), np.arange(n_te)]),
                       "label": np.concatenate([d["y_train"], d["y_test"]])})
    part = np.full(len(df), "excluded", dtype=object)
    part[train] = "train"
    part[dev] = "dev"
    part[n_tr + d["test_idx"]] = "test"
    df["partition"] = part
    meta = d.get("metadata")
    if meta is not None and "subset" in meta.columns:
        m = meta.copy()
        m["row"] = m.groupby("subset").cumcount()
        keep = [c for c in ("subset", "row", "sha256", "appeared", "avclass") if c in m.columns]
        df = df.merge(m[keep], on=["subset", "row"], how="left")
    return df


def write_split_files(out_dir: str | Path, track: str, table: pd.DataFrame | None = None) -> None:
    """``splits/<track>_{train,dev,test}.csv``: the identifiers of every partition, ready to publish."""
    out = Path(out_dir) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    if table is None:   # custody: the manifest written by save_histories
        table = pd.read_csv(Path(out_dir) / "manifest.csv", keep_default_na=False)
    for name in ("train", "dev", "test"):
        table[table["partition"] == name].to_csv(out / f"{track}_{name}.csv", index=False)
    counts = table["partition"].value_counts().rename_axis("partition").reset_index(name="rows")
    counts.to_csv(out / f"{track}_summary.csv", index=False)


class StaticTrack:
    """Static PE track (BODMAS or EMBER) with row indices per partition."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.info = json.loads((self.dir / "track.json").read_text())
        self.name = self.info["track"]
        if self.name == "bodmas":
            self.X = np.load(self.dir / "X.npy", mmap_mode="r")
            self.meta = pd.read_csv(self.dir / "meta.csv", keep_default_na=False, dtype={"family": str})
            self.meta["timestamp"] = pd.to_datetime(self.meta["timestamp"], utc=True, errors="coerce", format="mixed")
            self.y = self.meta["label"].to_numpy().astype(np.int64)
            self.partition = self.meta["partition"].to_numpy()
        else:
            from ..data.ember import read_vectorized
            raw = Path(self.info["raw_dir"])
            Xtr, ytr = read_vectorized(raw, "train")
            Xte, yte = read_vectorized(raw, "test")
            self.X = _ConcatRows(Xtr, Xte)
            self.y = np.concatenate([ytr, yte]).astype(np.int64)
            n_tr = len(ytr)
            part = np.full(len(self.y), "excluded", dtype=object)
            part[np.load(self.dir / "train_idx.npy")] = "train"
            part[np.load(self.dir / "dev_idx.npy")] = "dev"
            part[n_tr + np.load(self.dir / "test_idx.npy")] = "test"
            self.partition = part
            self.meta = None

    def indices(self, name: str) -> np.ndarray:
        return np.where(self.partition == name)[0]


class _ConcatRows:
    """Row-wise view over two memory-mapped matrices (EMBER train + test)."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.shape = (a.shape[0] + b.shape[0], a.shape[1])

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        n = self.a.shape[0]
        out = np.empty((len(idx), self.shape[1]), dtype=np.float32)
        ia = idx < n
        if ia.any():
            out[ia] = self.a[idx[ia]]
        if (~ia).any():
            out[~ia] = self.b[idx[~ia] - n]
        return out

    def __len__(self):
        return self.shape[0]


def apply_preprocessing(raw_samples: Sequence[dict], prep: dict) -> list[dict]:
    """Apply frozen (training-derived) vocabularies and scalers to new raw samples,
    e.g. test histories re-derived under a robustness condition."""
    vocabs = [Vocabulary().load_state_dict(v) for v in prep["vocabs"]]
    scalers = {k: Standardizer().load_state_dict(v) for k, v in prep["scalers"].items()}
    out = []
    for s in raw_samples:
        c = dict(s)
        c["cat"] = np.array([[vocabs[k](row[k]) for k in range(len(vocabs))] for row in s["raw_cat"]],
                            dtype=np.int64).reshape(len(s["raw_cat"]), len(vocabs))
        c.pop("raw_cat", None)
        out.append(c)
    apply_event_standardizers(out, scalers)
    return out


def split(samples: Sequence[dict], partition: Sequence[str]) -> dict:
    parts: dict[str, list] = {}
    for s, p in zip(samples, partition):
        parts.setdefault(str(p), []).append(s)
    return parts
