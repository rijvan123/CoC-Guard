"""Table XI: scalability with investigation size and analyst workload.

Latency protocol (Section IV-E, Table A8): batch size of one investigation,
100 warm-up executions followed by 1,000 timed runs; the median latency is
reported with throughput, peak GPU memory (CUDA) and the trainable-parameter
count. Report-level intervention cost is measured separately and excluded from
routine latency. Investigation sizes follow Table XI(a) (10-200 events with
about five nodes per event).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from ..data.collate import collate_investigations
from ..data.schema import TrackSpec
from ..models.baselines.classic import LightGBMBaseline
from ..models.baselines.mlp import MLP
from ..models.cocguard import CoCGuard, count_parameters
from .common import base_parser, setup

PAPER_SIZES = {10: (50, 120), 25: (125, 320), 50: (250, 680), 100: (500, 1650), 200: (1000, 5000)}


def synthetic_investigation(T: int, n_nodes: int, n_edges: int, d_b: int = 96, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_nodes = max(n_nodes, T + 1)
    node_type = np.concatenate([np.zeros(T, np.int64), rng.integers(1, 10, size=n_nodes - T)])
    src = rng.integers(0, n_nodes, size=n_edges)
    dst = rng.integers(0, n_nodes, size=n_edges)
    return {
        "xb": rng.normal(size=(T, d_b)).astype(np.float16), "mb": np.ones((T, d_b), np.uint8),
        "xc": np.zeros((T, 0), np.float16), "mc": np.zeros((T, 0), np.uint8),
        "cat": rng.integers(1, 50, size=(T, 4)), "v": np.ones((T, 5), np.float32),
        "dtau": rng.exponential(1.0, size=T).astype(np.float32), "tau_avail": np.ones(T, np.uint8),
        "pair": np.stack([rng.integers(0, 4, T), np.zeros(T, np.int64), rng.integers(-1, 20, T), rng.integers(0, 16, T)], 1),
        "prev": np.arange(T) - 1, "node_type": node_type, "node_x": rng.normal(size=(n_nodes, 32)).astype(np.float32),
        "event_node": np.arange(T), "edge_index": np.stack([src, dst]), "edge_type": rng.integers(0, 12, size=n_edges),
        "edge_support": rng.uniform(0.5, 1.0, size=(n_edges, 4)).astype(np.float32), "loc": np.zeros(T, np.float32),
        "y_m": 0, "y_c": -1, "a_T": 1, "a_G": 1, "meta": {"id": f"synthetic-{T}"},
    }


def time_model(fn, warmup: int, runs: int, device) -> dict:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    lat = float(np.median(times)) * 1000.0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else float("nan")
    return {"latency_ms": lat, "throughput_per_s": 1000.0 / lat, "peak_gpu_memory_gb": peak}


def main(argv=None):
    p = base_parser(__doc__)
    args = p.parse_args(argv)
    cfg, out, device, seeds = setup(args, "scalability")
    ev = cfg.evaluation
    warm, runs = int(ev.latency_warmup), int(ev.latency_runs)
    spec = TrackSpec("cape", d_b=96, d_c=0, cat_vocab_sizes=[60, 60, 60, 20], n_operations=16, num_malware_classes=10)
    model = CoCGuard(spec, cfg.model).to(device).eval()
    concat = CoCGuard(spec, cfg.model, {"fusion": "concat"}).to(device).eval()
    rows = []
    for T in ev.latency_sizes:
        nodes, edges = PAPER_SIZES.get(int(T), (5 * T, int(T * (11.6 + 0.067 * T))))
        batch = collate_investigations([synthetic_investigation(int(T), nodes, edges)]).to(device)
        with torch.no_grad():
            res = time_model(lambda: model(batch), warm, runs, device)
        rows.append({"events": int(T), "nodes": nodes, "edges": edges, **res})
    pd.DataFrame(rows).to_csv(out / "table_XI_a.csv", index=False)
    # routine latency at the median configuration (~32 events, 160 nodes, 410 relations)
    med = collate_investigations([synthetic_investigation(32, 160, 410)]).to(device)
    routine = {}
    with torch.no_grad():
        routine["CoC-Guard"] = time_model(lambda: model(med), warm, runs, device)["latency_ms"]
        routine["Direct concatenation without RCF"] = time_model(lambda: concat(med), warm, runs, device)["latency_ms"]
        mlp = MLP(2381, 2).to(device).eval()
        x = torch.randn(1, 2381, device=device)
        routine["MLP"] = time_model(lambda: mlp(x), warm, runs, device)["latency_ms"]
    rng = np.random.default_rng(0)
    lgb = LightGBMBaseline(n_estimators=int(cfg.get("latency_lgb_trees", 1000)), early_stopping_rounds=None, n_jobs=1)
    lgb.fit(rng.normal(size=(2000, 2381)).astype(np.float32), rng.integers(0, 2, 2000))
    xr = rng.normal(size=(1, 2381)).astype(np.float32)
    routine["LightGBM"] = time_model(lambda: lgb.predict_proba(xr), max(1, warm // 10), max(5, runs // 10),
                                     torch.device("cpu"))["latency_ms"]
    summary = {"routine_latency_ms": routine, "trainable_parameters_cocguard": count_parameters(model),
               "device": str(device), "warmup": warm, "runs": runs}
    pd.Series(routine, name="latency_ms").to_csv(out / "routine_latency.csv")
    import json
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
