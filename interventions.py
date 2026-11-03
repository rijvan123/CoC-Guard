"""Validity-constrained interventions on isolated analytical copies (Eq. 25).

    S~^(j)_i = I_j(S_i; Omega),   V_{i,j} = V(S~^(j)_i) in {0, 1}

Candidates are *event factors* (one evidence event) and *relation factors* (an
evidence entity such as a file, registry key, network endpoint, process,
artifact, custodian or device, together with all events related to it; at
least two). Interventions never touch the original evidence: they operate on a
deep copy of the report-derived event sequence and evidence graph.

``mask``        behavioural attributes of the factor's events become unavailable (schema,
                order, entity references and relations are kept; attribute availability of
                the affected relations is reduced);
``substitute``  the events' behaviour is replaced by a frozen neutral template of the same
                schema (seeded choice among templates);
``correct``     custody transactions are restored to their verified state (rule outcomes and
                the rule-derived support of incident relations);
``delete``      unconstrained removal of the factor's events (used only by the "without
                validity constraints" ablation; it can break references and timestamp order).

Frozen constraints Omega (:func:`validity`): report schema, timestamp order,
required entity references, relation consistency and policy constraints
(protected events such as the initial execution context or the acquisition
record, and process-creation events whose process acts later). Each
realisation also applies a seeded, order-preserving timestamp jitter.
"""
from __future__ import annotations

import copy

import numpy as np

from ..data.schema import NODE_TYPE_INDEX

ARRAY_KEYS = ("xb", "mb", "xc", "mc", "cat", "v", "dtau", "tau_avail", "pair", "prev", "loc")
EVENT_TYPE = NODE_TYPE_INDEX["event"]
RELATION_FACTOR_TYPES = {NODE_TYPE_INDEX[t] for t in ("artifact", "process", "actor", "device", "file", "registry",
                                                      "network_endpoint")}


def isolated_copy(sample: dict) -> dict:
    """Deep copy of all evidence arrays (the original is never modified)."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in sample.items() if k != "_events"}
    out["_events"] = sample.get("_events")
    return out


def as_factor(target) -> dict:
    if isinstance(target, dict):
        return target
    return {"id": f"e{int(target)}", "events": [int(target)], "node": None, "level": "event"}


def candidate_factors(sample: dict, min_events: int = 2, relations: bool = True) -> list[dict]:
    """Event factors for every event plus entity-centred relation factors."""
    T = len(sample["v"])
    cands = [as_factor(t) for t in range(T)]
    if not relations or sample["edge_index"].shape[1] == 0:
        return cands
    node_to_event = {int(n): t for t, n in enumerate(sample["event_node"]) if n >= 0}
    nt = sample["node_type"]
    links: dict[int, set] = {}
    for s_, d_ in sample["edge_index"].T:
        s_, d_ = int(s_), int(d_)
        if s_ in node_to_event and int(nt[d_]) in RELATION_FACTOR_TYPES:
            links.setdefault(d_, set()).add(node_to_event[s_])
        if d_ in node_to_event and int(nt[s_]) in RELATION_FACTOR_TYPES:
            links.setdefault(s_, set()).add(node_to_event[d_])
    for node, evs in sorted(links.items()):
        if min_events <= len(evs) < T:
            cands.append({"id": f"n{node}", "events": sorted(evs), "node": int(node), "level": "relation"})
    return cands


def incident_edges(sample: dict, j: int) -> np.ndarray:
    node = int(sample["event_node"][j]) if len(sample["event_node"]) > j else -1
    if node < 0 or sample["edge_index"].shape[1] == 0:
        return np.zeros(0, dtype=np.int64)
    ei = sample["edge_index"]
    return np.where((ei[0] == node) | (ei[1] == node))[0]


def node_edges(sample: dict, node: int | None) -> np.ndarray:
    if node is None or sample["edge_index"].shape[1] == 0:
        return np.zeros(0, dtype=np.int64)
    ei = sample["edge_index"]
    return np.where((ei[0] == node) | (ei[1] == node))[0]


def protected_events(sample: dict) -> set:
    """Policy constraints: events that may not be intervened upon."""
    T = len(sample["v"])
    prot = {0} if T else set()
    evs = sample.get("_events")
    if evs:
        first_of_pid: dict = {}
        for t, e in enumerate(evs):
            first_of_pid.setdefault(e.get("pid"), t)
        for t, e in enumerate(evs):
            if e.get("op") == "process_create":
                # creating a process that acts later is a dependency of those later events
                if any(pid != e.get("pid") and t0 > t for pid, t0 in first_of_pid.items()):
                    prot.add(t)
    return prot


def _jitter(s: dict, j: int, rng: np.random.Generator, seconds: float) -> None:
    if seconds <= 0 or not s["tau_avail"][j]:
        return
    T = len(s["dtau"])
    delta = float(rng.uniform(-seconds, seconds))
    lo = -float(s["dtau"][j]) if j > 0 else 0.0
    hi = float(s["dtau"][j + 1]) if j + 1 < T else seconds
    delta = float(np.clip(delta, min(0.0, lo), max(0.0, hi)))
    s["dtau"][j] += delta
    if j + 1 < T:
        s["dtau"][j + 1] -= delta


def apply_intervention(sample: dict, target, kind: str, rng: np.random.Generator, templates: np.ndarray | None = None,
                       jitter_seconds: float = 1.0, floor: float = 0.1) -> dict:
    f = as_factor(target)
    if kind == "delete":
        s = isolated_copy(sample)
        for j in sorted(f["events"], reverse=True):
            s = delete_event(s, j)
        return s
    s = isolated_copy(sample)
    T = len(s["v"])
    support = s["edge_support"].copy()
    for j in f["events"]:
        if not 0 <= j < T:
            raise IndexError(j)
        edges = incident_edges(s, j)
        if kind == "mask":
            s["xb"][j] = 0
            s["mb"][j] = 0
            if s["cat"].shape[1] >= 1:
                s["cat"][j, 0] = 1                     # behavioural token -> UNK
            support[edges, 3] = np.minimum(support[edges, 3], 0.5)
        elif kind == "substitute":
            if templates is None or len(templates) == 0:
                raise ValueError("substitution requires a frozen template bank")
            tpl = templates[int(rng.integers(0, len(templates)))]
            s["xb"][j] = tpl[: s["xb"].shape[1]].astype(s["xb"].dtype)
            s["mb"][j] = 1
            if s["cat"].shape[1] >= 1:
                s["cat"][j, 0] = 1
        elif kind == "correct":
            s["v"][j] = 1.0
            support[edges, :3] = 1.0
        else:
            raise ValueError(f"unknown intervention {kind!r}")
    if f.get("node") is not None and kind in ("mask", "substitute"):
        ne = node_edges(s, f["node"])
        support[ne, 3] = np.minimum(support[ne, 3], 0.5)
    s["edge_support"] = support
    _jitter(s, f["events"][0], rng, jitter_seconds)
    return s


def delete_event(s: dict, j: int) -> dict:
    """Unconstrained deletion (no repair of references or timestamps)."""
    keep = np.ones(len(s["v"]), bool)
    keep[j] = False
    for k in ARRAY_KEYS:
        if k in s and isinstance(s[k], np.ndarray) and len(s[k]) == len(keep):
            s[k] = s[k][keep]
    s["prev"] = np.where(s["prev"] > j, s["prev"] - 1, s["prev"])     # a pointer to j now points elsewhere
    node = int(s["event_node"][j])
    s["event_node"] = np.delete(s["event_node"], j)
    if node >= 0:
        ei = s["edge_index"]
        keep_e = (ei[0] != node) & (ei[1] != node)
        ei = ei[:, keep_e]
        ei = np.where(ei > node, ei - 1, ei)
        s["edge_index"], s["edge_type"] = ei, s["edge_type"][keep_e]
        s["edge_support"] = s["edge_support"][keep_e]
        s["node_type"] = np.delete(s["node_type"], node)
        s["node_x"] = np.delete(s["node_x"], node, axis=0)
        s["event_node"] = np.where(s["event_node"] > node, s["event_node"] - 1, s["event_node"])
    if s.get("_events"):
        s["_events"] = [e for i, e in enumerate(s["_events"]) if i != j]
    return s


def validity(original: dict, s: dict, target, kind: str, constrained: bool = True, floor: float = 0.1) -> tuple[bool, str]:
    """V(S~) under the frozen constraints Omega. Returns (valid, reason)."""
    if not constrained:
        return True, "constraints disabled"
    if kind == "delete":
        return False, "deletion breaks report schema"
    f = as_factor(target)
    T = len(original["v"])
    # report schema
    for k in ("xb", "mb", "xc", "mc", "v", "dtau", "tau_avail", "prev"):
        if s[k].shape != original[k].shape:
            return False, f"schema changed for {k}"
        if s[k].dtype.kind == "f" and not np.all(np.isfinite(s[k].astype(np.float64))):
            return False, f"non-finite values in {k}"
    # temporal order
    neg_before = int(np.sum((original["dtau"] < 0) & (original["tau_avail"] > 0)))
    neg_after = int(np.sum((s["dtau"] < 0) & (s["tau_avail"] > 0)))
    if neg_after > neg_before:
        return False, "intervention breaks timestamp order"
    # required entity references
    prev = s["prev"]
    if np.any(prev >= np.arange(T)) or np.any(prev < -1):
        return False, "dangling predecessor reference"
    en = s["event_node"]
    if np.any(en >= len(s["node_type"])) or np.any((en >= 0) & (s["node_type"][np.clip(en, 0, None)] != EVENT_TYPE)):
        return False, "event-node reference broken"
    if s["edge_index"].size and s["edge_index"].max() >= len(s["node_type"]):
        return False, "dangling relation"
    # relation consistency and policy constraints, per event of the factor
    prot = protected_events(original)
    for j in f["events"]:
        if len(incident_edges(s, j)) != len(incident_edges(original, j)):
            return False, "relations of the candidate changed"
        if j in prot:
            return False, "protected event (initial context or dependency of later events)"
        inc = incident_edges(original, j)
        if len(inc) and np.all(original["edge_support"][inc, 3] <= floor + 1e-9):
            return False, "required dependency missing"
    if kind in ("mask", "substitute") and int(sum(original["mb"][j].sum() for j in f["events"])) == 0:
        return False, "behaviour already unavailable"
    if kind == "correct" and all(float(original["v"][j].min()) >= 1.0 for j in f["events"]):
        return False, "transaction already verified"
    return True, ""


def neutral_templates(train_samples, op_indices=(14, 15), k: int = 8, seed: int = 11) -> np.ndarray:
    """Frozen bank of neutral behaviour templates (library/system events of the training partition)."""
    rng = np.random.default_rng(seed)
    pool = []
    for s in train_samples:
        if s["pair"].shape[0] == 0:
            continue
        sel = np.isin(s["pair"][:, 3], op_indices) & (s["mb"].sum(1) > 0)
        if sel.any():
            pool.append(s["xb"][sel].astype(np.float32))
        if sum(len(p) for p in pool) > 5000:
            break
    if not pool:
        return np.zeros((1, train_samples[0]["xb"].shape[1]), np.float32)
    pool = np.concatenate(pool)
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return pool[idx]
