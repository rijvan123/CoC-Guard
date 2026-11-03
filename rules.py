"""Frozen deterministic custody rules R1-R6 and the verification vector (Eq. 5).

    v_{i,t} = [v^h, v^tau, v^a, v^tr, v^p]^T,   delta_{i,t} = 1 - v_{i,t}

R4 (transfer completeness) and R5 (predecessor validity) jointly form v^tr.
Every evaluation is logged as a :class:`RuleOutcome` with rule identifier,
evaluated field, binary outcome, affected event and failure reason, which is
what the structured audit record stores (Section III-C).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from . import policy
from .hashing import verify_chain


@dataclass(frozen=True)
class RuleOutcome:
    rule_id: str
    rule: str
    field: str
    event_index: int
    event_id: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    v: np.ndarray            # [T, 5] verification vector (1 = pass)
    rule_matrix: np.ndarray  # [T, 6] outcome of R1..R6 (1 = pass)
    outcomes: list           # list[RuleOutcome] (failures and passes)

    @property
    def delta(self) -> np.ndarray:
        return 1.0 - self.v

    def failures(self) -> list:
        return [o for o in self.outcomes if not o.passed]


class ActorRegistry:
    """Registry of custodians: canonical id -> role, lab and aliases."""

    def __init__(self, actors: Mapping[str, Mapping]):
        self.actors = {k: dict(v) for k, v in actors.items()}
        self.alias_to_id: dict[str, str] = {}
        for aid, info in self.actors.items():
            self.alias_to_id[aid] = aid
            for alias in info.get("aliases", []):
                self.alias_to_id[alias] = aid

    def resolve(self, actor: str | None) -> str | None:
        if actor is None:
            return None
        return self.alias_to_id.get(actor)

    def role(self, actor: str | None) -> str | None:
        aid = self.resolve(actor)
        return None if aid is None else self.actors[aid]["role"]

    def to_dict(self) -> dict:
        return self.actors


def _present(value) -> bool:
    return value is not None and value != ""


def evaluate_rules(events: Sequence[Mapping], registry: ActorRegistry) -> VerificationResult:
    """Apply the frozen rule set R to one ordered custody history."""
    T = len(events)
    rule_matrix = np.ones((T, 6), dtype=np.float32)
    outcomes: list[RuleOutcome] = []

    def record(rule_pos: int, t: int, passed: bool, reason: str = "") -> None:
        rid, name, field, _ = policy.RULES[rule_pos]
        if not passed:
            rule_matrix[t, rule_pos] = 0.0
        outcomes.append(RuleOutcome(rid, name, field, t, str(events[t].get("event_id")), bool(passed), reason))

    # ---- R1: cryptographic continuity (Eq. 4) -------------------------------
    vh = verify_chain(events)
    for t in range(T):
        reason = "" if vh[t] else ("digest unavailable" if events[t].get("digest") is None
                                   else "recorded digest differs from recomputation")
        record(0, t, bool(vh[t]), reason)

    # ---- R2: chronological consistency -------------------------------------
    first_ts = events[0].get("timestamp") if T else None
    last_transfer_ts: dict[str, float] = {}
    for t in range(T):
        ev = events[t]
        ts = ev.get("timestamp")
        ok, reason = True, ""
        if ts is None:
            ok, reason = False, "timestamp unavailable"
        else:
            prev_ts = events[t - 1].get("timestamp") if t > 0 else None
            if prev_ts is not None and ts < prev_ts:
                ok, reason = False, "timestamp precedes previous record"
            elif prev_ts is not None and ts - prev_ts > policy.MAX_GAP_SECONDS:
                ok, reason = False, "gap to previous record exceeds 30 days"
            elif first_ts is not None and ts < first_ts:
                ok, reason = False, "timestamp precedes acquisition"
            if ok and ev.get("operation") == "receive":
                src = last_transfer_ts.get(str(ev.get("evidence_id")))
                if src is not None and (ts - src > policy.MAX_TRANSFER_TO_RECEIVE):
                    ok, reason = False, "acknowledgement later than 72 h after transfer"
        if ev.get("operation") == "transfer" and ts is not None:
            last_transfer_ts[str(ev.get("evidence_id"))] = ts
        record(1, t, ok, reason)

    # ---- R3: actor authorisation ---------------------------------------------
    for t in range(T):
        ev = events[t]
        role = registry.role(ev.get("actor"))
        op = ev.get("operation")
        if role is None:
            record(2, t, False, "actor not in registry")
        elif role not in policy.AUTHORIZATION.get(op, set()):
            record(2, t, False, f"role '{role}' not authorised for '{op}'")
        else:
            record(2, t, True)

    # ---- R4: transfer completeness (custodian continuity) --------------------
    custodian: dict[str, str | None] = {}
    for t in range(T):
        ev = events[t]
        op = ev.get("operation")
        item = str(ev.get("evidence_id"))
        actor = registry.resolve(ev.get("actor"))
        ok, reason = True, ""
        if op == "acquire":
            custodian[item] = actor
        elif op == "extract":
            if actor != custodian.get(item):
                ok, reason = False, "extraction by non-custodian"
            derived = ev.get("derived_evidence_id")
            if _present(derived):
                custodian[str(derived)] = actor
        elif op == "transfer":
            receiver = registry.resolve(ev.get("receiver"))
            if actor != custodian.get(item):
                ok, reason = False, "transfer by non-custodian"
            elif receiver is None:
                ok, reason = False, "transfer without registered receiver"
            custodian[item] = receiver
        elif op == "receive":
            if actor != custodian.get(item):
                ok, reason = False, "receipt without matching transfer"
        elif op in policy.CUSTODY_HOLDING:
            if item not in custodian:
                # first event on an item that was never acquired or extracted
                ok, reason = False, "item handled before acquisition"
            elif actor != custodian.get(item):
                ok, reason = False, "custody-holding operation by non-custodian"
        record(3, t, ok, reason)

    # ---- R5: predecessor validity -------------------------------------------
    index_of = {str(ev.get("event_id")): i for i, ev in enumerate(events)}
    parent_of: dict[str, str] = {}
    for ev in events:
        if ev.get("operation") == "extract" and _present(ev.get("derived_evidence_id")):
            parent_of[str(ev.get("derived_evidence_id"))] = str(ev.get("evidence_id"))
    for t in range(T):
        ev = events[t]
        prev = ev.get("prev_event_id")
        ok, reason = True, ""
        if t == 0:
            if _present(prev):
                ok, reason = False, "first record declares a predecessor"
        elif not _present(prev):
            ok, reason = False, "predecessor missing"
        elif str(prev) not in index_of:
            ok, reason = False, "predecessor not found"
        else:
            j = index_of[str(prev)]
            if j >= t:
                ok, reason = False, "predecessor is not an earlier record"
            else:
                item = str(ev.get("evidence_id"))
                prev_item = str(events[j].get("evidence_id"))
                lineage_ok = prev_item == item or parent_of.get(item) == prev_item
                if not lineage_ok:
                    ok, reason = False, "predecessor belongs to an unrelated evidence item"
        record(4, t, ok, reason)

    # ---- R6: mandatory provenance ------------------------------------------
    for t in range(T):
        ev = events[t]
        missing = [f for f in policy.MANDATORY_PROVENANCE.get(ev.get("operation"), ())
                   if not _present(ev.get(f))]
        record(5, t, not missing, ("missing mandatory provenance: " + ",".join(missing)) if missing else "")

    v = np.stack([
        rule_matrix[:, 0],                              # v^h
        rule_matrix[:, 1],                              # v^tau
        rule_matrix[:, 2],                              # v^a
        np.minimum(rule_matrix[:, 3], rule_matrix[:, 4]),  # v^tr (R4 and R5)
        rule_matrix[:, 5],                              # v^p
    ], axis=1).astype(np.float32)
    return VerificationResult(v=v, rule_matrix=rule_matrix, outcomes=outcomes)


def rule_only_decision(result: VerificationResult) -> dict:
    """Decision of the rule-only custody detector (Section IV-C comparator)."""
    failed_events = np.where(result.rule_matrix.min(axis=1) < 1)[0]
    compromised = bool(len(failed_events))
    state = "intact"
    if compromised:
        failed_rules = {o.rule_id for o in result.failures()}
        mapping = {"R1": "hash_mismatch", "R2": "timestamp_perturbation", "R3": "unauthorized_actor",
                   "R4": "transfer_omission", "R5": "predecessor_corruption", "R6": "provenance_removal"}
        kinds = {mapping[r] for r in failed_rules}
        state = kinds.pop() if len(kinds) == 1 else "multiple"
    # Localisation score: number of failed rules per event, ties broken by order.
    loc_scores = (1.0 - result.rule_matrix).sum(axis=1)
    return {"compromised": compromised, "state": state, "loc_scores": loc_scores,
            "first_failure": int(failed_events[0]) if compromised else -1}
