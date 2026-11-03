"""Deterministic serialisation and cryptographic continuity (Eq. 4).

    h*_{i,t} = H( Ser[ q_i || eps_{i,t} || tau_{i,t} || a_{i,t} || d_{i,t} || o_{i,t} || z_{i,t} || h_{i,t-1} ] )
    v^h_{i,t} = I[ h_{i,t} == h*_{i,t} ]

``h_{i,t-1}`` is the recorded digest of the previous record in log order (the
first record chains to :data:`GENESIS`). The semantic custody predecessor
(``prev_event_id``) is part of the payload ``z`` and is checked separately by
rule R5, so that cryptographic continuity and predecessor validity remain two
distinct, auditable conditions.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

#: Digest that the first record of every history chains to.
GENESIS = "0" * 64

#: Fields of an event record that are *not* part of the payload z.
_HEADER_FIELDS = ("case_id", "event_id", "timestamp", "actor", "device", "operation", "digest")


def _canonical(value: Any) -> Any:
    """Map values onto a JSON-stable representation."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return format(value, ".9g")
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def format_timestamp(ts: float | None) -> str | None:
    """Timestamps are serialised as integer microseconds since the epoch."""
    if ts is None:
        return None
    if isinstance(ts, float) and (math.isnan(ts) or math.isinf(ts)):
        return None
    return str(int(round(float(ts) * 1_000_000)))


def payload_of(event: Mapping[str, Any]) -> dict:
    """Payload z_{i,t}: every recorded field except the digest and header fields.

    Keys starting with ``_`` are processing annotations (never recorded content)
    and are excluded.
    """
    return {k: v for k, v in event.items() if k not in _HEADER_FIELDS and not str(k).startswith("_")}


def serialize(case_id: str, event_id: str, timestamp: float | None, actor: str | None,
              device: str | None, operation: str | None, payload: Mapping[str, Any],
              prev_digest: str) -> bytes:
    """Ser[.] of Eq. (4): canonical, key-sorted, compact UTF-8 JSON."""
    record = {
        "q": case_id,
        "eps": event_id,
        "tau": format_timestamp(timestamp),
        "a": actor,
        "d": device,
        "o": operation,
        "z": _canonical(dict(payload)),
        "h_prev": prev_digest,
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_digest(event: Mapping[str, Any], prev_digest: str) -> str:
    """Recompute h*_{i,t} for one event record."""
    data = serialize(event.get("case_id"), event.get("event_id"), event.get("timestamp"),
                     event.get("actor"), event.get("device"), event.get("operation"),
                     payload_of(event), prev_digest)
    return hashlib.sha256(data).hexdigest()


def seal_chain(events: Sequence[dict]) -> None:
    """(Re)compute and store the digest of every event in log order (in place).

    Used when a history is *recorded*: intact histories and violations that are
    introduced at record time (all types except cryptographic mismatch) carry a
    self-consistent chain.
    """
    prev = GENESIS
    for ev in events:
        ev["digest"] = compute_digest(ev, prev)
        prev = ev["digest"]


def verify_chain(events: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return v^h for every event: 1 if the recorded digest matches the recomputation."""
    out: list[int] = []
    prev = GENESIS
    for ev in events:
        recorded = ev.get("digest")
        ok = recorded is not None and recorded == compute_digest(ev, prev)
        out.append(int(ok))
        prev = recorded if recorded is not None else GENESIS
    return out
