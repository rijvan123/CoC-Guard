"""Structured forensic audit record (Section III-K, Algorithm 1 stage 6).

The record keeps four kinds of finding separate:

* **policy-confirmed violation**  a failed frozen rule (auditable, deterministic);
* **model-suspected factor**      ranked by learned localisation, not tested by intervention;
* **counterfactually verified**   satisfies Eqs. (25)-(28) under the defined intervention;
* **unverifiable factor**         no valid intervention, or insufficient supporting evidence.

It also stores the malware decision and calibrated confidence, known-family
support status, custody state, branch availability and reliability,
cross-view disagreement, fusion gates and the versions of the model,
calibration parameters, thresholds and rule set. The record documents
technical custody-integrity assessment only; it is not a statement about
legal admissibility.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..version import MODEL_VERSION, RULESET_VERSION, __version__

CLAIM_BOUNDARY = ("Technical custody-integrity assessment under the frozen rule set and model. "
                  "This record does not determine legal admissibility, deliberate tampering or "
                  "unrestricted causal responsibility.")


@dataclass
class AuditRecord:
    investigation_id: str
    evidence_ids: list = field(default_factory=list)
    malware: dict = field(default_factory=dict)          # class, calibrated confidence, probabilities
    known_family_support: dict = field(default_factory=dict)  # s_u, tau_u, components, decision
    custody: dict = field(default_factory=dict)          # state, probabilities
    rule_outcomes: list = field(default_factory=list)    # failed rules with reasons
    branches: dict = field(default_factory=dict)         # availability, reliability, disagreement, gates
    localization: list = field(default_factory=list)     # ranked events / relations with scores
    responsibility: list = field(default_factory=list)   # event-level responsibility scores
    policy_confirmed_violations: list = field(default_factory=list)
    model_suspected_factors: list = field(default_factory=list)
    counterfactually_verified_factors: list = field(default_factory=list)
    unverifiable_factors: list = field(default_factory=list)
    verification: list = field(default_factory=list)     # per-candidate validity, R, effect, stability
    versions: dict = field(default_factory=dict)
    claim_boundary: str = CLAIM_BOUNDARY
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)


def version_block(temperatures: dict, thresholds: dict, extra: dict | None = None) -> dict:
    return {"package": __version__, "model": MODEL_VERSION, "rule_set": RULESET_VERSION,
            "calibration": {k: float(v) for k, v in temperatures.items()},
            "thresholds": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in thresholds.items()},
            **(extra or {})}
