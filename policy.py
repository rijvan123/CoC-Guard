"""Frozen custody policy shared by the history generator and the rule set.

The policy is part of the frozen rule set R (Algorithm 1) and is versioned by
:data:`cocguard.version.RULESET_VERSION`. Changing anything here changes the
technical meaning of "intact", so it must be frozen before final evaluation.
"""
from __future__ import annotations

OPERATIONS = (
    "acquire", "hash_verify", "seal", "transfer", "receive", "store",
    "retrieve", "examine", "extract", "analyze", "report", "archive",
)
OP_INDEX = {op: i for i, op in enumerate(OPERATIONS)}

ROLES = ("first_responder", "evidence_custodian", "examiner", "analyst", "supervisor", "courier")
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}

#: Operation -> roles authorised to perform it (rule R3).
AUTHORIZATION = {
    "acquire": {"first_responder", "examiner"},
    "hash_verify": {"first_responder", "examiner", "evidence_custodian", "analyst"},
    "seal": {"first_responder", "evidence_custodian", "examiner"},
    "transfer": {"first_responder", "evidence_custodian", "courier", "examiner", "analyst"},
    "receive": {"evidence_custodian", "examiner", "courier", "analyst"},
    "store": {"evidence_custodian"},
    "retrieve": {"evidence_custodian"},
    "examine": {"examiner"},
    "extract": {"examiner"},
    "analyze": {"analyst", "examiner"},
    "report": {"analyst", "supervisor"},
    "archive": {"evidence_custodian", "supervisor"},
}

#: Operations that require possession of the item by the current custodian (rule R4).
CUSTODY_HOLDING = {"seal", "store", "retrieve", "examine", "extract", "analyze", "archive", "transfer"}
#: Operations exempt from custodian continuity (witness verification, reporting).
CONTINUITY_EXEMPT = {"hash_verify", "report"}

#: Mandatory provenance fields per operation (rule R6).
MANDATORY_PROVENANCE = {
    "acquire": ("source_location", "acquisition_tool", "tool_version"),
    "extract": ("acquisition_tool", "derived_evidence_id"),
    "seal": ("seal_id",),
    "transfer": ("form_id",),
}
#: Optional provenance fields that intact histories normally carry.
OPTIONAL_PROVENANCE = ("form_id", "notes", "source_location", "tool_version", "seal_id", "case_reference")
PROVENANCE_FIELDS = (
    "source_location", "acquisition_tool", "tool_version", "derived_evidence_id",
    "seal_id", "form_id", "notes", "case_reference", "exception_code",
)

#: Chronological plausibility windows (rule R2), in seconds.
MAX_GAP_SECONDS = 30 * 24 * 3600          # no two consecutive records more than 30 days apart
MAX_TRANSFER_TO_RECEIVE = 72 * 3600       # a transfer must be acknowledged within 72 h

#: Custody states predicted by the custody head (C_c = 8).
CUSTODY_STATES = (
    "intact", "hash_mismatch", "timestamp_perturbation", "unauthorized_actor",
    "transfer_omission", "predecessor_corruption", "provenance_removal", "multiple",
)
VIOLATION_TYPES = CUSTODY_STATES[1:7]
STATE_INDEX = {s: i for i, s in enumerate(CUSTODY_STATES)}

#: Components of the verification vector v (Eq. 5) and the six frozen checks.
VERIFICATION_COMPONENTS = ("hash", "time", "auth", "transfer", "provenance")
RULES = (
    ("R1", "cryptographic_continuity", "digest", "hash"),
    ("R2", "chronological_consistency", "timestamp", "time"),
    ("R3", "actor_authorization", "actor", "auth"),
    ("R4", "transfer_completeness", "actor/receiver", "transfer"),
    ("R5", "predecessor_validity", "prev_event_id", "transfer"),
    ("R6", "provenance_availability", "provenance", "provenance"),
)
