# Structured audit record (Section III-K, Algorithm 1 stage 6)

`cocguard.pipeline.CoCGuardPipeline.investigate(sample)` returns a decision dictionary and an `AuditRecord` (`cocguard/audit/record.py`), serialisable with `record.to_json()`. Example records are written by `cocguard workload` to `results/workload/audit_examples/`.

| Field | Content |
|---|---|
| `investigation_id`, `evidence_ids` | identifiers of the investigation and its evidence |
| `malware` | decision (class or `unknown/review`), argmax class, calibrated confidence, probabilities |
| `known_family_support` | s^u, τ_u, the components [1 − max p, d^rep, d^TG], status (accepted known family or unknown/review) |
| `custody` | predicted technical custody state and probabilities |
| `rule_outcomes` | failed deterministic rules per event (retained unchanged) |
| `branches` | temporal and graph availability, branch reliability, cross-view disagreement, malware and custody fusion gates |
| `localization` | top-ranked transactions with scores (custody track) |
| `responsibility` | prioritised candidate factors (events or relations) with rank |
| `policy_confirmed_violations` | violations established by a frozen rule |
| `model_suspected_factors` | ranked by learned localisation, not tested by intervention |
| `counterfactually_verified_factors` | factors satisfying Eqs. (25)–(28): valid, Δ ≥ 0.10, S ≥ 0.80, consistent direction |
| `unverifiable_factors` | no valid intervention (with the failed constraint) |
| `verification` | per candidate: intervention, validity, repetitions R, effects per realisation, Δ, S, Q, r, status |
| `versions` | package, model and rule-set versions, temperatures, thresholds and coefficients |
| `claim_boundary` | fixed statement: technical assessment only, not legal admissibility |
| `created_utc` | timestamp |

The four finding categories are kept separate, so that deterministic rule results, predictive attributions and intervention-based evidence are never presented as equivalent conclusions.
