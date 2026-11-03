# Method → code map and implementation decisions

This document links every equation of the manuscript to its implementation and records each decision taken where the manuscript leaves a detail open. Equation numbers refer to the manuscript.

## 1. Evidence representation (Eqs. 1–3)

`cocguard/data/schema.py` defines the per-investigation sample. An event e_{i,t} keeps its behavioural attributes x^b, custody attributes x^c, timestamp, actor, device, operation, recorded digest and availability masks m as **separate arrays**. An unavailable attribute is therefore never confused with an observed zero. The model returns (p^m, p^c, s^u, r), as in Eq. (3). The unsupportedness s^u is computed by the screener, and r by the counterfactual module.

## 2. Deterministic verification (Eqs. 4–5)

* `custody/hashing.py`
  * `Ser[·]` is canonical, key-sorted JSON; timestamps are serialised as integer microseconds.
  * `H` is SHA-256.
  * `h_{i,t−1}` is the recorded digest of the previous record in **log order**.
  * The semantic custody predecessor (`prev_event_id`) is part of the payload z and is checked separately by rule R5. Cryptographic continuity and predecessor validity therefore remain two distinct, auditable conditions.
  * Keys starting with `_` are processing annotations and are never hashed.
* `custody/rules.py` contains the frozen rules R1–R6:

  | Rule | Check |
  |---|---|
  | R1 | hash continuity |
  | R2 | chronology: monotonic order, ≤ 30-day gaps, acknowledgement within 72 h of a transfer |
  | R3 | role-based authorisation |
  | R4 | custodian continuity across transfers |
  | R5 | predecessor existence, order and lineage |
  | R6 | mandatory provenance |

  R4 and R5 together form v^tr, so v has five components, as in Eq. (5). Every outcome is logged with the rule id, field, event, result and reason.
* `custody/policy.py` is the frozen policy. It is versioned as `custody-rules-1.0` in every audit record.

## 3. Reliability (Eqs. 6–7) — the core mechanism

* Eq. (6): `custody/reliability.py` (NumPy) and `models/reliability.py::rule_reliability` (PyTorch).
  * ω (severity) is hash 1.0, time 0.6, auth 0.8, transfer 0.8, provenance 0.5.
  * λ_b = λ_c = 0.5.
  * When a whole view is not native to a track (custody on CAPE or BODMAS, behaviour on none), its term is removed and the remaining λ renormalised, mirroring the renormalisation stated for Eq. (19).
* Eq. (7): `LearnedReliability` computes ρ = ρ^r·[α_ρ + (1 − α_ρ)·σ(w_ρᵀ[x̃^b‖x̃^c‖v‖m] + b_ρ)], with α_ρ = 0.60 (Table A7). Hence ρ ≤ ρ^r always: a learned estimate can lower reliability but cannot restore it after a failure. This is tested in `tests/test_model.py`.

## 4. Transaction-Consistency Learning (Eqs. 8–10) — `models/temporal.py`

* **u⁰ (Eq. 8):** W_u[…] plus sinusoidal positions p_t plus φ(Δτ). φ is a Time2Vec encoding of the signed log elapsed time together with an availability flag, so negative elapsed times (order inversions) stay visible to the model.
* **Attention (Eq. 9):** QKᵀ/√d_h + b^τ + b^c + log(ρ_j + ε).
  * b^τ is a learned per-head table over 16 signed log-time buckets, plus a bucket for "time unavailable".
  * b^c is a learned per-head combination of pairwise custody-compatibility features: same evidence item, j is the recorded predecessor of t, t is the predecessor of j, same actor, same device, chronologically feasible, time unavailable. It also includes a learned operation-pair table.
* **Pooling (Eq. 10):** π_t ∝ exp(w_Tᵀu_t)·ρ_t, computed in a numerically stable form.
* **Configuration:** pre-LayerNorm Transformer layers; L_T = 3, 4 heads, d = 128, FFN 512, dropout 0.2 (Table A4).

## 5. Heterogeneous evidence graph (Eqs. 11–14)

* `data/graph.py` builds the 10 node types and 12 relation types of Table A2 / Eq. (11).
  * The support components (r^id, r^τ, r^p, r^a) of Eq. (12) are stored per edge, and η = their product is formed in the model. A failed component is floored at 0.10, so explicitly invalid relations stay in the graph with reduced weight.
  * The "without deterministic verification" ablation keeps only r^a.
* **Custody graphs** (`data/custody_features.py`): event, artifact, actor, device, operation and storage-location nodes, with temporal-predecessor, handled-by, executed-on, performed-operation, stored-at, transferred-to, shared-identity and derived-from relations.
* **CAPE graphs** (`data/cape.py`):
  * Nodes: event, process, actor (process image), sandbox device, operation category, file, registry key, network endpoint, storage location (directory class), and artifact (root sample and dropped executables).
  * Relations: all 12 types.
  * Pruning to 512 nodes and 2,048 edges keeps, in order: events, then high-support entities, then temporal relations, then the best-supported relations, then chronological coverage.
* **Eq. (13)** (`models/graph.py`): relation-specific, 4-head GAT-style attention β̄ normalised over N_r(v). Each message is multiplied by η_uv, and h ← GELU(LN[W₀h + Σ_r Σ_u β̄·η·W_r h_u]). Every relation is propagated in both directions with separate weights (r and r⁻¹).
* **Eq. (14):** type-aware reliability pooling.
  * Event nodes use ρ_t. Entity nodes use the mean of ρ_event·η over incident relations.
  * The output is the concatenation over the 10 node types plus the node-type availability mask.

## 6. Reliability-Controlled Fusion and heads (Eqs. 15–18)

* **Quality vector (Eq. 15):** q^k = [a^k, 1 − u^k, ρ̄^k, 1 − d^TG].
  * u^k is the normalised entropy of a branch-level auxiliary head.
  * ρ̄^T is the mean event reliability; ρ̄^G is the mean edge support.
  * d^TG is the base-2 Jensen–Shannon divergence between the temporal and graph auxiliary predictions, averaged over the active tasks.
  * Quality vectors enter the gates with a stop-gradient, so the gates cannot be trained by manipulating the auxiliary predictions.
* **Gates (Eq. 16) and fusion (Eq. 17):** `models/fusion.py`, with task-specific gates for malware and custody. Availability masking assigns exactly zero weight to an unavailable branch (tested).
* **Comparators** (Table IX), selected with `fusion:`: `concat`, `equal`, `attention`, `uncertainty`, `availability`, `integrity`.
* **Eq. (18):** `models/heads.py`. Separate heads; the temperatures T_m and T_c are fitted by L-BFGS on development NLL only (`openworld/calibration.py`).

## 7. Open-World Screening (Eqs. 19–21) — `openworld/screening.py`

* **Score:** s^u = λ_p(1 − max p) + λ_d·d^rep + λ_x·d^TG. When the cross-view term is unavailable it is masked and the remaining λ renormalised.
* **d^rep (Eq. 20)** is a regularised Mahalanobis distance. Σ_c is the Ledoit–Wolf estimate shrunk 10% towards the pooled covariance.
* **Decision:** an observation is rejected (unknown/review) if s^u > τ_u (Eq. 21).
* *Decision:* μ_c and Σ_c are development-derived, as the paper states. The normaliser κ_c is the 95th percentile of **out-of-fold** development distances (2-fold cross-fitting). Measuring development samples against prototypes they helped build would otherwise make κ_c optimistic, and the threshold would not transfer to the test partition.
* *Decision:* (λ, τ_u) are selected on development data by **pseudo-unknown hold-out**.
  * Each known family takes a turn as the pseudo-unknown family: its prototype is excluded from Eq. (20) and its probability mass from the confidence term.
  * The configuration with the best mean known–unknown H-score is kept. The genuinely withheld family is never used.
  * `screening.selection: fixed` uses the Table A7 values (0.40/0.35/0.25, τ_u = 0.81) instead.
  * Baselines use the same hold-out on their own rejection scores.

## 8. Evidence-Controlled Replay Memory (Eqs. 22–24) — `replay/memory.py`

* **Admission (Eq. 23)** requires all of the following:
  * confidence ≥ τ_p (0.90) and s^u ≤ τ_u
  * ρ̄ ≥ τ_ρ (0.75) and an intact custody state (when the track has custody)
  * not final-test and not withheld-family
  * a verified label: a released label that agrees with the accepted decision. Disputed and unverified pseudo-labels are excluded.
* **Retention (Eq. 24)** is max-sum diversification with cosine distance on L2-normalised z^m, solved greedily (a ½-approximation). B_c = 25, γ = 0.10.
* Every admission, rejection, retention and eviction is logged.
* *Decision (chronological protocol):* the last 20% of the BODMAS timeline is split into four contiguous intervals. Each interval is split by duplicate group into an adaptation stream and a final-test subset. Evaluation is prequential, and only adaptation-stream observations can enter memory. LightGBM and MLP are trained once.

## 9. Counterfactual verification (Eqs. 25–28) — `counterfactual/`

* **Candidates** are *event factors* and *relation factors*. A relation factor is an entity (file, registry key, endpoint, process, artifact, custodian or device) together with all its related events, and needs at least two events. This reflects the manuscript's "events and relations" and the fact that behavioural evidence is often spread over several events of one entity.
* **Interventions** act on isolated deep copies: `mask`, `substitute` (frozen neutral templates), `correct` (custody), and `delete` (the unconstrained ablation only).
* **Validity** V checks schema, timestamp order, entity references, relation consistency and policy. Policy protects the initial context and process creations whose process acts later. An invalid intervention is reported as **unverifiable**, never as a zero effect.
* *Decision:* realisations are **paired**. In realisation r, the original and the intervened copy are evaluated under the same seeded MC-dropout state of the frozen model and the same report-level realisation noise (order-preserving timestamp jitter; neighbour swaps in the nondeterministic condition). Δ^(r) therefore isolates the intervention effect, and S (Eq. 27) measures how consistently it reproduces.
* **Scoring:** D is base-2 JS divergence; β_m = 0.30; R = 5.
  * Q is the fraction of realisations whose change has the expected direction.
  * A factor is verified when it is valid, Δ ≥ 0.10, S ≥ 0.80 and Q ≥ 0.80.
* **Selection** (`selection.py`): random; attention received; Shapley values from antithetic permutation sampling with an all-masked baseline; and responsibility (sensitivity, pooling weight, rule violations, localisation, with a validity prior). Factor scores aggregate event scores.

## 10. Objective and training (Eq. 29) — `training/`

* **Losses:**
  * L_m and L_c are cross-entropy.
  * L_loc is a listwise softmax loss over event positions of compromised histories.
  * L_agree is JS agreement of the branch predictions, applied only when both branches are available and have ρ̄ ≥ 0.75.
  * L_cal is the differentiable DCA penalty |mean confidence − accuracy|.
  * The L2 term covers all weight matrices.
* **Weights:** 1.0 / 1.0 / 0.5 / 0.1 / 0.05 / 1e-5, plus 0.2 for the branch auxiliary heads.
* **Stages:**
  1. Branches are initialised with their auxiliary objectives.
  2. The full Eq. (29) is optimised with AdamW (lr 3e-4, weight decay 1e-5), gradient clipping 1.0, a 5-epoch warm-up, cosine decay to 1e-6, at most 120 epochs, and patience 15 on the development objective.
  3. Temperatures, prototypes, screening coefficients and thresholds are selected on development data only.

## 11. Static configuration (BODMAS, EMBER) — `models/static.py`

* Encoder: [x·m ‖ m] → 512 → 256 → 128 with LayerNorm, GELU and dropout 0.2, followed by the calibrated head.
* Screening uses λ_x = 0, with λ_p and λ_d renormalised.
* For admission, the investigation reliability is the availability-aware Eq. (6).
* Heavy-tailed count columns (training max > 100) receive a signed log1p, and all columns are z-scored with training statistics, computed in chunks for EMBER.

## 12. Controlled custody histories (Table A3) — `data/custody_generator.py`

* **Workflow grammar:**
  * acquisition: acquire, verify, seal
  * intake: transfer and receive, optionally via a courier, then store
  * 0–4 examination cycles: retrieve → transfer → receive → examine/extract/analyse → transfer → receive → store
  * closing: report, optionally archive
* **Population:** 12 labs with registered staff, devices and locations.
* **Length:** 8 + Gamma(2.0, 7.05) events (rounded), truncated to [8, 40] by rejection so that no mass piles up at the maximum. This gives a mean of 20.43 recorded events over the 20,000 histories (≈ 20.4).
* **Record-time violations:** every violation is introduced at record time with the chain re-sealed, **except** hash mismatches, which are post-hoc content changes.
* **Subtle violations:** 15% of the non-hash violations are rule-invisible. Examples include an authorised-role actor from another lab, a predecessor that skips a record of the same item, a missing acknowledgement, an order-preserving timestamp shift, and removed optional provenance. This gives the rule-only detector about 88–90% recall, close to Table VI.
* **Benign irregularities:** 2.5% of intact histories contain a documented one (clock skew, holiday delay, courier witness). Each triggers a rule and gives the intact false-positive rate of about 2.3%.
* **Robustness conditions** (Table VI) degrade the *records* of test histories without changing labels: clock noise, missing optional attributes, alias-recorded custodians, predecessor-link noise, an extra violation, and all of these combined.
* **Integrity accuracy** is the accuracy of the technical custody *state* (8 states). In leave-one-violation-type-out, where the held-out state cannot be predicted by name, integrity accuracy is binary (intact vs compromised).
