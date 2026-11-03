#!/usr/bin/env bash
# End-to-end smoke reproduction on format-faithful SYNTHETIC stand-ins.
# Purpose: verify that every stage (data preparation -> experiments -> statistics -> figures)
# runs on your machine in minutes on a CPU. The numbers it produces are NOT comparable to the
# paper; use scripts/reproduce_paper.sh with the official datasets for that.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/smoke.yaml}
SYN=${SYN:-data/synthetic}
PROC=data/processed_smoke
RES=results_smoke
PY=${PYTHON:-python}

$PY -m cocguard synthetic --out "$SYN" --cape-reports 300 --bodmas-rows 1500 --ember-train 2000 --ember-test 500
$PY -m cocguard prepare --track cape    --raw "$SYN/cape"      --out $PROC/cape    --config $CFG
$PY -m cocguard prepare --track bodmas  --raw "$SYN/bodmas"    --out $PROC/bodmas  --config $CFG
$PY -m cocguard prepare --track ember   --raw "$SYN/ember2018" --out $PROC/ember   --config $CFG
$PY -m cocguard prepare --track custody --out $PROC/custody --config $CFG --sources "$SYN/cape/public_labels.csv"

for exp in closed-world open-world chronological custody lovo counterfactual fusion ablation; do
  echo "=== $exp ==="
  $PY -m cocguard $exp --config $CFG
done
$PY -m cocguard sensitivity  --config $CFG --groups tau_u repetitions
$PY -m cocguard scalability  --config $CFG
$PY -m cocguard workload     --config $CFG --cases 5
$PY -m cocguard statistics   --config $CFG
$PY -m cocguard figures --results $RES --out $RES/figures
$PY -m cocguard compare --results $RES
echo "Smoke reproduction finished: see $RES/"
