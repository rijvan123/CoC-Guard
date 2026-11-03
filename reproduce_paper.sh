#!/usr/bin/env bash
# Full experimental protocol of the paper on the OFFICIAL datasets (see data/README.md).
# Expected layout:
#   data/raw/bodmas/{bodmas.npz,bodmas_metadata.csv[,bodmas_malware_category.csv]}
#   data/raw/cape/{public_labels.csv,public_full_reports/*.json}
#   data/raw/ember2018/{X_train.dat,y_train.dat,X_test.dat,y_test.dat,metadata.csv}  (vectorised, feature v2)
# Hardware used in the paper (Table A8): 1x RTX 3090 (24 GB), Xeon Gold 6248R, 128 GB RAM.
# Runtime is dominated by the CAPE track (open-world, fusion and ablation retrain several models per seed).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/default.yaml}
RAW=${RAW:-data/raw}
PROC=data/processed
WORKERS=${WORKERS:-8}
PY=${PYTHON:-python}

$PY -m cocguard prepare --track bodmas  --raw "$RAW/bodmas"    --out $PROC/bodmas  --config $CFG
$PY -m cocguard prepare --track cape    --raw "$RAW/cape"      --out $PROC/cape    --config $CFG --workers $WORKERS
$PY -m cocguard prepare --track ember   --raw "$RAW/ember2018" --out $PROC/ember   --config $CFG
$PY -m cocguard prepare --track custody --out $PROC/custody --config $CFG --sources "$RAW/cape/public_labels.csv"

$PY -m cocguard closed-world   --config $CFG                       # Table IV, Fig. 5
$PY -m cocguard open-world     --config $CFG ${OPEN_WORLD_ARGS:-}  # Table V, Fig. 6 (add OPEN_WORLD_ARGS="--targets all" for every family)
$PY -m cocguard chronological  --config $CFG                       # Fig. 7
$PY -m cocguard custody        --config $CFG                       # Table VI, Fig. 8
$PY -m cocguard lovo           --config $CFG                       # Table VII
$PY -m cocguard counterfactual --config $CFG                       # Table VIII, Fig. 9
$PY -m cocguard fusion         --config $CFG                       # Table IX
$PY -m cocguard ablation       --config $CFG                       # Table X, Fig. 10
$PY -m cocguard sensitivity    --config $CFG                       # Section V-H
$PY -m cocguard scalability    --config $CFG                       # Table XI(a)
$PY -m cocguard workload       --config $CFG                       # Table XI(b) + example audit records
$PY -m cocguard statistics     --config $CFG                       # Table XII
$PY -m cocguard figures --results results --out results/figures    # Figs. 5-10
$PY -m cocguard compare --results results                          # re-run vs. manuscript values
