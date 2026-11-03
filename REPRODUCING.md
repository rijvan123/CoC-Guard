# Reproducing the experiments

All experiments read processed tracks from `paths.processed` and write to `paths.results/<experiment>/`. They use the matched seeds {11, 23, 37, 53, 71} unless `--seeds` is given. Each run directory contains:

* `run.log`: the complete, time-stamped log of the run
* `config_used.yaml` and `environment.json` (versions, hardware, git commit, command line)
* `training_logs/`: one per-epoch CSV and one JSON summary for every trained model
* `runs.csv`: one row per method, seed and condition
* the aggregated table as `.csv` and `.md` (mean ± SD over seeds, values in %)
* `predictions/`: per-observation outputs (`.npz` and `.csv`) used by the statistical tests

`cocguard prepare` writes `splits/<track>_{train,dev,test}.csv` with the identifiers of each partition. The custody set can be released with `cocguard export-custody` (see `docs/DATA_PACKAGE.md`).

## Order of execution

```
prepare (4 tracks) → closed-world → open-world → chronological → custody → lovo → counterfactual
                   → fusion → ablation → sensitivity → scalability → workload → statistics → figures → compare
```

`counterfactual` and `workload` reuse the CAPE checkpoints saved by `closed-world` (`results/closed_world/checkpoints/`). They train a model themselves if no checkpoint is found. `sensitivity --groups tau_u` reads the saved family-withheld scores, and `statistics` reads the saved predictions of the other experiments.

## Protocols

| Experiment | Track | Protocol |
|---|---|---|
| Closed world (Table IV) | BODMAS (binary, chronological 70/10/20), CAPE (10 families, 70/10/20 SHA-256 groups), EMBER (official split; development 60k rows from train) | Temperatures fitted on development data; accuracy, macro P/R/F1 and AUROC on the final test set |
| Family-withheld (Table V) | CAPE | The target family is removed from train and dev and appears only in the test partition. `--targets auto` withholds one family per seed; `--targets all` withholds every family. The rare, future, 30% behaviour-loss and cross-view-disagreement conditions are run for DOMR-style, Mecha-style and CoC-Guard |
| Chronological (Fig. 7) | BODMAS | Four future intervals of the test period; adaptation streams and final-test subsets; prequential evaluation; replay with B_c = 25 and γ = 0.10 |
| Custody (Table VI) | Controlled histories | 14,000/2,000/4,000 split by base history; rule-only, BiLSTM, GCN, direct concatenation and CoC-Guard; six record-degradation conditions on the test histories |
| LOVO (Table VII) | Controlled histories | Six rounds; the held-out type never enters train or dev; the test set has held-out-type histories and the intact versions of the same base histories; 10,000 type-stratified bootstrap replicates |
| Counterfactual (Table VIII) | CAPE | Frozen calibrated model; reference positives from exhaustive constrained intervention; budget of \|ref\| + 5 per investigation; five challenging conditions; unconstrained ablation |
| Fusion (Table IX) | CAPE + custody | Seven fusion modes with identical encoders, partitions and seeds |
| Ablation (Table X) | CAPE + custody | One component removed at a time. Values that cannot change are flagged *a* (not activated on the track) or *b* (downstream, unchanged by design) |
| Sensitivity | CAPE + custody | τ_u from 0.60 to 0.95 (from saved scores); graph depth 1–4; behavioural window 32–256; R ∈ {1, 3, 5, 7, 10} |
| Scalability (Table XI) | synthetic sizes | Batch size 1; 100 warm-up and 1,000 timed runs; median latency; CUDA peak memory; parameter count; routine latency at about 32 events, 160 nodes and 410 relations |
| Statistics (Table XII) | saved predictions | Paired stratified bootstrap (10,000 replicates), McNemar, exact Wilcoxon over seeds (exploratory; the minimum p with 5 pairs is 0.0625), Holm within comparison families |

## Compute

The paper used one RTX 3090 (24 GB), a 24-core Xeon and 128 GB of RAM (Table A8). CAPE preprocessing of the full reports benefits from `--workers`. Some full reports are very large; the reduced reports can be used with a warning. The open-world, fusion and ablation experiments train several CAPE models per seed and dominate the runtime. Use `--seeds 11` and `--targets auto` for a first pass.

## Determinism

Seeds fix the data order, initialisation and all sampling. For bit-wise repeatable GPU runs, set `deterministic: true`; this is slower. LightGBM uses `random_state=seed`.

## Comparing with the manuscript

`cocguard compare --results results` writes `comparison_with_manuscript.csv` with the reported value, the re-run value and their difference for Tables IV, V(a), VI, VIII, X and Figure 7. The reported values are in `paper_results/reported_values.json`. Entries read from Figure 7 rather than tabulated are flagged as approximate.
