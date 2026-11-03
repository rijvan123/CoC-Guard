# Custody data package, run logs and figure files

This page describes the three kinds of files that the code produces for archiving (for example on Zenodo): the controlled custody data package, the log files of every experiment run, and the figure files.

## 1. Controlled custody data package

```bash
cocguard export-custody --out CoC-Guard_custody_data --zip                      # generate (seed 11) and export
cocguard export-custody --out CoC-Guard_custody_data --from data/processed/custody   # export an existing track
cocguard export-custody --out CoC-Guard_custody_data --sources data/raw/cape/public_labels.csv
```

The package is self-describing: it carries its own `README.md`, `PROVENANCE.json` and `CHECKSUMS.sha256`.

| Folder / file | Content |
|---|---|
| `custody_histories/histories.jsonl.gz` | Canonical data. One JSON object per history: every custody record with its SHA-256 digest chain, ground-truth state, violation records and affected record indices. Loaded by `cocguard.data.custody_generator.load_histories`. |
| `custody_histories/manifest.csv` | One row per history: identifiers, partition, state, length, affected records, violation types, subtlety flags, benign irregularity, seed. |
| `custody_histories/generation.json`, `actors.csv`, `labs.csv` | Actor registry, laboratory map, generator configuration and seed. |
| `provenance_logs/events_{train,dev,test}.csv.gz` | The custody provenance logs: one row per custody record, with all recorded fields, the integer microsecond time stamp that enters the digest, the outcome of rules R1–R6, the verification vector of Eq. (5), failure reasons and the ground truth for the record. |
| `provenance_logs/rule_audit.csv.gz` | Every failed rule evaluation: rule, evaluated field, reason, and whether the record was actually touched by an injected violation. |
| `provenance_logs/chain_verification.csv` | One row per history: head digest, digest-chain verification, failed rules, rule-only decision, ground truth. |
| `splits/custody_{train,dev,test}.csv` | The 70/10/20 split of `cocguard custody` (both versions of a base history share a partition). |
| `splits/lovo/<type>.csv`, `splits/lovo/summary.csv` | Leave-one-violation-type-out assignment of every history for each of the six types, as used by `cocguard lovo`. |
| `perturbations/violation_records.csv` | One row per injected violation with its type, subtle flag, affected records and injection details (shift in seconds, replaced actor, removed field, omitted transfer, ...). |
| `perturbations/benign_irregularities.csv` | Intact histories with documented benign irregularities and the rules they trigger. |
| `perturbations/robustness/<condition>_seed<k>.csv` | For each robustness condition and training seed: what was altered in each test history (clock shifts, removed attributes, aliased identifiers, predecessor noise, added violation), the state after the condition and the rule outcome before and after. |
| `perturbations/robustness/histories/<condition>_seed<k>.jsonl.gz` | Degraded test histories (first seed by default; `--history-seeds` for more). |
| `statistics/*.csv` | Counts per state and partition, length distribution, injected and subtle violations per type, rule failure rates, rule-only detector on the test split. |
| `figures/*.{png,pdf,svg}` | Descriptive figures of the data (`cocguard custody-figures --package <dir>` redraws them). |
| `config/config_used.yaml` | Complete configuration. |
| `PROVENANCE.json` | Code version, git commit and whether the checkout had uncommitted changes, command line, configuration, seeds, software environment, inputs and every output with bytes, rows and SHA-256. |
| `CHECKSUMS.sha256` | `sha256sum -c CHECKSUMS.sha256` |

**Reproducibility.** Compressed files are written with a fixed gzip header, so regenerating the package with the same configuration and the same software versions reproduces every data file byte for byte (`tests/test_export_custody.py` checks this). Only `PROVENANCE.json`, `README.md` and `CHECKSUMS.sha256` change, because they record time and environment.

**Source identifiers.** Without `--sources`, the source identifier of each base history is a synthetic SHA-256 and its malware label is a placeholder family (`family_00` … `family_09`). With `--sources data/raw/cape/public_labels.csv`, they are drawn from the public CAPE label file. The custody records themselves do not depend on this choice: the label is carried by the history object and never enters a record.

**Using a released package in the experiments.**

```bash
cocguard prepare --track custody --histories CoC-Guard_custody_data/custody_histories --out data/processed/custody
cocguard custody
cocguard lovo
```

**Claim boundary.** The histories are controlled technical ground truth. They are not court records, contain no personal data and say nothing about legal admissibility in any jurisdiction.

## 2. Log files of every run

Every experiment command (`cocguard closed-world`, `custody`, `lovo`, ...) writes, next to its tables:

| File | Content |
|---|---|
| `run.log` | Complete log of the run (time-stamped). |
| `config_used.yaml` | Complete configuration after command-line overrides. |
| `environment.json` | Python, NumPy, pandas, scikit-learn, LightGBM and PyTorch versions, CUDA and GPU, platform, `cocguard` version, git commit and uncommitted-change flag, command line, configuration files and overrides. |
| `training_logs/<model>[_<tag>]_seed<k>_runNN.csv` | One row per epoch (episode blocks for the DOMR-style baseline): stage (branch initialisation or joint training), learning rate, training loss, development objective, elapsed seconds. The tag names the held-out violation type (LOVO), the condition and withheld family (open world) or the track (fusion and ablation). |
| `training_logs/<model>_seed<k>_runNN.json` | Summary: best development objective and epoch, epochs run, training time, trainable parameters, training and development sizes. |
| `runs.csv` | One row per method, seed and condition with every metric. |
| `predictions/<method>_seed<k>.npz` and `.csv` | Per-observation test outputs (true label, decision, scores, calibrated probability of the predicted class); the `.npz` also keeps full probability vectors. |

`cocguard prepare` also writes `splits/<track>_{train,dev,test}.csv` with the identifiers of every partition (SHA-256 and family for CAPE and BODMAS, memmap row and `metadata.csv` fields for EMBER, history identifiers for the custody track), so the exact splits of a run can be published without the underlying data.

## 3. Figure files

```bash
cocguard figures --source paper   --out figures_reported      # Figures 5–10 from the values reported in the manuscript
cocguard figures --source results --results results --out results/figures   # Figures 5–10 from your own runs
cocguard custody-figures --package CoC-Guard_custody_data     # figures of the custody data
```

Each figure is written as PNG (300 dpi), PDF (embedded TrueType fonts) and SVG, together with `figureN_source_data.csv`, the numbers drawn in it. Figures drawn with `--source paper` reproduce values reported in the manuscript; they are not new results.
