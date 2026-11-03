"""Command-line interface: ``cocguard <command> [options]``.

=================  ======================================================  ==================
command            purpose                                                  paper artefact
=================  ======================================================  ==================
synthetic          write format-faithful synthetic stand-ins (CI/smoke)     -
prepare            build a processed track from the official datasets       Tables III, A1-A3
export-custody     custody data package: provenance logs, splits, CSV        Table A3, Section IV-A
custody-figures    descriptive figures of an exported custody package        -
closed-world       modality-compatible closed-world recognition             Table IV, Fig. 5
open-world         completely family-withheld recognition                   Table V, Fig. 6
chronological      chronological stability with Replay Memory               Fig. 7
custody            custody integrity, localisation, robustness              Table VI, Fig. 8
lovo               leave-one-violation-type-out generalisation              Table VII
counterfactual     report-level intervention verification                   Table VIII, Fig. 9
fusion             fusion-strategy comparison                               Table IX
ablation           objective-specific ablation                              Table X, Fig. 10
sensitivity        sensitivity analysis                                     Section V-H
scalability        latency, throughput, memory, parameters                  Table XI(a)
workload           analyst workload and example audit records               Table XI(b)
statistics         paired bootstrap, McNemar, Wilcoxon, Holm                Table XII
figures            Figures 5-10 from results or from reported values        Figs. 5-10
compare            compare re-run tables with the manuscript                -
=================  ======================================================  ==================
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "synthetic": ("cocguard.experiments.prepare", "synthetic_main"),
    "prepare": ("cocguard.experiments.prepare", "main"),
    "export-custody": ("cocguard.experiments.export_custody", "main"),
    "custody-figures": ("cocguard.experiments.dataset_figures", "main"),
    "closed-world": ("cocguard.experiments.closed_world", "main"),
    "open-world": ("cocguard.experiments.open_world", "main"),
    "chronological": ("cocguard.experiments.chronological", "main"),
    "custody": ("cocguard.experiments.custody_exp", "main"),
    "lovo": ("cocguard.experiments.lovo", "main"),
    "counterfactual": ("cocguard.experiments.counterfactual_exp", "main"),
    "fusion": ("cocguard.experiments.fusion_ablation", "main_fusion"),
    "ablation": ("cocguard.experiments.fusion_ablation", "main_ablation"),
    "sensitivity": ("cocguard.experiments.sensitivity", "main"),
    "scalability": ("cocguard.experiments.scalability", "main"),
    "workload": ("cocguard.experiments.workload", "main"),
    "statistics": ("cocguard.experiments.statistics_exp", "main"),
    "figures": ("cocguard.experiments.figures", "main"),
    "compare": ("cocguard.experiments.figures", "compare_main"),
}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: cocguard <command> [--help]")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        raise SystemExit(f"unknown command {cmd!r}; run 'cocguard --help'")
    module, fn = COMMANDS[cmd]
    getattr(importlib.import_module(module), fn)(rest)


if __name__ == "__main__":
    main()
