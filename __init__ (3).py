"""CoC-Guard: reliability-aware open-world malware analysis with technical
chain-of-custody assessment.

Reference implementation of the framework described in

    N. Nigam, R. Beg, G. Patidar, A. Sahu, C. Devkar, M. S. Khan, R. Soni,
    "Reliability Aware Open World Malware Analysis with Technical Chain of
    Custody Assessment".

The package is organised by the stages of Algorithm 1:

* :mod:`cocguard.custody`        deterministic verification and rule reliability (Eq. 4-6)
* :mod:`cocguard.models`         learned reliability, temporal and graph encoders,
                                 Reliability-Controlled Fusion and calibrated heads (Eq. 7-18)
* :mod:`cocguard.openworld`      calibration and Open-World Screening (Eq. 19-21)
* :mod:`cocguard.replay`         Evidence-Controlled Replay Memory (Eq. 22-24)
* :mod:`cocguard.counterfactual` report-level intervention verification (Eq. 25-28)
* :mod:`cocguard.audit`          structured audit records (Section III-K)

CoC-Guard performs *technical* custody-integrity assessment only. It does not
determine legal admissibility.
"""
from .version import __version__

__all__ = ["__version__"]
