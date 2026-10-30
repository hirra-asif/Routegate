# Retired experiments

The guarded/angular/SIGR pilot was removed from the active source tree on
14 September 2026. Its six package modules, five tests, five experiment files
and three pilot manifests are archived in the installation backup outside the
repository. Completed model outputs and judgements are preserved separately.

The independent H−E rotation diagnostic remains part of the repository. Frozen
final-evaluation and H-transition records retain their historical method names
and exclusions; their contents and hashes have not been edited.

The H-transition manifest builder now reads the historical residual/routing
vetoes from `routegate.historicalveto`. The extracted loader reproduces the
original reader's scores, thresholds, decisions and provenance on all 480 rows.
It retains the original input hashes and 1e-9 threshold-boundary convention.
This historical two-veto diagnostic is distinct from the final RouteGate gate.

The source-cleanup installer writes an exact inventory and checksums into its
backup beside the repository, including the archived pilot files and originals
of modified files. Its location is printed at installation time.
