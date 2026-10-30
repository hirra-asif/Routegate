#!/usr/bin/env bash
# Run from the Myriad shell. The container receives credentials via APPTAINERENV_.
set -euo pipefail
cd /home/ucabh21/routegate
ACTION="${1:-progress}"
PHASE="${2:-baseline}"
case "$PHASE" in baseline|responses) ;; *) echo "Phase must be baseline or responses"; exit 2;; esac
RUN=(apptainer exec --cleanenv --bind "$HOME:$HOME"
     "$HOME/containers/vllm-openai-v0.10.1.1.sif"
     /home/ucabh21/routegate/.venv/bin/python
     /home/ucabh21/routegate/experiments/final_evaluation/run_sealed_judging.py
     --phase "$PHASE")
case "$ACTION" in
  prepare)
    "${RUN[@]}" --stage prepare --dry-run
    "${RUN[@]}" --stage prepare
    "${RUN[@]}" --stage preflight
    "${RUN[@]}" --stage build --dry-run
    "${RUN[@]}" --stage build
    "${RUN[@]}" --stage dryrun
    ;;
  judge)
    "${RUN[@]}" --stage judge --judge sonnet --dry-run
    "${RUN[@]}" --stage judge --judge opus --dry-run
    "${RUN[@]}" --stage judge --judge sonnet
    "${RUN[@]}" --stage judge --judge opus
    "${RUN[@]}" --stage progress
    ;;
  review)
    "${RUN[@]}" --stage join --dry-run
    "${RUN[@]}" --stage join
    "${RUN[@]}" --stage export --dry-run
    "${RUN[@]}" --stage export
    ;;
  import)
    EXTRA=()
    if [ "$#" -ge 3 ]; then EXTRA=(--completed "$3"); fi
    "${RUN[@]}" --stage import "${EXTRA[@]}" --dry-run
    "${RUN[@]}" --stage import "${EXTRA[@]}"
    "${RUN[@]}" --stage report
    ;;
  progress)
    "${RUN[@]}" --stage progress
    ;;
  *) echo "Action must be prepare, judge, review, import or progress"; exit 2;;
esac
