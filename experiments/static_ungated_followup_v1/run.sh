#!/usr/bin/env bash
# Compatible with Myriad's older Bash: no expansion of an empty flags array.
set -euo pipefail
task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_container="$HOME/containers/vllm-openai-v0.10.1.1.sif"
task_action="${1:-status}"
if [ "$#" -gt 0 ]; then shift; fi
cd "$task_repo_root"

run_cpu() {
  apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_container" \
    "$task_repo_root/.venv/bin/python" \
    "$task_repo_root/experiments/static_ungated_followup_v1/run.py" "$@"
}

case "$task_action" in
  selftest)
    apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_container" \
      /bin/bash --noprofile --norc -c \
      'cd "$1"; export PYTHONPATH="$1/src"; exec "$1/.venv/bin/python" -m unittest discover -s tests -p test_static_ungated_followup.py -v' \
      bash "$task_repo_root"
    ;;
  generate)
    apptainer exec --nv --cleanenv --bind "$HOME:$HOME" "$task_container" \
      "$task_repo_root/.venv/bin/python" \
      "$task_repo_root/experiments/static_ungated_followup_v1/run.py" --stage generate "$@"
    ;;
  preflight|init|status|progress|report)
    run_cpu --stage "$task_action" "$@"
    ;;
  prepare-judging)
    run_cpu --stage prepare --dry-run "$@"
    run_cpu --stage prepare "$@"
    run_cpu --stage build --dry-run "$@"
    run_cpu --stage build "$@"
    run_cpu --stage dryrun "$@"
    ;;
  judge)
    task_judge="${1:?Specify sonnet or opus}"
    shift
    run_cpu --stage judge --judge "$task_judge" "$@"
    ;;
  review)
    run_cpu --stage join --dry-run "$@"
    run_cpu --stage join "$@"
    run_cpu --stage export --dry-run "$@"
    run_cpu --stage export "$@"
    ;;
  import)
    run_cpu --stage import --dry-run "$@"
    run_cpu --stage import "$@"
    run_cpu --stage report
    ;;
  export-results)
    tar -czf "$HOME/routegate_static_ungated_results.tar.gz" \
      -C "$HOME/routegate-data" static_ungated_followup_v1
    echo "Archive: $HOME/routegate_static_ungated_results.tar.gz"
    ;;
  *) echo "Use selftest, preflight, init, generate, status, prepare-judging, judge sonnet|opus, progress, review, import, report, or export-results."; exit 2;;
esac
