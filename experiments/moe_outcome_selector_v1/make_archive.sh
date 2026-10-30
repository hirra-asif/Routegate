#!/usr/bin/env bash
# Archive the experiment code together with the lightweight results of one run version.
# Excludes activation arrays, router logits and anything unrelated to this study.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo="$(cd -- "$task_dir/../.." && pwd)"
task_version="${1:-v1}"
task_data="${ROUTEGATE_DATA_DIR:-$HOME/routegate-data}"
task_results="$task_data/moe_outcome_selector_v1/$task_version"
test -d "$task_results" || { echo "No such run version: $task_results" >&2; exit 1; }
test -f "$task_results/COMPLETE.json" || echo "warning: $task_version has no COMPLETE.json" >&2
task_out="$task_data/moe_outcome_selector_v1_${task_version}.tar.gz"
task_stage="$(mktemp -d)"
trap 'rm -rf "$task_stage"' EXIT
task_root="$task_stage/moe_outcome_selector_v1_$task_version"
mkdir -p "$task_root/code" "$task_root/results"
cp "$task_dir"/*.py "$task_dir"/*.json "$task_dir"/*.md "$task_dir"/*.sh "$task_root/code/"
# Lightweight results only: no .npy, no activation or logit array, nothing over 32 MB.
find "$task_results" -maxdepth 1 -type f \
  ! -name '*.npy' ! -name '*.npz' ! -name '*.pt' ! -name '*.safetensors' \
  -size -32M -exec cp {} "$task_root/results/" \;
tar --numeric-owner --owner=0 --group=0 -czf "$task_out" -C "$task_stage" "moe_outcome_selector_v1_$task_version"
echo "archive: $task_out"
echo "bytes:   $(stat -c %s "$task_out")"
echo "sha256:  $(sha256sum "$task_out" | cut -d' ' -f1)"
