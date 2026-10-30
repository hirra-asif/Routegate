#!/usr/bin/env bash
# An isolated Opus workspace; original frozen scripts and locks stay intact.
set -euo pipefail
cd /home/ucabh21/routegate
RG_ACTION="${1:-judge}"
RG_SOURCE_DATA=/home/ucabh21/routegate-data
RG_OPUS_DATA="$RG_SOURCE_DATA/sealed_opus_parallel_v1"
RG_PYTHON=(apptainer exec --cleanenv --bind "$HOME:$HOME"
  "$HOME/containers/vllm-openai-v0.10.1.1.sif"
  /home/ucabh21/routegate/.venv/bin/python)
RG_HELPER=/home/ucabh21/routegate/experiments/final_evaluation/parallel_opus_workspace.py
RG_RUNNER=/home/ucabh21/routegate/experiments/final_evaluation/run_sealed_judging.py

case "$RG_ACTION" in
  judge)
    "${RG_PYTHON[@]}" "$RG_HELPER" prepare
    MODEL_API_KEY="${MODEL_API_KEY-}"
    if [ "${#MODEL_API_KEY}" -ne 51 ]; then
      unset MODEL_API_KEY APPTAINERENV_MODEL_API_KEY
      IFS= read -r -s -p "Judge API key: " MODEL_API_KEY
      echo
    fi
    if [ "${#MODEL_API_KEY}" -ne 51 ]; then
      echo "ERROR: expected 51 key characters; no judge calls made."
      unset MODEL_API_KEY APPTAINERENV_MODEL_API_KEY
      exit 1
    fi
    export MODEL_API_KEY
    export APPTAINERENV_MODEL_API_KEY="$MODEL_API_KEY"
    export ROUTEGATE_JUDGE_ENDPOINT="${ROUTEGATE_JUDGE_ENDPOINT:-https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke}"
    export APPTAINERENV_ROUTEGATE_JUDGE_ENDPOINT="$ROUTEGATE_JUDGE_ENDPOINT"
    echo "Starting Opus in its separate workspace. Sonnet can continue."
    "${RG_PYTHON[@]}" "$RG_RUNNER" --phase responses --stage judge --judge opus --data-root "$RG_OPUS_DATA"
    ;;
  progress)
    "${RG_PYTHON[@]}" "$RG_RUNNER" --phase responses --stage progress --data-root "$RG_OPUS_DATA"
    ;;
  merge)
    "${RG_PYTHON[@]}" "$RG_HELPER" merge
    "${RG_PYTHON[@]}" "$RG_RUNNER" --phase responses --stage progress --data-root "$RG_SOURCE_DATA"
    ;;
  *) echo "Usage: bash experiments/final_evaluation/run_opus_parallel.sh [judge|progress|merge]"; exit 2 ;;
esac
