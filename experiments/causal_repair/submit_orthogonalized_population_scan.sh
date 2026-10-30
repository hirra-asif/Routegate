#!/bin/bash -l
# The layer-wise causal activation steering scan, as a GPU array job.
#
# NOT SUBMITTED. Read it, decide the shape is right, then submit it yourself:
#
#     qsub experiments/causal_repair/submit_orthogonalized_population_scan.sh
#
# ---------------------------------------------------------------------------
# What this job runs
#
# 302 held-out targets (89 E + 213 H) x 48 transformer-block outputs = 14 496
# steered generations, plus 302 baselines, for 14 798 in total. B rows build the
# directions and are not targets here.
#
# State 0 is the embedding output before transformer block 0. It is not a
# transformer layer, it is not scanned, and it is not in any of the ranges
# below. The run script refuses it on the command line as well.
#
# Why an array job
#
# At greedy decoding with max_new_tokens=768 on a 30B-A3B MoE a generation is
# tens of seconds, so the whole scan is several days of GPU time -- far more than
# one job's wall clock. It splits cleanly along the state axis: each task loads
# the model once and sweeps its own block-output states over every target.
#
# The split is seven tasks of seven states, except the last, which has six:
#
#     task 1  states  1- 7   + every baseline
#     task 2  states  8-14
#     task 3  states 15-21
#     task 4  states 22-28
#     task 5  states 29-35
#     task 6  states 36-42
#     task 7  states 43-48   (six states)
#
# Together they cover states 1-48 exactly once. No state is chosen, weighted or
# ordered by anything from the earlier small intervention pilot; the tasks are a
# contiguous partition of the block axis and nothing else.
#
# The baseline does not depend on the state, so it is generated once per target
# and only by task 1. Every other task runs the intervention condition alone.
#
# Each task writes its own records file, so no two processes ever append to the
# same JSONL and the resume logic never sees a half-written line from a sibling.
# The records carry their own state, so merging afterwards is a concatenation.
# ---------------------------------------------------------------------------

#$ -N routegate_ortho_steer_scan
#$ -t 1-7
#$ -l h_rt=36:00:00
#$ -l mem=96G
#$ -l gpu=4
#$ -l tmpfs=30G
#$ -ac allow=L
#$ -cwd
#$ -o logs/ortho_steer_scan.$JOB_ID.$TASK_ID.out
#$ -j y

set -euo pipefail

# Submit from the repository root; -cwd starts the job there. Only $HOME is
# bound into the container, so use its spelling of the path.
REPO="$PWD"
HOME_PHYSICAL="$(cd "$HOME" && pwd -P)"
case "$REPO" in "$HOME_PHYSICAL"/*) REPO="$HOME/${REPO#"$HOME_PHYSICAL"/}" ;; esac
test -f "$REPO/pyproject.toml" && test -d "$REPO/src/routegate" || { echo "submit this script from the RouteGate repository root" >&2; exit 1; }
CONTAINER="${ROUTEGATE_CONTAINER_FILE:-$HOME/containers/vllm-openai-v0.10.1.1.sif}"

SCAN_ROOT="$REPO/outputs/causal_repair/orthogonalized_population_scan"
BANK="$SCAN_ROOT/direction_bank"

# --- the direction bank this scan is pinned to -----------------------------
# Build it first, on a login node -- it loads no model and needs no GPU -- and
# paste the manifest_sha256 it prints here. The run refuses any other bank, so
# an empty value stops the job rather than steering with directions nobody
# froze:
#     .venv/bin/python \
#       experiments/causal_repair/build_orthogonalized_direction_bank.py
BANK_SHA256="6e945d7349cb18c31a2d59e1984a309b72de05e601dc1493cb17a741329d570c"

if [ -z "$BANK_SHA256" ]; then
  echo "BANK_SHA256 is empty: build and freeze the direction bank first, then" >&2
  echo "paste its manifest_sha256 into this script. Refusing to run unpinned." >&2
  exit 2
fi

if [ ! -f "$BANK/manifest.json" ]; then
  echo "no direction bank at $BANK; build it before submitting." >&2
  exit 2
fi

# --- this task's slice of the block axis -----------------------------------
# States are 1-based block outputs, so the first state of task t is
# 7*(t-1) + 1. The last task is clamped to 48 and therefore holds six states.
STATES_PER_TASK=7
LAST_STATE=48
TASK="${SGE_TASK_ID:-1}"
FIRST=$(( (TASK - 1) * STATES_PER_TASK + 1 ))
LAST=$(( FIRST + STATES_PER_TASK - 1 ))
if [ "$LAST" -gt "$LAST_STATE" ]; then LAST="$LAST_STATE"; fi

if [ "$FIRST" -gt "$LAST_STATE" ]; then
  echo "task $TASK has no states to run; the array is 1-7 for 48 states." >&2
  exit 2
fi

# Only task 1 generates the state-independent baseline, once per target.
if [ "$TASK" -eq 1 ]; then
  CONDITIONS="baseline,orthogonalized_population_repair"
else
  CONDITIONS="orthogonalized_population_repair"
fi

OUTDIR="$SCAN_ROOT/scan/task_${TASK}"

echo "[task $TASK] block-output states ${FIRST}-${LAST}, conditions ${CONDITIONS}"
echo "[task $TASK] bank ${BANK} @ ${BANK_SHA256}"
echo "[task $TASK] output ${OUTDIR}"

apptainer exec --nv --cleanenv --bind "$HOME:$HOME" "$CONTAINER" \
  /bin/bash --noprofile --norc -lc "
    set -euo pipefail
    cd $REPO
    .venv/bin/python experiments/causal_repair/run_orthogonalized_population_scan.py \
      --all-targets \
      --states ${FIRST}-${LAST} \
      --conditions ${CONDITIONS} \
      --strength 0.2 \
      --budget-reference block_input \
      --direction-bank '${BANK}' \
      --expect-bank-sha256 '${BANK_SHA256}' \
      --output-dir '${OUTDIR}' \
      --resume
  "

# Re-running the same task resumes it: completed generations are skipped by key
# and a mismatched configuration, bank or dataset stops the job instead of
# appending rows that are not comparable to the ones already there.
