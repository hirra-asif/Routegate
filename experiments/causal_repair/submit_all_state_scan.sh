#!/bin/bash -l
# The all-state causal-patching discovery scan, as a GPU array job.
#
# NOT SUBMITTED. Read it, decide the shape is right, then submit it yourself:
#
#     qsub experiments/causal_repair/submit_all_state_scan.sh
#
# ---------------------------------------------------------------------------
# Why an array job
#
# The full scan is 89 E targets x 49 states = 4361 patched generations, plus 89
# baselines and whatever self-patch controls are asked for. At greedy decoding
# with max_new_tokens=768 on a 30B-A3B MoE, a generation is tens of seconds, so
# the whole scan is on the order of a day and a half of GPU time -- more than one
# job's wall clock. It splits cleanly by state: each task loads the model once
# and sweeps its own seven states over every target.
#
# Each task writes its own records file, so no two processes ever append to the
# same JSONL and the resume logic never sees a half-written line from a sibling.
# Merge afterwards; the records carry their own state, so a merge is a
# concatenation.
#
# The baseline does not depend on the state, so only task 1 generates it. The
# zero-delta self-patch control also runs on task 1, at its own states.
# ---------------------------------------------------------------------------

#$ -N routegate_patch_scan
#$ -t 1-7
#$ -l h_rt=36:00:00
#$ -l mem=96G
#$ -l gpu=4
#$ -l tmpfs=30G
#$ -ac allow=L
#$ -cwd
#$ -o logs/patch_scan.$JOB_ID.$TASK_ID.out
#$ -j y

set -euo pipefail

# Submit from the repository root; -cwd starts the job there. Only $HOME is
# bound into the container, so use its spelling of the path.
REPO="$PWD"
HOME_PHYSICAL="$(cd "$HOME" && pwd -P)"
case "$REPO" in "$HOME_PHYSICAL"/*) REPO="$HOME/${REPO#"$HOME_PHYSICAL"/}" ;; esac
test -f "$REPO/pyproject.toml" && test -d "$REPO/src/routegate" || { echo "submit this script from the RouteGate repository root" >&2; exit 1; }
CONTAINER="${ROUTEGATE_CONTAINER_FILE:-$HOME/containers/vllm-openai-v0.10.1.1.sif}"

# --- the manifest this scan is pinned to -----------------------------------
# Build it first, on a login node, and paste its hash here. The run refuses any
# other manifest, so an empty value stops the job rather than pairing at run
# time:
#     python experiments/causal_repair/build_donor_manifest.py
MANIFEST="$REPO/outputs/causal_repair/donor_manifest.json"
MANIFEST_SHA256="ea1741bea8b63b3cd20f8096ed4f1879fe7582e008560557ad93dfb9a6867c80"

if [ -z "$MANIFEST_SHA256" ]; then
  echo "MANIFEST_SHA256 is empty: build and freeze the donor manifest first," >&2
  echo "then paste its hash into this script. Refusing to run unpinned." >&2
  exit 2
fi

# --- this task's slice of the state axis -----------------------------------
STATES_PER_TASK=7
TASK="${SGE_TASK_ID:-1}"
FIRST=$(( (TASK - 1) * STATES_PER_TASK ))
LAST=$(( FIRST + STATES_PER_TASK - 1 ))
if [ "$LAST" -gt 48 ]; then LAST=48; fi

# Every task runs one zero-delta self-patch control near the centre of its
# state range. Only task 1 runs the state-independent baseline.
CONTROL_STATE=$(( (FIRST + LAST) / 2 ))
if [ "$TASK" -eq 1 ]; then
  CONDITIONS="baseline,self_patch,donor_patch"
else
  CONDITIONS="self_patch,donor_patch"
fi

OUTDIR="$REPO/outputs/causal_repair/scan/task_${TASK}"

echo "[task $TASK] states ${FIRST}-${LAST}, conditions ${CONDITIONS}"
echo "[task $TASK] output ${OUTDIR}"

apptainer exec --nv --cleanenv --bind "$HOME:$HOME" "$CONTAINER" \
  /bin/bash --noprofile --norc -lc "
    set -euo pipefail
    cd $REPO
    .venv/bin/python experiments/causal_repair/run_patching.py \
      --all-development-targets \
      --states ${FIRST}-${LAST} \
      --self-patch-states ${FIRST} \
      --conditions ${CONDITIONS} \
      --strength 0.2 \
      --budget-reference block_input \
      --donor-manifest '${MANIFEST}' \
      --expect-manifest-sha256 '${MANIFEST_SHA256}' \
      --output-dir '${OUTDIR}' \
      --resume
  "

# Re-running the same task resumes it: completed generations are skipped by key
# and a mismatched configuration, manifest or dataset stops the job instead of
# appending rows that are not comparable to the ones already there.
