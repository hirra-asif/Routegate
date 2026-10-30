#!/bin/bash -l
# The preregistered dual-direction causal screen, as a GPU array job.
#
# NOT SUBMITTED. Read it, decide the shape is right, then submit it yourself:
#
#     qsub experiments/dual_direction_controller/submit_causal_screen.sh
#
# ---------------------------------------------------------------------------
# What this job runs
#
# 3,214 generations over the 480-row development corpus:
#
#      89 E rows x  5 arms  =   445   baseline + repair 0.05/0.10/0.15/0.20
#     213 H rows x 13 arms  = 2 769   the same repair ladder, plus the safety
#                                     ladder 0.05/0.10/0.15/0.20, plus four
#                                     combined arms at alpha_R = 0.15
#                             -----
#                             3 214
#
# B rows built the historical directions and are not steered here.
#
# The edit, at historical layers 31/32/33 = states 32/33/34, at the final
# prompt token, prefill only:
#
#     delta = ||block_input|| * (alpha_R * d_R + alpha_S * d_S)
#
# The combination is not renormalised and neither direction is orthogonalised.
# The run refuses to start if that formula disagrees with the package's own
# single-direction actuator convention at alpha_S = 0.
#
# This stage generates. It does not judge: no response is scored, labelled or
# classified here, and the judging stage is separate and not yet specified.
#
# Why an array job
#
# At greedy decoding with max_new_tokens=768 on a 30B-A3B MoE a generation is
# tens of seconds, so 3,214 of them is well beyond one job's wall clock. The
# inventory splits along the row axis into eight contiguous shards of about 402
# generations each. Contiguous, not round-robin: a row's prompt states are
# captured once and reused across all of that row's arms, so keeping a row's
# arms together in one shard avoids re-capturing them.
#
# Each task writes its own records file, so no two processes ever append to the
# same JSONL and the resume logic never sees a half-written line from a sibling.
# The records carry their own row, class and arm, so merging afterwards is a
# concatenation.
# ---------------------------------------------------------------------------

#$ -N routegate_dual_direction_screen
#$ -t 1-8
#$ -l h_rt=36:00:00
#$ -l mem=96G
#$ -l gpu=4
#$ -l tmpfs=30G
#$ -ac allow=L
#$ -wd /home/ucabh21/routegate
#$ -o logs/dual_direction_screen.$JOB_ID.$TASK_ID.out
#$ -j y

set -euo pipefail

REPO="/home/ucabh21/routegate"
CONTAINER="$HOME/containers/vllm-openai-v0.10.1.1.sif"

BANK="$HOME/routegate-data/frozen/validated_direction_bank"
SCREEN_ROOT="$HOME/routegate-data/dual_direction_causal_screen"

# --- the direction bank this screen is pinned to ---------------------------
# Built on a login node by build_validated_direction_bank.py, which loads no
# model. The screen refuses any other artefact, so a wrong or empty value stops
# the job rather than steering with directions nobody froze.
BANK_SHA256="d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb"

if [ -z "$BANK_SHA256" ]; then
  echo "BANK_SHA256 is empty: build and freeze the direction bank first, then" >&2
  echo "paste its validated_direction_bank.npz digest here. Refusing to run." >&2
  exit 2
fi

if [ ! -f "$BANK/validated_direction_bank.npz" ]; then
  echo "no direction bank at $BANK; build it before submitting." >&2
  exit 2
fi

# --- the commit every task must run ----------------------------------------
# All eight tasks read the same working tree on a shared filesystem, so an edit
# made while the array is queued would silently split the run across two
# versions of the code. Pin the commit that carries the screen and refuse both
# a different HEAD and a dirty tree.
#
# Set this to the commit that carries run_causal_screen.py, dualsteer.py and
# the frozen causal_screen configuration. Until that commit exists the array
# refuses to start, which is the intended behaviour: the screen is not run from
# uncommitted code.
EXPECT_HEAD=""

if [ -z "$EXPECT_HEAD" ]; then
  echo "EXPECT_HEAD is empty: commit the causal-screen code, then paste its" >&2
  echo "commit here. The screen is not run from an uncommitted tree." >&2
  exit 2
fi

cd "$REPO"
HEAD_NOW="$(git rev-parse HEAD)"
case "$HEAD_NOW" in
  "$EXPECT_HEAD"*) ;;
  *) echo "HEAD is $HEAD_NOW, and this array is pinned to $EXPECT_HEAD." >&2
     echo "Every task must run the same committed code; refusing." >&2
     exit 2 ;;
esac

if [ -n "$(git status --porcelain)" ]; then
  echo "the working tree is dirty:" >&2
  git status --porcelain >&2
  echo "All eight tasks share this tree. Commit or stash first; refusing." >&2
  exit 2
fi

SHARD_COUNT=8
TASK="${SGE_TASK_ID:-1}"

if [ "$TASK" -lt 1 ] || [ "$TASK" -gt "$SHARD_COUNT" ]; then
  echo "task $TASK is outside the 1-$SHARD_COUNT array." >&2
  exit 2
fi

OUTDIR="$SCREEN_ROOT/shard_${TASK}"

echo "[task $TASK] shard ${TASK}/${SHARD_COUNT} of 3214 generations"
echo "[task $TASK] HEAD ${HEAD_NOW} (pinned ${EXPECT_HEAD}), tree clean"
echo "[task $TASK] bank ${BANK} @ ${BANK_SHA256}"
echo "[task $TASK] output ${OUTDIR}"

# One distinct directory per task, so no two processes append to one JSONL and
# the resume logic never sees a sibling's half-written line.
if [ -e "$OUTDIR" ] && [ ! -d "$OUTDIR" ]; then
  echo "$OUTDIR exists and is not a directory." >&2
  exit 2
fi

apptainer exec --nv --cleanenv --bind "$HOME:$HOME" "$CONTAINER" \
  /bin/bash --noprofile --norc -lc "
    set -euo pipefail
    cd $REPO
    .venv/bin/python experiments/dual_direction_controller/run_causal_screen.py \
      --direction-bank '${BANK}' \
      --expect-bank-sha256 '${BANK_SHA256}' \
      --output-dir '${OUTDIR}' \
      --shard-index ${TASK} \
      --shard-count ${SHARD_COUNT} \
      --device auto \
      --resume
  "

# No --only-class, --only-arm, --only-row-id or --limit is passed here. Those
# make a run a smoke test, every record it writes is marked as one, and
# consolidate_causal_screen.py refuses to treat such a shard as the screen.

# Re-running the same task resumes it: completed generations are skipped by the
# seven-field key -- row, class, arm, both coefficients, the bank digest and the
# generation-configuration digest -- and a mismatched bank, corpus, model,
# tokenizer or chat template stops the job instead of appending rows that are
# not comparable to the ones already there.
#
# Results land under routegate-data, outside git, as the repository's output
# policy requires. Nothing here is committed.
#
# The sealed evaluation set is not reachable from this job: no path above names
# it, the runner exposes no option that could, and it calls assert_not_sealed
# on the corpus it does read.
#
# When all eight tasks are done:
#
#     .venv/bin/python \
#       experiments/dual_direction_controller/consolidate_causal_screen.py \
#       --shards-root "$SCREEN_ROOT" --output-dir "$SCREEN_ROOT/consolidated"
