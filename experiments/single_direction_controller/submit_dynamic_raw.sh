#!/bin/bash -l
# The one-direction dynamic controller arm, as a single four-GPU job.
#
# NOT SUBMITTED. Read it, decide the shape is right, then submit it yourself:
#
#     qsub -v ROUTEGATE_EXPECT_HEAD=$(git rev-parse HEAD) \
#          experiments/single_direction_controller/submit_dynamic_raw.sh
#
# ---------------------------------------------------------------------------
# What this job runs
#
# 480 generations: every frozen development prompt, once, under dynamic_raw.
#
# Per prompt, in the pass that generates the response:
#
#   1. capture the UNSTEERED state-32 final-prompt-token activation, before
#      either site is written;
#   2. compute one g_R from that vector, through the fold checkpoint that
#      never trained on this row;
#   3. add  0.15 * g_R * ||h^(l)|| * d_R,l  at historical blocks 31 and 32 --
#      the same coefficient at both sites.
#
# Historical block 33 is NOT a site: an edit at 31 changes the activation
# entering it, so its unsteered budget does not exist in the same pass.
#
# The coefficient is never looked up by row id. gate_oof_v1's coefficient
# column is not opened at generation time; the same checkpoints score the
# frozen extraction's own state-32 activations offline, and that reference is
# what the live coefficient is CHECKED against -- per fold before the model
# loads, per row while it runs. A row whose live coefficient does not
# reproduce its reference within 32 float32 epsilons stops the job.
#
# This stage generates. It does not judge: no response is scored, labelled or
# classified here, nothing is trained, and the held-out split is not reachable
# from any path this job names.
#
# Why one task and not an array
#
# The runner shards (--shard-index k --shard-count n) and every shard writes
# its own records file. This job is the unsharded form the result is defined
# by: one process, one deterministic 480-row order, one records file. If the
# wall clock proves too short, re-submit the identical job -- it resumes, and
# a row already recorded is never regenerated.
# ---------------------------------------------------------------------------

#$ -N routegate_dynamic_raw
#$ -l h_rt=24:00:00
#$ -l mem=96G
#$ -l gpu=4
#$ -l tmpfs=30G
#$ -ac allow=L
#$ -cwd
#$ -o logs/dynamic_raw.$JOB_ID.out
#$ -j y

set -euo pipefail

# Submit from the repository root; -cwd starts the job there. Only $HOME is
# bound into the container, so use its spelling of the path.
REPO="$PWD"
HOME_PHYSICAL="$(cd "$HOME" && pwd -P)"
case "$REPO" in "$HOME_PHYSICAL"/*) REPO="$HOME/${REPO#"$HOME_PHYSICAL"/}" ;; esac
test -f "$REPO/pyproject.toml" && test -d "$REPO/src/routegate" || { echo "submit this script from the RouteGate repository root" >&2; exit 1; }
CONTAINER="${ROUTEGATE_CONTAINER_FILE:-$HOME/containers/vllm-openai-v0.10.1.1.sif}"

BANK="$HOME/routegate-data/frozen/validated_direction_bank"
CONTROLLERS="$HOME/routegate-data/gate_oof_v1_repro"
OUTDIR="$HOME/routegate-data/dynamic_arms_v1"

if [ ! -f "$BANK/validated_direction_bank.npz" ]; then
  echo "no direction bank at $BANK; this arm reads it as built." >&2
  exit 2
fi
if [ ! -f "$CONTROLLERS/controller_fold_0.json" ]; then
  echo "no controller bundle at $CONTROLLERS. Run" >&2
  echo "  reproduce_gate_controller.py --output-dir $CONTROLLERS" >&2
  echo "first: gate_oof_v1 saved no weights." >&2
  exit 2
fi

# --- the commit this job must run ------------------------------------------
# The job reads the working tree on a shared filesystem, so an edit made while
# it is queued would silently change what ran. The commit is therefore pinned,
# and the pin comes from the SUBMISSION rather than from a line in this file:
#
#     qsub -v ROUTEGATE_EXPECT_HEAD=$(git rev-parse HEAD) \
#          experiments/single_direction_controller/submit_dynamic_raw.sh
#
# Editing a constant here instead would be circular -- setting it dirties the
# tree this script refuses, and committing the edit moves the hash it was set
# to. Passing it at submission pins the commit that is actually checked out,
# with no edit at all. ROUTEGATE_EXPECT_HEAD may be any prefix of the commit.
EXPECT_HEAD="${ROUTEGATE_EXPECT_HEAD:-}"

if [ -z "$EXPECT_HEAD" ]; then
  echo "ROUTEGATE_EXPECT_HEAD is unset. Submit with" >&2
  echo "  qsub -v ROUTEGATE_EXPECT_HEAD=\$(git rev-parse HEAD) \\" >&2
  echo "       experiments/single_direction_controller/submit_dynamic_raw.sh" >&2
  echo "The arm is not run from an unpinned tree." >&2
  exit 2
fi

cd "$REPO"
HEAD_NOW="$(git rev-parse HEAD)"
case "$HEAD_NOW" in
  "$EXPECT_HEAD"*) ;;
  *) echo "HEAD is $HEAD_NOW, and this job is pinned to $EXPECT_HEAD." >&2
     exit 2 ;;
esac

if [ -n "$(git status --porcelain)" ]; then
  echo "the working tree is dirty:" >&2
  git status --porcelain >&2
  echo "Commit or stash first; refusing." >&2
  exit 2
fi

echo "[dynamic_raw] HEAD ${HEAD_NOW} (pinned ${EXPECT_HEAD}), tree clean"
echo "[dynamic_raw] bank        ${BANK}"
echo "[dynamic_raw] controllers ${CONTROLLERS}"
echo "[dynamic_raw] output      ${OUTDIR}"

if [ -e "$OUTDIR" ] && [ ! -d "$OUTDIR" ]; then
  echo "$OUTDIR exists and is not a directory." >&2
  exit 2
fi

# --resume is always passed. On a first run the directory is absent and it
# changes nothing; on a re-submission it is what makes the job continue rather
# than refuse a directory it already wrote. Records already present are
# revalidated -- locked fields, prompt digest, class, sites, direction digest,
# fold, and the coefficient against its own reference -- before being skipped.
apptainer exec --nv --cleanenv --bind "$HOME:$HOME" "$CONTAINER" \
  /bin/bash --noprofile --norc -lc "
    set -euo pipefail
    cd $REPO
    .venv/bin/python experiments/single_direction_controller/run_dynamic_arms.py \
      --direction-bank '${BANK}' \
      --controllers '${CONTROLLERS}' \
      --output-dir '${OUTDIR}' \
      --arm dynamic_raw \
      --shard-index 1 \
      --shard-count 1 \
      --device auto \
      --resume
  "

# When it finishes, verify the result. This loads no model, generates nothing
# and writes nothing; it re-runs every gate and requires exactly 480 completed
# records:
#
#   apptainer exec --cleanenv --bind $HOME:$HOME "$CONTAINER" \
#     /bin/bash --noprofile --norc -lc \
#     "cd $REPO && .venv/bin/python \
#        experiments/single_direction_controller/run_dynamic_arms.py \
#        --direction-bank '$BANK' --controllers '$CONTROLLERS' \
#        --output-dir '$OUTDIR' --arm dynamic_raw \
#        --shard-index 1 --shard-count 1 --verify"
#
# Results land under routegate-data, outside git, as the repository's output
# policy requires. Nothing here is committed, and no judge runs here.
