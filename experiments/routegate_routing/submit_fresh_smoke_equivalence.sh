#!/bin/bash -l
# Phase 2 step 10B -- the fresh two-row smoke that certifies the runtime cutover.
#
# NOT SUBMITTED. Read it, decide the shape is right, then submit it yourself,
# pinning the commit the evidence will belong to:
#
#     qsub -v EXPECTED_COMMIT="$(git rev-parse HEAD)" \
#          experiments/routegate_routing/submit_fresh_smoke_equivalence.sh
#
# EXPECTED_COMMIT is required, not decorative. A GPU job that runs minutes or
# hours after submission would otherwise silently certify whatever happened to
# be in the working tree when it started, which is not necessarily what was
# reviewed. The job refuses to run unless the tree is clean and HEAD is exactly
# the commit named at submission.
# ---------------------------------------------------------------------------
# What this proves, and why it needs a GPU
#
# Step 10 replaced the extraction's rq3_dev_generation import with
# routegate.runtime. Step 9 already proved, on CPU, that the two render and
# tokenise all 480 development prompts identically -- same bytes, same token
# ids, same final prompt token, same pinned snapshot and template digests.
#
# What step 9 cannot reach is the GPU code path: the model load, the router
# hooks, and the arrays derived from them. This job re-runs the two-row smoke
# under the new import and requires the numerical artifacts to come back
# byte-identical to the historical smoke.
#
# TWO ROWS ONLY. This job never runs the 480-row extraction. The frozen full
# run in outputs/routing_extraction_v1 is the input to the routing-versus-
# residual analysis and is not reproduced, re-derived or touched here.
#
# The historical smoke is hashed before and after the job and must be identical.
# The comparison report is written BESIDE the candidate directory, never inside
# it, so the fresh extraction is exactly what the extractor produced.
# ---------------------------------------------------------------------------

#$ -N routegate_fresh_smoke
#$ -l h_rt=2:00:00
#$ -l mem=96G
#$ -l gpu=4
#$ -l tmpfs=30G
#$ -ac allow=L
#$ -cwd
#$ -o logs/fresh_smoke.$JOB_ID.out
#$ -j y

set -euo pipefail

CONTAINER="$HOME/containers/vllm-openai-v0.10.1.1.sif"
REFERENCE="outputs/routing_extraction_v1_smoke"
CANDIDATE_ROOT="outputs/fresh_smoke_candidates"
# Keyed by JOB_ID so two submissions can never collide, and so the directory is
# always new -- which the extractor requires: it refuses an existing path and
# offers no overwrite.
CANDIDATE="$CANDIDATE_ROOT/routing_extraction_smoke_${JOB_ID:-manual}"
REPORT="$CANDIDATE_ROOT/equivalence_report_${JOB_ID:-manual}.txt"
REF_HASHES_BEFORE="$CANDIDATE_ROOT/reference_hashes_before_${JOB_ID:-manual}.sha256"
REF_HASHES_AFTER="$CANDIDATE_ROOT/reference_hashes_after_${JOB_ID:-manual}.sha256"

# ------------------------------------------------- 1. provenance of this run
echo "=== provenance ==="
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required; submit with -v EXPECTED_COMMIT=\"\$(git rev-parse HEAD)\"}"

DIRTY="$(git status --porcelain)"
if [ -n "$DIRTY" ]; then
    echo "[FATAL] the working tree is not clean; the evidence would not correspond"
    echo "        to any commit:"
    echo "$DIRTY"
    exit 2
fi

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]; then
    echo "[FATAL] HEAD is $ACTUAL_COMMIT but submission named $EXPECTED_COMMIT;"
    echo "        refusing to certify a tree other than the one reviewed"
    exit 2
fi

echo "  commit            : $ACTUAL_COMMIT (clean tree, matches submission)"
echo "  extractor  sha256 : $(sha256sum experiments/routegate_routing/extract_routing_v1.py | cut -d' ' -f1)"
echo "  runtime    sha256 : $(sha256sum src/routegate/runtime.py | cut -d' ' -f1)"
echo "  comparator sha256 : $(sha256sum tools/migration/compare_smoke_extractions.py | cut -d' ' -f1)"

# ------------------------------------------------- 2. the reference, before
echo
echo "=== fresh smoke equivalence check ==="
echo "  reference : $REFERENCE   (historical; read-only, never rewritten)"
echo "  candidate : $CANDIDATE   (fresh; must not already exist)"
echo "  report    : $REPORT      (beside the candidate, never inside it)"
echo "  rows      : 2 (one E, one H) -- the 480-row extraction is NOT run here"

test -d "$REFERENCE" || { echo "[FATAL] historical smoke reference is missing"; exit 2; }
test -e "$CANDIDATE" && { echo "[FATAL] candidate directory already exists"; exit 2; }
mkdir -p "$CANDIDATE_ROOT"

find "$REFERENCE" -type f -exec sha256sum {} + | sort -k2 > "$REF_HASHES_BEFORE"
echo "  hashed $(wc -l < "$REF_HASHES_BEFORE") reference files before the run"

# ------------------------------------------------- 3. extract and compare
apptainer exec --nv --cleanenv --bind "$HOME:$HOME" "$CONTAINER" \
  /bin/bash --noprofile --norc -lc "
    set -euo pipefail
    cd '$PWD'

    # The pinned snapshot is already in the local cache. Resolving it must not
    # reach the network, and refs/main must never be consulted.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

    # joblib is not involved here, but a single BLAS thread keeps any host-side
    # reduction order fixed, matching the discipline the original run used.
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

    echo
    echo '--- 1/2  fresh two-row extraction under routegate.runtime ---'
    .venv/bin/python experiments/routegate_routing/extract_routing_v1.py \
        --smoke --smoke-output-dir '$CANDIDATE'

    echo
    echo '--- 2/2  compare against the historical smoke ---'
    .venv/bin/python tools/migration/compare_smoke_extractions.py \
        --reference '$REFERENCE' \
        --candidate '$CANDIDATE' \
        --report '$REPORT'
  " && status=0 || status=$?

# ------------------------------------------------- 4. the reference, after
echo
echo "=== the historical smoke must be untouched ==="
find "$REFERENCE" -type f -exec sha256sum {} + | sort -k2 > "$REF_HASHES_AFTER"
if ! diff -q "$REF_HASHES_BEFORE" "$REF_HASHES_AFTER" >/dev/null; then
    echo "[FATAL] the historical smoke changed during this job:"
    diff "$REF_HASHES_BEFORE" "$REF_HASHES_AFTER" || true
    exit 3
fi
echo "  all $(wc -l < "$REF_HASHES_AFTER") reference files re-hashed identical"

if [ "$status" -ne 0 ]; then
    echo "[FAIL] the fresh smoke does not match the historical smoke; step 10 is NOT certified"
    exit "$status"
fi

echo
echo "[PASS] fresh smoke is equivalent to the historical smoke"
echo "       commit : $ACTUAL_COMMIT"
echo "       report : $REPORT"
