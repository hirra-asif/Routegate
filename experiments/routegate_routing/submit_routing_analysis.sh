#!/bin/bash -l
# RouteGate: routing-only vs residual-only vs residual+routing, as a batch job.
#
# NOT SUBMITTED. Read it, decide the shape is right, then submit it yourself:
#
#     qsub experiments/routegate_routing/submit_routing_analysis.sh
#
# ---------------------------------------------------------------------------
# What this job runs
#
# Three feature families over the same frozen 480 development rows and the same
# frozen outer folds as routegate_residual_v1:
#
#     residual   state 45 (= block 44 OUTPUT, not "layer 45")     2048 features
#     routing    48 MoE blocks x 128 experts, RAW pre-selection   6144 features
#                router logits at the final prompt token
#     combined   residual ++ routing                              8192 features
#
# times three tasks (E-vs-B sensor, H-vs-EB veto, direct E-vs-H diagnostic),
# each with a nested 5-fold C search inside every outer training fold, plus the
# topic control and a 2000-replicate paired bootstrap.
#
# Cost: measured at ~280 s per outer fold for the 8192-feature family on one
# core, so roughly 3.5-4 CPU-hours in total. That is compute-node work. The
# login node is for --verify-only and --smoke, both of which are cheap.
#
# --jobs only parallelises the inner grid search. Every fit is independent and
# deterministic, so the numbers do not depend on it; only the wall clock does.
# ---------------------------------------------------------------------------

#$ -N routegate_routing_analysis
#$ -l h_rt=8:00:00
#$ -l mem=4G
#$ -pe smp 8
#$ -cwd
#$ -o logs/routing_analysis.$JOB_ID.out
#$ -j y

# logs/ must already exist: the scheduler opens the -o path above before this
# script runs. It is kept in the repository by logs/.gitkeep.

set -euo pipefail

# Submit from the repository root; -cwd starts the job there. Only $HOME is
# bound into the container, so use its spelling of the path.
REPO="$PWD"
HOME_PHYSICAL="$(cd "$HOME" && pwd -P)"
case "$REPO" in "$HOME_PHYSICAL"/*) REPO="$HOME/${REPO#"$HOME_PHYSICAL"/}" ;; esac
test -f "$REPO/pyproject.toml" && test -d "$REPO/src/routegate" || { echo "submit this script from the RouteGate repository root" >&2; exit 1; }
CONTAINER="${ROUTEGATE_CONTAINER_FILE:-$HOME/containers/vllm-openai-v0.10.1.1.sif}"
JOBS="${NSLOTS:-8}"

# No GPU is requested and none is used: this stage loads two .npy arrays and
# fits scikit-learn logistic regressions. Qwen is never loaded, nothing is
# generated, nothing is steered, and the sealed set is never opened.

apptainer exec --cleanenv --bind "$HOME:$HOME" "$CONTAINER" \
  /bin/bash --noprofile --norc -lc "
    set -euo pipefail
    cd $REPO

    # one BLAS thread per worker: joblib already owns the parallelism, and
    # letting both layers expand oversubscribes the slot allocation
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

    # The tests gate the analysis. tests/test_routing_analysis_rules.py replays
    # routegate_residual_v1's own out-of-fold predictions through this stage's
    # threshold/gate/metric rules and demands the frozen answers back, so a
    # drifted rule stops the job in seconds instead of after four hours.
    .venv/bin/python -m pytest -q

    # Input verification is also re-run inside --run and aborts before any fit,
    # but doing it first keeps the failure at the top of the log.
    .venv/bin/python experiments/routegate_routing/analyze_routing_vs_residual_v1.py \
      --verify-only

    .venv/bin/python experiments/routegate_routing/analyze_routing_vs_residual_v1.py \
      --run --jobs $JOBS
  "

# Output lands in outputs/routegate_routing_analysis_v1/. The job writes nothing
# into routing_extraction_v1, routegate_residual_v1 or routegate_topic_analysis_v1;
# it re-hashes all three afterwards and fails if any byte moved.
