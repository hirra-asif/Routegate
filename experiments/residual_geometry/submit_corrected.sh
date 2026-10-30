#!/bin/bash -l
# The corrected residual-geometry analysis, as a batch job.
#
# It costs roughly 2.5 CPU-hours -- 49 states x 4 activation variants x 20
# repeated five-fold splits, a nested selection inside every training fold, and
# 200 permutations each repeating the whole state search. That is compute-node
# work; a login node is for editing and for the smoke test.
#
#   qsub experiments/residual_geometry/submit_corrected.sh
#
# The approval flag is passed here rather than prompted for, so submitting this
# script *is* the approval. Read configs/residual_geometry.json first.

#$ -N routegate_corrected
#$ -l h_rt=3:00:00
#$ -l mem=4G
#$ -pe smp 8
#$ -wd /home/ucabh21/routegate
#$ -o logs/corrected.$JOB_ID.out
#$ -j y

# logs/ must already exist: the scheduler opens the -o path above before this
# script runs. It is kept in the repository by logs/.gitkeep.

set -euo pipefail

REPO="/home/ucabh21/routegate"
CONTAINER="$HOME/containers/vllm-openai-v0.10.1.1.sif"
JOBS="${NSLOTS:-8}"

# The tests gate the analysis: `set -e` inside the container shell too, so a
# failing test stops the run instead of being printed and stepped over.
apptainer exec --cleanenv --bind "$HOME:$HOME" "$CONTAINER" \
  /bin/bash --noprofile --norc -lc "
    set -euo pipefail
    cd $REPO
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    .venv/bin/python -m pytest -q
    .venv/bin/python experiments/residual_geometry/run_analysis.py \
      --i-have-approved-the-corrected-protocol \
      --jobs $JOBS
  "
