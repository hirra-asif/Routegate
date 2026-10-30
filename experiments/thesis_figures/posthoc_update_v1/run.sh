#!/usr/bin/env bash
set -euo pipefail
RG_FIG_HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RG_FIG_OUTPUT="${1:-$RG_FIG_HERE/Figures}"
RG_CLUSTER_PYTHON=/shared/ucl/apps/python/bundles/1687356269-311/venv/bin/python3
RG_CLUSTER_LIB=/shared/ucl/apps/python/3.11.4/gnu-4.9.2/lib
if [[ -n "${ROUTEGATE_FIGURE_PYTHON:-}" ]]; then
    RG_FIG_PYTHON="$ROUTEGATE_FIGURE_PYTHON"
elif [[ -x "$RG_CLUSTER_PYTHON" ]]; then
    RG_FIG_PYTHON="$RG_CLUSTER_PYTHON"
else
    RG_FIG_PYTHON=python3
fi
if [[ "$RG_FIG_PYTHON" == "$RG_CLUSTER_PYTHON" ]]; then
    export LD_LIBRARY_PATH="$RG_CLUSTER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$RG_FIG_OUTPUT"
"$RG_FIG_PYTHON" "$RG_FIG_HERE/build_outcome_figures.py" --output-dir "$RG_FIG_OUTPUT"
"$RG_FIG_PYTHON" "$RG_FIG_HERE/build_selector_figures.py" --output-dir "$RG_FIG_OUTPUT"
"$RG_FIG_PYTHON" "$RG_FIG_HERE/build_current_runtime.py" --output-dir "$RG_FIG_OUTPUT"
"$RG_FIG_PYTHON" "$RG_FIG_HERE/build_activation_steering.py" --output-dir "$RG_FIG_OUTPUT"
printf 'All ten figures written to: %s\n' "$RG_FIG_OUTPUT"
