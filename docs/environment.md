# Environment

The study ran on UCL Myriad (SGE scheduler) inside one Apptainer image. This page
records that environment and how to reproduce or approximate it.

## Container image

| Field | Value |
| --- | --- |
| File | `vllm-openai-v0.10.1.1.sif` (10,442,780,672 bytes) |
| SHA256 of the `.sif` | `6029470a855ab84ecc8ff1a36f2f47f42ef47499aeb94255de08aded699af111` |
| Source | `docker://vllm/vllm-openai:v0.10.1.1` (label `org.label-schema.usage.singularity.deffile.from`) |
| Built | 4 July 2026 with Apptainer 1.2.4 (image labels) |
| Base OS | Ubuntu 22.04 (NVIDIA CUDA base image) |
| Python | 3.12.11 (`/usr/bin/python3.12`) |

The registry digest of the Docker tag was not recorded when the image was built,
so the tag alone does not guarantee identical bytes today. The `.sif` SHA256
above identifies the image actually used. To build an equivalent image:

```bash
apptainer build vllm-openai-v0.10.1.1.sif docker://vllm/vllm-openai:v0.10.1.1
```

Wrappers look for the image at `$HOME/containers/vllm-openai-v0.10.1.1.sif`;
set `ROUTEGATE_CONTAINER_FILE` to use another location. Wrappers bind only
`$HOME` into the container, so the repository and data root must live under
`$HOME` (on Myriad, use the `/home/<user>` spelling rather than `/myriadfs/...`).

## Python packages

The repository `.venv` was created inside the image with
`python3 -m venv --system-site-packages`, so it sees the image's packages, the
user site-packages under `$HOME/.local`, and its own installs. Versions and
origins as verified inside the container on 9 October 2026:

| Package | Version | Origin |
| --- | --- | --- |
| numpy | 2.2.6 | image |
| scipy | 1.16.1 | image |
| torch | 2.7.1+cu128 | image |
| transformers | 4.55.2 | image |
| huggingface-hub | 0.34.4 | image |
| PyYAML | 6.0.2 | image |
| vllm | 0.10.1.1 | image (not used by RouteGate code) |
| pandas | 3.0.3 | `$HOME/.local` |
| matplotlib | 3.11.0 | `$HOME/.local` |
| pyarrow | 24.0.0 | `$HOME/.local` |
| scikit-learn | 1.7.1 | `.venv` |
| pytest | 8.4.1 | `.venv` |

`pyproject.toml` declares these versions: core analysis dependencies, plus the
extras `dev` (pytest, PyYAML), `evaluation` (pyarrow, for the tinyBenchmarks
Parquet files) and `patching` (torch, transformers, huggingface-hub, for model
execution). The `tinyBenchmarks` package is not installed and is not needed; the
optional IRT estimate it provides was not computed.

The `.venv` interpreter symlinks resolve to the image's Python only inside the
container. On the Myriad login node they resolve to the host's Python 3.6, which
cannot run this code.

## Setting up

Inside the image on Myriad, from the repository root:

```bash
apptainer exec --cleanenv --bind "$HOME:$HOME" "$HOME/containers/vllm-openai-v0.10.1.1.sif" \
  python3 -m venv --system-site-packages .venv
apptainer exec --cleanenv --bind "$HOME:$HOME" "$HOME/containers/vllm-openai-v0.10.1.1.sif" \
  .venv/bin/python -m pip install -e '.[dev,evaluation]'
```

Elsewhere, with Python 3.12, a plain virtual environment is sufficient for the
offline tests and figures:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,evaluation]'
```

Add `patching` (and a CUDA-capable GPU) only for model execution.

## Model and hardware

Generation and activation capture used `Qwen/Qwen3-30B-A3B` at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, eager attention, bfloat16, batch 1,
greedy decoding and thinking mode disabled, on Myriad GPU nodes
(`-l gpu=4,mem=96G,tmpfs=30G -ac allow=L`). The weights are downloaded to the
local Hugging Face cache and are never stored in the repository; model runs use
`HF_HUB_OFFLINE=1`.

## Paths

| Setting | Purpose | Default |
| --- | --- | --- |
| `configs/paths.local.json` | Absolute paths to development inputs (copy from `configs/paths.local.example.json`; gitignored) | none |
| `ROUTEGATE_DATA_DIR` | Data root for experiment records outside Git | `$HOME/routegate-data` |
| `ROUTEGATE_CONTAINER_FILE` | Apptainer image | `$HOME/containers/vllm-openai-v0.10.1.1.sif` |
| `ROUTEGATE_FIGURE_PYTHON` | Interpreter for the figure package | Myriad bundle, then `python3` |

SGE submit scripts use `#$ -cwd` and must be submitted from the repository root.
Some hash-pinned scripts and frozen records keep the absolute Myriad paths they
were run with (for example `run_sealed_judging.sh` and `run_opus_parallel.sh`);
changing them would break their recorded integrity checks.
