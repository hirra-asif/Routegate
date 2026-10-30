"""Shared fixtures.

Tests that need the real corpus are skipped if the local path file is absent or
its inputs are unreachable, so the suite still runs on a machine that does not
carry the historical workspace. Everything else runs on synthetic fixtures.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MANIFEST_PATH = REPO_ROOT / "manifests" / "development_inputs.json"
PILOT_MANIFEST_PATH = REPO_ROOT / "manifests" / "pilot_reference.json"
PATHS_PATH = REPO_ROOT / "configs" / "paths.local.json"


def _local_paths():
    if not PATHS_PATH.exists():
        return None
    try:
        return json.loads(PATHS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def real_inputs_available() -> bool:
    paths = _local_paths()
    if paths is None:
        return False
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dataset = Path(paths.get("dataset", ""))
    extraction = Path(paths.get("extraction_dir", ""))
    return (dataset.exists()
            and (extraction / manifest["activation"]["filename"]).exists()
            and (extraction / manifest["extraction_manifest"]["filename"]).exists())


def pilot_reference_available() -> bool:
    # Committed artifact (Phase 2 step 5): resolved from the repository, so this
    # no longer depends on configs/paths.local.json being set on this machine.
    from routegate.data import PILOT_REFERENCE_DIR
    reference = PILOT_REFERENCE_DIR
    if not reference.is_dir():
        return False
    manifest = json.loads(PILOT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return all((reference / spec["filename"]).exists()
               for spec in manifest["reference_files"].values())


def data_root():
    """The private data root, as ``twositecontinuation.data_root`` derives it.

    Machine-local artefacts that no manifest addresses by absolute path -- the
    validated direction bank and the controller bundle -- hang off it, so a
    test can find them without a new entry in the path file.
    """
    paths = _local_paths()
    if paths is None or not paths.get("dataset"):
        return None
    root = Path(paths["dataset"]).parent.parent.parent
    return root if root.is_dir() else None


VALIDATED_DIRECTION_BANK = "frozen/validated_direction_bank"
CONTROLLER_BUNDLE = "gate_oof_v1_repro"


def dynamic_arm_inputs_available() -> bool:
    """The corpus, the validated bank and the reproduced controller bundle."""
    root = data_root()
    if root is None or not real_inputs_available():
        return False
    bank = root / VALIDATED_DIRECTION_BANK / "validated_direction_bank.npz"
    bundle = root / CONTROLLER_BUNDLE
    return (bank.is_file()
            and (bundle / "controller_fold_0.json").is_file()
            and (bundle / "reproduction_summary.json").is_file())


# Test modules that import torch at module level but cannot use
# pytest.importorskip because a completed study manifest pins their bytes
# (manifests/static_ungated_followup_v1.json). Without the `patching` extra they
# are reported as skipped, exactly as importorskip would report them.
TORCH_PINNED_MODULES = frozenset({"test_static_ungated_followup.py"})


class _TorchAbsentModule(pytest.Module):
    def _getobj(self):
        pytest.skip("could not import 'torch': install the `patching` extra",
                    allow_module_level=True)


def pytest_pycollect_makemodule(module_path, parent):
    if (module_path.name in TORCH_PINNED_MODULES
            and importlib.util.find_spec("torch") is None):
        return _TorchAbsentModule.from_parent(parent, path=module_path)
    return None


needs_real_inputs = pytest.mark.skipif(
    not real_inputs_available(),
    reason="the accepted development inputs are not reachable on this machine")

needs_dynamic_arm_inputs = pytest.mark.skipif(
    not dynamic_arm_inputs_available(),
    reason="the validated direction bank or the reproduced controller bundle "
           "is not reachable on this machine")

# The frozen gate thresholds live in a completed, gitignored sensor run
# (outputs/routegate_residual_v1/thresholds.json). load_thresholds still checks
# its pinned SHA256 wherever the file exists.
needs_gate_thresholds = pytest.mark.skipif(
    not (REPO_ROOT / "outputs" / "routegate_residual_v1" / "thresholds.json").is_file(),
    reason="the frozen gate thresholds (outputs/routegate_residual_v1) are not present")

needs_pilot_reference = pytest.mark.skipif(
    not (real_inputs_available() and pilot_reference_available()),
    reason="the frozen pilot reference run is not reachable on this machine")


@pytest.fixture(scope="session")
def input_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pilot_manifest():
    return json.loads(PILOT_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def local_paths():
    paths = _local_paths()
    if paths is None:
        pytest.skip("configs/paths.local.json is absent")
    return paths


@pytest.fixture(scope="session")
def inputs():
    """The real, fully validated development corpus."""
    if not real_inputs_available():
        pytest.skip("the accepted development inputs are not reachable")
    from routegate.data import load_development_inputs
    return load_development_inputs()


@pytest.fixture(scope="session")
def frozen_folds(inputs, local_paths, pilot_manifest):
    if not pilot_reference_available():
        pytest.skip("the frozen pilot reference run is not reachable")
    from routegate.splits import load_frozen_folds
    spec = pilot_manifest["reference_files"]["fold_assignment_and_out_of_fold_scores"]
    from routegate.data import pilot_reference_dir
    path = pilot_reference_dir() / spec["filename"]
    return load_frozen_folds(path, inputs.row_ids,
                             row_id_column=spec["row_id_column"],
                             fold_column=spec["fold_column"])


@pytest.fixture
def synthetic_problem():
    """A small separable E/B problem with a planted linear direction.

    The plant is deliberately strong. The probe is regularised at C = 0.1 and
    fits 24 coefficients from roughly 72 training rows, so a weak plant is
    shrunk far enough that a recovery test would fail on the regularisation
    rather than on anything the test is about. At this separation the
    out-of-fold recovery is unambiguous, which keeps the leakage and
    determinism tests -- which need scores that actually respond to the data --
    meaningful too.
    """
    rng = np.random.default_rng(20260805)
    n_e, n_b, dim = 30, 60, 24
    y = np.array([1] * n_e + [0] * n_b, dtype=int)
    X = rng.normal(size=(n_e + n_b, dim))
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    X[y == 1] += 3.0 * direction
    return X, y


@pytest.fixture(scope="session")
def direction_bank_dir():
    """The validated direction bank the whole study is pinned to."""
    if not dynamic_arm_inputs_available():
        pytest.skip("the validated direction bank is not reachable")
    return data_root() / VALIDATED_DIRECTION_BANK


@pytest.fixture(scope="session")
def controller_bundle_dir():
    """The reproduced controller bundle: five fold checkpoints and the refit."""
    if not dynamic_arm_inputs_available():
        pytest.skip("the reproduced controller bundle is not reachable")
    return data_root() / CONTROLLER_BUNDLE
