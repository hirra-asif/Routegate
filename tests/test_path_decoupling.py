"""The refactored experiment scripts carry no machine-specific path.

Phase 2 steps 6-8 removed the hard-coded ``SM``/``RG`` constants from four
scripts so they resolve inputs through ``configs/paths.local.json`` plus the
committed input manifest, and resolve outputs from their own ``__file__``.
These tests are the standing guard that the decoupling does not regress.

All five scripts are covered. ``extract_routing_v1.py`` was exempt through
steps 6-8 -- it still imported ``rq3_dev_generation`` from the historical
workspace, and its SHA256 was pinned here so it could not drift before its turn.
Step 10 cut that import over to ``routegate.runtime``, gated on the seven-check
prompt-equivalence test, so the exemption and the pin are retired and the script
now stands under the same guard as the others. The step-10-specific checks --
the fresh-smoke guard and the smoke comparator -- live in
``tests/test_runtime_cutover.py``.

Everything runs offline: these tests read source text and one JSON example.
None of them resolves a data path, loads a model or needs
``configs/paths.local.json`` to exist.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

REFACTORED = [
    "experiments/routegate_residual/run_baseline_v1.py",
    "experiments/routegate_residual/run_topic_analysis_v1.py",
    "experiments/routegate_routing/analyze_routing_vs_residual_v1.py",
    "experiments/routegate_routing/validate_routing_prep_v1.py",
    # Joined the set at step 10, when its SteerMoE import was cut.
    "experiments/routegate_routing/extract_routing_v1.py",
]

EXAMPLE_PATHS_FILE = REPO_ROOT / "configs" / "paths.local.example.json"
# `rq2_actuator_dir` joined the set on 2026-08-21, when the frozen RQ2 actuator
# was migrated into the machine-local data root. It is a runtime key like the
# others: gitignored, absolute only in paths.local.json, and hash-gated on read
# (manifests/rq2_actuator.json). See tests/test_frozen_actuator.py.
# `triplet_validation_dir` joined on 2026-08-21 with the retrospective triplet
# validation. It is a runtime key like the others: gitignored, absolute only in
# paths.local.json, and it holds the frozen judge-protocol text, the judging
# packets, the private join maps and the raw API records -- prompt-bearing and
# de-anonymising material, none of which is ever committed.
# `focused_direction_validation_dir` joined on 2026-08-24 with the blind phase-1
# judging of focused_direction_validation_v1. It is a runtime key like the
# others: gitignored, absolute only in paths.local.json, and it holds the 871
# pinned screen records, the blind judging packet, the private row map, the raw
# API bodies, the joined cells and the adjudicated cells -- prompt-bearing and
# de-anonymising material, none of which is ever committed. Read by
# routegate.focusedjudging.judging_root via its PATHS_KEY.
# `two_site_validation_v2_consolidated_dir` and `two_site_judging_dir` joined on
# 2026-08-27 with the blind judging of the two-site static actuator candidate.
# The first names the authoritative consolidated package -- 480 candidate screen
# records and 480 referenced baselines -- and is only ever opened read-only,
# hash-gated by manifests/two_site_actuator_judging_v1.json. The second is that
# stage's own working root: the frozen B quality-rubric text, the blind packet,
# the private row map that de-anonymises the opaque ids AND records the A/B
# orientation, the raw API bodies, the joined cells and the adjudicated cells --
# prompt-bearing and de-anonymising material, none of which is ever committed.
# Read by routegate.twositejudging.consolidated_root and judging_root via their
# PATHS_KEYs.
REQUIRED_RUNTIME_KEYS = {"dataset", "extraction_dir", "static_steering_master",
                         "rq2_actuator_dir", "triplet_validation_dir",
                         "focused_direction_validation_dir",
                         "two_site_validation_v2_consolidated_dir",
                         "two_site_judging_dir"}

SEALED_TOKENS = ("rq3_sealed", "sealed_final_manifest", "rq3_prompt_splits")


def source_of(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


# --------------------------------------------------- the four refactored files
@pytest.mark.parametrize("relative", REFACTORED)
def test_the_script_does_not_name_the_historical_workspace(relative):
    assert "SteerMoE-main" not in source_of(relative)


@pytest.mark.parametrize("relative", REFACTORED)
def test_the_script_contains_no_absolute_home_path(relative):
    text = source_of(relative)
    for needle in ("/home/ucabh21", "/myriadfs", "/scratch/"):
        assert needle not in text, f"{relative} contains {needle!r}"


@pytest.mark.parametrize("relative", REFACTORED)
def test_the_script_defines_no_SM_constant(relative):
    """Not merely absent as a string: absent as a module-level assignment."""
    tree = ast.parse(source_of(relative))
    assigned = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
    assert "SM" not in assigned, f"{relative} still defines SM"


@pytest.mark.parametrize("relative", REFACTORED)
def test_the_script_derives_its_repository_root_from_dunder_file(relative):
    text = source_of(relative)
    assert re.search(r"(REPO_ROOT|RG)\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[2\]", text), (
        f"{relative} must locate the repository from its own __file__")


@pytest.mark.parametrize("relative", REFACTORED)
def test_the_script_resolves_inputs_through_the_shared_loader(relative):
    """No second JSON parser, no duplicate path-configuration system."""
    text = source_of(relative)
    assert "from routegate.data import" in text, relative
    assert "load_local_paths" in text, relative
    assert "load_input_manifest" in text, relative


@pytest.mark.parametrize("relative", REFACTORED)
def test_the_script_introduces_no_sealed_set_locator(relative):
    text = source_of(relative).lower()
    for token in SEALED_TOKENS:
        assert token not in text, f"{relative} names {token!r}"


def test_no_refactored_script_writes_to_an_absolute_output_path():
    for relative in REFACTORED:
        text = source_of(relative)
        assert not re.search(r'["\']/[a-z]+/[a-z0-9_]+/routegate/outputs', text), relative


# --------------------------------------- the deferred dependency is now gone
def test_no_script_imports_the_historical_generation_module():
    """The step-10 cutover, held as a standing guard over the whole set.

    Structural: the prep validator legitimately mentions the old module name in
    the text of a check, and naming it there is not importing it.
    """
    for relative in REFACTORED:
        tree = ast.parse(source_of(relative))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("rq3_dev_generation" not in a.name for a in node.names), relative
            elif isinstance(node, ast.ImportFrom):
                assert "rq3_dev_generation" not in (node.module or ""), relative


# ------------------------------------------------------ the local-path example
def test_the_example_has_exactly_the_required_runtime_keys():
    config = json.loads(EXAMPLE_PATHS_FILE.read_text(encoding="utf-8"))
    runtime = {k for k in config if not k.startswith("_")}
    assert runtime == REQUIRED_RUNTIME_KEYS


def test_the_example_no_longer_mentions_a_pilot_reference_directory():
    """Pilot references are committed artifacts now, not a configured path."""
    assert "pilot_reference_dir" not in EXAMPLE_PATHS_FILE.read_text(encoding="utf-8")


def test_the_example_contains_no_real_absolute_path():
    raw = EXAMPLE_PATHS_FILE.read_text(encoding="utf-8")
    for needle in ("ucabh21", "/myriadfs", "/scratch/", "/Users/", "SteerMoE"):
        assert needle not in raw, f"the example leaks {needle!r}"
    config = json.loads(raw)
    for key in REQUIRED_RUNTIME_KEYS:
        assert config[key].startswith("/absolute/path/to/"), key


def test_the_example_names_no_held_out_material():
    raw = EXAMPLE_PATHS_FILE.read_text(encoding="utf-8").lower()
    for token in SEALED_TOKENS:
        assert token not in raw


# ------------------------------------------- what a record is allowed to name
def test_recorded_source_paths_come_from_the_committed_manifest():
    """A scientific record names the canonical source, never the resolved path.

    The resolved path points into whichever private data root this machine
    happens to use; the canonical path is the one the completed run recorded.
    Only the second may reach an output, or relocating the data would silently
    change what an output says it was built from.
    """
    for relative in ("experiments/routegate_residual/run_baseline_v1.py",
                     "experiments/routegate_routing/analyze_routing_vs_residual_v1.py"):
        assert "canonical_source_path" in source_of(relative), relative


def test_the_analysis_declares_canonical_input_names_not_resolved_paths():
    text = source_of("experiments/routegate_routing/analyze_routing_vs_residual_v1.py")
    assert "declared_inputs=list(DECLARED_INPUT_NAMES)" in text
    assert "declared_inputs=[str(p) for p in DECLARED_INPUTS]" not in text


def test_the_manifest_supplies_a_relative_canonical_path_for_every_frozen_input():
    manifest = json.loads(
        (REPO_ROOT / "manifests" / "development_inputs.json").read_text(encoding="utf-8"))
    for key in ("dataset", "extraction_manifest", "activation"):
        canonical = manifest[key]["canonical_source_path"]
        assert canonical, key
        assert not canonical.startswith("/"), f"{key} canonical path is absolute"
        assert "ucabh21" not in canonical, key
