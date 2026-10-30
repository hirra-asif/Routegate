"""The two stage guards replaced after the Phase 2 steps 6-8 review.

Both originals were environment heuristics that could not survive a legitimate
pipeline, and both were replaced with checks on the thing actually being
protected.

``neither_dir_exists_yet`` (preparation validator)
    Asserted that both routing output directories were absent. True only before
    the very first extraction; a permanent, uninformative failure afterwards.
    Replaced by ``routing_output_path_state``, which accepts *absent* or *plain
    directory at the exact expected location* and refuses a file, a symlink, or
    any other location.

``frozen_residual_v1_not_modified`` (routing analysis)
    Flagged any file in the frozen baseline whose mtime was under an hour old.
    Blind to a content change that preserved the timestamp, and a false alarm
    whenever the stage that wrote the directory had legitimately just run.
    Replaced by an exact before/after filename-to-SHA256 comparison.

Everything here is CPU-only, offline, and writes only into tmp_path. No model,
no GPU, no API, no sealed set. The source-level and predicate tests need no
configuration at all; the handful that exercise ``snapshot_dir_sha256``
behaviourally must load the analysis module, so they skip where
configs/paths.local.json is absent -- the same pattern
tests/test_routing_analysis_rules.py already uses.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from routegate.validation import routing_output_path_state

REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "experiments/routegate_routing/validate_routing_prep_v1.py"
ANALYSIS = REPO_ROOT / "experiments/routegate_routing/analyze_routing_vs_residual_v1.py"


@pytest.fixture
def outputs(tmp_path):
    """A stand-in repository outputs/ directory."""
    d = tmp_path / "outputs"
    d.mkdir()
    return d


# ------------------------------------------------- the obsolete check is gone
def test_neither_dir_exists_yet_no_longer_exists():
    assert "neither_dir_exists_yet" not in PREP.read_text(encoding="utf-8")


def test_no_check_is_emitted_under_the_obsolete_name_anywhere():
    """The name survives in one docstring, explaining what was replaced and why.
    It must not survive as an emitted check in any script or module."""
    for path in sorted((REPO_ROOT / "experiments").rglob("*.py")) + \
                sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r'(?:P|chk)\(\s*["\']neither_dir_exists_yet', text), path


def test_the_misleading_fixed_literal_is_gone():
    """The old detail text asserted a state it had not observed."""
    assert "both output directories are absent" not in PREP.read_text(encoding="utf-8")


def test_the_prep_validator_uses_the_replacement_predicate():
    text = PREP.read_text(encoding="utf-8")
    assert "from routegate.validation import routing_output_path_state" in text
    assert text.count("routing_output_path_state(") == 2   # OUT_FULL and OUT_SMOKE


# ------------------------------------------------- the replacement: passing
def test_absent_directory_passes(outputs):
    expected = outputs / "routing_extraction_v1"
    ok, detail = routing_output_path_state(expected, expected, outputs)
    assert ok
    assert "absent" in detail


def test_existing_plain_directory_passes(outputs):
    expected = outputs / "routing_extraction_v1"
    expected.mkdir()
    (expected / "router_logits.npy").write_bytes(b"")
    ok, detail = routing_output_path_state(expected, expected, outputs)
    assert ok
    assert "directory" in detail and "1 entry" in detail


def test_an_empty_existing_directory_passes(outputs):
    expected = outputs / "routing_extraction_v1_smoke"
    expected.mkdir()
    ok, detail = routing_output_path_state(expected, expected, outputs)
    assert ok
    assert "0 entries" in detail


# ------------------------------------------------- the replacement: refusing
def test_an_unexpected_file_fails(outputs):
    expected = outputs / "routing_extraction_v1"
    expected.write_text("not a directory")
    ok, detail = routing_output_path_state(expected, expected, outputs)
    assert not ok
    assert "not a directory" in detail


def test_a_symlink_to_a_directory_fails(outputs, tmp_path):
    """Refused even though it *looks* like a valid directory: a symlink can
    redirect a write out of the repository entirely."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    expected = outputs / "routing_extraction_v1"
    expected.symlink_to(elsewhere, target_is_directory=True)
    assert expected.is_dir()                      # the trap this check exists for
    ok, detail = routing_output_path_state(expected, expected, outputs)
    assert not ok
    assert "symlink" in detail


def test_a_dangling_symlink_fails(outputs, tmp_path):
    expected = outputs / "routing_extraction_v1"
    expected.symlink_to(tmp_path / "does_not_exist", target_is_directory=True)
    ok, detail = routing_output_path_state(expected, expected, outputs)
    assert not ok
    assert "symlink" in detail


def test_a_path_outside_the_outputs_directory_fails(outputs, tmp_path):
    stray = tmp_path / "routing_extraction_v1"
    stray.mkdir()
    ok, detail = routing_output_path_state(stray, stray, outputs)
    assert not ok
    assert "outside" in detail


def test_the_right_name_in_the_wrong_place_fails(outputs):
    expected = outputs / "routing_extraction_v1"
    actual = outputs / "nested" / "routing_extraction_v1"
    actual.mkdir(parents=True)
    ok, detail = routing_output_path_state(actual, expected, outputs)
    assert not ok
    assert "expected" in detail


def test_the_detail_always_reports_the_observed_state(outputs):
    """Never a fixed literal: each branch names what was actually found."""
    expected = outputs / "routing_extraction_v1"
    details = []
    ok, d = routing_output_path_state(expected, expected, outputs); details.append((ok, d))
    expected.mkdir()
    ok, d = routing_output_path_state(expected, expected, outputs); details.append((ok, d))
    assert details[0][1] != details[1][1]
    assert all(expected.name in d for _, d in details)


# ------------------------------------------------- the analysis-stage guard
def test_the_analysis_no_longer_uses_st_mtime_anywhere():
    assert "st_mtime" not in ANALYSIS.read_text(encoding="utf-8")


def test_the_analysis_guard_makes_no_stat_call():
    """Structural, not textual: no `.stat()` survives in the module."""
    tree = ast.parse(ANALYSIS.read_text(encoding="utf-8"))
    stat_calls = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "stat"]
    assert not stat_calls


def test_the_guard_compares_complete_before_and_after_mappings():
    text = ANALYSIS.read_text(encoding="utf-8")
    assert "resid_v1_before = snapshot_dir_sha256(RESID_V1)" in text
    assert "resid_v1_after = snapshot_dir_sha256(RESID_V1)" in text
    assert 'chk("frozen_residual_v1_not_modified", resid_v1_after == resid_v1_before' in text


def test_the_snapshot_is_taken_before_any_fitting():
    """Ordering matters: a snapshot taken afterwards would prove nothing."""
    text = ANALYSIS.read_text(encoding="utf-8")
    assert (text.index("resid_v1_before = snapshot_dir_sha256(RESID_V1)")
            < text.index("feats = build_features(row_ids)"))


def test_the_before_snapshot_is_threaded_into_final_validation():
    text = ANALYSIS.read_text(encoding="utf-8")
    assert re.search(r"def final_validation\([^)]*resid_v1_before", text, re.S)


# ------------------------------- snapshot_dir_sha256 behaviour (module load)
needs_local_paths = pytest.mark.skipif(
    not (REPO_ROOT / "configs" / "paths.local.json").is_file(),
    reason="configs/paths.local.json is not present on this machine")


def _analysis_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyze_routing_vs_residual_v1", ANALYSIS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # no heavy imports at module scope
    return mod


@pytest.fixture(scope="module")
def snapshot():
    return _analysis_module().snapshot_dir_sha256


@pytest.fixture
def baseline(tmp_path):
    d = tmp_path / "routegate_residual_v1"
    d.mkdir()
    (d / "oof_predictions.csv").write_text("row_id,p_E\nr0,0.5\n")
    (d / "thresholds.json").write_text('{"tau_E": 0.5}')
    return d


@needs_local_paths
def test_snapshot_maps_every_filename_to_its_sha256(snapshot, baseline):
    import hashlib
    snap = snapshot(baseline)
    assert set(snap) == {"oof_predictions.csv", "thresholds.json"}
    assert snap["thresholds.json"] == hashlib.sha256(
        (baseline / "thresholds.json").read_bytes()).hexdigest()


@needs_local_paths
def test_snapshot_is_stable_when_nothing_changes(snapshot, baseline):
    assert snapshot(baseline) == snapshot(baseline)


@needs_local_paths
def test_snapshot_detects_a_content_change(snapshot, baseline):
    before = snapshot(baseline)
    (baseline / "thresholds.json").write_text('{"tau_E": 0.9}')   # same length
    assert snapshot(baseline) != before


@needs_local_paths
def test_snapshot_detects_a_content_change_that_preserves_the_mtime(snapshot, baseline):
    """The exact case the old mtime heuristic could not see."""
    import os
    target = baseline / "thresholds.json"
    stamp = (target.stat().st_atime, target.stat().st_mtime)
    before = snapshot(baseline)
    target.write_text('{"tau_E": 0.9}')
    os.utime(target, stamp)                       # timestamp restored, content changed
    assert target.stat().st_mtime == stamp[1]
    assert snapshot(baseline) != before


@needs_local_paths
def test_snapshot_detects_an_added_file(snapshot, baseline):
    before = snapshot(baseline)
    (baseline / "sneaked_in.json").write_text("{}")
    after = snapshot(baseline)
    assert after != before
    assert set(after) - set(before) == {"sneaked_in.json"}


@needs_local_paths
def test_snapshot_detects_a_deleted_file(snapshot, baseline):
    before = snapshot(baseline)
    (baseline / "thresholds.json").unlink()
    after = snapshot(baseline)
    assert after != before
    assert set(before) - set(after) == {"thresholds.json"}


@needs_local_paths
def test_snapshot_ignores_subdirectories(snapshot, baseline):
    (baseline / "nested").mkdir()
    assert set(snapshot(baseline)) == {"oof_predictions.csv", "thresholds.json"}
