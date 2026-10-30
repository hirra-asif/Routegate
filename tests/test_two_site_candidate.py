"""The two-site actuator candidate, and the result that produced it.

Nothing here loads the frozen model, generates a token, judges a response or
touches held-out material. The committed records are read read-only; every
mutation is made on a copy in ``tmp_path``.

What is under test is the part of this step that is easy to get silently
wrong: whether the candidate's sites are the sites the budget test actually
cleared, whether historical block numbers and package state numbers stay apart,
and whether the record binding the two together can drift without anything
noticing.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import twosite as T                                  # noqa: E402
from routegate.onepassbudget import (                               # noqa: E402
    ALPHA_R,
    TOLERANCE,
    WORST_CASE_G_R,
    block_input_state,
    block_output_state,
)
from routegate.validation import (                                  # noqa: E402
    SealedSetError,
    ValidationError,
)

RUNNER = (REPO_ROOT / "experiments" / "single_direction_controller"
          / "run_two_site_validation.py")
FREEZER = (REPO_ROOT / "experiments" / "single_direction_controller"
           / "freeze_two_site_candidate.py")
V2 = REPO_ROOT / T.V2_RUN_DIRECTORY

needs_v2 = pytest.mark.skipif(
    not (V2 / "summary.json").is_file(),
    reason="the completed v2 budget run is not reachable on this machine")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _summary(per_block):
    """A minimal v2-shaped summary, for the binding tests."""
    return {"per_block": {str(b): {"within_tolerance": ok}
                          for b, ok in per_block.items()}}


# =========================================== 1. block / state mapping
def test_historical_blocks_map_to_package_states_one_higher():
    assert T.CANDIDATE_BLOCKS == (31, 32)
    assert T.CANDIDATE_STATES == (32, 33)
    assert T.CANDIDATE_BUDGET_STATES == (31, 32)
    for block, state in zip(T.CANDIDATE_BLOCKS, T.CANDIDATE_STATES):
        assert block_output_state(block) == state == block + 1
        assert block_input_state(block) == block


def test_the_dropped_block_and_its_state_are_recorded_together():
    assert T.DROPPED_BLOCKS == (33,)
    assert T.DROPPED_STATES == (34,)
    assert block_output_state(33) == 34


def test_the_ablation_is_historical_block_32_which_is_package_state_33():
    """The trap this project has warned about since the probe stage: block 32
    is NOT state 32. State 32 is the output of block 31."""
    assert T.ABLATION_BLOCKS == (32,)
    assert T.ABLATION_STATES == (33,)
    assert block_output_state(32) == 33
    assert block_output_state(31) == 32          # state 32 belongs to block 31
    record = T.build_candidate_record()["preregistered_ablation"]
    assert record["historical_blocks"] == [32]
    assert record["package_states_edited"] == [33]
    assert record["status"] == "PRESERVED_NOT_RUN"
    assert "NOT package state 32" in record["numbering_warning"]


# ============================================= 2. block 33 is excluded
def test_block_33_is_not_a_candidate_site():
    intervention = T.candidate_intervention()
    assert 33 not in intervention["historical_blocks"]
    assert 34 not in intervention["package_states_edited"]
    assert 33 not in intervention["budget_reference_states"]
    assert intervention["dropped_blocks"] == [33]


def test_a_candidate_that_includes_the_failing_block_is_refused():
    summary = _summary({31: True, 32: True, 33: False})
    with pytest.raises(ValidationError, match="no_failing_site_is_included"):
        T.assert_sites_are_the_passing_sites((31, 32, 33), summary)


def test_a_candidate_that_smuggles_in_an_unmeasured_block_is_refused():
    summary = _summary({31: True, 32: True, 33: False})
    with pytest.raises(ValidationError,
                       match="every_candidate_site_passed"):
        T.assert_sites_are_the_passing_sites((31, 32, 30), summary)


def test_dropping_a_block_that_passed_is_refused_rather_than_ignored():
    summary = _summary({31: True, 32: True, 33: False})
    with pytest.raises(ValidationError,
                       match="no_passing_site_is_dropped"):
        T.assert_sites_are_the_passing_sites((31,), summary)


def test_the_sites_are_derived_from_the_summary_not_asserted_beside_it():
    assert T.passing_blocks_from_summary(
        _summary({31: True, 32: True, 33: False})) == (31, 32)
    assert T.passing_blocks_from_summary(
        _summary({31: True, 32: False, 33: False})) == (31,)


def test_a_summary_clearing_other_blocks_makes_this_a_different_candidate():
    """The binding is mechanical: the package's pinned sites must equal what
    the summary cleared, so a different result cannot reuse this candidate."""
    summary = _summary({31: True, 32: False, 33: False})
    with pytest.raises(ValidationError):
        T.assert_sites_are_the_passing_sites(T.CANDIDATE_BLOCKS, summary)


# ============================================ 3. the v2 digest binding
def test_the_candidate_is_bound_to_the_exact_v2_summary_digest():
    record = T.build_candidate_record()
    assert record["bound_to"]["summary_sha256"] == T.V2_SUMMARY_SHA256
    assert T.V2_SUMMARY_SHA256 == (
        "30caa46961ee254a9b7faa3b10b4820cfe46c00a3015d5df3f33d013c3b13276")
    assert len(T.V2_ARTIFACTS) == 5
    for name, digest in T.V2_ARTIFACTS.items():
        assert len(digest) == 64, name


@needs_v2
def test_the_pinned_digests_are_the_artifacts_on_disk():
    from routegate.validation import sha256_file
    for name, expected in T.V2_ARTIFACTS.items():
        assert sha256_file(V2 / name) == expected, name


@needs_v2
def test_the_recorded_numbers_are_the_artifacts_numbers():
    outcome = T.verify_v2_result()
    assert outcome["reachable"] is True
    assert outcome["checks"].all_passed
    summary = outcome["summary"]
    assert summary["decision"]["decision"] == "BLOCKED"
    assert T.passing_blocks_from_summary(summary) == T.CANDIDATE_BLOCKS


@needs_v2
def test_an_altered_v2_artifact_is_refused(tmp_path):
    fake = tmp_path / "v2"
    fake.mkdir()
    for name in T.V2_ARTIFACTS:
        (fake / name).write_text("not the pinned artifact", encoding="utf-8")
    with pytest.raises(ValidationError, match="v2_artifact_sha256"):
        T.verify_v2_result(fake)


def test_an_unreachable_v2_run_is_a_fact_not_a_failure(tmp_path):
    assert T.verify_v2_result(tmp_path / "absent")["reachable"] is False


def test_the_committed_records_pass_every_gate():
    outcome = T.verify_records()
    assert outcome["checks"].all_passed
    assert outcome["checks"].as_dict()["checks_run"] > 20


def test_an_edited_record_fails_its_own_digest(tmp_path):
    for path, name in ((T.CLOSURE_MANIFEST, "closure"),
                       (T.CANDIDATE_MANIFEST, "candidate")):
        copy = tmp_path / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["note_added_later"] = "hello"
        copy.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        with pytest.raises(ValidationError, match="digest_recomputes"):
            if name == "closure":
                T.verify_records(closure_path=copy)
            else:
                T.verify_records(candidate_path=copy)


def test_a_re_digested_edit_is_still_refused(tmp_path):
    """Recomputing the digest does not launder an edit: the package is the
    second witness and it did not move."""
    copy = tmp_path / "candidate.json"
    payload = json.loads(T.CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    payload["bound_to"]["summary_sha256"] = "0" * 64
    payload[T.DIGEST_FIELD] = T.record_sha256(payload)
    copy.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValidationError, match="candidate_bound_to"):
        T.verify_records(candidate_path=copy)


# ================================= 4. alpha and direction identity
def test_alpha_and_the_direction_are_unchanged():
    intervention = T.candidate_intervention()
    assert intervention["alpha_R"] == ALPHA_R == 0.15
    assert intervention["g_R"] == WORST_CASE_G_R == 1.0
    assert intervention["applied_strength"] == 0.15
    assert intervention["alpha_S"] == 0.0

    direction = T.build_candidate_record()["direction"]
    assert direction["repair"] == "d_R = B_d1 - E"
    assert direction["artifact_sha256"] == (
        "d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb")
    assert direction["unchanged"] is True
    assert direction["safety_direction_is_not_used"] is True


def test_the_pinned_direction_digest_is_the_one_the_repository_pins():
    config = json.loads(
        (REPO_ROOT / "configs" / "dual_direction_controller.json")
        .read_text(encoding="utf-8"))
    assert T.build_candidate_record()["direction"]["artifact_sha256"] == (
        config["causal_screen"]["direction_bank"]["artifact_sha256"])


def test_no_safety_coefficient_and_no_controller_anywhere():
    record = T.build_candidate_record()
    assert record["controller"]["used_in_this_validation"] is False
    assert record["intervention"]["alpha_S"] == 0.0
    text = json.dumps(record)
    assert "H - E" in text          # named, so its absence is deliberate
    assert record["direction"]["safety_direction_is_not_used"] is True


def test_a_strength_above_the_frozen_one_is_refused():
    class _Bank:
        def directions_for_state(self, state):
            raise AssertionError("must refuse before reading a direction")
    with pytest.raises(T.TwoSiteError, match="outside"):
        T.build_two_site_plans(_Bank(), {}, alpha=0.2, g_r=1.0)
    with pytest.raises(T.TwoSiteError, match="outside"):
        T.build_two_site_plans(_Bank(), {}, alpha=0.15, g_r=0.0)


def test_planning_refuses_a_capture_missing_a_budget_reference():
    class _Bank:
        def directions_for_state(self, state):
            raise AssertionError("must refuse before reading a direction")
    with pytest.raises(T.TwoSiteError, match="capture"):
        T.build_two_site_plans(_Bank(), {32: object()})   # state 31 absent


# ================================================ 5. the one-pass requirement
def test_the_candidate_declares_one_forward_pass_and_prefill_only():
    intervention = T.candidate_intervention()
    assert intervention["forward_passes"] == 1
    assert intervention["prefill_only"] is True
    assert intervention["generated_token_positions_are_never_edited"] is True
    assert intervention["token_position"] == "final prompt token"
    assert intervention["budget_reference"] == "block_input"


def test_the_closure_records_why_one_pass_fails_at_block_33():
    closure = T.build_closure_record()
    assert closure["decision"] == "BLOCKED"
    assert closure["block_33_is_infeasible"]["block"] == 33
    assert closure["block_33_is_infeasible"]["rows_over_tolerance"] == 382
    assert "no ordering of hooks recovers it" in \
        closure["block_33_is_infeasible"]["structural_reason"].lower()
    assert closure["failed_specification"]["historical_blocks"] == [31, 32, 33]


def test_the_rejected_runtime_input_alternative_is_recorded_as_worse():
    variant = T.build_closure_record()["rejected_alternative"]
    assert variant["status"] == "REJECTED"
    assert variant["n_exceeding_tolerance"] == 432
    assert variant["max"] > T.V2_PER_BLOCK[33]["max"]


def test_no_tolerance_was_changed():
    closure = T.build_closure_record()
    assert closure["tolerance"] == TOLERANCE == 1e-2
    assert "was not changed" in closure["tolerance_note"]


def test_the_consequence_is_a_candidate_and_claims_nothing_yet():
    consequence = T.build_closure_record()["consequence"]
    assert consequence["remove_block"] == 33
    assert consequence["keep_blocks"] == [31, 32]
    assert consequence["status"] == "CANDIDATE, NOT VALIDATED"
    assert consequence["nothing_is_claimed_yet"] is True
    claims = T.build_candidate_record()["what_this_does_not_claim"]
    assert claims["a_causal_result"] is False
    assert claims["that_the_31_33_band_is_invalid"] is False


# ============================== 6-9. the runner's guards
def test_the_runner_refuses_a_sealed_output_directory(tmp_path):
    module = _load(RUNNER, "_two_site_runner")
    with pytest.raises(SealedSetError):
        module.claim_output_dir(tmp_path / "sealed_run", create=False)


def test_the_runner_refuses_a_non_empty_output_directory(tmp_path):
    module = _load(RUNNER, "_two_site_runner")
    out = tmp_path / "run"
    out.mkdir()
    assert module.claim_output_dir(out, create=False) == out
    (out / "screen_records.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="already holds"):
        module.claim_output_dir(out, create=False)


def test_checking_the_output_directory_does_not_create_it(tmp_path):
    module = _load(RUNNER, "_two_site_runner")
    out = tmp_path / "run"
    module.claim_output_dir(out, create=False)
    assert not out.exists()


def test_a_reference_run_with_any_different_digest_is_refused(tmp_path):
    from routegate.validation import Checklist
    module = _load(RUNNER, "_two_site_runner")
    reference = tmp_path / "phase1"
    reference.mkdir()
    expected = {"model": "Qwen/Qwen3-30B-A3B", "revision": "abc",
                "dataset_sha256": "d" * 64}
    (reference / "run_manifest.json").write_text(
        json.dumps({**expected, "revision": "MOVED"}), encoding="utf-8")
    with pytest.raises(ValidationError, match="reference_revision_matches"):
        module.assert_reusable_reference(reference, expected, Checklist("t"))


def test_a_matching_reference_run_is_reused(tmp_path):
    from routegate.validation import Checklist
    module = _load(RUNNER, "_two_site_runner")
    reference = tmp_path / "phase1"
    reference.mkdir()
    expected = {"model": "Qwen/Qwen3-30B-A3B", "revision": "abc",
                "dataset_sha256": "d" * 64}
    (reference / "run_manifest.json").write_text(json.dumps(expected),
                                                 encoding="utf-8")
    checks = Checklist("t")
    module.assert_reusable_reference(reference, expected, checks)
    assert checks.all_passed


def test_a_reference_without_its_own_manifest_is_refused(tmp_path):
    from routegate.validation import Checklist
    module = _load(RUNNER, "_two_site_runner")
    reference = tmp_path / "phase1"
    reference.mkdir()
    with pytest.raises(ValidationError, match="does not exist"):
        module.assert_reusable_reference(reference, {}, Checklist("t"))


def test_the_runner_exposes_no_option_reaching_held_out_material():
    from routegate.validation import looks_sealed
    module = _load(RUNNER, "_two_site_runner")
    options = [o for a in module.build_parser()._actions
               for o in a.option_strings]
    assert options
    for option in options:
        assert looks_sealed(option) is None, option
    for required in ("--direction-bank", "--output-dir", "--dry-run",
                     "--reuse-baseline"):
        assert required in options, required


def test_the_runner_judges_nothing_and_trains_nothing():
    """Scanned as CODE, with docstrings stripped.

    The prose necessarily says what the runner does not do -- it names the
    rubric and the adjudication procedure in order to state that neither is
    touched -- so a substring scan over the source would fail on the runner's
    own disclaimer.
    """
    import ast

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]
    code = ast.unparse(tree).lower()
    for banned in ("adjudicat", "rubric", "optimizer", "backward()",
                   "requires_grad_(true)", "import torch.optim"):
        assert banned not in code, banned
    # judging is named only as a flag that says it did not happen
    assert "judges_responses" in code
    assert "trains_a_controller" in code


def test_the_freezer_loads_no_model_and_generates_nothing():
    source = FREEZER.read_text(encoding="utf-8")
    for banned in ("torch", "load_model_and_tokenizer", "generate(",
                   "PatchController", "cuda"):
        assert banned not in source, banned


def test_the_records_hold_no_prompt_or_response_text():
    for build in (T.build_closure_record, T.build_candidate_record):
        payload = build()
        assert payload["contains_prompt_or_response_text"] is False
        payload.pop("contains_prompt_or_response_text")
        text = json.dumps(payload)
        for banned in ("prompt_text", "response_text"):
            assert banned not in text, banned


def test_the_records_are_derivable_and_carry_no_timestamp():
    for build in (T.build_closure_record, T.build_candidate_record):
        assert build() == build()
        assert "created_utc" not in build()
