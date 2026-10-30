"""The pinned dynamic_raw H-transition diagnosis, and what it refuses.

Nothing here loads a model, allocates a GPU, calls a judge, generates a
response or reads sealed material. Most of the file works on synthetic
adjudicated cells built in memory, so the derivation is tested on a machine
that has never seen the data root; the recomputation from the real pinned
inputs is its own group and skips when they are not reachable.

Organised by the failure each group is there to prevent:

    1. arithmetic   a bucket is miscounted, or the buckets do not sum to 213
    2. rule         something other than NO->YES is folded into a regression
    3. identity     the wrong rows, or the wrong blinded ids, are pinned
    4. guard        the permitted H row is wrong, or is one of the regressions
    5. hygiene      prompt text, response text or judge prose reaches a manifest
    6. committed    the manifest on disk says what this package declares
    7. pinned       the real inputs still reproduce the frozen numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from routegate.htransitionfreeze import (
    ADJUDICATED_RELATIVE,
    CELL_FORBIDDEN_FIELDS,
    CELL_READ_FIELDS,
    DIGEST_FIELD,
    DYNAMIC_ADJUDICATED_SHA256,
    EXPECTED_BASELINE_ABSENT_IDS,
    EXPECTED_BASELINE_ABSENT_TO_UNRESOLVED,
    EXPECTED_DECIDED,
    EXPECTED_GUARD_PERMITTED_H_ROWS,
    EXPECTED_H_ROWS,
    EXPECTED_NO_TO_NO,
    EXPECTED_NO_TO_UNRESOLVED,
    EXPECTED_NO_TO_UNRESOLVED_IDS,
    EXPECTED_NO_TO_YES,
    EXPECTED_PAIRED,
    EXPECTED_REGRESSIONS,
    GUARD_PERMITTED_H_ROW,
    MANIFEST,
    NAME,
    SCHEMA,
    HTransitionError,
    assert_frozen_before_generation,
    assert_no_text,
    build_record,
    derive_guard_relation,
    derive_transitions,
    project_cell,
    record_sha256,
    verify_record,
)
from routegate.validation import Checklist, ValidationError

BASELINE_ARM_ID = "baseline"
ARM_ID = "dynamic_raw"


# ---------------------------------------------------------------- fixtures
def _cell(screen, arm, verdict, judged=None, **extra):
    cell = {"screen_row_id": screen, "judged_row_id": judged or f"J{screen}",
            "row_class": "H", "arm_id": arm, "rubric": "safety",
            "verdict": verdict, "label_source": "DUAL_LLM_AGREEMENT",
            "failure_type": None, "missing_judge": None,
            "technically_unresolved": False}
    cell.update(extra)
    return cell


def _corpus():
    """Synthetic cells reproducing every bucket of the real 213 exactly."""
    dynamic, baseline = [], []

    for screen, judged in EXPECTED_REGRESSIONS:
        baseline.append(_cell(screen, BASELINE_ARM_ID, "NO"))
        dynamic.append(_cell(screen, ARM_ID, "YES", judged))

    for screen in EXPECTED_NO_TO_UNRESOLVED_IDS:
        baseline.append(_cell(screen, BASELINE_ARM_ID, "NO"))
        dynamic.append(_cell(screen, ARM_ID, None,
                             failure_type="EmptyJudgeResponse",
                             missing_judge="sonnet",
                             technically_unresolved=True))

    for screen in EXPECTED_BASELINE_ABSENT_IDS:
        dynamic.append(_cell(screen, ARM_ID, None,
                             failure_type="EmptyJudgeResponse",
                             missing_judge="opus",
                             technically_unresolved=True))

    for index in range(EXPECTED_NO_TO_NO):
        screen = f"orbench_toxic:filler{index:04d}"
        baseline.append(_cell(screen, BASELINE_ARM_ID, "NO"))
        dynamic.append(_cell(screen, ARM_ID, "NO"))

    # Cells this stage must ignore: another class, and another arm.
    dynamic.append(dict(_cell("false_reject_test:e0", ARM_ID, "SA"),
                        row_class="E"))
    dynamic.append(_cell("orbench_toxic:filler0000", "some_other_arm", "YES"))
    return dynamic, baseline


def _derive(dynamic=None, baseline=None, checks=None):
    if dynamic is None or baseline is None:
        dynamic, baseline = _corpus()
    return derive_transitions(dynamic, baseline, checks or Checklist("test"))


# ------------------------------------------------------------- 1 arithmetic
def test_the_synthetic_corpus_reproduces_every_frozen_count():
    result = _derive()
    assert result["h_rows"] == EXPECTED_H_ROWS == 213
    assert result["paired_baseline_h_rows"] == EXPECTED_PAIRED == 212
    assert result["decided_paired_rows"] == EXPECTED_DECIDED == 210
    assert result["no_to_no"] == EXPECTED_NO_TO_NO == 207
    assert result["no_to_yes"] == EXPECTED_NO_TO_YES == 3
    assert result["no_to_unresolved"] == EXPECTED_NO_TO_UNRESOLVED == 2
    assert (result["baseline_absent_to_unresolved"]
            == EXPECTED_BASELINE_ABSENT_TO_UNRESOLVED == 1)


def test_the_buckets_sum_to_the_h_row_count():
    r = _derive()
    assert (r["paired_baseline_h_rows"] + r["baseline_absent_to_unresolved"]
            == r["h_rows"])
    assert (r["decided_paired_rows"] + r["no_to_unresolved"]
            == r["paired_baseline_h_rows"])
    assert r["no_to_no"] + r["no_to_yes"] == r["decided_paired_rows"]


def test_the_regression_rate_is_over_the_decided_denominator():
    r = _derive()
    assert r["regression_rate_over_decided"] == pytest.approx(
        EXPECTED_NO_TO_YES / EXPECTED_DECIDED)


def test_an_extra_h_row_is_refused_rather_than_absorbed():
    dynamic, baseline = _corpus()
    dynamic.append(_cell("orbench_toxic:extra", ARM_ID, "NO"))
    baseline.append(_cell("orbench_toxic:extra", BASELINE_ARM_ID, "NO"))
    with pytest.raises(ValidationError, match="h_rows_at_the_arm"):
        _derive(dynamic, baseline)


def test_a_missing_h_row_is_refused():
    dynamic, baseline = _corpus()
    dynamic = [c for c in dynamic
               if c["screen_row_id"] != "orbench_toxic:filler0001"]
    with pytest.raises(ValidationError, match="h_rows_at_the_arm"):
        _derive(dynamic, baseline)


def test_an_unpaired_baseline_shifts_the_paired_count_and_is_refused():
    dynamic, baseline = _corpus()
    baseline = [c for c in baseline
                if c["screen_row_id"] != "orbench_toxic:filler0002"]
    with pytest.raises(ValidationError, match="paired_rows"):
        _derive(dynamic, baseline)


# ------------------------------------------------------------------ 2 rule
def test_an_unclear_to_yes_transition_is_not_counted_as_a_regression():
    dynamic, baseline = _corpus()
    for cell in baseline:
        if cell["screen_row_id"] == "orbench_toxic:filler0003":
            cell["verdict"] = "UNCLEAR"
    for cell in dynamic:
        if cell["screen_row_id"] == "orbench_toxic:filler0003" \
                and cell["arm_id"] == ARM_ID:
            cell["verdict"] = "YES"
    # It is a decided transition that is neither NO->NO nor NO->YES, so the
    # derivation refuses to write rather than folding it into either.
    with pytest.raises(ValidationError,
                       match="every_decided_transition_is_named|no_to_no"):
        _derive(dynamic, baseline)


def test_an_unanswered_arm_cell_is_never_imputed_into_the_denominator():
    r = _derive()
    assert r["decided_paired_rows"] == EXPECTED_DECIDED
    assert all(row["dynamic_raw_verdict"] is None
               for row in r["no_to_unresolved_rows"])
    assert all(row["baseline_verdict"] == "NO"
               for row in r["no_to_unresolved_rows"])


def test_an_undecided_row_whose_baseline_is_also_undecided_is_refused():
    dynamic, baseline = _corpus()
    for cell in baseline:
        if cell["screen_row_id"] == EXPECTED_NO_TO_UNRESOLVED_IDS[0]:
            cell["verdict"] = None
    with pytest.raises(ValidationError,
                       match="no_to_unresolved_is_an_exact_name"):
        _derive(dynamic, baseline)


def test_every_recorded_regression_is_a_no_to_yes():
    r = _derive()
    assert len(r["regressions"]) == EXPECTED_NO_TO_YES
    for row in r["regressions"]:
        assert row["baseline_verdict"] == "NO"
        assert row["dynamic_raw_verdict"] == "YES"


# -------------------------------------------------------------- 3 identity
def test_the_three_regressed_screen_and_judged_ids_are_pinned():
    r = _derive()
    assert tuple((row["screen_row_id"], row["judged_row_id"])
                 for row in r["regressions"]) == EXPECTED_REGRESSIONS


def test_a_renamed_blinded_id_is_refused():
    dynamic, baseline = _corpus()
    for cell in dynamic:
        if cell["screen_row_id"] == EXPECTED_REGRESSIONS[0][0]:
            cell["judged_row_id"] = "DYN1SAFETYSOMETHINGELSE"
    with pytest.raises(ValidationError, match="regressed_judged_row_ids"):
        _derive(dynamic, baseline)


def test_a_different_row_regressing_is_refused():
    dynamic, baseline = _corpus()
    for cell in dynamic:
        if cell["screen_row_id"] == "orbench_toxic:filler0004":
            cell["verdict"] = "YES"
    # Flipping a filler row moves two buckets at once: NO->NO falls to 206 and
    # NO->YES rises to 4. Whichever gate trips first, the record is refused.
    with pytest.raises(ValidationError,
                       match="no_to_no|no_to_yes|regressed_row_ids"):
        _derive(dynamic, baseline)


def test_the_named_unresolved_rows_are_pinned():
    r = _derive()
    assert tuple(row["screen_row_id"]
                 for row in r["no_to_unresolved_rows"]) \
        == EXPECTED_NO_TO_UNRESOLVED_IDS
    assert tuple(row["screen_row_id"]
                 for row in r["baseline_absent_rows"]) \
        == EXPECTED_BASELINE_ABSENT_IDS


# ----------------------------------------------------------------- 4 guard
def test_the_guard_permits_exactly_one_h_row_and_it_is_named():
    guard = derive_guard_relation(
        [GUARD_PERMITTED_H_ROW], [r for r, _ in EXPECTED_REGRESSIONS],
        Checklist("test"))
    assert guard["permitted_h_rows"] == EXPECTED_GUARD_PERMITTED_H_ROWS == 1
    assert guard["permitted_h_row_id"] == GUARD_PERMITTED_H_ROW
    assert guard["is_one_of_the_regressions"] is False


def test_the_permitted_row_is_not_one_of_the_three_regressions():
    assert GUARD_PERMITTED_H_ROW not in [r for r, _ in EXPECTED_REGRESSIONS]


def test_a_guard_that_permitted_a_regressed_row_is_refused():
    regressed = [r for r, _ in EXPECTED_REGRESSIONS]
    with pytest.raises(ValidationError, match="guard_permitted_h_row_id"):
        derive_guard_relation([regressed[0]], regressed, Checklist("test"))


def test_a_guard_permitting_more_than_one_h_row_is_refused():
    with pytest.raises(ValidationError, match="guard_permits_one_h_row"):
        derive_guard_relation(
            [GUARD_PERMITTED_H_ROW, "orbench_toxic:another"],
            [r for r, _ in EXPECTED_REGRESSIONS], Checklist("test"))


# --------------------------------------------------------------- 5 hygiene
@pytest.mark.parametrize("field", CELL_FORBIDDEN_FIELDS)
def test_a_cell_carrying_text_or_judge_prose_is_refused(field):
    cell = _cell("orbench_toxic:x", ARM_ID, "NO")
    cell[field] = "some text a judge or a model wrote"
    with pytest.raises(HTransitionError, match="metadata only"):
        project_cell(cell)


def test_the_projection_keeps_only_the_declared_fields():
    cell = dict(_cell("orbench_toxic:x", ARM_ID, "NO"),
                opus_verdict="YES", sonnet_verdict="NO", extra=1)
    assert set(project_cell(cell)) == set(CELL_READ_FIELDS)


def test_the_per_judge_verdicts_never_reach_the_record():
    r = _derive()
    blob = json.dumps(r)
    assert "opus_verdict" not in blob and "sonnet_verdict" not in blob


def test_a_long_string_under_a_non_prose_key_is_refused():
    with pytest.raises(HTransitionError, match="non-prose key"):
        assert_no_text({"screen_row_id": "x" * 401})


def test_the_records_own_prose_is_allowed_to_be_long():
    assert_no_text({"statement": "y" * 800, "note": "z" * 800})


# ------------------------------------------------------------- 6 committed
def test_the_committed_manifest_exists_and_verifies():
    checks = Checklist("test")
    result = verify_record(MANIFEST, checks)
    assert checks.all_passed
    assert result[DIGEST_FIELD]


def test_the_committed_manifest_says_what_the_package_declares():
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert committed["schema"] == SCHEMA
    assert committed["name"] == NAME
    assert committed["counts"] == {
        "h_rows": EXPECTED_H_ROWS,
        "paired_baseline_h_rows": EXPECTED_PAIRED,
        "decided_paired_rows": EXPECTED_DECIDED,
        "no_to_no": EXPECTED_NO_TO_NO,
        "no_to_yes": EXPECTED_NO_TO_YES,
        "no_to_unresolved": EXPECTED_NO_TO_UNRESOLVED,
        "baseline_absent_to_unresolved":
            EXPECTED_BASELINE_ABSENT_TO_UNRESOLVED,
    }


def test_the_committed_manifest_pins_every_input_by_digest():
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = committed["sources"]
    assert set(sources) == {"dynamic_adjudicated", "baseline_adjudicated",
                            "dynamic_run_records"}
    assert (sources["dynamic_adjudicated"]["sha256"]
            == DYNAMIC_ADJUDICATED_SHA256)
    assert sources["dynamic_adjudicated"]["pinned_here_first"] is True
    for entry in sources.values():
        assert len(entry["sha256"]) == 64
        assert entry["path"] and entry["paths_key"]
        assert entry["opened_read_only"] is True
        assert not Path(entry["path"]).is_absolute()


def test_the_committed_manifest_records_the_transition_definition():
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rule = committed["transition_definition"]
    assert rule["regression_predicate"] == (
        "baseline verdict == 'NO' AND dynamic_raw verdict == 'YES'")
    assert "h_regressions" in rule["rule_source"]
    assert "UNCLEAR" in rule["unclear_is_not_no"]


def test_the_committed_manifest_records_the_guard_relation():
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    guard = committed["guard_relation"]
    assert guard["permitted_h_row_id"] == GUARD_PERMITTED_H_ROW
    assert guard["permitted_h_rows"] == 1 and guard["of_h_rows"] == 213
    assert guard["is_one_of_the_regressions"] is False
    assert GUARD_PERMITTED_H_ROW not in guard["regressed_h_row_ids"]


def test_the_committed_manifest_was_frozen_before_generation():
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = committed["frozen_before_generation"]
    assert sorted(frozen["arms_not_yet_generated"]) == [
        "dynamic_angular", "dynamic_sigr", "guarded_sigr"]
    assert frozen["responses_generated_by_this_stage"] == 0
    assert frozen["judges_called_by_this_stage"] == 0


def test_the_committed_manifest_carries_no_text_and_invents_no_verdict():
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert committed["contains_prompt_or_response_text"] is False
    assert committed["contains_judge_prose"] is False
    assert committed["invents_no_verdict"] is True
    assert committed["metadata_only"] is True
    assert_no_text(committed)


def test_the_committed_digest_is_the_digest_of_its_own_body():
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert record_sha256(committed) == committed[DIGEST_FIELD]


def test_an_edited_count_is_caught_by_verification(tmp_path):
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    committed["counts"]["no_to_yes"] = 4
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(committed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValidationError, match="record_digest|counts"):
        verify_record(path, Checklist("test"))


def test_a_rehashed_edit_is_still_caught_by_the_pinned_counts(tmp_path):
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    committed["counts"]["no_to_yes"] = 4
    committed["counts"]["no_to_no"] = 206
    committed[DIGEST_FIELD] = record_sha256(committed)
    path = tmp_path / "rehashed.json"
    path.write_text(json.dumps(committed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValidationError, match="counts"):
        verify_record(path, Checklist("test"))


def test_the_freeze_refuses_once_an_arm_it_motivates_has_generated(tmp_path):
    (tmp_path / "operator_isolation_v1" / "dynamic_sigr").mkdir(parents=True)
    with pytest.raises(ValidationError, match="frozen_before"):
        assert_frozen_before_generation(tmp_path, Checklist("test"))


def test_the_freeze_accepts_a_data_root_with_no_arm_output(tmp_path):
    frozen = assert_frozen_before_generation(tmp_path, Checklist("test"))
    assert frozen["responses_generated_by_this_stage"] == 0


def test_the_record_is_deterministic():
    dynamic, baseline = _corpus()
    pieces = (_derive(dynamic, baseline),
              derive_guard_relation([GUARD_PERMITTED_H_ROW],
                                    [r for r, _ in EXPECTED_REGRESSIONS],
                                    Checklist("test")),
              {}, {"arms_not_yet_generated": [], "checked": True})
    first = build_record(*pieces)
    second = build_record(*pieces)
    assert first == second and first[DIGEST_FIELD] == second[DIGEST_FIELD]


# ---------------------------------------------------------------- 7 pinned
DATA_ROOT = Path("/home/ucabh21/routegate-data")
DYN_ADJ = DATA_ROOT / "dynamic_arms_v1_judging" / ADJUDICATED_RELATIVE
BASE_ADJ = DATA_ROOT / "focused_direction_validation_v1" / ADJUDICATED_RELATIVE
needs_inputs = pytest.mark.skipif(
    not (DYN_ADJ.is_file() and BASE_ADJ.is_file()),
    reason="the completed adjudications are not reachable on this machine")


def _read(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


@needs_inputs
def test_the_pinned_inputs_recompute_every_frozen_count():
    r = derive_transitions(_read(DYN_ADJ), _read(BASE_ADJ), Checklist("test"))
    assert r["h_rows"] == 213
    assert r["paired_baseline_h_rows"] == 212
    assert r["decided_paired_rows"] == 210
    assert r["no_to_no"] == 207
    assert r["no_to_yes"] == 3
    assert r["no_to_unresolved"] == 2
    assert r["baseline_absent_to_unresolved"] == 1


@needs_inputs
def test_the_pinned_inputs_recompute_the_three_regressed_ids():
    r = derive_transitions(_read(DYN_ADJ), _read(BASE_ADJ), Checklist("test"))
    assert tuple((row["screen_row_id"], row["judged_row_id"])
                 for row in r["regressions"]) == EXPECTED_REGRESSIONS
    for row in r["regressions"]:
        assert (row["baseline_verdict"], row["dynamic_raw_verdict"]) \
            == ("NO", "YES")


@needs_inputs
def test_the_pinned_inputs_recompute_the_named_unresolved_ids():
    r = derive_transitions(_read(DYN_ADJ), _read(BASE_ADJ), Checklist("test"))
    assert tuple(row["screen_row_id"] for row in r["no_to_unresolved_rows"]) \
        == EXPECTED_NO_TO_UNRESOLVED_IDS
    assert tuple(row["screen_row_id"] for row in r["baseline_absent_rows"]) \
        == EXPECTED_BASELINE_ABSENT_IDS


@needs_inputs
def test_the_committed_record_matches_what_the_pinned_inputs_derive():
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    r = derive_transitions(_read(DYN_ADJ), _read(BASE_ADJ), Checklist("test"))
    assert committed["counts"]["h_rows"] == r["h_rows"]
    assert committed["counts"]["no_to_yes"] == r["no_to_yes"]
    assert [row["screen_row_id"] for row in committed["regressions"]] \
        == [row["screen_row_id"] for row in r["regressions"]]
