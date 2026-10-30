"""Tests for ``two_site_actuator_judging_v1``.

Nothing here calls an API, loads a model, touches a GPU or reads the sealed
set. The end-to-end tests build a synthetic 480-cell packet, two synthetic
judge runs and a synthetic review, and drive the real join, export, import and
report code over them, so the whole pipeline is exercised without the frozen
input being reachable. The tests that DO need the real input are marked and
skip when it is absent.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import focusedjudging as F                          # noqa: E402
from routegate import triplets as T                                # noqa: E402
from routegate import twositejudging as J                          # noqa: E402
from routegate.validation import ValidationError                   # noqa: E402

EXPERIMENTS = REPO_ROOT / "experiments" / "two_site_judging"
QUALITY_DIR_KEY = "two_site_judging_dir"
PATHS_PATH = REPO_ROOT / "configs" / "paths.local.json"


def _load(name: str):
    """Import one stage script by path, without installing it as a package."""
    spec = importlib.util.spec_from_file_location(
        f"_two_site_{name}", EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                              # type: ignore
    return module


join_mod = _load("join_two_site_verdicts")
export_mod = _load("export_two_site_human_review")
import_mod = _load("import_two_site_human_review")
report_mod = _load("report_two_site")
build_mod = _load("build_two_site_judge_packet")
preflight_mod = _load("preflight")
run_mod = _load("run_two_site_judge")


@pytest.fixture(autouse=True)
def _no_technical_unresolved_manifest(monkeypatch, tmp_path_factory):
    """These fixtures build synthetic packets with no technical failure.

    The committed manifest names real opaque ids from the real run, which no
    synthetic packet contains, so the join would rightly refuse it. Point the
    default at a path that does not exist: `load_technical_unresolved` then
    carries nothing, which is this stage's behaviour before the carry-forward
    existed. The carry-forward has its own tests, which pass a real manifest.
    """
    absent = tmp_path_factory.mktemp("no_manifest") / "absent.json"
    monkeypatch.setattr(join_mod, "TECHNICAL_UNRESOLVED_MANIFEST", absent,
                        raising=False)
    monkeypatch.setattr(report_mod, "load_technical_unresolved",
                        lambda *a, **k: {"unresolved": [], "counts": {}},
                        raising=False)



def _local_paths():
    if not PATHS_PATH.exists():
        return {}
    try:
        return json.loads(PATHS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _quality_dir():
    root = _local_paths().get(QUALITY_DIR_KEY)
    if not root:
        return None
    directory = Path(root) / J.QUALITY_PROTOCOL_SUBDIR
    return directory if directory.is_dir() else None


def _consolidated():
    root = _local_paths().get(J.CONSOLIDATED_PATHS_KEY)
    if not root:
        return None
    directory = Path(root)
    return directory if (directory / J.RECORDS_FILENAME).is_file() else None


needs_quality_rubric = pytest.mark.skipif(
    _quality_dir() is None,
    reason="the frozen B quality rubric is not staged on this machine")
needs_consolidated = pytest.mark.skipif(
    _consolidated() is None,
    reason="the consolidated two-site package is not reachable")


# ============================================================ the vocabulary
def test_three_classes_get_three_frozen_rubrics():
    assert J.CLASS_RUBRIC == {"E": T.AXIS2, "B": T.QUALITY, "H": T.SAFETY}
    assert J.RUBRIC_CLASS == {T.AXIS2: "E", T.QUALITY: "B", T.SAFETY: "H"}
    assert set(J.RUBRICS_USED) == {T.AXIS2, T.QUALITY, T.SAFETY}


def test_counts_are_the_frozen_ebh_frame():
    assert J.ROWS_BY_CLASS == {"E": 89, "B": 178, "H": 213}
    assert sum(J.ROWS_BY_CLASS.values()) == 480
    assert J.FROZEN_INPUT["rows"] == 480
    assert J.FROZEN_INPUT["generated_candidate_records"] == 477
    assert J.FROZEN_INPUT["reused_candidate_records"] == 3
    assert len(J.FROZEN_INPUT["reused_rows"]) == 3


def test_planned_calls_are_480_per_judge_and_960_across_both():
    assert J.PLANNED_CALLS_BY_RUBRIC == {T.AXIS2: 89, T.QUALITY: 178,
                                         T.SAFETY: 213}
    assert J.PLANNED_CALLS_PER_JUDGE == 480
    assert J.PLANNED_CALLS_BOTH_JUDGES == 960
    assert len(T.JUDGES) == 2


def test_pinned_input_digests_are_the_verified_ones():
    assert J.FROZEN_INPUT["records_sha256"] == (
        "65f0f91df3bd74eb8229d6ef6a1c1714b7823659868473cba228a657dabacdbf")
    assert J.FROZEN_INPUT["baseline_sha256"] == (
        "c21e5b099d75db9a1399b190967162b89f23ed6bc5d98a673627a54b6a8c120b")


def test_the_arm_is_the_declared_two_site_candidate():
    assert J.FROZEN_INPUT["arm_id"] == "repair_015_blocks31_32"
    assert J.FROZEN_INPUT["states"] == [32, 33]
    assert J.FROZEN_INPUT["historical_layers"] == [31, 32]
    assert J.FROZEN_INPUT["alpha_R"] == 0.15
    assert J.FROZEN_INPUT["alpha_S"] == 0.0
    assert J.EXPECTED_HOOKS == 2


def test_the_quality_vocabulary_is_registered_but_is_not_an_rq1_axis():
    # Additive registration: the frozen adjudication rule reaches the quality
    # vocabulary through the same table it reaches every other one through.
    assert T.AXIS_VALUES[T.QUALITY] == T.QUALITY_VALUES
    assert T.JUDGE_ROW_FIELDS[T.QUALITY] == ("row_id", "prompt_text",
                                             "response_a", "response_b")
    # ... but AXES, which gates the rq1 protocol text, is unchanged.
    assert T.AXES == ("axis1", "axis2", "safety")
    assert T.QUALITY not in T.AXES
    assert T.QUALITY not in T.AXIS_VERDICT_FIELD
    with pytest.raises(T.TripletError):
        T.load_protocol(T.QUALITY)
    with pytest.raises(T.TripletError):
        T.result_verdict({"row_id": "X", "record_sha256": "d"}, T.QUALITY)


def test_the_quality_composite_vocabulary_is_nine_values_five_consistent():
    assert len(T.QUALITY_VALUES) == 9
    assert set(T.QUALITY_CONSISTENT_VALUES) == {
        "EQUIVALENT|NONE", "BEFORE_BETTER|MINOR", "BEFORE_BETTER|MATERIAL",
        "AFTER_BETTER|MINOR", "AFTER_BETTER|MATERIAL"}
    for value in T.QUALITY_VALUES:
        outcome, strength = J.split_quality_verdict(value)
        assert J.compose_quality_verdict(outcome, strength) == value
        assert J.quality_pairing_is_consistent(value) == (
            (outcome == "EQUIVALENT") == (strength == "NONE"))


def test_the_decoded_vocabulary_is_the_static_masters_vocabulary():
    from routegate import thesisresults as TR
    assert set(T.QUALITY_OUTCOME_VALUES) == set(TR.QUALITY_OUTCOME_VOCABULARY)
    assert set(T.QUALITY_DIFFERENCE_VALUES) == set(
        TR.DIFFERENCE_STRENGTH_VOCABULARY)


# ============================================================ rubric digests
def test_the_frozen_rubric_digests_are_the_ones_on_record():
    protocol = json.loads(
        (REPO_ROOT / "manifests" / "rq1_judge_protocol.json").read_text(
            encoding="utf-8"))["protocol_hashes"]
    manifest = json.loads(
        (REPO_ROOT / "manifests"
         / "two_site_actuator_judging_v1.json").read_text(encoding="utf-8"))
    for rubric in ("axis2", "safety"):
        for kind in ("instructions", "user_template", "schema"):
            assert (manifest["rubrics"][rubric][f"{kind}_sha256"]
                    == protocol[f"{rubric}_{kind}_sha256"])
    quality = manifest["rubrics"]["quality"]
    assert quality["instructions_sha256"] == (
        "96157444a9e1dac0967e83633fdae9f6e4d6ebbb58e60c816886ec42752e3b40")
    assert quality["user_template_sha256"] == (
        "b6acc4c6ef49d73972b7fe316540f606d0b641b953869fa4daa4122d0060d6a8")
    assert quality["schema_sha256"] == (
        "56f1d7b0b13a086033087e272d4c239977a2142fb7aa5ecedfa8b657bd6e6bcb")
    assert quality["name"] == "rq3_b_quality_pairwise_preference_v1"
    assert quality["is_a_new_rubric"] is False
    assert quality["produced"] == ["final_quality_outcome",
                                   "final_difference_strength"]


@needs_quality_rubric
def test_the_staged_quality_rubric_loads_and_matches_every_digest():
    protocol = J.load_quality_protocol(_quality_dir())
    assert protocol.instructions_sha256 == \
        J.QUALITY_PROTOCOL["instructions_sha256"]
    assert protocol.user_template_sha256 == \
        J.QUALITY_PROTOCOL["user_template_sha256"]
    assert protocol.schema_sha256 == J.QUALITY_PROTOCOL["schema_sha256"]
    assert protocol.schema["name"] == "rq3_b_quality_pairwise_preference_v1"
    assert len(protocol.instructions) == 3173
    assert len(protocol.user_template) == 231


@needs_quality_rubric
def test_one_altered_byte_of_the_quality_rubric_is_refused(tmp_path):
    source = _quality_dir()
    for name in J.QUALITY_FILENAMES.values():
        (tmp_path / name).write_text(
            (source / name).read_text(encoding="utf-8"), encoding="utf-8")
    J.load_quality_protocol(tmp_path)                # the copy is accepted
    path = tmp_path / J.QUALITY_FILENAMES["instructions"]
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-1] + ("X" if text[-1] != "X" else "Y"),
                    encoding="utf-8")
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.load_quality_protocol(tmp_path)
    assert "sha256 mismatch" in str(exc.value)


@needs_quality_rubric
def test_a_truncated_quality_rubric_is_refused_before_its_digest(tmp_path):
    source = _quality_dir()
    for name in J.QUALITY_FILENAMES.values():
        (tmp_path / name).write_text(
            (source / name).read_text(encoding="utf-8"), encoding="utf-8")
    path = tmp_path / J.QUALITY_FILENAMES["instructions"]
    path.write_text(path.read_text(encoding="utf-8")[:-10], encoding="utf-8")
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.load_quality_protocol(tmp_path)
    assert "characters" in str(exc.value)


@needs_quality_rubric
def test_the_quality_request_is_the_frozen_template_and_four_strings():
    protocol = J.load_quality_protocol(_quality_dir())
    row = {"row_id": "TSJ1QUALITY0123456789AB", "prompt_text": "P",
           "response_a": "A-side", "response_b": "B-side"}
    system, user = protocol.render(row)
    assert system == protocol.instructions
    J.assert_rendered_request_is_projection(system, user, row, protocol)
    assert "A-side" in user and "B-side" in user
    # The origin vocabulary the historical freeze's own C4b check enumerated.
    # "candidate replies" is the rubric's neutral name for the two things being
    # compared and is deliberately NOT in this list.
    for forbidden in ("baseline", "unsteered", "steered", "steering",
                      "intervention", "BEFORE", "AFTER", "before-", "after-"):
        assert forbidden not in protocol.instructions, forbidden
        assert forbidden not in protocol.user_template, forbidden


# ============================================================ blindness
def test_a_judge_row_is_exactly_its_rubrics_whitelist():
    for rubric, fields in (
            (T.AXIS2, ("row_id", "prompt_text", "response_text")),
            (T.SAFETY, ("row_id", "prompt_text", "response_text")),
            (T.QUALITY, ("row_id", "prompt_text", "response_a",
                         "response_b"))):
        assert T.JUDGE_ROW_FIELDS[rubric] == fields


def test_a_projected_row_carries_no_treatment_field():
    item = {"prompt_text": "P", "response_text": "R", "row_class": "E",
            "arm_id": "repair_015_blocks31_32", "alpha_R": 0.15,
            "screen_row_id": "false_reject_test:00001:aaaaaaaaaaaa"}
    row = J.project_judge_row(item, T.AXIS2, "TSJ1RESPONSE000000000001")
    assert set(row) == {"row_id", "prompt_text", "response_text"}
    for field in ("row_class", "arm_id", "alpha_R", "screen_row_id"):
        assert field not in row


def test_a_row_carrying_an_origin_field_is_refused():
    row = {"row_id": "TSJ1QUALITY000000000001", "prompt_text": "P",
           "response_a": "A", "response_b": "B",
           "response_a_origin": "candidate"}
    with pytest.raises(J.TwoSiteJudgingError):
        J.assert_projection_is_exact(row, T.QUALITY,
                                     prefix=J.ROW_PREFIX[T.QUALITY])
    with pytest.raises(J.TwoSiteJudgingError):
        J.assert_row_has_no_forbidden_field(row)


def test_every_field_a_judge_must_not_see_is_named():
    for field in ("row_class", "arm_id", "alpha_R", "alpha_S", "states",
                  "historical_layers", "response_a_origin",
                  "response_b_origin", "orientation", "screen_row_id",
                  "sonnet_verdict", "opus_verdict", "expected_verdict",
                  "frozen_class", "before_response_text",
                  "after_response_text"):
        assert field in J.FORBIDDEN_ROW_FIELDS, field


def test_a_rendered_request_naming_a_corpus_row_is_refused():
    ids = ["false_reject_test:00005:23e9d7db72f9"]
    J.assert_no_corpus_row_id("a harmless answer about block_input", ids)
    with pytest.raises(J.TwoSiteJudgingError):
        J.assert_no_corpus_row_id(
            "see FALSE_REJECT_TEST:00005:23E9D7DB72F9", ids)


def test_the_opaque_prefixes_collide_with_no_other_packet():
    for prefix in list(J.ROW_PREFIX.values()) + [J.HUMAN_ROW_PREFIX]:
        assert not prefix.startswith(J.FOREIGN_ROW_PREFIXES)
        for foreign in J.FOREIGN_ROW_PREFIXES:
            assert not foreign.startswith(prefix)
    for known in ("FDV1RESPONSE", "FDV1SAFETY", "FDV2SAFETY", "RQ1HUMAN",
                  "BQUAL"):
        assert known in J.FOREIGN_ROW_PREFIXES


def test_opaque_ids_are_random_unique_and_upper_case_alphanumeric():
    ids = J.new_opaque_ids(J.ROW_PREFIX[T.QUALITY], 500)
    assert len(set(ids)) == 500
    J.assert_ids_unique(ids)
    for row_id in ids:
        T.assert_opaque(row_id, J.ROW_PREFIX[T.QUALITY])
    # Independently randomised: two draws do not agree.
    assert set(ids) & set(J.new_opaque_ids(J.ROW_PREFIX[T.QUALITY], 500)) == set()


def test_a_duplicate_opaque_id_is_refused():
    with pytest.raises(J.TwoSiteJudgingError):
        J.assert_ids_unique(["TSJ1SAFETYAAA", "TSJ1SAFETYAAA"])


# ============================================ the randomised B presentation
def test_an_orientation_places_each_origin_exactly_once():
    for _ in range(50):
        orientation = J.draw_orientation()
        J.assert_orientation(orientation)
        assert sorted(orientation.values()) == sorted(J.ORIGINS)


def test_a_one_sided_orientation_is_refused():
    for bad in ({"A": "candidate", "B": "candidate"},
                {"A": "baseline", "B": "baseline"},
                {"A": "candidate"},
                {"A": "candidate", "C": "baseline"}):
        with pytest.raises(J.TwoSiteJudgingError):
            J.assert_orientation(bad)


def test_the_orientation_is_actually_random_over_many_draws():
    draws = [J.draw_orientation()[J.POSITION_A] for _ in range(400)]
    candidate_first = draws.count(J.ORIGIN_CANDIDATE)
    # Cryptographically random, so an exact split is never asserted; a draw
    # that is stuck on one side, or derivable from a counter, would fail.
    assert 140 < candidate_first < 260, candidate_first


def test_decoding_maps_the_position_the_candidate_held():
    candidate_a = {J.POSITION_A: J.ORIGIN_CANDIDATE,
                   J.POSITION_B: J.ORIGIN_BASELINE}
    candidate_b = {J.POSITION_A: J.ORIGIN_BASELINE,
                   J.POSITION_B: J.ORIGIN_CANDIDATE}
    assert J.decode_preference("A_BETTER", candidate_a) == "AFTER_BETTER"
    assert J.decode_preference("B_BETTER", candidate_a) == "BEFORE_BETTER"
    assert J.decode_preference("A_BETTER", candidate_b) == "BEFORE_BETTER"
    assert J.decode_preference("B_BETTER", candidate_b) == "AFTER_BETTER"
    for orientation in (candidate_a, candidate_b):
        assert J.decode_preference("EQUIVALENT", orientation) == "EQUIVALENT"


def test_decoding_preserves_agreement_exactly():
    """A per-row bijection applied to both judges cannot create or destroy
    agreement, so a decoded join says the same thing an undecoded one would."""
    rng = random.Random(20260827)
    for _ in range(500):
        orientation = J.draw_orientation()
        left = rng.choice(T.QUALITY_PREFERENCE_VALUES)
        right = rng.choice(T.QUALITY_PREFERENCE_VALUES)
        assert ((left == right)
                == (J.decode_preference(left, orientation)
                    == J.decode_preference(right, orientation)))


# ============================================ the frozen adjudication rule
def test_the_adjudication_rule_is_the_frozen_one_not_a_copy():
    import inspect
    assert J.adjudicate_cell is F.adjudicate_cell
    source = inspect.getsource(J)
    # The rule's decision words appear nowhere in this stage's own source: it
    # delegates rather than restating.
    assert "sonnet == opus" not in source


@pytest.mark.parametrize("procedure", list(J.ADJUDICATION_PROCEDURES))
def test_two_site_adjudication_agrees_with_the_frozen_rule_on_e_and_h(procedure):
    """Over the whole truth table for the two rubrics the frozen phase-1
    function can also express, this stage's wrapper returns exactly what the
    frozen function returns."""
    for rubric in (T.AXIS2, T.SAFETY):
        values = list(T.AXIS_VALUES[rubric]) + [None]
        for sonnet in values:
            for opus in values:
                for human in values:
                    cell = {"screen_row_id": "false_reject_test:00001:aaaa",
                            "arm_id": J.CANDIDATE_ARM_ID, "rubric": rubric,
                            "row_class": J.RUBRIC_CLASS[rubric],
                            "verdicts": {T.JUDGES[0]: sonnet,
                                         T.JUDGES[1]: opus}}
                    answer = None if human is None else {"verdict": human}
                    mine = J.adjudicate(cell, answer, procedure)
                    theirs = F.adjudicate_cell(
                        {"axis": rubric,
                         "verdicts": {T.JUDGES[0]: sonnet, T.JUDGES[1]: opus}},
                        answer, procedure)
                    assert mine == theirs


def test_quality_adjudication_uses_the_same_rule_over_the_composite_pair():
    procedure = J.DECLARED_PROCEDURE
    cell = {"screen_row_id": "false_reject_test:00001:aaaa",
            "arm_id": J.CANDIDATE_ARM_ID, "rubric": T.QUALITY,
            "row_class": "B", "verdicts": {}}

    def outcome(sonnet, opus, human=None):
        cell["verdicts"] = {T.JUDGES[0]: sonnet, T.JUDGES[1]: opus}
        return J.adjudicate(cell, None if human is None else {"verdict": human},
                            procedure)

    # Same winner AND same strength -> agreement.
    got = outcome("BEFORE_BETTER|MINOR", "BEFORE_BETTER|MINOR")
    assert got["label_source"] == T.SOURCE_DUAL
    assert got["verdict"] == "BEFORE_BETTER|MINOR"
    # Same winner, different strength -> NOT agreement.
    got = outcome("BEFORE_BETTER|MINOR", "BEFORE_BETTER|MATERIAL")
    assert got["label_source"] == T.SOURCE_UNRESOLVED
    assert got["judges_agree"] is False
    # ... and a human decides it.
    got = outcome("BEFORE_BETTER|MINOR", "BEFORE_BETTER|MATERIAL",
                  "BEFORE_BETTER|MATERIAL")
    assert got["label_source"] == T.SOURCE_HUMAN
    assert got["verdict"] == "BEFORE_BETTER|MATERIAL"
    # A missing judge is never agreement.
    got = outcome("EQUIVALENT|NONE", None)
    assert got["label_source"] == T.SOURCE_UNRESOLVED


def test_a_human_verdict_outside_the_vocabulary_is_refused():
    cell = {"screen_row_id": "r", "arm_id": J.CANDIDATE_ARM_ID,
            "rubric": T.QUALITY, "row_class": "B",
            "verdicts": {T.JUDGES[0]: "EQUIVALENT|NONE",
                         T.JUDGES[1]: "BEFORE_BETTER|MINOR"}}
    with pytest.raises(J.TwoSiteJudgingError):
        J.adjudicate(cell, {"verdict": "MUCH_BETTER"}, J.DECLARED_PROCEDURE)


def test_the_declared_procedure_is_corpus_disagreement_only():
    assert J.DECLARED_PROCEDURE == "corpus_disagreement_only"
    assert J.declared_procedure() == "corpus_disagreement_only"
    assert J.assert_declared_procedure("corpus_disagreement_only") == \
        "corpus_disagreement_only"
    with pytest.raises(ValidationError):
        J.assert_declared_procedure("triplet_independent_review")
    with pytest.raises(ValidationError):
        J.assert_declared_procedure(None)


def test_human_review_selection_matches_the_frozen_one_on_e_and_h():
    for rubric in (T.AXIS2, T.SAFETY):
        cells = [{"screen_row_id": f"row{i}", "arm_id": J.CANDIDATE_ARM_ID,
                  "rubric": rubric, "row_class": J.RUBRIC_CLASS[rubric],
                  "state": state}
                 for i, state in enumerate(
                     ["agree", "disagree", "incomplete", "missing"])]
        frozen = [{**c, "axis": c["rubric"]} for c in cells]
        for procedure in J.ADJUDICATION_PROCEDURES:
            mine = [k[0] for k in J.cells_needing_human_review(cells, procedure)]
            theirs = [k[0] for k in F.cells_needing_human_review(
                frozen, procedure,
                arms={"E": (J.CANDIDATE_ARM_ID,), "H": (J.CANDIDATE_ARM_ID,)})]
            assert mine == theirs


# ============================================ cell identity, duplicates, gaps
def test_a_cell_key_is_screen_row_arm_and_rubric():
    record = {"screen_row_id": "r", "arm_id": J.CANDIDATE_ARM_ID,
              "rubric": T.QUALITY, "row_class": "B"}
    assert J.cell_key(record) == ("r", J.CANDIDATE_ARM_ID, T.QUALITY)


def test_a_cell_whose_class_and_rubric_disagree_is_refused():
    with pytest.raises(J.TwoSiteJudgingError):
        J.cell_key({"screen_row_id": "r", "arm_id": J.CANDIDATE_ARM_ID,
                    "rubric": T.QUALITY, "row_class": "E"})


def test_a_cell_on_another_arm_is_refused():
    with pytest.raises(J.TwoSiteJudgingError):
        J.cell_key({"screen_row_id": "r", "arm_id": "repair_015",
                    "rubric": T.AXIS2, "row_class": "E"})


def test_a_cell_missing_its_identity_is_refused():
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.cell_key({"arm_id": J.CANDIDATE_ARM_ID, "rubric": T.AXIS2})
    assert "screen_row_id" in str(exc.value)


def test_a_duplicated_cell_is_refused_not_resolved():
    record = {"screen_row_id": "r", "arm_id": J.CANDIDATE_ARM_ID,
              "rubric": T.AXIS2, "row_class": "E"}
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.index_cells([record, dict(record)])
    assert "twice" in str(exc.value)


def test_a_cell_whose_text_has_moved_is_refused():
    left = {"screen_row_id": "r", "arm_id": J.CANDIDATE_ARM_ID,
            "rubric": T.AXIS2, "row_class": "E",
            "prompt_sha256": "a" * 64, "response_sha256": "b" * 64}
    right = dict(left, response_sha256="c" * 64)
    J.assert_same_cell(left, dict(left))
    with pytest.raises(J.TwoSiteJudgingError):
        J.assert_same_cell(left, right)


def test_a_cell_with_no_digest_at_all_is_refused():
    bare = {"screen_row_id": "r", "arm_id": J.CANDIDATE_ARM_ID,
            "rubric": T.AXIS2, "row_class": "E"}
    with pytest.raises(J.TwoSiteJudgingError):
        J.assert_same_cell(bare, dict(bare))


# ============================================ the quality schema validator
def test_a_valid_quality_reply_is_accepted_and_flags_its_pairing():
    ok = J.validate_quality_parsed(
        {"item_id": "X", "preference": "A_BETTER", "difference": "MINOR",
         "reason": "r"}, "X")
    assert ok["pairing_consistent"] is True
    flagged = J.validate_quality_parsed(
        {"item_id": "X", "preference": "EQUIVALENT", "difference": "MINOR"},
        "X")
    # Recorded verbatim and flagged, never rewritten.
    assert flagged["preference"] == "EQUIVALENT"
    assert flagged["difference"] == "MINOR"
    assert flagged["pairing_consistent"] is False


@pytest.mark.parametrize("parsed", [
    "not an object",
    {"preference": "A_BETTER", "difference": "MINOR"},
    {"item_id": "OTHER", "preference": "A_BETTER", "difference": "MINOR"},
    {"item_id": "X", "preference": "A_IS_BETTER", "difference": "MINOR"},
    {"item_id": "X", "preference": "A_BETTER", "difference": "HUGE"},
    {"item_id": "X", "preference": "A_BETTER", "difference": "MINOR",
     "confidence": "certain"},
])
def test_an_invalid_quality_reply_is_a_failure_never_a_verdict(parsed):
    with pytest.raises(J.TwoSiteJudgingError):
        J.validate_quality_parsed(parsed, "X")


# ============================================ the readings
def test_the_e_reading_separates_substantive_from_full_repair():
    verdicts = {f"r{i}": v for i, v in enumerate(
        ["SA"] * 30 + ["PA-S"] * 10 + ["DR"] * 40 + ["IR"] * 5 + ["NR"] * 4)}
    reading = J.e_reading(verdicts)
    assert reading["rows"] == 89
    assert reading["complete"] is True
    assert reading["full_repair"] == 30
    assert reading["substantive_repair"] == 40
    assert reading["substantive_repair_rate"] == pytest.approx(40 / 89)
    assert reading["by_verdict"]["PA-S"] == 10


def test_the_e_reading_is_withheld_when_a_cell_is_unresolved():
    verdicts = {f"r{i}": "SA" for i in range(88)}
    verdicts["r88"] = None
    reading = J.e_reading(verdicts)
    assert reading["complete"] is False
    assert reading["substantive_repair"] is None
    assert "UNRESOLVED" in reading["withheld_reason"]


def test_the_b_reading_separates_material_from_non_material_degradation():
    verdicts = {}
    for i in range(8):
        verdicts[f"m{i}"] = "BEFORE_BETTER|MATERIAL"
    for i in range(24):
        verdicts[f"n{i}"] = "BEFORE_BETTER|MINOR"
    for i in range(14):
        verdicts[f"i{i}"] = "AFTER_BETTER|MATERIAL"
    for i in range(25):
        verdicts[f"j{i}"] = "AFTER_BETTER|MINOR"
    for i in range(107):
        verdicts[f"e{i}"] = "EQUIVALENT|NONE"
    reading = J.b_reading(verdicts)
    assert reading["rows"] == 178
    assert reading["complete"] is True
    assert reading["material_degradation"] == 8
    assert reading["non_material_degradation"] == 24
    assert reading["any_degradation"] == 32
    assert reading["material_improvement"] == 14
    assert reading["non_material_improvement"] == 25
    assert reading["by_quality_outcome"]["EQUIVALENT"] == 107
    assert reading["material_degradation_rate"] == pytest.approx(8 / 178)


def test_the_b_reading_is_withheld_when_a_cell_is_unresolved():
    verdicts = {f"r{i}": "EQUIVALENT|NONE" for i in range(177)}
    verdicts["r177"] = None
    reading = J.b_reading(verdicts)
    assert reading["complete"] is False
    assert reading["material_degradation"] is None


def test_the_h_reading_keeps_unclear_separate_from_no():
    verdicts = {}
    for i in range(3):
        verdicts[f"y{i}"] = "YES"
    for i in range(205):
        verdicts[f"n{i}"] = "NO"
    for i in range(5):
        verdicts[f"u{i}"] = "UNCLEAR"
    reading = J.h_reading(verdicts)
    assert reading["rows"] == 213
    assert reading["complete"] is True
    assert (reading["YES"], reading["NO"], reading["UNCLEAR"]) == (3, 205, 5)
    assert reading["actionable_harmful_help_rate"] == pytest.approx(3 / 213)
    assert reading["undetermined_rate"] == pytest.approx(5 / 213)


def test_the_readings_were_frozen_before_any_verdict_exists():
    for name in J.CLASSES:
        assert J.READING_RULES[name]["frozen_before_any_verdict_exists"] is True


# ============================================ output-directory refusal
def test_a_non_empty_output_directory_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    J.assert_output_dir_is_new(out)                       # empty is fine
    (out / "already_here.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.assert_output_dir_is_new(out)
    assert "already holds files" in str(exc.value)


def test_writing_over_an_input_is_refused(tmp_path):
    review = tmp_path / "review"
    review.mkdir()
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.assert_output_dir_is_new(review, (review,))
    assert "never written over" in str(exc.value)


def test_writing_over_an_input_through_a_symlink_is_refused(tmp_path):
    review = tmp_path / "review"
    review.mkdir()
    link = tmp_path / "link"
    link.symlink_to(review, target_is_directory=True)
    with pytest.raises(J.TwoSiteJudgingError):
        J.assert_output_dir_is_new(link, (review,))


# ============================================ nothing reaches the sealed set
def test_no_stage_file_reads_the_sealed_set_commitment():
    for name in preflight_mod.STAGE_FILES:
        path = EXPERIMENTS / name
        assert path.is_file(), name
        assert not preflight_mod._reads_token(
            path, preflight_mod.SEALED_COMMITMENT_TOKEN), name
    assert not preflight_mod._reads_token(
        REPO_ROOT / "src" / "routegate" / "twositejudging.py",
        preflight_mod.SEALED_COMMITMENT_TOKEN)


def test_no_stage_file_imports_the_phase1_packet_builder():
    for name in preflight_mod.STAGE_FILES:
        assert not preflight_mod._imports_token(
            EXPERIMENTS / name, preflight_mod.PHASE1_BUILDER_TOKENS), name


def test_no_configured_path_of_this_stage_is_sealed():
    from routegate.validation import looks_sealed
    paths = _local_paths()
    for key in (J.PATHS_KEY, J.CONSOLIDATED_PATHS_KEY):
        if key in paths:
            assert looks_sealed(paths[key]) is None, key


def test_a_sealed_path_is_refused_by_the_root_resolvers(tmp_path):
    paths = {J.PATHS_KEY: "/data/sealed_set/two_site",
             J.CONSOLIDATED_PATHS_KEY: str(tmp_path)}
    with pytest.raises(ValidationError):
        J.judging_root(paths=paths)


def test_a_missing_path_key_is_refused(tmp_path):
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.judging_root(paths={"something_else": str(tmp_path)})
    assert J.PATHS_KEY in str(exc.value)


def test_the_manifest_states_the_sealed_set_is_inaccessible():
    manifest = json.loads(
        (REPO_ROOT / "manifests"
         / "two_site_actuator_judging_v1.json").read_text(encoding="utf-8"))
    assert manifest["sealed_set"]["accessible"] is False
    assert manifest["contains_prompt_or_response_text"] is False
    T.assert_no_text_fields(manifest, "the committed judging manifest")


# ============================================ the synthetic end-to-end run
CORPUS_IDS = {
    "E": [f"false_reject_test:{i:05d}:{i:012x}" for i in range(89)],
    "B": [f"orbench_hard:{i:012x}" for i in range(178)],
    "H": [f"strongreject:{i:012x}" for i in range(213)],
}


def _sha(text: str) -> str:
    from routegate.validation import sha256_text
    return sha256_text(text)


@pytest.fixture
def pipeline(tmp_path):
    """A complete synthetic packet, two judge runs and their private map."""
    rng = random.Random(20260827)
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True)
    runs = tmp_path / "runs"

    rows_by_rubric = {r: [] for r in J.RUBRICS_USED}
    private = []
    verdicts = {}                       # row_id -> {judge: parsed_result}
    for row_class in J.CLASSES:
        rubric = J.CLASS_RUBRIC[row_class]
        for n, screen_row_id in enumerate(CORPUS_IDS[row_class]):
            row_id = J.new_opaque_ids(J.ROW_PREFIX[rubric], 1)[0]
            prompt = f"prompt number {n} for a {rubric} task"
            entry = {"row_id": row_id, "rubric": rubric,
                     "screen_row_id": screen_row_id, "row_class": row_class,
                     "arm_id": J.CANDIDATE_ARM_ID,
                     "prompt_sha256": _sha(prompt),
                     "hit_max_new_tokens": False,
                     "completion_token_count": 100 + n}
            if rubric in J.PAIRWISE_RUBRICS:
                # Deliberately neutral text: the fixture must not name the
                # origins, or the blindness scan below would be checking the
                # fixture rather than the export.
                baseline = f"reply P about topic {n}"
                candidate = f"reply Q about topic {n}"
                orientation = J.draw_orientation()
                texts = {J.ORIGIN_BASELINE: baseline,
                         J.ORIGIN_CANDIDATE: candidate}
                row = {"row_id": row_id, "prompt_text": prompt,
                       "response_a": texts[orientation[J.POSITION_A]],
                       "response_b": texts[orientation[J.POSITION_B]]}
                entry.update({
                    "baseline_response_sha256": _sha(baseline),
                    "candidate_response_sha256": _sha(candidate),
                    "response_a_origin": orientation[J.POSITION_A],
                    "response_b_origin": orientation[J.POSITION_B],
                    "responses_are_byte_identical": False})
                preference = rng.choice(T.QUALITY_PREFERENCE_VALUES)
                difference = ("NONE" if preference == "EQUIVALENT"
                              else rng.choice(["MINOR", "MATERIAL"]))
                parsed = {"item_id": row_id, "preference": preference,
                          "difference": difference, "pairing_consistent": True,
                          "reason": ""}
                other = dict(parsed)
                if n % 7 == 0:          # a genuine split on one row in seven
                    other["difference"] = ("MINOR"
                                           if difference != "MINOR"
                                           else "MATERIAL")
                    if preference == "EQUIVALENT":
                        other["preference"] = "A_BETTER"
            else:
                response = f"response number {n}"
                row = {"row_id": row_id, "prompt_text": prompt,
                       "response_text": response}
                entry["response_sha256"] = _sha(response)
                entry["candidate_response_sha256"] = _sha(response)
                field = T.AXIS_VERDICT_FIELD[rubric]
                value = rng.choice(T.AXIS_VALUES[rubric])
                parsed = {"item_id": row_id, field: value,
                          "confidence": "high", "reason": ""}
                other = dict(parsed)
                if n % 7 == 0:
                    others = [v for v in T.AXIS_VALUES[rubric] if v != value]
                    other[field] = rng.choice(others)
            entry["rendered_user_sha256"] = _sha(json.dumps(row,
                                                            sort_keys=True))
            entry["rendered_system_sha256"] = _sha(rubric)
            rows_by_rubric[rubric].append(row)
            private.append(entry)
            verdicts[row_id] = {T.JUDGES[0]: parsed, T.JUDGES[1]: other}

    for rubric, rows in rows_by_rubric.items():
        T.write_jsonl(packet / f"{rubric}_judge_input.jsonl",
                      sorted(rows, key=lambda r: r["row_id"]))
    T.write_jsonl(packet / "private" / J.ROW_MAP_FILE,
                  sorted(private, key=lambda r: r["row_id"]))

    for judge in T.JUDGES:
        for rubric in J.RUBRICS_USED:
            out = runs / f"packet__{rubric}__{judge}"
            out.mkdir(parents=True)
            records = []
            for row in rows_by_rubric[rubric]:
                record = {"row_id": row["row_id"], "rubric": rubric,
                          "judge": judge, "status": "OK",
                          "parsed_result": verdicts[row["row_id"]][judge]}
                record["record_sha256"] = T.record_hash(record)
                records.append(record)
            T.write_jsonl(out / "results.jsonl", records)
    return {"tmp": tmp_path, "packet": packet, "runs": runs,
            "private": {e["row_id"]: e for e in private}}


def test_the_join_produces_480_terminal_cells(pipeline):
    summary = join_mod.join(pipeline["packet"], pipeline["runs"],
                            pipeline["tmp"] / "joined")
    assert summary["cells"] == 480
    assert summary["records_by_judge"] == {"sonnet": 480, "opus": 480}
    assert set(summary["by_state"]) <= {"agree", "disagree"}
    assert sum(summary["by_state"].values()) == 480
    assert summary["coverage_observed"]["cells_by_rubric"] == {
        "axis2": 89, "quality": 178, "safety": 213}
    assert summary["coverage_observed"]["cells_by_class"] == {
        "E": 89, "B": 178, "H": 213}
    assert summary["coverage_observed"]["cells_by_arm"] == {
        J.CANDIDATE_ARM_ID: 480}


def test_the_join_decodes_quality_through_the_private_map(pipeline):
    join_mod.join(pipeline["packet"], pipeline["runs"],
                  pipeline["tmp"] / "joined")
    cells = T.read_jsonl(pipeline["tmp"] / "joined" / "private"
                         / J.JOINED_CELLS_FILE)
    sonnet_raw = {}
    for rubric in J.PAIRWISE_RUBRICS:
        for record in T.read_jsonl(
                pipeline["runs"] / f"packet__{rubric}__sonnet"
                / "results.jsonl"):
            sonnet_raw[record["row_id"]] = record["parsed_result"]
    checked = 0
    for cell in cells:
        if cell["rubric"] not in J.PAIRWISE_RUBRICS:
            continue
        entry = pipeline["private"][cell["row_id"]]
        orientation = {J.POSITION_A: entry["response_a_origin"],
                       J.POSITION_B: entry["response_b_origin"]}
        raw = sonnet_raw[cell["row_id"]]
        expected = J.compose_quality_verdict(
            J.decode_preference(raw["preference"], orientation),
            raw["difference"])
        assert cell["verdicts"]["sonnet"] == expected
        checked += 1
    assert checked == 178


def test_a_missing_judge_row_fails_the_join_closed(pipeline):
    path = (pipeline["runs"] / "packet__safety__opus" / "results.jsonl")
    kept = T.read_jsonl(path)[:-1]
    T.write_jsonl(path, kept)
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        join_mod.join(pipeline["packet"], pipeline["runs"],
                      pipeline["tmp"] / "joined")
    assert "479" in str(exc.value) or "incomplete" in str(exc.value)


def test_a_duplicate_judge_record_fails_the_join_closed(pipeline):
    path = (pipeline["runs"] / "packet__axis2__sonnet" / "results.jsonl")
    records = T.read_jsonl(path)
    T.write_jsonl(path, records + [records[0]])
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        join_mod.join(pipeline["packet"], pipeline["runs"],
                      pipeline["tmp"] / "joined")
    assert "two successful records" in str(exc.value)


def test_a_duplicate_private_map_row_fails_the_join_closed(pipeline):
    path = pipeline["packet"] / "private" / J.ROW_MAP_FILE
    entries = T.read_jsonl(path)
    T.write_jsonl(path, entries + [entries[0]])
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        join_mod.join(pipeline["packet"], pipeline["runs"],
                      pipeline["tmp"] / "joined")
    assert "twice" in str(exc.value)


def test_the_join_refuses_a_non_empty_output_directory(pipeline):
    out = pipeline["tmp"] / "joined"
    out.mkdir()
    (out / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        join_mod.join(pipeline["packet"], pipeline["runs"], out)
    assert "already holds files" in str(exc.value)


@needs_quality_rubric
def test_the_review_export_is_disagreement_only_and_blind(pipeline):
    join_mod.join(pipeline["packet"], pipeline["runs"],
                  pipeline["tmp"] / "joined")
    cells = T.read_jsonl(pipeline["tmp"] / "joined" / "private"
                         / J.JOINED_CELLS_FILE)
    disagreements = sum(1 for c in cells if c["state"] == "disagree")
    summary = export_mod.export(
        pipeline["tmp"] / "joined", pipeline["packet"],
        pipeline["tmp"] / "review", J.DECLARED_PROCEDURE, 20260827,
        _rq1_protocol_dir(), _quality_dir(), None)
    assert summary["rows_exported"] == disagreements
    assert summary["cells_in_join"] == 480
    rows = T.read_jsonl(pipeline["tmp"] / "review" / "human_review_input.jsonl")
    assert len(rows) == disagreements
    for row in rows:
        assert set(row) == set(
            export_mod.REVIEWER_ROW_FIELDS[row["rubric_task"]])
        for forbidden in ("verdict", "verdicts", "agree", "row_class",
                          "arm_id", "screen_row_id", "response_a_origin",
                          "response_b_origin", "state"):
            assert forbidden not in row
    # The orientation is nowhere a reviewer can reach. The manifest is
    # deliberately excluded: it NAMES the withheld fields in
    # reviewer_does_not_see, which is a declaration, not a leak.
    review_dir = pipeline["tmp"] / "review"
    for name in ("human_review_input.jsonl", "human_review_TEMPLATE.jsonl",
                 "reviewer_instructions.json"):
        blob = (review_dir / name).read_text(encoding="utf-8")
        for marker in ("response_a_origin", "response_b_origin",
                       "orientation", J.ORIGIN_BASELINE):
            assert marker not in blob, (name, marker)
    # And the private map, which does carry the join, carries no orientation
    # either: only the packet's own private map does.
    private = (review_dir / "private" / J.HUMAN_ROW_MAP_FILE).read_text(
        encoding="utf-8")
    assert "response_a_origin" not in private
    assert "response_b_origin" not in private


@needs_quality_rubric
def test_the_review_export_refuses_an_undeclared_procedure(pipeline):
    join_mod.join(pipeline["packet"], pipeline["runs"],
                  pipeline["tmp"] / "joined")
    with pytest.raises(ValidationError):
        export_mod.export(
            pipeline["tmp"] / "joined", pipeline["packet"],
            pipeline["tmp"] / "review", "triplet_independent_review", 1,
            _rq1_protocol_dir(), _quality_dir(), None)


def test_a_joined_dir_without_joined_cells_is_refused(tmp_path):
    """A missing join output is a diagnosis, not a traceback.

    The export and the adjudication both decode through the joined cells. If
    step 6 never ran, the stage must say so in the same fail-closed voice every
    other stage uses -- not surface a bare FileNotFoundError from inside the
    reader, which reads as a crash rather than as a refusal.
    """
    joined = tmp_path / "joined"
    (joined / "private").mkdir(parents=True)

    with pytest.raises(J.TwoSiteJudgingError) as exc:
        export_mod.export(joined, tmp_path / "packet", tmp_path / "review",
                          J.DECLARED_PROCEDURE, 1, tmp_path / "protocol",
                          tmp_path / "quality", None)
    assert J.JOINED_CELLS_FILE in str(exc.value)
    assert "--joined-dir" in str(exc.value)

    with pytest.raises(J.TwoSiteJudgingError) as exc:
        import_mod.run(joined, tmp_path / "packet", None, None,
                       tmp_path / "adjudicated", J.DECLARED_PROCEDURE)
    assert J.JOINED_CELLS_FILE in str(exc.value)
    assert "--joined-dir" in str(exc.value)


@needs_quality_rubric
def test_the_full_pipeline_reports_480_ebh_rows(pipeline):
    joined = pipeline["tmp"] / "joined"
    review = pipeline["tmp"] / "review"
    join_mod.join(pipeline["packet"], pipeline["runs"], joined)
    export_mod.export(joined, pipeline["packet"], review,
                      J.DECLARED_PROCEDURE, 20260827, _rq1_protocol_dir(),
                      _quality_dir(), None)

    # The reviewer answers every exported row, in the A/B space they were shown.
    rng = random.Random(7)
    answers = []
    for row in T.read_jsonl(review / "human_review_input.jsonl"):
        rubric = row["rubric_task"]
        if rubric in J.PAIRWISE_RUBRICS:
            preference = rng.choice(T.QUALITY_PREFERENCE_VALUES)
            difference = ("NONE" if preference == "EQUIVALENT"
                          else rng.choice(["MINOR", "MATERIAL"]))
            answers.append({"row_id": row["row_id"], "preference": preference,
                            "difference": difference, "note": ""})
        else:
            answers.append({"row_id": row["row_id"],
                            "verdict": rng.choice(T.AXIS_VALUES[rubric]),
                            "confidence": "high", "note": ""})
    completed = review / "human_review_COMPLETED.jsonl"
    T.write_jsonl(completed, answers)

    adjudicated = pipeline["tmp"] / "adjudicated"
    summary = import_mod.run(joined, pipeline["packet"], review, completed,
                             adjudicated, J.DECLARED_PROCEDURE)
    assert summary["cells"] == 480
    assert summary["unresolved"] == 0
    assert summary["complete_adjudication"] is True
    assert summary["by_class"] == {"B": 178, "E": 89, "H": 213}
    assert (summary["by_label_source"].get("DUAL_LLM_AGREEMENT", 0)
            + summary["by_label_source"].get("HUMAN_ADJUDICATION", 0)) == 480

    result = report_mod.report(adjudicated, pipeline["tmp"] / "report",
                               J.DECLARED_PROCEDURE)
    assert result["complete"] is True
    assert result["withheld"] == []
    assert result["rows_by_class"] == {"E": 89, "B": 178, "H": 213}
    e, b, h = (result["readings"][c] for c in ("E", "B", "H"))
    assert e["full_repair"] + sum(
        v for k, v in e["by_verdict"].items() if k != "SA") == 89
    assert e["substantive_repair"] >= e["full_repair"]
    assert (b["material_degradation"] + b["non_material_degradation"]
            == b["any_degradation"])
    assert h["YES"] + h["NO"] + h["UNCLEAR"] == 213

    # Deterministic: a second report over the same file is byte-identical.
    again = report_mod.report(adjudicated, pipeline["tmp"] / "report2",
                              J.DECLARED_PROCEDURE)
    assert again["readings"] == result["readings"]


@needs_quality_rubric
def test_an_unreviewed_disagreement_is_unresolved_and_withholds_its_class(
        pipeline):
    joined = pipeline["tmp"] / "joined"
    join_mod.join(pipeline["packet"], pipeline["runs"], joined)
    adjudicated = pipeline["tmp"] / "adjudicated"
    summary = import_mod.run(joined, pipeline["packet"], None, None,
                             adjudicated, J.DECLARED_PROCEDURE)
    assert summary["unresolved"] > 0
    assert summary["complete_adjudication"] is False
    result = report_mod.report(adjudicated, pipeline["tmp"] / "report",
                              J.DECLARED_PROCEDURE)
    assert result["complete"] is False
    assert set(result["withheld"]) == {"E", "B", "H"}
    for name in J.CLASSES:
        assert "UNRESOLVED" in result["readings"][name]["withheld_reason"]


@needs_quality_rubric
def test_a_reviewer_answer_is_decoded_through_the_packet_map(pipeline):
    joined = pipeline["tmp"] / "joined"
    review = pipeline["tmp"] / "review"
    join_mod.join(pipeline["packet"], pipeline["runs"], joined)
    export_mod.export(joined, pipeline["packet"], review,
                      J.DECLARED_PROCEDURE, 3, _rq1_protocol_dir(),
                      _quality_dir(), None)
    rows = [r for r in T.read_jsonl(review / "human_review_input.jsonl")
            if r["rubric_task"] in J.PAIRWISE_RUBRICS]
    assert rows, "the synthetic run produced no quality disagreement"
    completed = review / "human_review_COMPLETED.jsonl"
    T.write_jsonl(completed, [{"row_id": r["row_id"], "preference": "A_BETTER",
                               "difference": "MATERIAL", "note": ""}
                              for r in rows])
    adjudicated = pipeline["tmp"] / "adjudicated"
    import_mod.run(joined, pipeline["packet"], review, completed, adjudicated,
                   J.DECLARED_PROCEDURE)

    human_map = {r["row_id"]: r for r in T.read_jsonl(
        review / "private" / J.HUMAN_ROW_MAP_FILE)}
    by_cell = {(r["screen_row_id"], r["arm_id"], r["rubric"]): r
               for r in T.read_jsonl(adjudicated / J.ADJUDICATED_CELLS_FILE)}
    for row in rows:
        mapping = human_map[row["row_id"]]
        record = by_cell[(mapping["screen_row_id"], mapping["arm_id"],
                          mapping["rubric"])]
        entry = pipeline["private"][mapping["judged_row_id"]]
        expected = ("AFTER_BETTER"
                    if entry["response_a_origin"] == J.ORIGIN_CANDIDATE
                    else "BEFORE_BETTER")
        assert record["final_quality_outcome"] == expected
        assert record["final_difference_strength"] == "MATERIAL"
        assert record["label_source"] == T.SOURCE_HUMAN


@needs_quality_rubric
def test_a_half_answered_quality_row_is_refused(pipeline):
    joined = pipeline["tmp"] / "joined"
    review = pipeline["tmp"] / "review"
    join_mod.join(pipeline["packet"], pipeline["runs"], joined)
    export_mod.export(joined, pipeline["packet"], review,
                      J.DECLARED_PROCEDURE, 5, _rq1_protocol_dir(),
                      _quality_dir(), None)
    rows = [r for r in T.read_jsonl(review / "human_review_input.jsonl")
            if r["rubric_task"] in J.PAIRWISE_RUBRICS]
    completed = review / "human_review_COMPLETED.jsonl"
    T.write_jsonl(completed, [{"row_id": rows[0]["row_id"],
                               "preference": "A_BETTER", "difference": "",
                               "note": ""}])
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        import_mod.run(joined, pipeline["packet"], review, completed,
                       pipeline["tmp"] / "adjudicated", J.DECLARED_PROCEDURE)
    assert "half a decision" in str(exc.value)


@needs_quality_rubric
def test_an_answer_for_a_row_the_export_did_not_carry_is_refused(pipeline):
    joined = pipeline["tmp"] / "joined"
    review = pipeline["tmp"] / "review"
    join_mod.join(pipeline["packet"], pipeline["runs"], joined)
    export_mod.export(joined, pipeline["packet"], review,
                      J.DECLARED_PROCEDURE, 11, _rq1_protocol_dir(),
                      _quality_dir(), None)
    completed = review / "human_review_COMPLETED.jsonl"
    T.write_jsonl(completed, [{"row_id": "TSJ1HUMANDEADBEEF0001",
                               "verdict": "SA", "confidence": "high"}])
    with pytest.raises(J.TwoSiteJudgingError):
        import_mod.run(joined, pipeline["packet"], review, completed,
                       pipeline["tmp"] / "adjudicated", J.DECLARED_PROCEDURE)


def _rq1_protocol_dir():
    root = _local_paths().get("triplet_validation_dir")
    if not root:
        pytest.skip("the frozen rq1 protocol text is not reachable")
    directory = Path(root) / "frozen" / "rq1_judge_protocol"
    if not directory.is_dir():
        pytest.skip("the frozen rq1 protocol text is not reachable")
    return directory


# ============================================ the real, pinned input
@needs_consolidated
def test_the_pinned_input_verifies_by_digest_and_by_count():
    from routegate.validation import Checklist
    checks = Checklist("test")
    records = J.load_screen_records(_consolidated(), checks)
    baselines = J.load_referenced_baselines(_consolidated(), checks)
    J.load_consolidation_claims(_consolidated(), checks)
    assert checks.all_passed, checks.failures()
    assert len(records) == 480
    assert len(baselines) == 480
    by_class = {}
    for record in records:
        by_class[record["row_class"]] = by_class.get(record["row_class"], 0) + 1
    assert by_class == {"E": 89, "B": 178, "H": 213}
    assert {r["arm_id"] for r in records} == {J.CANDIDATE_ARM_ID}
    assert {r["arm_id"] for r in baselines.values()} == {"baseline"}


@needs_consolidated
def test_one_altered_byte_of_the_pinned_records_is_refused(tmp_path):
    from routegate.validation import Checklist
    source = _consolidated()
    for name in (J.RECORDS_FILENAME, J.BASELINE_FILENAME,
                 J.CONSOLIDATION_MANIFEST, J.CONSOLIDATION_VALIDATION):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    J.load_screen_records(tmp_path, Checklist("copy"))       # the copy passes
    path = tmp_path / J.RECORDS_FILENAME
    path.write_bytes(path.read_bytes() + b"\n")
    # A trailing newline changes the digest and nothing else; refused anyway.
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        J.load_screen_records(tmp_path, Checklist("altered"))
    assert "expected sha256" in str(exc.value)


@needs_consolidated
def test_a_dropped_record_is_refused_by_count(tmp_path):
    from routegate.validation import Checklist
    source = _consolidated()
    records = T.read_jsonl(source / J.RECORDS_FILENAME)[:-1]
    T.write_jsonl(tmp_path / J.RECORDS_FILENAME, records)
    with pytest.raises(J.TwoSiteJudgingError):
        J.load_screen_records(tmp_path, Checklist("short"))


@needs_consolidated
@needs_quality_rubric
def test_the_real_packet_builds_480_blind_rows(tmp_path):
    """The builder, over the real pinned input, at full scale."""
    dataset = Path(_local_paths()["dataset"])
    if not dataset.is_file():
        pytest.skip("the accepted corpus is not reachable")
    out = tmp_path / "packet"
    summary = build_mod.build(_consolidated(), out, _rq1_protocol_dir(),
                              _quality_dir(), dataset, write=True)
    assert summary["rows_total"] == 480
    assert summary["rows_by_rubric"] == {"axis2": 89, "quality": 178,
                                         "safety": 213}
    assert summary["calls_per_judge"] == 480
    assert summary["calls_both_judges"] == 960
    assert summary["reuses_the_phase1_packet_builder"] is False
    assert summary["input"]["records_sha256"] == \
        J.FROZEN_INPUT["records_sha256"]
    assert summary["input"]["baseline_sha256"] == \
        J.FROZEN_INPUT["baseline_sha256"]

    orientation = summary["b_orientation"]
    assert (orientation["candidate_in_position_a"]
            + orientation["candidate_in_position_b"]) == 178
    # Randomised, not fixed: an all-one-side draw over 178 rows would be a bug.
    assert 0 < orientation["candidate_in_position_a"] < 178

    ids = []
    for rubric in J.RUBRICS_USED:
        rows = T.read_jsonl(out / f"{rubric}_judge_input.jsonl")
        assert len(rows) == J.PLANNED_CALLS_BY_RUBRIC[rubric]
        assert [r["row_id"] for r in rows] == sorted(r["row_id"] for r in rows)
        for row in rows:
            assert set(row) == set(T.JUDGE_ROW_FIELDS[rubric])
            J.assert_row_has_no_forbidden_field(row)
            T.assert_opaque(row["row_id"], J.ROW_PREFIX[rubric])
            ids.append(row["row_id"])
    assert len(set(ids)) == 480
    assert not any(i.startswith(J.FOREIGN_ROW_PREFIXES) for i in ids)

    entries = T.read_jsonl(out / "private" / J.ROW_MAP_FILE)
    assert len({J.cell_key(e) for e in entries}) == 480
    T.assert_no_text_fields(
        json.loads((out / "manifest.json").read_text(encoding="utf-8")),
        "the packet manifest")


@needs_consolidated
@needs_quality_rubric
def test_the_builder_refuses_a_non_empty_output_directory(tmp_path):
    dataset = Path(_local_paths()["dataset"])
    if not dataset.is_file():
        pytest.skip("the accepted corpus is not reachable")
    out = tmp_path / "packet"
    out.mkdir()
    (out / "stale.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        build_mod.build(_consolidated(), out, _rq1_protocol_dir(),
                        _quality_dir(), dataset, write=True)
    assert "already holds files" in str(exc.value)


@needs_consolidated
@needs_quality_rubric
def test_the_builder_refuses_to_write_into_its_own_input(tmp_path):
    dataset = Path(_local_paths()["dataset"])
    if not dataset.is_file():
        pytest.skip("the accepted corpus is not reachable")
    with pytest.raises(J.TwoSiteJudgingError) as exc:
        build_mod.build(_consolidated(), Path(_quality_dir()),
                        _rq1_protocol_dir(), _quality_dir(), dataset,
                        write=True)
    assert "never written over" in str(exc.value)


@needs_consolidated
@needs_quality_rubric
def test_two_builds_mint_different_ids_and_different_orientations(tmp_path):
    """Nothing about the packet is derivable from the corpus row id."""
    dataset = Path(_local_paths()["dataset"])
    if not dataset.is_file():
        pytest.skip("the accepted corpus is not reachable")
    maps = []
    for name in ("one", "two"):
        out = tmp_path / name
        build_mod.build(_consolidated(), out, _rq1_protocol_dir(),
                        _quality_dir(), dataset, write=True)
        maps.append({e["screen_row_id"]: e
                     for e in T.read_jsonl(out / "private" / J.ROW_MAP_FILE)})
    left, right = maps
    assert set(left) == set(right)
    # Independently randomised ids: no row keeps its opaque id.
    assert not ({e["row_id"] for e in left.values()}
                & {e["row_id"] for e in right.values()})
    # Independently randomised orientation: over 178 rows the two draws differ.
    flipped = sum(1 for k in left
                  if left[k].get("response_a_origin")
                  and left[k]["response_a_origin"]
                  != right[k]["response_a_origin"])
    assert flipped > 0
