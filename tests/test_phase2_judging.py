"""Blind phase-2 judging: every gate, on synthetic fixtures.

Nothing here calls an API, loads a model, allocates a GPU, reads the sealed set
or opens the historical workspace. Every fixture is invented in ``tmp_path``.

Organised by the failure each group is there to prevent:

    1. vocabulary   one rubric, eight generated arms, two inherited ones
    2. counts       144 / 36 / 180 are derived from the protocol, not typed
    3. input        a wrong, short, duplicated, smoke or off-selection record
                    stops the build, and an inherited arm can never be judged
    4. blindness    arm, coefficient, grouping and SELECTION STATUS cannot leak
    5. cells        a phase-2 decision cannot be filed under a phase-1 arm
    6. inherited    the 36 references reconcile or nothing proceeds
    7. counting     UNCLEAR and a gap are undetermined, never a quiet NO
    8. decision     the frozen rule is called, not reimplemented
    9. join         a short run, an inherited arm or a one-judge cell fails
   10. phase 1      every phase-1 behaviour this work touched still holds
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import dualsteer as D                                 # noqa: E402
from routegate import focusedjudging as F                            # noqa: E402
from routegate import phase2 as P2                                   # noqa: E402
from routegate import phase2judging as P                             # noqa: E402
from routegate import triplets as T                                  # noqa: E402
from routegate.dualsteer import CLASS_E, CLASS_H, protocol_for       # noqa: E402
from routegate.validation import Checklist, sha256_file, sha256_text  # noqa: E402

STAGE_DIR = REPO_ROOT / "experiments" / "focused_direction_judging"
PHASE2_FILES = ("build_phase2_judge_packet.py", "join_phase2_verdicts.py",
                "export_phase2_human_review.py",
                "import_phase2_human_review.py", "report_phase2.py")


def _stage_module(name: str):
    """Load a stage BY PATH, under a name that cannot collide."""
    key = f"_phase2_stage_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, STAGE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def _code_without_prose(path: Path) -> str:
    """A module's executable source, docstrings removed.

    The prose in these files deliberately names the things they must not do. A
    substring scan over the raw text would fire on the documentation rather
    than on the behaviour, and deleting the documentation would make it pass.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                body.pop(0)
    return ast.unparse(tree)


# ============================================================== 1. vocabulary
def test_phase2_judges_one_rubric_and_it_is_the_frozen_safety_one():
    assert P.PHASE2_AXIS == T.SAFETY
    assert P.AXES_USED == (T.SAFETY,)
    assert P.PHASE2_ROW_CLASS == CLASS_H


def test_axis1_and_axis2_are_not_part_of_phase2():
    for axis in (T.AXIS1, T.AXIS2):
        assert axis not in P.AXES_USED
        assert axis not in P.PHASE2_ROW_PREFIX


def test_the_arm_table_is_derived_from_the_frozen_protocol():
    protocol = protocol_for(P.PROTOCOL_NAME)
    assert P.PHASE2_ARM_IDS == tuple(
        a.arm_id for a in protocol.arms_by_class[CLASS_H])
    assert len(P.PHASE2_ARM_IDS) == 8
    assert P.PHASE2_CLASS_ARMS[CLASS_E] == ()
    assert P.PHASE2_REFERENCE_ARM_IDS == ("baseline", "repair_015")
    assert not set(P.PHASE2_ARM_IDS) & set(P.PHASE2_REFERENCE_ARM_IDS)


def test_the_two_ladders_are_paired_at_every_strength():
    assert P.PHASE2_ALPHAS == (0.05, 0.10, 0.15, 0.20)
    assert sorted(P.SAFETY_ARM_BY_ALPHA) == sorted(P.COMBINED_ARM_BY_ALPHA)
    for alpha in P.PHASE2_ALPHAS:
        safety = D.ARM_BY_ID[P.SAFETY_ARM_BY_ALPHA[alpha]]
        combined = D.ARM_BY_ID[P.COMBINED_ARM_BY_ALPHA[alpha]]
        assert safety.alpha_repair == 0.0
        assert combined.alpha_repair == 0.15      # pinned; only alpha_S moves
        assert safety.alpha_safety == combined.alpha_safety == alpha


def test_phase2_ids_cannot_collide_with_phase1_or_either_retrospective_packet():
    prefix = P.PHASE2_ROW_PREFIX[P.PHASE2_AXIS]
    assert prefix not in P.PHASE2_FOREIGN_ROW_PREFIXES
    assert P.PHASE2_HUMAN_ROW_PREFIX not in P.PHASE2_FOREIGN_ROW_PREFIXES
    for foreign in F.ROW_PREFIX.values():
        assert foreign in P.PHASE2_FOREIGN_ROW_PREFIXES
        assert not prefix.startswith(foreign)
    assert F.HUMAN_ROW_PREFIX in P.PHASE2_FOREIGN_ROW_PREFIXES


# ================================================================== 2. counts
def test_the_counts_are_144_36_and_180_and_are_derived():
    assert P.FROZEN_PHASE2_INPUT["rows"] == D.PHASE2_NEW_GENERATIONS == 144
    assert D.PHASE2_ROWS * len(P.PHASE2_ARM_IDS) == 144
    assert D.PHASE2_ROWS * len(P.PHASE2_REFERENCE_ARM_IDS) == 36
    assert P.FROZEN_COVERAGE["cells"] == 144
    assert P.FROZEN_COVERAGE["inherited_reference_cells"] == 36
    assert P.FROZEN_COVERAGE["analysis_table_records"] == 180
    assert P.PLANNED_CALLS_PER_JUDGE == 144
    assert P.PLANNED_CALLS_BOTH_JUDGES == 288
    assert sum(P.FROZEN_COVERAGE["cells_by_arm"].values()) == 144


def test_the_pinned_input_is_the_pinned_phase2_records():
    assert P.FROZEN_PHASE2_INPUT["records_sha256"] == (
        "1d113bb0cd35ce2073cc963c4bf7a26c537f0d12202099f15009af0c4ecad919")
    assert P.FROZEN_PHASE2_INPUT["protocol"] == D.PHASE2_PROTOCOL
    assert P.FROZEN_PHASE2_INPUT["smoke"] is False
    assert P.FROZEN_PHASE2_INPUT["tests_safety_direction"] is True


def test_the_pins_agree_with_the_configuration_phase2_inherits():
    checks = Checklist("t")
    P.assert_inherited_configuration(checks)
    assert checks.all_passed


def test_this_packet_is_not_deduplicated_and_says_so():
    assert P.DEDUPLICATION["deduplicated"] is False


# =================================================================== 3. input
def _record(row_id, arm_id, response=None, **over):
    arm = D.ARM_BY_ID[arm_id]
    text = response if response is not None else f"synthetic {row_id} {arm_id}"
    record = {
        "row_id": row_id, "row_class": CLASS_H, "arm_id": arm_id,
        "protocol": P.PROTOCOL_NAME,
        "prompt_sha256": sha256_text(f"prompt::{row_id}"),
        "response_text": text, "response_sha256": sha256_text(text),
        "alpha_R": arm.alpha_repair, "alpha_S": arm.alpha_safety,
        "smoke": False, "hit_max_new_tokens": False,
        "completion_token_count": 12, "error_message": "",
    }
    for field in P.RUN_LOCKED_RECORD_FIELDS:
        if field != "protocol":
            record[field] = P.FROZEN_PHASE2_INPUT[field]
    record.update(over)
    return record


ROW_A = D.PHASE2_SELECTED_ROWS[0]
ROW_B = D.PHASE2_SELECTED_ROWS[1]


def _tiny(monkeypatch, rows=(ROW_A,)):
    """One or two bound rows instead of eighteen. Every arm, still."""
    monkeypatch.setattr(P, "PHASE2_SELECTED_ROWS", tuple(sorted(rows)))
    monkeypatch.setattr(P, "PHASE2_ROWS", len(rows))
    monkeypatch.setattr(P, "PHASE2_REFERENCE_RECORDS",
                        len(rows) * len(P.PHASE2_REFERENCE_ARM_IDS))
    monkeypatch.setattr(P, "PHASE2_ANALYSIS_TABLE_RECORDS",
                        len(rows) * (len(P.PHASE2_ARM_IDS)
                                     + len(P.PHASE2_REFERENCE_ARM_IDS)))
    monkeypatch.setitem(P.FROZEN_COVERAGE, "cells_by_arm",
                        {arm: len(rows) for arm in P.PHASE2_ARM_IDS})


def _write_records(tmp_path, records, monkeypatch):
    directory = tmp_path / "consolidated"
    directory.mkdir(parents=True, exist_ok=True)
    T.write_jsonl(directory / P.RECORDS_FILENAME, records)
    path = directory / P.RECORDS_FILENAME
    monkeypatch.setitem(P.FROZEN_PHASE2_INPUT, "records_sha256",
                        sha256_file(path))
    monkeypatch.setitem(P.FROZEN_PHASE2_INPUT, "records_bytes",
                        path.stat().st_size)
    monkeypatch.setitem(P.FROZEN_PHASE2_INPUT, "rows", len(records))
    return directory


def _complete(rows=(ROW_A,)):
    return [_record(row, arm) for row in rows for arm in P.PHASE2_ARM_IDS]


def test_a_complete_pinned_phase2_package_loads(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    directory = _write_records(tmp_path, _complete(), monkeypatch)
    checks = Checklist("t")
    assert len(P.load_phase2_screen_records(directory, checks)) == 8
    assert checks.all_passed


def test_a_digest_mismatch_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    directory = _write_records(tmp_path, _complete(), monkeypatch)
    monkeypatch.setitem(P.FROZEN_PHASE2_INPUT, "records_sha256", "0" * 64)
    with pytest.raises(P.Phase2JudgingError, match="sha256"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_an_inherited_reference_arm_in_the_package_is_refused(tmp_path,
                                                              monkeypatch):
    """The one thing phase 2 must never do: regenerate what it inherits."""
    _tiny(monkeypatch)
    records = _complete() + [_record(ROW_A, "repair_015")]
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="INHERITED"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_row_outside_the_bound_selection_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    records = _complete() + [_record("strongreject:not_selected", "safety_005")]
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="frozen phase-1 rule"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_an_e_record_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    records = _complete()
    records[0]["row_class"] = CLASS_E
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="steers"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_phase1_repair_ladder_arm_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    records = _complete() + [_record(ROW_A, "repair_005")]
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="not an arm"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_duplicated_row_and_arm_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    records = _complete()
    records.append(dict(records[0]))
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="already appeared"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_missing_arm_is_refused_rather_than_judged_short(tmp_path,
                                                           monkeypatch):
    _tiny(monkeypatch)
    records = [r for r in _complete() if r["arm_id"] != "safety_015"]
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(Exception):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_smoke_record_is_never_judged_as_a_protocol_record(tmp_path,
                                                             monkeypatch):
    _tiny(monkeypatch)
    records = _complete()
    records[0]["smoke"] = True
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="smoke"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_record_carrying_an_error_is_not_judged(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    records = _complete()
    records[0]["error_message"] = "cuda oom"
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="error_message"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_coefficient_that_disagrees_with_its_arm_is_refused(tmp_path,
                                                              monkeypatch):
    _tiny(monkeypatch)
    records = _complete()
    records[0]["alpha_S"] = 0.99
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="alpha_S"):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_response_that_does_not_match_its_own_digest_is_refused(tmp_path,
                                                                  monkeypatch):
    _tiny(monkeypatch)
    records = _complete()
    records[0]["response_sha256"] = "0" * 64
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match="response_sha256"):
        P.load_phase2_screen_records(directory, Checklist("t"))


@pytest.mark.parametrize("field", ["direction_bank_sha256",
                                   "generation_config_sha256",
                                   "tokenizer_config_sha256",
                                   "chat_template_sha256", "revision"])
def test_a_record_from_another_run_is_refused(field, tmp_path, monkeypatch):
    """Phase 2 pins the generation configuration too: the inherited references
    are only comparable because nothing about the generation moved."""
    _tiny(monkeypatch)
    records = _complete()
    records[0][field] = "f" * 64
    directory = _write_records(tmp_path, records, monkeypatch)
    with pytest.raises(P.Phase2JudgingError, match=field):
        P.load_phase2_screen_records(directory, Checklist("t"))


def test_a_consolidation_claiming_incompleteness_is_refused(tmp_path):
    directory = tmp_path / "consolidated"
    directory.mkdir()
    (directory / P.CONSOLIDATION_VALIDATION).write_text(json.dumps({
        "protocol": P.PROTOCOL_NAME, "protocol_expected_generations": 144,
        "complete_protocol": False, "reportable": True, "smoke": False,
        "records_sha256": P.FROZEN_PHASE2_INPUT["records_sha256"],
        "checklist": {"all_passed": True},
        "protocol_tests_safety_direction": True,
        "decides_no_behavioural_outcome": True}), encoding="utf-8")
    (directory / P.CONSOLIDATION_MANIFEST).write_text(json.dumps({
        "dataset_sha256": P.FROZEN_PHASE2_INPUT["dataset_sha256"],
        "direction_bank_sha256":
            P.FROZEN_PHASE2_INPUT["direction_bank_sha256"],
        "generation_config_sha256":
            P.FROZEN_PHASE2_INPUT["generation_config_sha256"]}),
        encoding="utf-8")
    with pytest.raises(Exception):
        P.load_phase2_consolidation_claims(directory, Checklist("t"))


# =============================================================== 4. blindness
def test_a_phase2_judge_row_carries_exactly_three_fields():
    assert T.JUDGE_ROW_FIELDS[P.PHASE2_AXIS] == ("row_id", "prompt_text",
                                                 "response_text")


@pytest.mark.parametrize("leak", ["arm_id", "alpha_S", "screen_row_id",
                                  "prompt_sha256", "selected", "selection",
                                  "is_selected", "regressed", "source",
                                  "inherited", "recovered"])
def test_a_forbidden_field_on_a_phase2_row_is_refused(leak):
    """Phase 2 has one extra secret: that every row here was SELECTED."""
    row = {"row_id": "FDV2SAFETY0123", "prompt_text": "p",
           "response_text": "r", leak: "x"}
    with pytest.raises(P.Phase2JudgingError):
        P.assert_phase2_row_has_no_forbidden_field(row)
    with pytest.raises(T.TripletError):
        T.assert_row_is_blind(row, P.PHASE2_AXIS)


def test_the_selection_status_is_in_the_forbidden_set():
    for name in ("selected", "selection", "selection_status", "is_selected",
                 "regressed", "regression_row"):
        assert name in P.PHASE2_FORBIDDEN_ROW_FIELDS


def test_a_projected_phase2_row_is_the_whitelist_and_nothing_else():
    item = {"prompt_text": "p", "response_text": "r",
            "prompt_sha256": sha256_text("p"),
            "response_sha256": sha256_text("r"),
            "arm_id": "safety_005", "screen_row_id": ROW_A}
    row = F.project_judge_row(item, P.PHASE2_AXIS, "FDV2SAFETY000000000001")
    assert set(row) == {"row_id", "prompt_text", "response_text"}
    P.assert_phase2_row_has_no_forbidden_field(row)


def test_the_builder_projects_and_never_mutates_a_screen_record():
    code = _code_without_prose(STAGE_DIR / "build_phase2_judge_packet.py")
    assert "project_judge_row" in code
    for forbidden in ("del row[", "dict(record)", "**record", "copy.deepcopy"):
        assert forbidden not in code, forbidden


# =================================================================== 5. cells
def _cell(row=ROW_A, arm="safety_005", axis=None, **over):
    cell = {"screen_row_id": row, "arm_id": arm,
            "axis": axis or P.PHASE2_AXIS, "row_class": CLASS_H}
    cell.update(over)
    return cell


def test_a_phase2_cell_key_refuses_an_inherited_arm():
    """A phase-2 decision can never be filed under baseline or repair_015."""
    for arm in P.PHASE2_REFERENCE_ARM_IDS:
        with pytest.raises(F.FocusedJudgingError):
            P.phase2_cell_key(_cell(arm=arm))


def test_the_analysis_arm_table_accepts_both_and_only_both():
    for arm in P.PHASE2_ARM_IDS + P.PHASE2_REFERENCE_ARM_IDS:
        assert P.analysis_cell_key(_cell(arm=arm))[1] == arm
    with pytest.raises(F.FocusedJudgingError):
        P.analysis_cell_key(_cell(arm="repair_005"))


def test_a_phase2_cell_key_refuses_the_wrong_rubric():
    with pytest.raises(F.FocusedJudgingError):
        P.phase2_cell_key(_cell(axis=T.AXIS2))


def test_one_cell_carries_one_record():
    with pytest.raises(F.FocusedJudgingError, match="twice"):
        P.index_phase2_cells([_cell(), _cell()])


# =============================================================== 6. inherited
def _inherited(row, arm, verdict=None, **over):
    if verdict is None:
        verdict = "NO" if arm == "baseline" else "YES"
    record = {
        "screen_row_id": row, "arm_id": arm, "axis": T.SAFETY,
        "row_class": CLASS_H, "verdict": verdict,
        "label_source": T.SOURCE_DUAL,
        "adjudication_procedure": F.PROCEDURE_CORPUS,
        "prompt_sha256": sha256_text(f"prompt::{row}"),
        "response_sha256": sha256_text(f"response::{row}::{arm}"),
        "hit_max_new_tokens": False, "completion_token_count": 12,
    }
    record.update(over)
    return record


def _phase1_file(tmp_path, records):
    path = tmp_path / F.ADJUDICATED_CELLS_FILE
    T.write_jsonl(path, records)
    return path


def _prompts(rows):
    return {row: sha256_text(f"prompt::{row}") for row in rows}


def test_the_inherited_references_reconcile(tmp_path, monkeypatch):
    _tiny(monkeypatch, (ROW_A, ROW_B))
    rows = (ROW_A, ROW_B)
    path = _phase1_file(tmp_path, [_inherited(r, a) for r in rows
                                   for a in P.PHASE2_REFERENCE_ARM_IDS])
    checks = Checklist("t")
    out = P.load_inherited_references(path, _prompts(rows),
                                      F.PROCEDURE_CORPUS, checks)
    assert len(out) == 4
    assert checks.all_passed


def test_a_short_inherited_set_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch, (ROW_A, ROW_B))
    rows = (ROW_A, ROW_B)
    records = [_inherited(r, a) for r in rows
               for a in P.PHASE2_REFERENCE_ARM_IDS][:-1]
    path = _phase1_file(tmp_path, records)
    with pytest.raises(Exception):
        P.load_inherited_references(path, _prompts(rows), F.PROCEDURE_CORPUS,
                                    Checklist("t"))


def test_an_inherited_cell_from_another_procedure_is_refused(tmp_path,
                                                             monkeypatch):
    _tiny(monkeypatch)
    records = [_inherited(ROW_A, a) for a in P.PHASE2_REFERENCE_ARM_IDS]
    records[0]["adjudication_procedure"] = F.PROCEDURE_INDEPENDENT
    path = _phase1_file(tmp_path, records)
    with pytest.raises(P.Phase2JudgingError, match="procedure"):
        P.load_inherited_references(path, _prompts([ROW_A]),
                                    F.PROCEDURE_CORPUS, Checklist("t"))


def test_an_unresolved_inherited_cell_is_refused(tmp_path, monkeypatch):
    _tiny(monkeypatch)
    records = [_inherited(ROW_A, a) for a in P.PHASE2_REFERENCE_ARM_IDS]
    records[0]["label_source"] = T.SOURCE_UNRESOLVED
    records[0]["verdict"] = None
    path = _phase1_file(tmp_path, records)
    with pytest.raises(P.Phase2JudgingError, match="UNRESOLVED"):
        P.load_inherited_references(path, _prompts([ROW_A]),
                                    F.PROCEDURE_CORPUS, Checklist("t"))


@pytest.mark.parametrize("arm,verdict", [("baseline", "YES"),
                                         ("repair_015", "NO"),
                                         ("repair_015", "UNCLEAR")])
def test_an_inherited_verdict_that_is_not_the_selection_rule_is_refused(
        arm, verdict, tmp_path, monkeypatch):
    """The selection is NO at baseline and YES at repair_015. If the phase-1
    adjudication says otherwise, the two have parted company."""
    _tiny(monkeypatch)
    records = [_inherited(ROW_A, a) for a in P.PHASE2_REFERENCE_ARM_IDS]
    for record in records:
        if record["arm_id"] == arm:
            record["verdict"] = verdict
    path = _phase1_file(tmp_path, records)
    with pytest.raises(P.Phase2JudgingError, match="selection rule"):
        P.load_inherited_references(path, _prompts([ROW_A]),
                                    F.PROCEDURE_CORPUS, Checklist("t"))


def test_an_inherited_cell_judged_over_a_different_prompt_is_refused(
        tmp_path, monkeypatch):
    _tiny(monkeypatch)
    records = [_inherited(ROW_A, a) for a in P.PHASE2_REFERENCE_ARM_IDS]
    records[0]["prompt_sha256"] = "0" * 64
    path = _phase1_file(tmp_path, records)
    with pytest.raises(P.Phase2JudgingError, match="prompt"):
        P.load_inherited_references(path, _prompts([ROW_A]),
                                    F.PROCEDURE_CORPUS, Checklist("t"))


def test_reading_the_inherited_references_never_writes_to_phase1(tmp_path,
                                                                 monkeypatch):
    _tiny(monkeypatch)
    records = [_inherited(ROW_A, a) for a in P.PHASE2_REFERENCE_ARM_IDS]
    path = _phase1_file(tmp_path, records)
    before = sha256_file(path)
    P.load_inherited_references(path, _prompts([ROW_A]), F.PROCEDURE_CORPUS,
                                Checklist("t"))
    assert sha256_file(path) == before


#: Every call that can create or overwrite a file.
_WRITING_CALLS = ("write_jsonl", "write_text", "write_bytes", "open", "mkdir",
                  "touch", "unlink", "rename", "replace", "rmtree", "copy",
                  "copyfile", "dump")


def test_no_phase2_file_writes_to_the_phase1_adjudication():
    """Immutability, structurally.

    Not a substring scan: every call that can create or overwrite a file is
    found in the AST and its arguments are re-rendered. If the phase-1
    adjudication -- the only thing this stage inherits and the only thing it
    could destroy -- ever reached one of them, the expression naming it would
    be in those arguments.
    """
    offenders = []
    for name in PHASE2_FILES:
        tree = ast.parse((STAGE_DIR / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (func.attr if isinstance(func, ast.Attribute)
                      else getattr(func, "id", ""))
            if called not in _WRITING_CALLS:
                continue
            rendered = " ".join(
                [ast.unparse(func)] + [ast.unparse(a) for a in node.args])
            if "phase1" in rendered or "inherited" in rendered:
                offenders.append(f"{name}: {rendered[:90]}")
    assert not offenders, offenders


def test_the_inherited_reference_loader_opens_nothing_for_writing():
    tree = ast.parse((REPO_ROOT / "src" / "routegate"
                      / "phase2judging.py").read_text(encoding="utf-8"))
    loader = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "load_inherited_references")
    for node in ast.walk(loader):
        if isinstance(node, ast.Call):
            func = node.func
            called = (func.attr if isinstance(func, ast.Attribute)
                      else getattr(func, "id", ""))
            assert called not in _WRITING_CALLS, called


# =============================================== 7. the analysis table + counts
def _table(rows, verdicts):
    """A 10-arm table for each row, from ``{(row, arm): verdict}``."""
    out = []
    for row in rows:
        for arm in P.PHASE2_REFERENCE_ARM_IDS + P.PHASE2_ARM_IDS:
            out.append({"screen_row_id": row, "arm_id": arm,
                        "axis": T.SAFETY,
                        "verdict": verdicts.get((row, arm)),
                        "label_source": T.SOURCE_DUAL,
                        "source": P.PHASE2_SOURCE})
    return out


def _all(rows, baseline="NO", treated="YES", safety="NO", combined="NO"):
    verdicts = {}
    for row in rows:
        verdicts[(row, "baseline")] = baseline
        verdicts[(row, "repair_015")] = treated
        for arm in P.SAFETY_ARM_BY_ALPHA.values():
            verdicts[(row, arm)] = safety
        for arm in P.COMBINED_ARM_BY_ALPHA.values():
            verdicts[(row, arm)] = combined
    return verdicts


def test_the_analysis_table_is_144_plus_36_and_nothing_is_rewritten(
        tmp_path, monkeypatch):
    _tiny(monkeypatch, (ROW_A, ROW_B))
    rows = (ROW_A, ROW_B)
    phase2 = {}
    for row in rows:
        for arm in P.PHASE2_ARM_IDS:
            cell = _inherited(row, arm, verdict="NO")
            cell["arm_id"] = arm
            phase2[(row, arm, T.SAFETY)] = cell
    inherited = {(row, arm, T.SAFETY): _inherited(row, arm)
                 for row in rows for arm in P.PHASE2_REFERENCE_ARM_IDS}
    checks = Checklist("t")
    table = P.analysis_table(phase2, inherited, checks)
    assert len(table) == 2 * 10
    assert sum(1 for r in table if r["is_inherited"]) == 4
    assert checks.all_passed
    for row in table:
        if row["is_inherited"]:
            source = inherited[(row["screen_row_id"], row["arm_id"],
                                row["axis"])]
            assert row["source_record_sha256"] == T.record_hash(source)
            assert row["source"] == P.INHERITED_SOURCE


def test_a_cell_claimed_by_both_sources_is_refused(monkeypatch):
    _tiny(monkeypatch)
    shared = {(ROW_A, "safety_005", T.SAFETY): _inherited(
        ROW_A, "safety_005", verdict="NO")}
    with pytest.raises(P.Phase2JudgingError, match="both"):
        P.analysis_table(shared, shared, Checklist("t"))


def test_recovery_is_yes_at_repair015_becoming_no_at_the_combined_arm():
    rows = [ROW_A, ROW_B]
    counts = P.counts_by_strength(_all(rows), rows)
    for alpha in P.PHASE2_ALPHAS:
        assert counts[alpha]["combined_recovered"] == 2
        assert counts[alpha]["combined_undetermined"] == 0
        assert counts[alpha]["safety_only_harmful"] == 0


def test_a_row_still_harmful_at_the_combined_arm_is_not_a_recovery():
    rows = [ROW_A]
    counts = P.counts_by_strength(_all(rows, combined="YES"), rows)
    assert counts[0.05]["combined_recovered"] == 0
    assert counts[0.05]["combined_undetermined"] == 0


@pytest.mark.parametrize("value", ["UNCLEAR", None])
def test_unclear_or_missing_is_undetermined_and_never_a_quiet_recovery(value):
    rows = [ROW_A]
    verdicts = _all(rows)
    verdicts[(ROW_A, P.COMBINED_ARM_BY_ALPHA[0.10])] = value
    counts = P.counts_by_strength(verdicts, rows)
    assert counts[0.10]["combined_recovered"] == 0
    assert counts[0.10]["combined_undetermined"] == 1
    assert counts[0.05]["combined_recovered"] == 1      # other strengths intact


def test_safety_only_harm_is_no_at_baseline_becoming_yes_alone():
    rows = [ROW_A, ROW_B]
    verdicts = _all(rows, safety="YES")
    counts = P.counts_by_strength(verdicts, rows)
    for alpha in P.PHASE2_ALPHAS:
        assert counts[alpha]["safety_only_harmful"] == 2


@pytest.mark.parametrize("value", ["UNCLEAR", None])
def test_an_undecided_safety_only_arm_is_undetermined_not_harmless(value):
    rows = [ROW_A]
    verdicts = _all(rows)
    verdicts[(ROW_A, P.SAFETY_ARM_BY_ALPHA[0.20])] = value
    counts = P.counts_by_strength(verdicts, rows)
    assert counts[0.20]["safety_only_harmful"] == 0
    assert counts[0.20]["safety_only_undetermined"] == 1


def test_the_per_row_table_covers_every_arm_inherited_and_generated():
    rows = [ROW_A]
    table = P.per_row_transitions(_all(rows), rows)
    assert set(table[ROW_A]) == set(P.PHASE2_REFERENCE_ARM_IDS
                                    + P.PHASE2_ARM_IDS)


# ================================================================ 8. decision
def test_the_decision_is_the_frozen_rule_and_is_not_reimplemented():
    """The threshold, the veto, the tie-break and the branches live in
    routegate.phase2. Nothing in this stage may hold a second copy."""
    sources = [REPO_ROOT / "src" / "routegate" / "phase2judging.py"]
    sources += [STAGE_DIR / name for name in PHASE2_FILES]
    for path in sources:
        code = _code_without_prose(path)
        assert "PHASE2_RECOVERY_THRESHOLD" not in code, path.name
        assert "PHASE2_SAFETY_ONLY_HARM_LIMIT" not in code, path.name
        assert "NO_STRENGTH_QUALIFIES" not in code, path.name
    report = _code_without_prose(STAGE_DIR / "report_phase2.py")
    assert "evaluate_phase2_decision" in report


def test_the_counts_feed_the_frozen_rule_unchanged():
    rows = list(D.PHASE2_SELECTED_ROWS)
    counts = P.counts_by_strength(_all(rows), rows)
    decision = P2.evaluate_phase2_decision(counts)
    assert decision["outcome"] == "STRENGTH_SELECTED"
    # every strength recovers all 18 and causes none, so the tie breaks to the
    # smallest alpha_S -- the rule's own statement, not this file's
    assert decision["selected"]["alpha_S"] == min(P.PHASE2_ALPHAS)
    assert decision["selected"]["combined_arm_recovered"] == len(rows)


def test_one_undecided_row_makes_that_strength_undetermined_overall():
    rows = list(D.PHASE2_SELECTED_ROWS)
    verdicts = _all(rows)
    verdicts[(rows[0], P.COMBINED_ARM_BY_ALPHA[0.15])] = "UNCLEAR"
    decision = P2.evaluate_phase2_decision(P.counts_by_strength(verdicts, rows))
    assert decision["outcome"] == "UNDETERMINED"
    assert decision["selected"] is None


def test_a_direction_that_harms_alone_cannot_qualify():
    rows = list(D.PHASE2_SELECTED_ROWS)
    decision = P2.evaluate_phase2_decision(
        P.counts_by_strength(_all(rows, safety="YES"), rows))
    assert decision["outcome"] == "NO_STRENGTH_QUALIFIES"
    assert decision["selected"] is None


# ==================================================================== 9. join
def test_the_join_demands_every_verdict_from_every_judge():
    cells = {(f"R{i}", "safety_005", T.SAFETY): {"state": "agree"}
             for i in range(P.FROZEN_COVERAGE["cells"])}
    runs = {"sonnet": {f"r{i}": {} for i in range(144)},
            "opus": {f"r{i}": {} for i in range(143)}}
    with pytest.raises(P.Phase2JudgingError, match="opus_record_count"):
        P.check_phase2_coverage(cells, runs, Checklist("t"))


def test_a_cell_with_only_one_judge_is_never_terminal():
    assert P.FROZEN_COVERAGE["terminal_states"] == ("agree", "disagree")
    for state in ("incomplete", "missing"):
        assert state not in P.FROZEN_COVERAGE["terminal_states"]


def test_phase2_reads_no_technical_exclusion_manifest():
    """Phase 1 has a pinned exclusion manifest. Phase 2 has none, and a
    one-judge cell here is a failure rather than an invented exclusion."""
    for name in PHASE2_FILES:
        code = _code_without_prose(STAGE_DIR / name)
        assert "load_technical_exclusions" not in code, name
        assert "verify_technical_exclusions" not in code, name


def test_an_inherited_arm_reaching_the_join_is_refused():
    rows = list(D.PHASE2_SELECTED_ROWS)
    cells = {(row, arm, T.SAFETY): {"state": "agree"}
             for row in rows for arm in P.PHASE2_ARM_IDS}
    cells[(rows[0], "repair_015", T.SAFETY)] = {"state": "agree"}
    runs = {"sonnet": {f"r{i}": {} for i in range(144)},
            "opus": {f"r{i}": {} for i in range(144)}}
    with pytest.raises(P.Phase2JudgingError):
        P.check_phase2_coverage(cells, runs, Checklist("t"))


def test_a_complete_join_passes_coverage():
    rows = list(D.PHASE2_SELECTED_ROWS)
    cells = {(row, arm, T.SAFETY): {"state": "agree"}
             for row in rows for arm in P.PHASE2_ARM_IDS}
    runs = {"sonnet": {f"r{i}": {} for i in range(144)},
            "opus": {f"r{i}": {} for i in range(144)}}
    checks = Checklist("t")
    observed = P.check_phase2_coverage(cells, runs, checks)
    assert observed["cells"] == 144 and observed["rows"] == 18
    assert checks.all_passed


def test_no_phase2_file_calls_an_api_or_reads_a_credential():
    for name in PHASE2_FILES:
        code = _code_without_prose(STAGE_DIR / name)
        for forbidden in ("requests", "urllib", "http.client", "socket",
                          "MODEL_API_KEY", "ROUTEGATE_JUDGE_API_KEY",
                          "ROUTEGATE_JUDGE_ENDPOINT", "os.environ"):
            assert forbidden not in code, f"{name}: {forbidden}"


def test_no_phase2_file_reads_the_legacy_stress_manifest():
    for name in PHASE2_FILES:
        assert not F.reads_legacy_stress_manifest(STAGE_DIR / name), name
    assert not F.reads_legacy_stress_manifest(
        REPO_ROOT / "src" / "routegate" / "phase2judging.py")


# ================================================================= 10. phase 1
def test_the_phase1_arm_table_is_still_the_default_and_still_refuses_phase2():
    """The shared cell key gained an optional arm table. Phase 1's behaviour
    is the default, and a phase-2 arm still cannot enter a phase-1 cell."""
    for arm in P.PHASE2_ARM_IDS:
        with pytest.raises(F.FocusedJudgingError):
            F.cell_key({"screen_row_id": ROW_A, "arm_id": arm,
                        "axis": T.SAFETY})
    assert F.cell_key({"screen_row_id": ROW_A, "arm_id": "repair_015",
                       "axis": T.SAFETY}) == (ROW_A, "repair_015", T.SAFETY)


def test_the_phase1_coverage_and_pins_are_untouched():
    assert F.FROZEN_COVERAGE["cells"] == 871
    assert F.FROZEN_INPUT["records_sha256"] == (
        "f0ee90a804268984367a4558ed82cf9d50f4e144cabaf71a88803cd760cc3918")
    assert F.CLASS_ARMS[CLASS_H] == ("baseline", "repair_015")


def test_the_declared_procedure_is_unchanged_and_shared():
    assert F.declared_adjudication_procedure() == F.PROCEDURE_CORPUS
    assert P.REUSED_UNCHANGED["adjudication_procedure"] == F.PROCEDURE_CORPUS


def test_the_adjudication_rule_is_phase1s_applied_to_phase2_arms():
    """Same function, same truth table, different arm table."""
    cell = {"screen_row_id": ROW_A, "arm_id": "safety_005", "axis": T.SAFETY,
            "row_class": CLASS_H, "verdicts": {"sonnet": "NO", "opus": "NO"}}
    outcome = F.adjudicate_cell(cell, None, F.PROCEDURE_CORPUS,
                                arms=P.PHASE2_CLASS_ARMS)
    assert outcome["verdict"] == "NO"
    assert outcome["label_source"] == T.SOURCE_DUAL
    split = dict(cell, verdicts={"sonnet": "NO", "opus": "YES"})
    assert F.adjudicate_cell(split, None, F.PROCEDURE_CORPUS,
                             arms=P.PHASE2_CLASS_ARMS)["label_source"] == (
        T.SOURCE_UNRESOLVED)
    missing = dict(cell, verdicts={"sonnet": "NO", "opus": None})
    assert F.adjudicate_cell(missing, {"verdict": "NO"}, F.PROCEDURE_CORPUS,
                             arms=P.PHASE2_CLASS_ARMS)["verdict"] is None


# ================================================ 11. the five stages, chained
#
# A rehearsal of the whole offline chain on invented text: join -> export ->
# import -> report. It proves the five files actually wire together and that
# the 180-record table and the decision come out of the far end. No real
# prompt, response, protocol text or corpus row is read, and no API is called.
def _fake_packet(tmp_path):
    """A packet-shaped directory: 144 blind rows and their private map."""
    packet = tmp_path / "packet"
    (packet / "private").mkdir(parents=True)
    cells = [(row, arm) for row in D.PHASE2_SELECTED_ROWS
             for arm in P.PHASE2_ARM_IDS]
    ids = T.new_opaque_ids(P.PHASE2_ROW_PREFIX[P.PHASE2_AXIS], len(cells))
    public, private = [], []
    for index, (row_id, (corpus_row, arm)) in enumerate(zip(ids, cells)):
        # deliberately NOT derived from the corpus row id: the export scans
        # reviewer text for synthetic identifiers and would refuse one
        row_index = index // len(P.PHASE2_ARM_IDS)
        prompt, response = f"prompt {row_index}", f"response {index}"
        public.append({"row_id": row_id, "prompt_text": prompt,
                       "response_text": response})
        private.append({
            "row_id": row_id, "axis": T.SAFETY, "screen_row_id": corpus_row,
            "row_class": CLASS_H, "arm_id": arm,
            "prompt_sha256": sha256_text(prompt),
            "response_sha256": sha256_text(response),
            "hit_max_new_tokens": False, "completion_token_count": 11,
            "rendered_user_sha256": sha256_text(f"user {index}"),
            "rendered_system_sha256": sha256_text("system"),
        })
    T.write_jsonl(packet / f"{T.SAFETY}_judge_input.jsonl", public)
    T.write_jsonl(packet / "private" / F.ROW_MAP_FILE, private)
    return packet, private


def _fake_runs(tmp_path, private, split_row_id):
    """Two judges, every row, and exactly one genuine disagreement."""
    runs = tmp_path / "runs"
    for judge in T.JUDGES:
        directory = runs / f"packet__{T.SAFETY}__{judge}"
        directory.mkdir(parents=True)
        records = []
        for entry in private:
            verdict = "NO"
            if entry["row_id"] == split_row_id and judge == T.JUDGES[1]:
                verdict = "YES"
            record = {"row_id": entry["row_id"], "axis": T.SAFETY,
                      "judge": judge, "status": "OK",
                      "parsed_result": {"actionable_harmful_help": verdict}}
            record["record_sha256"] = T.record_hash(record)
            records.append(record)
        T.write_jsonl(directory / "results.jsonl", records)
    return runs


def _fake_phase1(tmp_path, private):
    """The 36 inherited references, over the same prompts."""
    prompt_by_row = {}
    for entry in private:
        prompt_by_row.setdefault(entry["screen_row_id"], entry["prompt_sha256"])
    records = []
    for row, digest in sorted(prompt_by_row.items()):
        for arm in P.PHASE2_REFERENCE_ARM_IDS:
            records.append(_inherited(row, arm, prompt_sha256=digest))
    path = tmp_path / "phase1"
    path.mkdir()
    T.write_jsonl(path / F.ADJUDICATED_CELLS_FILE, records)
    return path


class _StubProtocol:
    instructions = "apply the frozen safety rubric"
    allowed_values = T.SAFETY_VALUES
    verdict_field = "actionable_harmful_help"
    instructions_sha256 = "a" * 64
    schema_sha256 = "b" * 64


def test_join_export_import_report_chain_end_to_end(tmp_path, monkeypatch):
    join_mod = _stage_module("join_phase2_verdicts")
    export_mod = _stage_module("export_phase2_human_review")
    import_mod = _stage_module("import_phase2_human_review")
    report_mod = _stage_module("report_phase2")

    packet, private = _fake_packet(tmp_path)
    split = private[0]["row_id"]
    runs = _fake_runs(tmp_path, private, split)
    phase1 = _fake_phase1(tmp_path, private)

    # ---- join
    joined = tmp_path / "joined"
    summary = join_mod.join(packet, runs, joined, write=True)
    assert summary["cells"] == 144
    assert summary["by_state"] == {"agree": 143, "disagree": 1}

    # ---- export: only the disagreement, and the reviewer sees four fields
    monkeypatch.setattr(export_mod, "load_protocol",
                        lambda axis, directory=None, manifest=None:
                        _StubProtocol())
    monkeypatch.setattr(export_mod, "load_prompt_texts",
                        lambda dataset, checks: {
                            row: {} for row in D.PHASE2_SELECTED_ROWS})
    review = tmp_path / "human_review"
    exported = export_mod.export(joined, packet, review, F.PROCEDURE_CORPUS,
                                 seed=1, protocol_dir=tmp_path / "protocol",
                                 dataset=tmp_path / "corpus.csv", write=True)
    assert exported["rows_exported"] == 1
    reviewer_rows = T.read_jsonl(review / "human_review_input.jsonl")
    assert set(reviewer_rows[0]) == set(export_mod.REVIEWER_FIELDS)

    # ---- the reviewer answers
    completed = review / "human_review_COMPLETED.jsonl"
    T.write_jsonl(completed, [{"row_id": reviewer_rows[0]["row_id"],
                               "verdict": "NO", "confidence": "high",
                               "note": ""}])

    # ---- import
    adjudicated = tmp_path / "adjudicated"
    result = import_mod.run(joined, review, completed, adjudicated,
                            F.PROCEDURE_CORPUS, write=True)
    assert result["cells"] == 144
    assert result["unresolved"] == 0
    assert result["by_label_source"][T.SOURCE_HUMAN] == 1
    assert result["by_label_source"][T.SOURCE_DUAL] == 143

    # ---- report
    readings = report_mod.report(adjudicated, phase1, tmp_path / "readings",
                                 write=True)
    assert readings["analysis_table_records"] == 180
    assert readings["cells_inherited_from_phase1"] == 36
    assert readings["unresolved_cells"] == 0
    table = T.read_jsonl(tmp_path / "readings" / "analysis_table.jsonl")
    assert len(table) == 180
    assert sum(1 for r in table if r["is_inherited"]) == 36
    # every combined arm returned every row, and no safety-only arm harmed one
    decision = readings["d_decision"]
    assert decision["outcome"] == "STRENGTH_SELECTED"
    assert decision["selected"]["alpha_S"] == 0.05
    assert decision["selected"]["combined_arm_recovered"] == 18
    assert readings["checklist"]["all_passed"]
    T.assert_no_text_fields(readings, "the phase-2 readings")


def test_the_chain_refuses_a_join_missing_one_judges_verdict(tmp_path):
    """One judge short is a failure, not an exclusion: phase 2 has no
    exclusion manifest and the join may not invent one."""
    join_mod = _stage_module("join_phase2_verdicts")
    packet, private = _fake_packet(tmp_path)
    runs = _fake_runs(tmp_path, private, private[0]["row_id"])
    opus = runs / f"packet__{T.SAFETY}__{T.JUDGES[1]}" / "results.jsonl"
    kept = T.read_jsonl(opus)[:-1]
    T.write_jsonl(opus, kept)
    with pytest.raises(P.Phase2JudgingError):
        join_mod.join(packet, runs, tmp_path / "joined", write=False)


def _adjudicated_cells(tmp_path, mutate=None):
    """144 phase-2 adjudicated cells, as the importer would have written them."""
    records = []
    for index, row in enumerate(D.PHASE2_SELECTED_ROWS):
        for arm in P.PHASE2_ARM_IDS:
            record = _inherited(row, arm, verdict="NO",
                                prompt_sha256=sha256_text(f"prompt {index}"))
            record["arm_id"] = arm
            records.append(record)
    if mutate is not None:
        mutate(records)
    path = tmp_path / "adjudicated"
    path.mkdir()
    T.write_jsonl(path / F.ADJUDICATED_CELLS_FILE, records)
    return path


def test_the_report_refuses_two_prompts_for_one_row(tmp_path):
    """A row is one question. If two of its arms were judged over different
    prompts, the transition table would compare different questions."""
    report_mod = _stage_module("report_phase2")

    def flip(records):
        records[1]["prompt_sha256"] = "0" * 64

    adjudicated = _adjudicated_cells(tmp_path, flip)
    with pytest.raises(P.Phase2JudgingError, match="One row is one prompt"):
        report_mod.report(adjudicated, tmp_path / "absent",
                          tmp_path / "readings", write=False)


def test_the_report_refuses_an_absent_phase1_adjudication(tmp_path):
    """The 36 references are read from phase 1 or the reading is not made."""
    report_mod = _stage_module("report_phase2")
    adjudicated = _adjudicated_cells(tmp_path)
    with pytest.raises(P.Phase2JudgingError, match="never regenerated"):
        report_mod.report(adjudicated, tmp_path / "absent",
                          tmp_path / "readings", write=False)
