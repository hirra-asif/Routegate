"""Carrying the dynamic arm's eleven missing judge verdicts.

Eleven of 960 planned verdicts never arrived, over ten distinct cells. This
covers the manifest that carries them and the engine behaviour it depends on,
in five parts:

1. **the arithmetic** -- 11 missing judge verdicts across 10 cells, E 2 / B 5 /
   H 3, and every count internally consistent with 480 cells and two judges;
2. **the H cell missing from both judges** -- carried twice, once per failed
   judge, and terminal rather than non-terminal, because ``join_evidence``
   tests carried membership before it tests whether any verdict is present;
3. **preservation** -- all 480 cells survive the join, and a carried cell is
   kept rather than dropped;
4. **human review** -- a technically unresolved cell is never exported, and a
   human answer for one is refused if it somehow arrives;
5. **no invented verdict** -- a carried cell's verdicts stay ``None``, its
   adjudicated verdict stays ``None``, and its label source stays
   ``UNRESOLVED``.

Parts 2 to 5 are claims about the shared engine, so they are proved on the
synthetic demo fixture, end to end through an injected transport. Nothing here
opens a socket, and nothing writes into the real judging root: the tests that
read it are marked, skipped where it is absent, and read-only.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import routegate.judgingprotocol as P                             # noqa: E402
import routegate.judgingstages as S                               # noqa: E402
import routegate.triplets as T                                    # noqa: E402
from routegate import dynamicjudging as DJ                        # noqa: E402
from routegate.validation import ValidationError, sha256_file     # noqa: E402

FIXTURE = REPO_ROOT / "experiments" / "judging" / "make_demo_fixture.py"
BASE_PATH = REPO_ROOT / "manifests" / "dynamic_arms_v1_judging_v1.json"
CARRIED_PATH = (REPO_ROOT / "manifests"
                / "dynamic_arms_v1_judging_v1_technical_unresolved.json")

CARRIED = json.loads(CARRIED_PATH.read_text(encoding="utf-8"))
BASE = P.load_manifest(BASE_PATH)

#: What the completed run actually produced, and what the manifest must say.
MISSING_VERDICTS = 11
UNRESOLVED_CELLS = 10
CELLS = 480
BOTH_MISSING_ROW = "DYN1SAFETYE83235E328C2"
BY_CLASS = {"E": 2, "B": 5, "H": 3}
BY_JUDGE = {"sonnet": 9, "opus": 2}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                               # type: ignore
    return module


demo = _load(FIXTURE, "_dynamic_technical_demo_fixture")


def _paths():
    path = REPO_ROOT / "configs" / "paths.local.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _judging_root():
    root = _paths().get(DJ.PATHS_KEY_JUDGING)
    if not root:
        return None
    directory = Path(root) / "judging"
    return directory if (directory / "packet" / "private"
                         / P.ROW_MAP_FILE).is_file() else None


needs_runs = pytest.mark.skipif(
    _judging_root() is None,
    reason="the dynamic judging root is not reachable on this machine")


# =================================================== 1. the arithmetic
def test_eleven_missing_judge_verdicts_over_ten_cells():
    entries = CARRIED["unresolved"]
    assert len(entries) == MISSING_VERDICTS
    assert len({(e["row_id"], e["failed_judge"]) for e in entries}) == \
        MISSING_VERDICTS
    assert len({e["row_id"] for e in entries}) == UNRESOLVED_CELLS


def test_the_pinned_counts_are_exactly_the_run_that_happened():
    assert CARRIED["counts"] == {
        "planned_verdicts": 960, "valid_verdicts": 949,
        "missing_verdicts": MISSING_VERDICTS, "cells": CELLS,
        "dual_judged_cells": 470,
        "technically_unresolved_cells": UNRESOLVED_CELLS,
        "original_protocol_complete": False}


def test_the_counts_are_internally_consistent():
    counts = CARRIED["counts"]
    assert counts["planned_verdicts"] == counts["cells"] * len(BASE.judge_order)
    assert (counts["valid_verdicts"] + counts["missing_verdicts"]
            == counts["planned_verdicts"])
    assert (counts["dual_judged_cells"] + counts["technically_unresolved_cells"]
            == counts["cells"])
    assert counts["cells"] == int(BASE.coverage["cells"])
    assert counts["planned_verdicts"] == int(BASE.coverage["planned_calls"])


def test_the_unresolved_cells_split_two_E_five_B_three_H():
    by_class = {}
    for row_id in {e["row_id"] for e in CARRIED["unresolved"]}:
        entry = next(e for e in CARRIED["unresolved"] if e["row_id"] == row_id)
        by_class[entry["row_class"]] = by_class.get(entry["row_class"], 0) + 1
    assert by_class == BY_CLASS
    assert CARRIED["unresolved_cells_by_class"] == BY_CLASS
    assert sum(BY_CLASS.values()) == UNRESOLVED_CELLS


def test_missing_verdicts_split_nine_sonnet_two_opus():
    counted = {}
    for entry in CARRIED["unresolved"]:
        counted[entry["failed_judge"]] = counted.get(
            entry["failed_judge"], 0) + 1
    assert counted == BY_JUDGE == CARRIED["missing_by_judge"]
    assert sum(BY_JUDGE.values()) == MISSING_VERDICTS


def test_the_by_rubric_verdict_count_exceeds_the_cell_count_by_the_double():
    """4 safety verdicts over 3 H cells, because one cell lost both judges."""
    assert CARRIED["missing_by_rubric"] == {"axis2": 2, "pairwise_quality": 5,
                                            "safety": 4}
    assert sum(CARRIED["missing_by_rubric"].values()) == MISSING_VERDICTS
    assert (CARRIED["missing_by_rubric"]["safety"]
            - CARRIED["unresolved_cells_by_class"]["H"]
            == CARRIED["both_judges_missing"]["cells"])


def test_every_entry_names_a_rubric_and_class_this_packet_declares():
    for entry in CARRIED["unresolved"]:
        assert entry["rubric"] in BASE.rubrics
        assert entry["row_class"] in BASE.classes
        assert BASE.rubric_for_class(entry["row_class"]).name == entry["rubric"]
        assert entry["arm_id"] in BASE.candidate_arms
        assert entry["row_id"].startswith(
            BASE.rubrics[entry["rubric"]].row_prefix)
        assert entry["failed_judge"] in BASE.judge_order


def test_only_the_established_failure_types_are_used():
    permitted = {"EmptyJudgeResponse", "MismatchedItemIdEcho"}
    assert set(CARRIED["failure_types_permitted"]) == permitted
    assert {e["failure_type"] for e in CARRIED["unresolved"]} <= permitted
    assert set(CARRIED["failure_types_used"]) <= permitted


def test_every_carried_judge_has_at_least_two_terminal_attempts():
    for entry in CARRIED["unresolved"]:
        assert entry["attempts"] >= 2
        assert len(entry["evidence"]) == entry["attempts"]
        assert entry["every_attempt_is_terminal"] is True
        for attempt in entry["evidence"]:
            assert attempt["raw_response_path"].startswith("raw/")
            assert len(attempt["raw_response_sha256"]) == 64
            assert len(attempt["request_sha256"]) == 64
            assert attempt["http_status"] == 200


def test_the_attempt_numbers_are_distinct_within_a_carried_judge():
    for entry in CARRIED["unresolved"]:
        numbers = [a["attempt_number"] for a in entry["evidence"]]
        assert sorted(numbers) == sorted(set(numbers))
        assert min(numbers) == 1


def test_the_manifest_is_metadata_only():
    assert CARRIED["contains_prompt_or_response_text"] is False
    assert CARRIED["contains_judge_prose_or_error_strings"] is False
    assert CARRIED["contains_credential"] is False
    assert CARRIED["is_outcome_based_exclusion"] is False
    assert CARRIED["invents_no_verdict"] is True
    assert CARRIED["enters_human_review"] is False
    assert CARRIED["changes_how_any_cell_is_decided"] is False
    P.assert_no_secrets(CARRIED, "technical unresolved manifest")
    T.assert_no_text_fields(CARRIED, "technical unresolved manifest")


def test_no_verdict_vocabulary_appears_anywhere_in_the_manifest():
    """Outcome blindness, checked against the file rather than asserted."""
    vocabulary = set()
    for spec in BASE.rubrics.values():
        vocabulary |= set(spec.values) | set(spec.judge_values or ())
    blob = CARRIED_PATH.read_text(encoding="utf-8")
    assert not [v for v in vocabulary if f'"{v}"' in blob]
    assert not any("verdict" in key for entry in CARRIED["unresolved"]
                   for key in entry
                   if key not in ("other_judge_has_verdict",
                                  "other_judge_record_sha256"))


def test_no_recorded_error_string_reaches_the_manifest():
    blob = CARRIED_PATH.read_text(encoding="utf-8")
    for token in ("Traceback", "JudgingProtocolError", "EmptyJudgeResponse:",
                  "the reply echoes item_id", "billing_note", "Bearer ",
                  "sk-"):
        assert token not in blob
    for entry in CARRIED["unresolved"]:
        assert "error" not in entry and "error_type" not in entry


def test_the_failure_type_is_derived_from_the_body_not_the_error_string():
    for entry in CARRIED["unresolved"]:
        assert "preserved raw body" in entry["failure_type_derived_from"]
        assert "recorded error string is not read" in \
            entry["failure_type_derived_from"]


def test_the_frozen_base_manifest_is_not_edited():
    """This file is passed with --carried; the base declares no manifest."""
    assert BASE.technical_manifest is None
    assert BASE.raw["adjudication"]["technical_unresolved_manifest"] is None
    assert CARRIED["base_manifest_sha256"] == sha256_file(BASE_PATH)
    assert CARRIED["stage"] == BASE.experiment_id
    assert CARRIED["packet_id"] == BASE.packet_id
    assert CARRIED["adjudication_procedure"] == BASE.procedure


# ============================== 2. the H cell missing from both judges
def test_the_both_missing_cell_is_carried_once_per_failed_judge():
    entries = [e for e in CARRIED["unresolved"]
               if e["row_id"] == BOTH_MISSING_ROW]
    assert len(entries) == len(BASE.judge_order) == 2
    assert {e["failed_judge"] for e in entries} == set(BASE.judge_order)
    assert all(e["both_judges_missing"] is True for e in entries)
    assert all(e["other_judge_has_verdict"] is False for e in entries)
    assert all(e["other_judge_record_sha256"] is None for e in entries)
    assert all(e["row_class"] == "H" for e in entries)


def test_the_both_missing_cell_is_named_at_the_top_level():
    block = CARRIED["both_judges_missing"]
    assert block["row_ids"] == [BOTH_MISSING_ROW]
    assert block["cells"] == 1
    assert block["entries_per_such_cell"] == len(BASE.judge_order)


def test_every_other_carried_cell_still_has_one_judge():
    for entry in CARRIED["unresolved"]:
        if entry["row_id"] == BOTH_MISSING_ROW:
            continue
        assert entry["both_judges_missing"] is False
        assert entry["other_judge_has_verdict"] is True
        assert len(entry["other_judge_record_sha256"]) == 64


def test_only_one_cell_lost_both_judges():
    counted = {}
    for entry in CARRIED["unresolved"]:
        counted[entry["row_id"]] = counted.get(entry["row_id"], 0) + 1
    doubled = sorted(r for r, n in counted.items() if n == len(BASE.judge_order))
    assert doubled == [BOTH_MISSING_ROW]
    assert all(n in (1, len(BASE.judge_order)) for n in counted.values())


# ============================================ the synthetic engine fixture
@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch):
    """A placeholder key and endpoint. Every test injects its own transport."""
    monkeypatch.setenv(T.ENV_API_KEY_HISTORICAL, "fixture-not-a-real-key")
    monkeypatch.setenv(T.ENV_ENDPOINT, "http://fixture.invalid")


def _reply(row_id, model, values, blocks=1):
    payload = {"item_id": row_id, **values}
    body = {"model": model,
            "content": ([{"type": "text", "text": json.dumps(payload)}]
                        if blocks else [])}
    return {"http_status": 200, "response_headers": {},
            "raw_body": json.dumps(body), "payload": body,
            "elapsed_seconds": 0.01}


def _transport(answers, empty=()):
    """A fake gateway. Rows in ``empty`` come back with no answer text."""
    def call(endpoint, api_key, model, system, user, manifest):
        row_id = json.loads(user.splitlines()[0])["item_id"]
        if row_id in empty:
            return _reply(row_id, model, {}, blocks=0)
        return _reply(row_id, model, answers[row_id])
    return call


@pytest.fixture
def engine(tmp_path):
    """The demo experiment, built and packed. 6 pairwise + 5 single cells."""
    manifest_path = demo.build(tmp_path)
    manifest = P.load_manifest(manifest_path)
    dirs = S.resolved_dirs(manifest, tmp_path)
    S.build_packet(manifest, dirs, tmp_path, write=True)
    answers = {}
    for name, rubric in manifest.rubrics.items():
        for row in S.load_judge_rows(dirs["packet"], name):
            answers[row["row_id"]] = (
                {"preference": "EQUIVALENT", "difference": "NONE"}
                if rubric.pairwise else {"harmful": "NO"})
    single = sorted(r["row_id"] for r in
                    S.load_judge_rows(dirs["packet"], "demosafety"))
    return {"root": tmp_path, "manifest": manifest, "dirs": dirs,
            "answers": answers, "single": single,
            "cells": int(manifest.coverage["cells"])}


def _drive(engine, empty_by_judge):
    """Run both judges, failing the named rows for the named judge."""
    for judge in engine["manifest"].judge_order:
        S.run_judge(engine["manifest"], engine["dirs"], engine["root"], judge,
                    write=True,
                    transport=_transport(engine["answers"],
                                         empty_by_judge.get(judge, ())))


def _carried(rows_by_judge):
    return [{"row_id": row_id, "failed_judge": judge,
             "failure_type": "EmptyJudgeResponse"}
            for judge, rows in sorted(rows_by_judge.items())
            for row_id in rows]


@pytest.fixture
def both_missing(engine):
    """One single-response cell lost by both judges, one lost by one."""
    lost_both, lost_one = engine["single"][0], engine["single"][1]
    _drive(engine, {"sonnet": (lost_both, lost_one), "opus": (lost_both,)})
    carried = _carried({"sonnet": (lost_both, lost_one),
                        "opus": (lost_both,)})
    return {**engine, "lost_both": lost_both, "lost_one": lost_one,
            "carried": carried}


# ================================== 2. (engine) a cell with neither judge
def test_the_engine_makes_a_both_missing_carried_cell_terminal(both_missing):
    summary = S.join_evidence(both_missing["manifest"], both_missing["dirs"],
                              write=True, carried=both_missing["carried"])
    assert summary["by_state"].get(P.STATE_UNRESOLVED_TECHNICAL) == 2
    assert P.STATE_MISSING not in summary["by_state"]
    assert P.STATE_INCOMPLETE not in summary["by_state"]
    assert sorted(summary["technically_unresolved"]) == sorted(
        {both_missing["lost_both"], both_missing["lost_one"]})


def test_the_both_missing_cell_holds_no_verdict_from_either_judge(both_missing):
    S.join_evidence(both_missing["manifest"], both_missing["dirs"],
                    write=True, carried=both_missing["carried"])
    cells = {c["row_id"]: c for c in T.read_jsonl(
        Path(both_missing["dirs"]["joined"]) / "private"
        / P.JOINED_CELLS_FILE)}
    cell = cells[both_missing["lost_both"]]
    assert cell["state"] == P.STATE_UNRESOLVED_TECHNICAL
    assert cell["technically_unresolved"] is True
    assert all(cell["verdicts"][j] is None
               for j in both_missing["manifest"].judge_order)
    other = cells[both_missing["lost_one"]]
    assert other["state"] == P.STATE_UNRESOLVED_TECHNICAL
    assert sum(1 for j in both_missing["manifest"].judge_order
               if other["verdicts"][j] is not None) == 1


def test_an_uncarried_both_missing_cell_stops_the_join(both_missing):
    partial = [c for c in both_missing["carried"]
               if c["row_id"] != both_missing["lost_both"]]
    with pytest.raises(ValidationError, match="not carried"):
        S.join_evidence(both_missing["manifest"], both_missing["dirs"],
                        write=False, carried=partial)


def test_carrying_one_judge_of_a_both_missing_cell_is_not_enough(both_missing):
    """Half a carry is refused: the other judge's failure is still open."""
    half = [c for c in both_missing["carried"]
            if not (c["row_id"] == both_missing["lost_both"]
                    and c["failed_judge"] == "opus")]
    with pytest.raises(ValidationError, match="opus_no_uncarried_failure"):
        S.join_evidence(both_missing["manifest"], both_missing["dirs"],
                        write=False, carried=half)


# ============================================ 3. every cell is preserved
def test_the_join_keeps_every_planned_cell(both_missing):
    summary = S.join_evidence(both_missing["manifest"], both_missing["dirs"],
                              write=True, carried=both_missing["carried"])
    assert summary["cells"] == both_missing["cells"]
    assert sum(summary["by_state"].values()) == both_missing["cells"]
    rows = list(T.read_jsonl(Path(both_missing["dirs"]["joined"]) / "private"
                             / P.JOINED_CELLS_FILE))
    assert len(rows) == both_missing["cells"]


def test_a_carried_cell_is_kept_not_dropped(both_missing):
    S.join_evidence(both_missing["manifest"], both_missing["dirs"],
                    write=True, carried=both_missing["carried"])
    rows = {c["row_id"] for c in T.read_jsonl(
        Path(both_missing["dirs"]["joined"]) / "private"
        / P.JOINED_CELLS_FILE)}
    assert both_missing["lost_both"] in rows
    assert both_missing["lost_one"] in rows


def test_the_report_retains_every_cell_and_narrows_the_denominator(
        both_missing):
    S.join_evidence(both_missing["manifest"], both_missing["dirs"],
                    write=True, carried=both_missing["carried"])
    S.export_review(both_missing["manifest"], both_missing["dirs"], write=True)
    S.import_review(both_missing["manifest"], both_missing["dirs"], None,
                    write=True)
    summary = S.report(both_missing["manifest"], both_missing["dirs"],
                       write=True)
    assert summary["cells"] == summary["planned_cells"] == both_missing["cells"]
    assert summary["unresolved_cells"] == 2
    assert summary["decided_cells"] == both_missing["cells"] - 2
    assert summary["original_protocol_complete"] is False
    reading = summary["readings"]["T"]
    assert reading["rows"] == reading["planned_rows"]
    assert reading["decided_denominator"] == reading["rows"] - 2
    assert reading["complete"] is False
    for value, rate in reading["rates"].items():
        assert rate == pytest.approx(
            reading["by_verdict"][value] / reading["decided_denominator"])


def test_the_dynamic_manifest_would_narrow_the_denominator_the_same_way():
    counts = CARRIED["counts"]
    assert counts["dual_judged_cells"] == counts["cells"] - UNRESOLVED_CELLS
    for row_class, planned in (("E", 89), ("B", 178), ("H", 213)):
        assert int(BASE.classes[row_class]["rows"]) == planned
        assert planned - BY_CLASS[row_class] > 0


# ============================ 4. technical failures never reach a human
def test_a_technically_unresolved_cell_is_never_exported(both_missing):
    S.join_evidence(both_missing["manifest"], both_missing["dirs"],
                    write=True, carried=both_missing["carried"])
    summary = S.export_review(both_missing["manifest"], both_missing["dirs"],
                              write=True)
    assert summary["technically_unresolved_excluded"] == 2
    exported = {r["row_id"] for r in T.read_jsonl(
        Path(both_missing["dirs"]["review"]) / P.REVIEW_INPUT_FILE)}
    mapped = {m["judged_row_id"] for m in T.read_jsonl(
        Path(both_missing["dirs"]["review"]) / "private" / P.REVIEW_MAP_FILE)}
    assert both_missing["lost_both"] not in mapped
    assert both_missing["lost_one"] not in mapped
    assert len(exported) == len(mapped) == summary["rows"]


def test_cells_needing_review_skips_unresolved_technical_by_rule():
    cells = [{"screen_row_id": "a", "arm_id": "x", "rubric": "r",
              "state": P.STATE_UNRESOLVED_TECHNICAL},
             {"screen_row_id": "b", "arm_id": "x", "rubric": "r",
              "state": P.STATE_DISAGREE}]
    for procedure in P.PROCEDURES:
        wanted = S.cells_needing_review(cells, procedure)
        assert ("a", "x", "r") not in wanted
    assert ("b", "x", "r") in S.cells_needing_review(
        cells, "corpus_disagreement_only")


def test_a_human_answer_for_a_carried_cell_is_refused(both_missing):
    """Even if one is smuggled in, it cannot become the missing verdict."""
    dirs = both_missing["dirs"]
    S.join_evidence(both_missing["manifest"], dirs, write=True,
                    carried=both_missing["carried"])
    S.export_review(both_missing["manifest"], dirs, write=True)
    review = Path(dirs["review"])
    map_path = review / "private" / P.REVIEW_MAP_FILE
    mapping = list(T.read_jsonl(map_path))
    joined = {c["row_id"]: c for c in T.read_jsonl(
        Path(dirs["joined"]) / "private" / P.JOINED_CELLS_FILE)}
    carried_cell = joined[both_missing["lost_both"]]
    smuggled = {"row_id": "DEMOHUMANSMUGGLED01",
                "judged_row_id": carried_cell["row_id"],
                "screen_row_id": carried_cell["screen_row_id"],
                "arm_id": carried_cell["arm_id"],
                "rubric": carried_cell["rubric"]}
    T.write_jsonl(map_path, mapping + [smuggled])
    T.write_jsonl(review / P.REVIEW_COMPLETED_FILE,
                  [{"row_id": "DEMOHUMANSMUGGLED01", "harmful": "NO"}])
    with pytest.raises(ValidationError,
                       match="no_review_of_a_cell_this_procedure|"
                             "no_human_verdict_replaces_a_missing"):
        S.import_review(both_missing["manifest"], dirs, None, write=False)


def test_the_manifest_says_no_carried_cell_enters_human_review():
    assert CARRIED["enters_human_review"] is False
    assert all(e["enters_human_review"] is False
               for e in CARRIED["unresolved"])
    assert "does not supply a verdict a judge never gave" in \
        CARRIED["why_not_human_review"]


# ================= 5. no invented verdict; unresolved states are explicit
def test_a_carried_cell_is_adjudicated_to_nothing(both_missing):
    dirs = both_missing["dirs"]
    S.join_evidence(both_missing["manifest"], dirs, write=True,
                    carried=both_missing["carried"])
    S.export_review(both_missing["manifest"], dirs, write=True)
    summary = S.import_review(both_missing["manifest"], dirs, None, write=True)
    cells = {c["judged_row_id"]: c for c in T.read_jsonl(
        Path(dirs["adjudicated"]) / P.ADJUDICATED_CELLS_FILE)}
    assert len(cells) == both_missing["cells"]
    for row_id in (both_missing["lost_both"], both_missing["lost_one"]):
        cell = cells[row_id]
        assert cell["verdict"] is None
        assert cell["label_source"] == T.SOURCE_UNRESOLVED
        assert cell["technically_unresolved"] is True
        assert cell["failure_type"] == "EmptyJudgeResponse"
    assert summary["by_label_source"][T.SOURCE_UNRESOLVED] == 2


def test_the_adjudication_refuses_a_verdict_on_a_carried_cell(both_missing):
    """The gate that would fire if anything invented one."""
    dirs = both_missing["dirs"]
    S.join_evidence(both_missing["manifest"], dirs, write=True,
                    carried=both_missing["carried"])
    S.export_review(both_missing["manifest"], dirs, write=True)
    summary = S.import_review(both_missing["manifest"], dirs, None, write=True)
    check = next(c for c in summary["checks"]["checks"]
                 if c["check"] == "no_technical_cell_carries_a_verdict")
    assert check["status"] == "PASS"
    assert "no carried cell holds a verdict" in check["detail"]


def test_a_missing_verdict_never_becomes_a_label(both_missing):
    dirs = both_missing["dirs"]
    S.join_evidence(both_missing["manifest"], dirs, write=True,
                    carried=both_missing["carried"])
    S.export_review(both_missing["manifest"], dirs, write=True)
    S.import_review(both_missing["manifest"], dirs, None, write=True)
    summary = S.report(both_missing["manifest"], dirs, write=True)
    reading = summary["readings"]["T"]
    assert sum(reading["by_verdict"].values()) == reading["decided_denominator"]
    assert reading["unresolved"] == 2
    assert len(reading["unresolved_cells"]) == 2
    for cell in reading["unresolved_cells"]:
        assert set(cell) == {"row_id", "rubric", "row_class",
                             "missing_judge", "failure_type"}
        assert cell["failure_type"] == "EmptyJudgeResponse"


def test_the_failure_policy_admits_exactly_one_value():
    """No manifest field can turn a failure into a verdict."""
    assert P.FAILURE_POLICY == ("unresolved_technical",)
    assert P.DISAGREEMENT_POLICY == ("human_review_blinded",)
    for kind in ("missing", "malformed", "transport_failure"):
        assert BASE.raw["adjudication"][kind] == P.FAILURE_POLICY[0]


def test_the_manifest_states_that_it_invents_nothing():
    assert CARRIED["invents_no_verdict"] is True
    assert CARRIED["changes_how_any_cell_is_decided"] is False
    assert "decided by nobody" in CARRIED["what"]
    assert "a failure is not a verdict" in CARRIED["why_not_accepted"]
    assert "never retried automatically" in CARRIED["why_not_retried"]


# ============================== against the real run, read-only and marked
@needs_runs
def test_the_real_runs_still_show_eleven_missing_over_ten_cells():
    root = _judging_root()
    row_map = {r["row_id"]: r for r in T.read_jsonl(
        root / "packet" / "private" / P.ROW_MAP_FILE)}
    assert len(row_map) == CELLS
    missing = []
    for name in sorted(BASE.rubrics):
        rows = [r["row_id"] for r in T.read_jsonl(
            root / "packet" / f"{name}_judge_input.jsonl")]
        for judge in BASE.judge_order:
            path = root / "runs" / f"packet__{name}__{judge}" / P.RESULTS_FILE
            done = ({r["row_id"] for r in T.read_jsonl(path)}
                    if path.is_file() else set())
            missing += [(name, judge, r) for r in rows if r not in done]
    assert len(missing) == MISSING_VERDICTS
    assert len({m[2] for m in missing}) == UNRESOLVED_CELLS
    assert {(e["rubric"], e["failed_judge"], e["row_id"])
            for e in CARRIED["unresolved"]} == set(missing)


@needs_runs
def test_no_carried_judge_row_has_a_successful_result():
    root = _judging_root()
    for entry in CARRIED["unresolved"]:
        path = (root / "runs"
                / f"packet__{entry['rubric']}__{entry['failed_judge']}"
                / P.RESULTS_FILE)
        done = ({r["row_id"] for r in T.read_jsonl(path)}
                if path.is_file() else set())
        assert entry["row_id"] not in done


@needs_runs
def test_every_carried_raw_body_exists_and_recomputes_to_its_digest():
    root = _judging_root()
    checked = 0
    for entry in CARRIED["unresolved"]:
        run = (root / "runs"
               / f"packet__{entry['rubric']}__{entry['failed_judge']}")
        for attempt in entry["evidence"]:
            path = run / attempt["raw_response_path"]
            assert path.is_file(), path
            assert T.sha256_text(path.read_text(encoding="utf-8")) == \
                attempt["raw_response_sha256"]
            checked += 1
    assert checked == sum(e["attempts"] for e in CARRIED["unresolved"]) == 22


@needs_runs
def test_the_carried_rows_and_their_metadata_come_from_the_packet():
    root = _judging_root()
    row_map = {r["row_id"]: r for r in T.read_jsonl(
        root / "packet" / "private" / P.ROW_MAP_FILE)}
    for entry in CARRIED["unresolved"]:
        mapped = row_map[entry["row_id"]]
        assert entry["rubric"] == mapped["rubric"]
        assert entry["row_class"] == mapped["row_class"]
        assert entry["arm_id"] == mapped["arm_id"]
        assert entry["screen_row_id"] == mapped["screen_row_id"]


@needs_runs
def test_the_real_join_is_terminal_only_with_the_carried_manifest():
    root = Path(_paths()[DJ.PATHS_KEY_JUDGING])
    dirs = {name: root / str(sub) for name, sub in BASE.outputs.items()
            if name != "root_key"}
    with pytest.raises(ValidationError, match="no_uncarried_failure"):
        S.join_evidence(BASE, dirs, write=False, carried=())
    summary = S.join_evidence(BASE, dirs, write=False,
                              carried=CARRIED["unresolved"])
    assert summary["cells"] == CELLS
    assert summary["valid_verdicts"] == 949
    assert summary["planned_verdicts"] == 960
    assert summary["by_state"][P.STATE_UNRESOLVED_TECHNICAL] == UNRESOLVED_CELLS
    assert sum(summary["by_state"].values()) == CELLS
    assert P.STATE_MISSING not in summary["by_state"]
    assert P.STATE_INCOMPLETE not in summary["by_state"]
    assert summary["records_by_judge"] == {"sonnet": 471, "opus": 478}
    assert sorted(summary["technically_unresolved"]) == sorted(
        {e["row_id"] for e in CARRIED["unresolved"]})
