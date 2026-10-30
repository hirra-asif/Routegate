"""The thesis-results derivation: every way a figure could stop being a result.

Nothing here reads the real corpus, loads a model, generates a token, judges a
cell or calls a service. Every test builds a small synthetic world in
``tmp_path`` -- a handful of rows, a handful of cells -- so the gates are
exercised on inputs whose right answer is known by construction rather than by
looking up what the frozen files happen to say.

Organised by the failure each group prevents:

    1. sources     a wrong, sealed or stale file is read as if it were the one
    2. verdicts    a raw judge label or a prompt column reaches a reported count
    3. counting    a rate is written without the two integers under it
    4. phase 2     a transition or an anonymisation drifts
    5. provenance  a run is not reconstructible from what it wrote
    6. brakes      a moved number, or a second run, is absorbed instead of refused
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import thesisresults as T                            # noqa: E402
from routegate.validation import (                                  # noqa: E402
    Checklist,
    SealedSetError,
    StaleInputError,
    ValidationError,
    sha256_file,
)

BUILDER = (REPO_ROOT / "experiments" / "thesis_results"
           / "build_frozen_development_figures.py")


# ------------------------------------------------------------------ fixtures
#: A tiny corpus: 2 E rows, 2 B rows, 2 H rows. One E row is substantively
#: repaired, one B row is materially degraded, one H row goes harmful -- so
#: every numerator is 1 and every denominator is 2 when everything is steered.
SYNTHETIC_ROWS = [
    {"row_id": "e_repaired", "frozen_class": "E", "before_axis2": "DR",
     "final_after_axis2": "SA", "repair_status": "FULL_REPAIR_E_TO_B"},
    {"row_id": "e_refusing", "frozen_class": "E", "before_axis2": "DR",
     "final_after_axis2": "DR", "repair_status": "STILL_REFUSAL"},
    {"row_id": "b_degraded", "frozen_class": "B", "before_axis2": "SA",
     "final_quality_outcome": "BEFORE_BETTER", "final_difference_strength": "MATERIAL"},
    {"row_id": "b_intact", "frozen_class": "B", "before_axis2": "SA",
     "final_quality_outcome": "EQUIVALENT", "final_difference_strength": "NONE"},
    {"row_id": "h_harmful", "frozen_class": "H", "before_axis2": "DR",
     "final_actionable_harmful_help": "YES"},
    {"row_id": "h_safe", "frozen_class": "H", "before_axis2": "DR",
     "final_actionable_harmful_help": "NO"},
]

ALL_COLUMNS = (list(T.MASTER_READ_COLUMNS) + list(T.MASTER_TEXT_COLUMNS)
               + list(T.MASTER_RAW_JUDGE_COLUMNS))


def write_master(path: Path, rows=None, columns=None) -> Path:
    """A synthetic steering master, carrying the text and judge columns too."""
    rows = SYNTHETIC_ROWS if rows is None else rows
    columns = ALL_COLUMNS if columns is None else columns
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            record = {column: "" for column in columns}
            record.update({k: v for k, v in row.items() if k in columns})
            # The columns a caller must never read carry poison, not blanks.
            for column in T.MASTER_TEXT_COLUMNS:
                if column in record:
                    record[column] = f"PROMPT TEXT for {row['row_id']}"
            for column in T.MASTER_RAW_JUDGE_COLUMNS:
                if column in record:
                    record[column] = "RAW_JUDGE_LABEL"
            writer.writerow(record)
    return path


def steer(*row_ids) -> dict:
    """A gate that fires on exactly ``row_ids``."""
    wanted = set(row_ids)
    return {row["row_id"]: row["row_id"] in wanted for row in SYNTHETIC_ROWS}


def relaxed_checkpoints(monkeypatch, **overrides) -> None:
    """Point the checkpoints at the synthetic world, for the brake tests."""
    checkpoints = {
        "corpus_rows": 6,
        "class_sizes": {"E": 2, "B": 2, "H": 2},
        "systems": {"synthetic": {"E": [1, 2], "B": [1, 2], "H": [1, 2]}},
        "phase2": {"denominator": 2, "recovered": {s: 1 for s in T.PHASE2_STRENGTHS},
                   "harmful_alone": {s: 0 for s in T.PHASE2_STRENGTHS},
                   "outcome": "NO_STRENGTH_QUALIFIES"},
    }
    checkpoints.update(overrides)
    monkeypatch.setattr(T, "FROZEN_CHECKPOINTS", checkpoints)


PHASE2_ROWS = ("row_alpha", "row_beta", "row_gamma")


def phase2_verdicts() -> dict:
    """Three rows: one recovers everywhere, one nowhere, one is harmed alone."""
    verdicts = {}
    for row_id in PHASE2_ROWS:
        cell = {T.PHASE2_BASELINE_ARM: "NO", T.PHASE2_REPAIR_ARM: "YES"}
        for strength in T.PHASE2_STRENGTHS:
            cell[T.PHASE2_COMBINED_ARMS[strength]] = (
                "NO" if row_id == "row_alpha" else "YES")
            cell[T.PHASE2_SAFETY_ONLY_ARMS[strength]] = (
                "YES" if (row_id == "row_gamma" and strength != "0.05") else "NO")
        verdicts[row_id] = cell
    return verdicts


def write_analysis_table(path: Path, verdicts=None, **overrides) -> Path:
    verdicts = phase2_verdicts() if verdicts is None else verdicts
    lines = []
    for row_id, cell in verdicts.items():
        for arm, verdict in cell.items():
            record = {"screen_row_id": row_id, "arm_id": arm, "verdict": verdict,
                      "axis": T.PHASE2_AXIS, "row_class": "H",
                      "label_source": "DUAL_LLM_AGREEMENT",
                      "source": "phase2_adjudication", "is_inherited": False,
                      "adjudication_procedure": T.PHASE2_ADJUDICATION_PROCEDURE,
                      "prompt_sha256": "0" * 64, "response_sha256": "1" * 64}
            record.update(overrides)
            lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def synthetic_readings(verdicts=None, **overrides) -> dict:
    verdicts = phase2_verdicts() if verdicts is None else verdicts
    readings = {
        "schema": T.PHASE2_READINGS_SCHEMA,
        "protocol": T.PHASE2_PROTOCOL,
        "adjudication_procedure": T.PHASE2_ADJUDICATION_PROCEDURE,
        "complete_adjudication": True,
        "unresolved_cells": 0,
        "withheld": {"population_reportable": False},
        "b_per_row_transitions": verdicts,
        "c_counts_by_strength": {
            strength: {"combined_recovered": 1, "denominator": len(verdicts),
                       "safety_only_harmful": 0 if strength == "0.05" else 1}
            for strength in T.PHASE2_STRENGTHS},
        "d_decision": {"denominator": len(verdicts), "recovery_threshold": 2,
                       "safety_only_harm_limit": 0, "qualifying_strengths": [],
                       "outcome": "NO_STRENGTH_QUALIFIES"},
    }
    readings.update(overrides)
    return readings


def write_readings(path: Path, **kwargs) -> Path:
    path.write_text(json.dumps(synthetic_readings(**kwargs), indent=2),
                    encoding="utf-8")
    return path


# ------------------------------------------------------- 1. source validation
def test_a_master_missing_a_verdict_column_is_refused(tmp_path):
    path = write_master(tmp_path / "master.csv",
                        columns=[c for c in ALL_COLUMNS
                                 if c != "final_actionable_harmful_help"])
    with pytest.raises(T.ThesisResultsError, match="missing the column"):
        T.read_static_steering_master(path)


def test_a_master_with_a_repeated_row_is_refused(tmp_path):
    rows = SYNTHETIC_ROWS + [SYNTHETIC_ROWS[0]]
    path = write_master(tmp_path / "master.csv", rows=rows)
    with pytest.raises(T.ThesisResultsError, match="repeats row id"):
        T.read_static_steering_master(path)


def test_an_unknown_class_label_is_refused(tmp_path):
    rows = [dict(SYNTHETIC_ROWS[0], frozen_class="X")] + SYNTHETIC_ROWS[1:]
    path = write_master(tmp_path / "master.csv", rows=rows)
    with pytest.raises(T.ThesisResultsError, match="frozen_class"):
        T.read_static_steering_master(path)


def test_a_verdict_outside_the_vocabulary_is_refused(tmp_path):
    rows = [dict(row) for row in SYNTHETIC_ROWS]
    rows[4]["final_actionable_harmful_help"] = "PROBABLY"
    path = write_master(tmp_path / "master.csv", rows=rows)
    with pytest.raises(ValidationError, match="vocabulary_final_actionable"):
        T.read_static_steering_master(path, Checklist("t"))


def test_an_undecided_h_verdict_is_refused_rather_than_counted(tmp_path):
    rows = [dict(row) for row in SYNTHETIC_ROWS]
    rows[4]["final_actionable_harmful_help"] = "UNCLEAR"
    path = write_master(tmp_path / "master.csv", rows=rows)
    with pytest.raises(ValidationError, match="h_verdicts_are_decided"):
        T.read_static_steering_master(path, Checklist("t"))


def test_the_phase2_procedure_is_checked_not_assumed(tmp_path):
    path = write_readings(tmp_path / "readings.json",
                          adjudication_procedure="majority_vote")
    with pytest.raises(ValidationError, match="adjudication_procedure"):
        T.read_phase2_readings(path, Checklist("t"))


def test_a_readings_file_with_an_unresolved_cell_is_refused(tmp_path):
    path = write_readings(tmp_path / "readings.json", unresolved_cells=2)
    with pytest.raises(ValidationError, match="adjudication_complete"):
        T.read_phase2_readings(path, Checklist("t"))


def test_readings_that_claim_population_reportability_are_refused(tmp_path):
    path = write_readings(tmp_path / "readings.json",
                          withheld={"population_reportable": True})
    with pytest.raises(ValidationError, match="not_population_reportable"):
        T.read_phase2_readings(path, Checklist("t"))


# ---------------------------------------------------- 2. sealed and stale paths
@pytest.mark.parametrize("name", ["sealed_master.csv", "final_benchmark.csv",
                                  "holdout_benchmark.json"])
def test_a_sealed_looking_source_is_refused_before_it_is_opened(tmp_path, name):
    path = tmp_path / name
    with pytest.raises(SealedSetError):
        T.assert_source_is_usable(path, "input")
    assert not path.exists()


def test_a_sealed_output_directory_is_refused(tmp_path):
    with pytest.raises(SealedSetError):
        T.assert_output_directory_is_new(tmp_path / "sealed_figures")


def test_a_stale_dataset_revision_is_refused(tmp_path):
    path = tmp_path / "rq3_ebh_final_analysis_dataset.csv"
    path.write_text("row_id\n", encoding="utf-8")
    with pytest.raises(StaleInputError):
        T.assert_source_is_usable(path, "input")


def test_a_sealed_path_inside_a_record_is_refused(tmp_path):
    blob = {"source": "/data/sealed_benchmark/rows.csv"}
    with pytest.raises(SealedSetError):
        T.assert_no_sealed_material(blob, "record")


def test_a_sealed_key_inside_a_record_is_refused(tmp_path):
    with pytest.raises(SealedSetError):
        T.assert_no_sealed_material({"sealed_rows": 12}, "record")


def test_prose_that_disclaims_the_sealed_set_is_recorded_not_refused():
    checks = Checklist("t")
    disclaimers = T.assert_no_sealed_material(
        {"calibrated_on": "development only; the sealed set was not used"},
        "thresholds", checks)
    assert disclaimers == ["calibrated_on: development only; the sealed set "
                           "was not used"]
    assert checks.all_passed


# ------------------------------------------------ 3. adjudicated verdicts only
def test_a_raw_judge_column_may_not_be_read(tmp_path):
    for column in T.MASTER_RAW_JUDGE_COLUMNS:
        with pytest.raises(T.ThesisResultsError, match="raw per-judge labels"):
            T.assert_adjudicated_verdicts_only(["row_id", column])


def test_a_prompt_or_response_column_may_not_be_read(tmp_path):
    for column in T.MASTER_TEXT_COLUMNS:
        with pytest.raises(T.ThesisResultsError, match="prompt or response text"):
            T.assert_adjudicated_verdicts_only(["row_id", column])


def test_the_reader_keeps_no_text_and_no_raw_label(tmp_path):
    outcomes = T.read_static_steering_master(write_master(tmp_path / "m.csv"))
    kept = {column for record in outcomes.rows.values() for column in record}
    assert kept == set(T.MASTER_READ_COLUMNS)
    assert not kept & set(T.MASTER_TEXT_COLUMNS)
    assert not kept & set(T.MASTER_RAW_JUDGE_COLUMNS)
    flat = json.dumps(outcomes.rows)
    assert "PROMPT TEXT" not in flat and "RAW_JUDGE_LABEL" not in flat


def test_only_adjudicated_label_sources_are_accepted(tmp_path):
    path = write_analysis_table(tmp_path / "table.jsonl",
                                label_source="SONNET_ONLY")
    with pytest.raises(ValidationError, match="label_sources_are_adjudicated"):
        T.read_phase2_analysis_table(path, Checklist("t"))


def test_an_undecided_phase2_cell_is_refused(tmp_path):
    verdicts = phase2_verdicts()
    verdicts["row_beta"][T.PHASE2_COMBINED_ARMS["0.10"]] = "UNCLEAR"
    path = write_analysis_table(tmp_path / "table.jsonl", verdicts=verdicts)
    with pytest.raises(ValidationError, match="every_verdict_is_decided"):
        T.read_phase2_analysis_table(path, Checklist("t"))


# ---------------------------------------------- 4. numerator and denominator
def test_every_rate_carries_the_two_integers_it_came_from(tmp_path):
    outcomes = T.read_static_steering_master(write_master(tmp_path / "m.csv"))
    reading = T.derive_system_reading(T.STATIC_SYSTEM, outcomes)
    for label in T.DEVELOPMENT_CLASSES:
        proportion = reading.proportions[label]
        assert (proportion.numerator, proportion.denominator) == (1, 2)
        assert proportion.rate == proportion.numerator / proportion.denominator


def test_the_denominator_is_the_class_not_the_steered_rows(tmp_path):
    """A gate that fires on nothing scores 0/N, never 0/0 and never n/0."""
    outcomes = T.read_static_steering_master(write_master(tmp_path / "m.csv"))
    reading = T.derive_system_reading(T.SAFETY_FIRST_SYSTEM, outcomes,
                                      steered=steer())
    for label in T.DEVELOPMENT_CLASSES:
        proportion = reading.proportions[label]
        assert (proportion.numerator, proportion.denominator,
                proportion.steered) == (0, 2, 0)


def test_an_unsteered_row_cannot_enter_a_numerator(tmp_path):
    """Gating out the one damaged B row must move B from 1/2 to 0/2."""
    outcomes = T.read_static_steering_master(write_master(tmp_path / "m.csv"))
    reading = T.derive_system_reading(
        T.SAFETY_FIRST_SYSTEM, outcomes,
        steered=steer("e_repaired", "b_intact", "h_harmful"))
    assert reading.proportions["E"].numerator == 1
    assert reading.proportions["B"].numerator == 0
    assert reading.proportions["H"].numerator == 1


def test_a_gate_that_does_not_cover_the_corpus_is_refused(tmp_path):
    outcomes = T.read_static_steering_master(write_master(tmp_path / "m.csv"))
    partial = {"e_repaired": True}
    with pytest.raises(ValidationError, match="covers_the_corpus"):
        T.derive_system_reading(T.SAFETY_FIRST_SYSTEM, outcomes,
                                steered=partial, checks=Checklist("t"))


def test_the_wilson_interval_brackets_the_rate_and_closes_at_the_ends():
    low, high = T.wilson_interval(0, 40)
    assert low == 0.0 and 0.0 < high < 1.0
    low, high = T.wilson_interval(40, 40)
    assert high == 1.0 and 0.0 < low < 1.0
    low, high = T.wilson_interval(8, 178)
    assert low < 8 / 178 < high
    # The reference values for 8/178 at z = 1.96, to six places.
    assert round(low, 6) == 0.022947 and round(high, 6) == 0.086167


def test_a_wilson_interval_needs_a_real_denominator():
    with pytest.raises(T.ThesisResultsError):
        T.wilson_interval(0, 0)
    with pytest.raises(T.ThesisResultsError):
        T.wilson_interval(5, 4)


# ---------------------------------------------------- 5. phase-2 transitions
def test_recovery_is_the_yes_to_no_transition_not_the_arm_verdict(tmp_path):
    """A row already NO at repair_015 is not a recovery, only alpha is."""
    verdicts = T.read_phase2_analysis_table(
        write_analysis_table(tmp_path / "t.jsonl"), Checklist("t"))
    screen = T.derive_phase2_screen(verdicts, synthetic_readings(),
                                    Checklist("t"))
    for strength in T.PHASE2_STRENGTHS:
        assert screen.recovered[strength].numerator == 1
        assert screen.recovered[strength].denominator == 3


def test_harm_alone_is_counted_only_where_the_baseline_was_safe(tmp_path):
    verdicts = T.read_phase2_analysis_table(
        write_analysis_table(tmp_path / "t.jsonl"), Checklist("t"))
    screen = T.derive_phase2_screen(verdicts, synthetic_readings(),
                                    Checklist("t"))
    assert screen.harmful_alone["0.05"].numerator == 0
    assert all(screen.harmful_alone[s].numerator == 1
               for s in T.PHASE2_STRENGTHS[1:])


def test_a_screen_that_disagrees_with_its_own_readings_is_refused(tmp_path):
    verdicts = T.read_phase2_analysis_table(
        write_analysis_table(tmp_path / "t.jsonl"), Checklist("t"))
    readings = synthetic_readings()
    readings["c_counts_by_strength"]["0.10"]["combined_recovered"] = 3
    screen = T.derive_phase2_screen(verdicts, readings, Checklist("t"))
    with pytest.raises(ValidationError, match="recovered_agrees_at_0.10"):
        T.assert_screen_matches_readings(screen, readings, Checklist("t"))


def test_a_selection_row_that_did_not_regress_is_refused(tmp_path):
    verdicts = phase2_verdicts()
    verdicts["row_beta"][T.PHASE2_REPAIR_ARM] = "NO"
    path = write_analysis_table(tmp_path / "t.jsonl", verdicts=verdicts)
    parsed = T.read_phase2_analysis_table(path, Checklist("t"))
    with pytest.raises(ValidationError, match="selection_is_regressed_rows"):
        T.derive_phase2_screen(parsed, synthetic_readings(verdicts=verdicts),
                               Checklist("t"))


def test_a_missing_arm_is_refused_rather_than_treated_as_no(tmp_path):
    verdicts = phase2_verdicts()
    del verdicts["row_beta"][T.PHASE2_SAFETY_ONLY_ARMS["0.20"]]
    path = write_analysis_table(tmp_path / "t.jsonl", verdicts=verdicts)
    with pytest.raises(ValidationError, match="arm_coverage"):
        T.read_phase2_analysis_table(path, Checklist("t"))


# --------------------------------------------------- 6. deterministic labels
def test_only_rows_that_fail_somewhere_enter_the_matrix():
    matrix = T.derive_failure_matrix(synthetic_readings(), Checklist("t"))
    assert matrix.labels == ("H-01",)
    assert matrix.mapping["H-01"] == "row_gamma"


def test_the_matrix_is_ordered_by_failure_count_then_row_id():
    verdicts = phase2_verdicts()
    # zulu fails everywhere, alpha fails once, beta fails once. The two
    # one-failure rows must come back in row-id order.
    for strength in T.PHASE2_STRENGTHS:
        verdicts["row_alpha"][T.PHASE2_SAFETY_ONLY_ARMS[strength]] = "YES"
        verdicts["row_beta"][T.PHASE2_SAFETY_ONLY_ARMS[strength]] = (
            "YES" if strength == "0.05" else "NO")
    verdicts["row_gamma"] = dict(verdicts["row_gamma"])
    verdicts["row_gamma"][T.PHASE2_SAFETY_ONLY_ARMS["0.10"]] = "YES"
    for strength in ("0.05", "0.15", "0.20"):
        verdicts["row_gamma"][T.PHASE2_SAFETY_ONLY_ARMS[strength]] = "NO"
    matrix = T.derive_failure_matrix(synthetic_readings(verdicts=verdicts),
                                     Checklist("t"))
    assert matrix.labels == ("H-01", "H-02", "H-03")
    assert matrix.mapping["H-01"] == "row_alpha"          # 4 failures
    assert [matrix.mapping["H-02"], matrix.mapping["H-03"]] == [
        "row_beta", "row_gamma"]                          # 1 each, by row id
    assert matrix.failure_counts == {"H-01": 4, "H-02": 1, "H-03": 1}


def test_the_same_readings_always_produce_the_same_labels():
    first = T.derive_failure_matrix(synthetic_readings())
    reordered = synthetic_readings()
    reordered["b_per_row_transitions"] = dict(
        reversed(list(reordered["b_per_row_transitions"].items())))
    second = T.derive_failure_matrix(reordered)
    assert first.mapping == second.mapping


def test_the_plotted_labels_carry_no_source_identity():
    matrix = T.derive_failure_matrix(synthetic_readings())
    assert all(label.startswith("H-") and label[2:].isdigit()
               for label in matrix.labels)
    assert not any(matrix.mapping[label] in label for label in matrix.labels)


def test_a_matrix_column_must_equal_the_screens_harm_count(tmp_path):
    verdicts = T.read_phase2_analysis_table(
        write_analysis_table(tmp_path / "t.jsonl"), Checklist("t"))
    readings = synthetic_readings()
    screen = T.derive_phase2_screen(verdicts, readings, Checklist("t"))
    matrix = T.derive_failure_matrix(readings, Checklist("t"))
    T.assert_failure_matrix_matches_screen(matrix, screen, Checklist("t"))

    readings["b_per_row_transitions"]["row_beta"][
        T.PHASE2_SAFETY_ONLY_ARMS["0.20"]] = "YES"
    drifted = T.derive_failure_matrix(readings, Checklist("t"))
    with pytest.raises(ValidationError, match="column_agrees_at_0.20"):
        T.assert_failure_matrix_matches_screen(drifted, screen, Checklist("t"))


# ------------------------------------------------------------- 7. provenance
def test_every_input_digest_is_checked_against_what_was_recorded(tmp_path):
    path = write_master(tmp_path / "m.csv")
    digest = sha256_file(path)
    checks = Checklist("t")
    T.assert_input_digests({"master": digest}, {"master": digest}, checks)
    assert checks.all_passed
    with pytest.raises(ValidationError, match="input_digest_master"):
        T.assert_input_digests({"master": "0" * 64}, {"master": digest},
                               Checklist("t"))


def test_git_state_records_the_files_a_run_ran_from():
    state = T.git_state([REPO_ROOT / "src" / "routegate" / "thesisresults.py",
                         BUILDER])
    assert state["head"] and len(state["head"]) == 40
    for relative, entry in state["files"].items():
        assert entry["sha256"] and len(entry["sha256"]) == 64
        assert entry["git_blob"] and len(entry["git_blob"]) == 40
        assert entry["status"]


def test_a_completed_run_records_a_digest_for_every_file_it_wrote(tmp_path):
    """The output manifest must cover the directory, not a subset of it."""
    from routegate.runrecord import write_output_hashes
    for name in ("figure_1.png", "table_1.csv"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    digests = write_output_hashes(tmp_path)
    written = {p.name for p in tmp_path.iterdir()} - {"output_hashes.json"}
    assert set(digests) == written
    for name, digest in digests.items():
        assert digest == sha256_file(tmp_path / name)


# ----------------------------------------------------- 8. output immutability
def test_a_new_or_empty_output_directory_is_accepted(tmp_path):
    assert T.assert_output_directory_is_new(tmp_path / "fresh")
    (tmp_path / "empty").mkdir()
    assert T.assert_output_directory_is_new(tmp_path / "empty")


def test_a_directory_that_already_holds_a_run_is_refused(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "provenance_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(T.ThesisResultsError, match="not written over"):
        T.assert_output_directory_is_new(directory)


def test_an_output_path_that_is_a_file_is_refused(tmp_path):
    path = tmp_path / "run"
    path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(T.ThesisResultsError, match="not a directory"):
        T.assert_output_directory_is_new(path)


# -------------------------------------------------------- 9. frozen brakes
def _reading(tmp_path, numerators):
    outcomes = T.read_static_steering_master(write_master(tmp_path / "m.csv"))
    fired = steer(*numerators)
    specification = T.SystemSpecification(
        system_id="synthetic", display_name="synthetic", short_name="synthetic",
        kind="gated", outcome_source="synthetic")
    return outcomes, {"synthetic": T.derive_system_reading(
        specification, outcomes, steered=fired)}


def test_matching_checkpoints_pass(tmp_path, monkeypatch):
    relaxed_checkpoints(monkeypatch)
    outcomes, readings = _reading(tmp_path, SYNTHETIC_ROWS and
                                  ("e_repaired", "b_degraded", "h_harmful"))
    verdicts = T.read_phase2_analysis_table(
        write_analysis_table(tmp_path / "t.jsonl"), Checklist("t"))
    screen = T.derive_phase2_screen(verdicts, synthetic_readings(
        d_decision={"denominator": 3, "recovery_threshold": 2,
                    "safety_only_harm_limit": 0, "qualifying_strengths": [],
                    "outcome": "NO_STRENGTH_QUALIFIES"}), Checklist("t"))
    monkeypatch.setitem(T.FROZEN_CHECKPOINTS["phase2"], "denominator", 3)
    monkeypatch.setitem(T.FROZEN_CHECKPOINTS["phase2"], "harmful_alone",
                        {"0.05": 0, "0.10": 1, "0.15": 1, "0.20": 1})
    checks = Checklist("t")
    T.assert_frozen_checkpoints(readings, screen, outcomes.class_sizes(), checks)
    assert checks.all_passed


def test_a_moved_system_count_refuses_the_run(tmp_path, monkeypatch):
    relaxed_checkpoints(monkeypatch)
    outcomes, readings = _reading(tmp_path, ("b_degraded", "h_harmful"))
    verdicts = T.read_phase2_analysis_table(
        write_analysis_table(tmp_path / "t.jsonl"), Checklist("t"))
    screen = T.derive_phase2_screen(verdicts, synthetic_readings(), Checklist("t"))
    with pytest.raises(ValidationError, match="checkpoint_synthetic_E"):
        T.assert_frozen_checkpoints(readings, screen, outcomes.class_sizes(),
                                    Checklist("t"))


def test_a_moved_class_size_refuses_the_run(tmp_path, monkeypatch):
    relaxed_checkpoints(monkeypatch, class_sizes={"E": 89, "B": 178, "H": 213})
    outcomes, readings = _reading(tmp_path, ("e_repaired",))
    verdicts = T.read_phase2_analysis_table(
        write_analysis_table(tmp_path / "t.jsonl"), Checklist("t"))
    screen = T.derive_phase2_screen(verdicts, synthetic_readings(), Checklist("t"))
    with pytest.raises(ValidationError, match="checkpoint_class_sizes"):
        T.assert_frozen_checkpoints(readings, screen, outcomes.class_sizes(),
                                    Checklist("t"))


def test_a_moved_phase2_outcome_refuses_the_run(tmp_path, monkeypatch):
    relaxed_checkpoints(monkeypatch)
    outcomes, readings = _reading(tmp_path,
                                  ("e_repaired", "b_degraded", "h_harmful"))
    verdicts = T.read_phase2_analysis_table(
        write_analysis_table(tmp_path / "t.jsonl"), Checklist("t"))
    screen = T.derive_phase2_screen(verdicts, synthetic_readings(), Checklist("t"))
    monkeypatch.setitem(T.FROZEN_CHECKPOINTS["phase2"], "denominator", 3)
    monkeypatch.setitem(T.FROZEN_CHECKPOINTS["phase2"], "harmful_alone",
                        {"0.05": 0, "0.10": 1, "0.15": 1, "0.20": 1})
    monkeypatch.setitem(T.FROZEN_CHECKPOINTS["phase2"], "outcome", "SELECTED")
    with pytest.raises(ValidationError, match="checkpoint_phase2_outcome"):
        T.assert_frozen_checkpoints(readings, screen, outcomes.class_sizes(),
                                    Checklist("t"))


def test_a_system_without_a_derivation_cannot_be_checked_off(monkeypatch):
    relaxed_checkpoints(monkeypatch)
    screen = T.Phase2Screen(denominator=2, recovered={}, harmful_alone={},
                            recovery_threshold=2, safety_only_harm_limit=0,
                            outcome="NO_STRENGTH_QUALIFIES",
                            qualifying_strengths=(), per_strength={})
    with pytest.raises(T.ThesisResultsError, match="no reading was derived"):
        T.assert_frozen_checkpoints({}, screen, {"E": 2, "B": 2, "H": 2})


# ---------------------------------------------------------------- the runner
def _builder():
    spec = importlib.util.spec_from_file_location("build_frozen_figures", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_builder_refuses_a_sealed_output_path(tmp_path):
    module = _builder()
    assert module.main(["--output-dir", str(tmp_path / "sealed_run"),
                        "--dry-run"]) == 1


def test_the_builder_refuses_a_directory_that_already_holds_a_run(tmp_path):
    module = _builder()
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "figure_1_repair_safety_tradeoff.png").write_bytes(b"x")
    assert module.main(["--output-dir", str(directory), "--dry-run"]) == 1


def test_the_builder_never_names_the_sealed_set_as_an_input():
    text = BUILDER.read_text(encoding="utf-8")
    for token in ("final_benchmark", "benchmark_test", "holdout_benchmark"):
        assert token not in text


# ------------------------------------------------------------- 10. redrawing
def test_the_same_figure_drawn_twice_is_byte_identical(tmp_path):
    """A figure over frozen inputs must not carry a clock or a random id."""
    from routegate import plotting

    plt = plotting.configure()
    digests = []
    for run in ("first", "second"):
        directory = tmp_path / run
        directory.mkdir()
        figure = plt.figure(figsize=(3.0, 2.0))
        axis = figure.subplots()
        axis.bar([0, 1], [1, 2], color=plotting.SERIES_E)
        plotting.tidy(axis)
        written = plotting.save_figure(figure, directory, "f", dpi=90)
        plt.close(figure)
        digests.append({extension: sha256_file(path)
                        for extension, path in written.items()})
    assert digests[0] == digests[1]


# --------------------------------------------------- 11. system-name standing
def test_the_static_system_name_is_recorded_not_reconstructed():
    assert T.STATIC_SYSTEM.name_standing == T.NAME_STANDING_RECORDED
    assert not T.STATIC_SYSTEM.name_is_reconstructed
    assert "name_denotes" not in T.STATIC_SYSTEM.as_dict()


@pytest.mark.parametrize("specification",
                         [T.HIGHER_RECALL_SYSTEM, T.SAFETY_FIRST_SYSTEM])
def test_a_routegate_label_declares_itself_reconstructed(specification):
    assert specification.name_is_reconstructed
    payload = specification.as_dict()
    assert payload["name_standing"] == T.NAME_STANDING_RECONSTRUCTED
    assert payload["name_denotes"]
    # The disclaimer travels with the specification, so a consumer that
    # serialises the system cannot serialise the name without it.
    assert payload["name_is_not_a_quotation"] == T.RECONSTRUCTION_STATEMENT


def test_a_reconstructed_label_says_what_it_denotes_in_checkable_terms():
    """The denotation has to name the counts, or it is not checkable."""
    for specification, counts in ((T.HIGHER_RECALL_SYSTEM, ("54/89", "0/178", "1/213")),
                                  (T.SAFETY_FIRST_SYSTEM, ("37/89", "0/178", "0/213"))):
        for count in counts:
            assert count in specification.name_denotes, (
                f"{specification.system_id} does not say it denotes {count}")


def test_the_reconstruction_statement_disclaims_a_quotation():
    text = T.RECONSTRUCTION_STATEMENT.lower()
    assert "reconstruction" in text
    assert "not a quotation" in text
    assert "run-name field" in text


def test_the_reconstructed_names_are_disclaimed_in_what_is_not_claimed():
    joined = " ".join(T.DEVELOPMENT_ONLY_STATEMENTS).lower()
    assert "reconstructed label" in joined


# ------------------------------------------------------- 12. the probe heads
PROBE_RUN = REPO_ROOT / "outputs" / "dual_direction_controller" / "probe_heads_v1"


def synthetic_probe_metrics(**overrides) -> dict:
    """A metrics record for a two-row-per-class world, over the three states."""
    states = []
    for state in (T.PROBE_PRIMARY_STATE,) + T.PROBE_SENSITIVITY_STATES:
        is_primary = state == T.PROBE_PRIMARY_STATE
        heads = {}
        for head in T.PROBE_HEADS:
            task = T.PROBE_HEAD_TASKS[head]
            n_pos, n_neg = (2, 2) if head == "repair" else (2, 4)
            heads[head] = {
                "head": head, "state": state, "historical_layer": state - 1,
                "is_primary_read_state": is_primary,
                "role": "primary" if is_primary else "sensitivity_check",
                "positive_class": list(task["positive_class"]),
                "negative_class": list(task["negative_class"]),
                "excluded_from_training": list(task["excluded_from_training"]),
                "score_column": task["score_column"],
                "n_scored": n_pos + n_neg,
                "n_positive": n_pos, "n_negative": n_neg,
                "n_scored_without_a_target": 2 if head == "repair" else 0,
                "roc_auc": 1.0, "pr_auc": 1.0, "brier": 0.0,
                "balanced_accuracy": 1.0, "accuracy": 1.0,
                "decision_threshold": 0.5, "n_folds": 5,
                "chance": {"roc_auc": 0.5, "pr_auc": n_pos / (n_pos + n_neg),
                           "balanced_accuracy": 0.5,
                           "prevalence": n_pos / (n_pos + n_neg)},
                "calibration_model_fitted": False,
            }
        states.append({"state": state, "historical_layer": state - 1,
                       "role": "primary" if is_primary else "sensitivity_check",
                       "heads": heads})
    metrics = {
        "stage": T.PROBE_RUN_STAGE,
        "created_utc": "2026-08-21T21:38:27Z",
        "primary_read_state": T.PROBE_PRIMARY_STATE,
        "sensitivity_states": list(T.PROBE_SENSITIVITY_STATES),
        "state_selection": "none. The primary read state is predeclared.",
        "specification": {
            "calibration": {"model_fitted": False},
            "layer_state_convention": {
                "primary_read_layer": T.PROBE_PRIMARY_LAYER,
                "primary_read_state": T.PROBE_PRIMARY_STATE,
                "primary_read_state_standing":
                    "predeclared. State 32 is the primary controller read state.",
            },
        },
        "states": states,
    }
    metrics.update(overrides)
    return metrics


def synthetic_probe_validation(**overrides) -> dict:
    blob = {"stage": T.PROBE_RUN_STAGE, "all_passed": True, "checks_run": 12,
            "summary": {"class_counts": {"E": 2, "B": 2, "H": 2},
                        "calibration_model_fitted": False,
                        "model_loaded": False}}
    blob.update(overrides)
    return blob


#: Two E, two B, two H, scored at the extremes so that the synthetic metrics
#: declared above -- all 1.0, Brier exactly 0.0 -- are the values that
#: genuinely recompute from them. The recompute gate is real, so a fixture
#: whose metrics only roughly matched would fail it, which is the point.
#: Every probability sits in bin 0 or bin 9, leaving eight empty bins, so the
#: empty-bin path is exercised by the same fixture.
SYNTHETIC_PROBE_SCORES = [
    ("e_one", "E", 0, 1.0, 0.0), ("e_two", "E", 1, 1.0, 0.0),
    ("b_one", "B", 0, 0.0, 0.0), ("b_two", "B", 1, 0.0, 0.0),
    ("h_one", "H", 0, 0.5, 1.0), ("h_two", "H", 1, 0.5, 1.0),
]


def write_probe_predictions(path: Path, scores=None, states=None) -> Path:
    scores = SYNTHETIC_PROBE_SCORES if scores is None else scores
    states = ((T.PROBE_PRIMARY_STATE,) + T.PROBE_SENSITIVITY_STATES
              if states is None else states)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "row_id", "final_ebh", "fold", "state", "p_E", "p_H"])
        writer.writeheader()
        for state in states:
            for row_id, label, fold, p_e, p_h in scores:
                writer.writerow({"row_id": row_id, "final_ebh": label,
                                 "fold": fold, "state": state,
                                 "p_E": p_e, "p_H": p_h})
    return path


def probe_world(tmp_path, metrics=None, validation=None, scores=None):
    """Read the synthetic run the way the builder reads the real one."""
    checks = Checklist("t")
    path = write_probe_predictions(tmp_path / "oof.csv", scores=scores)
    predictions = T.read_probe_oof_predictions(path, checks)
    return (metrics or synthetic_probe_metrics(),
            validation or synthetic_probe_validation(),
            predictions, checks)


def test_the_primary_state_is_the_one_the_run_predeclared(tmp_path):
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    assert probes.primary_state == 32
    assert probes.primary_layer == 31
    assert probes.sensitivity_states == (33, 34)
    assert set(probes.primary) == set(T.PROBE_HEADS)


def test_a_run_that_did_not_predeclare_its_state_is_refused(tmp_path):
    metrics = synthetic_probe_metrics()
    metrics["specification"]["layer_state_convention"][
        "primary_read_state_standing"] = "chosen as the best of the three"
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValidationError, match="primary_state_is_predeclared"):
        T.read_probe_metrics(path, Checklist("t"))


def test_a_sensitivity_state_may_not_be_promoted_to_primary(tmp_path):
    metrics = synthetic_probe_metrics()
    metrics["primary_read_state"] = 33
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValidationError, match="primary_state_is_32"):
        T.read_probe_metrics(path, Checklist("t"))


def test_a_run_that_selected_a_state_from_its_metrics_is_refused(tmp_path):
    metrics = synthetic_probe_metrics()
    metrics["state_selection"] = "state 33, the highest PR-AUC of the three"
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValidationError, match="no_state_was_selected"):
        T.read_probe_metrics(path, Checklist("t"))


def test_a_fitted_calibrator_is_refused(tmp_path):
    metrics = synthetic_probe_metrics()
    metrics["specification"]["calibration"]["model_fitted"] = True
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValidationError, match="no_calibration_model_was_fitted"):
        T.read_probe_metrics(path, Checklist("t"))


def test_a_validation_record_that_did_not_pass_is_refused(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(synthetic_probe_validation(all_passed=False)),
                    encoding="utf-8")
    with pytest.raises(ValidationError, match="own_checklist_passed"):
        T.read_probe_validation(path, Checklist("t"))


def test_the_run_manifest_binds_the_files_that_were_actually_read(tmp_path):
    """The run recorded what its outputs hash to; a mismatch is a wrong file."""
    manifest = {"schema": T.PROBE_RUN_SCHEMA,
                "outputs": {"metrics.json": "a" * 64,
                            "oof_predictions.csv": "b" * 64}}
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    good = {"probe_metrics": "a" * 64, "probe_oof_predictions": "b" * 64}
    checks = Checklist("t")
    T.read_probe_run_manifest(path, good, checks)
    assert checks.all_passed
    with pytest.raises(ValidationError, match="binds_metrics.json"):
        T.read_probe_run_manifest(path, dict(good, probe_metrics="c" * 64),
                                  Checklist("t"))


def test_a_row_scored_twice_at_one_state_is_refused(tmp_path):
    scores = SYNTHETIC_PROBE_SCORES + [SYNTHETIC_PROBE_SCORES[0]]
    path = write_probe_predictions(tmp_path / "oof.csv", scores=scores)
    with pytest.raises(T.ThesisResultsError, match="twice at state"):
        T.read_probe_oof_predictions(path, Checklist("t"))


def test_a_row_that_changes_fold_between_states_is_refused(tmp_path):
    path = tmp_path / "oof.csv"
    write_probe_predictions(path)
    text = path.read_text(encoding="utf-8").splitlines()
    text[-1] = text[-1].replace(",1,34,", ",3,34,")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="folds_and_classes_are_stable"):
        T.read_probe_oof_predictions(path, Checklist("t"))


def test_the_repair_head_uses_only_e_and_b_rows(tmp_path):
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    repair = probes.primary["repair"]
    assert repair.n_positive == 2 and repair.n_negative == 2
    assert repair.n_scored == 4                 # E + B, never the H rows
    assert repair.n_scored_without_a_target == 2
    assert "H" not in "".join(T.PROBE_HEAD_TASKS["repair"]["positive_class"]
                              + T.PROBE_HEAD_TASKS["repair"]["negative_class"])
    assert T.PROBE_HEAD_TASKS["repair"]["excluded_from_training"] == ("H",)


def test_the_safety_head_uses_every_row_with_e_and_b_as_one_negative(tmp_path):
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    safety = probes.primary["safety"]
    assert safety.n_positive == 2 and safety.n_negative == 4
    assert safety.n_scored == 6                 # every row is eligible
    assert safety.n_scored_without_a_target == 0
    assert T.PROBE_HEAD_TASKS["safety"]["negative_class"] == ("E", "B")


def test_an_orientation_that_has_moved_is_refused(tmp_path):
    metrics = synthetic_probe_metrics()
    metrics["states"][0]["heads"]["repair"]["positive_class"] = ["B"]
    metrics["states"][0]["heads"]["repair"]["negative_class"] = ["E"]
    _, validation, predictions, checks = probe_world(tmp_path)
    with pytest.raises(ValidationError, match="probe_repair_32_orientation"):
        T.derive_probe_readings(metrics, validation, predictions, checks)


def test_a_stored_metric_that_no_longer_follows_from_the_scores_is_refused(tmp_path):
    metrics = synthetic_probe_metrics()
    metrics["states"][0]["heads"]["safety"]["roc_auc"] = 0.5
    _, validation, predictions, checks = probe_world(tmp_path)
    with pytest.raises(ValidationError,
                       match="probe_safety_32_stored_metrics_recompute"):
        T.derive_probe_readings(metrics, validation, predictions, checks)


def test_reported_metrics_are_the_stored_ones_not_the_recomputed_ones(tmp_path):
    """Recomputation is a check. What is reported is what the run stored."""
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    stored = metrics["states"][0]["heads"]["safety"]
    reading = probes.primary["safety"]
    for name in ("roc_auc", "pr_auc", "brier", "balanced_accuracy", "accuracy"):
        assert getattr(reading, name) == stored[name]


def test_a_class_count_that_disagrees_with_the_run_is_refused(tmp_path):
    metrics = synthetic_probe_metrics()
    metrics["states"][0]["heads"]["safety"]["n_positive"] = 5
    _, validation, predictions, checks = probe_world(tmp_path)
    with pytest.raises(ValidationError, match="probe_safety_32_class_sizes"):
        T.derive_probe_readings(metrics, validation, predictions, checks)


def test_only_the_primary_state_carries_a_calibration(tmp_path):
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    for head in T.PROBE_HEADS:
        assert probes.primary[head].bins
        assert probes.primary[head].ece is not None
        for state in probes.sensitivity_states:
            assert probes.sensitivity[state][head].bins == ()
            assert probes.sensitivity[state][head].ece is None


# ------------------------------------------------------ 13. fixed-bin binning
def test_the_grid_is_ten_fixed_equal_width_bins():
    bins = T.calibration_bins([0.05, 0.5, 0.95], [0, 1, 1])
    assert len(bins) == T.CALIBRATION_BIN_COUNT == 10
    assert [b.low for b in bins] == [i / 10 for i in range(10)]
    assert [b.high for b in bins] == [(i + 1) / 10 for i in range(10)]
    assert sum(b.count for b in bins) == 3


def test_every_probability_lands_in_exactly_one_bin():
    """The edges matter: 0.0, every interior edge, and 1.0."""
    probabilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bins = T.calibration_bins(probabilities, [0] * len(probabilities))
    assert sum(b.count for b in bins) == len(probabilities)
    # An interior edge belongs to the bin above it; 1.0 to the last bin.
    assert [b.count for b in bins] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]


def test_an_empty_bin_is_kept_with_no_statistics():
    bins = T.calibration_bins([0.05, 0.95], [0, 1])
    assert len(bins) == 10
    empty = [b for b in bins if b.is_empty]
    assert len(empty) == 8
    for b in empty:
        assert b.count == 0
        assert b.mean_predicted is None and b.observed_fraction is None
        # Written as empty, never as a number and never as NaN.
        payload = b.as_dict()
        assert payload["mean_predicted_probability"] == ""
        assert payload["observed_positive_fraction"] == ""
        assert payload["positives"] == ""
        assert payload["is_empty"] is True


def test_a_bin_reports_the_mean_prediction_and_the_observed_fraction():
    bins = T.calibration_bins([0.62, 0.66, 0.68], [1, 0, 1])
    seventh = bins[6]
    assert seventh.count == 3 and seventh.positives == 2
    assert seventh.observed_fraction == pytest.approx(2 / 3)
    assert seventh.mean_predicted == pytest.approx((0.62 + 0.66 + 0.68) / 3)


def test_binning_refuses_unpaired_inputs():
    with pytest.raises(T.ThesisResultsError, match="must be paired"):
        T.calibration_bins([0.1, 0.2], [1])


def test_a_perfectly_calibrated_head_has_zero_calibration_error():
    """20 rows at p = 0.25 of which 5 are positive: the bin is exactly right."""
    probabilities = [0.25] * 20
    targets = [1] * 5 + [0] * 15
    bins = T.calibration_bins(probabilities, targets)
    assert T.expected_calibration_error(bins) == pytest.approx(0.0)


def test_the_calibration_error_is_the_count_weighted_gap():
    # 10 rows at p = 0.9 with none positive, 90 at p = 0.1 with none positive.
    probabilities = [0.9] * 10 + [0.1] * 90
    targets = [0] * 100
    bins = T.calibration_bins(probabilities, targets)
    assert T.expected_calibration_error(bins) == pytest.approx(
        0.10 * 0.9 + 0.90 * 0.1)


def test_the_bin_rule_and_the_ece_definition_say_what_they_are():
    assert "10 fixed equal-width bins" in T.CALIBRATION_BIN_RULE
    assert "not chosen from the data" in T.CALIBRATION_BIN_RULE
    assert "derived HERE" in T.ECE_DEFINITION
    assert "descriptive statistic" in T.ECE_DEFINITION
    assert "10 fixed bins" in T.ECE_DEFINITION
    assert "No calibration model is fitted" in T.CALIBRATION_IS_NOT_A_MODEL
    assert "stored" in T.BRIER_IS_THE_STORED_SCORE


# --------------------------------------------------- 14. the probe checkpoint
def test_the_real_probe_metrics_meet_the_frozen_checkpoint():
    """The one test that reads the real run, and it only reads metrics.json."""
    path = PROBE_RUN / "metrics.json"
    if not path.is_file():
        pytest.skip("the frozen probe-head run is not reachable")
    blob = json.loads(path.read_text(encoding="utf-8"))
    primary = [e for e in blob["states"]
               if e["state"] == T.PROBE_PRIMARY_STATE][0]
    expected = T.FROZEN_CHECKPOINTS["probe_heads"]
    for head in T.PROBE_HEADS:
        for metric, want in expected[head].items():
            assert round(float(primary["heads"][head][metric]), 6) == want


def test_a_moved_probe_metric_refuses_the_run(tmp_path, monkeypatch):
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    monkeypatch.setitem(
        T.FROZEN_CHECKPOINTS, "probe_heads",
        {"primary_state": 32,
         "repair": {"roc_auc": 1.0, "pr_auc": 1.0, "brier": 0.0},
         "safety": {"roc_auc": 0.5, "pr_auc": 1.0, "brier": 0.0}})
    with pytest.raises(ValidationError, match="checkpoint_probe_safety_roc_auc"):
        T.assert_probe_checkpoints(probes, Checklist("t"))


def test_matching_probe_checkpoints_pass(tmp_path, monkeypatch):
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    monkeypatch.setitem(
        T.FROZEN_CHECKPOINTS, "probe_heads",
        {"primary_state": 32,
         "repair": {"roc_auc": 1.0, "pr_auc": 1.0, "brier": 0.0},
         "safety": {"roc_auc": 1.0, "pr_auc": 1.0, "brier": 0.0}})
    verified = Checklist("t")
    T.assert_probe_checkpoints(probes, verified)
    assert verified.all_passed


def test_the_probe_checkpoint_covers_the_primary_state_only():
    """A brake on a sensitivity state would be a second place to judge from."""
    expected = T.FROZEN_CHECKPOINTS["probe_heads"]
    assert expected["primary_state"] == 32
    assert set(expected) == {"primary_state"} | set(T.PROBE_HEADS)


def test_oof_coverage_states_what_it_rests_on(tmp_path):
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    repair = probes.primary["repair"].oof_coverage
    assert "4/4 eligible row(s) scored exactly once" in repair
    assert "2 further row(s)" in repair and "no target" in repair
    safety = probes.primary["safety"].oof_coverage
    assert "6/6 eligible row(s) scored exactly once" in safety
    assert "further row" not in safety


def test_the_serialised_reading_carries_the_task_and_the_disclaimers(tmp_path):
    metrics, validation, predictions, checks = probe_world(tmp_path)
    probes = T.derive_probe_readings(metrics, validation, predictions, checks)
    payload = probes.as_dict()
    assert payload["read_position"]["primary_state"] == 32
    assert "predeclared" in payload["read_position"]["primary_state_standing"]
    assert payload["no_calibration_model_is_fitted"] == T.CALIBRATION_IS_NOT_A_MODEL
    repair = payload["primary"]["repair"]
    assert repair["task"] == "E versus B"
    assert repair["positive_class"] == ["E"] and repair["negative_class"] == ["B"]
    assert repair["calibration_model_fitted"] is False
    assert repair["calibration"]["probabilities_are"] == (
        "uncalibrated out-of-fold probabilities")
    assert len(repair["calibration"]["bins"]) == 10
    assert repair["brier_standing"] == T.BRIER_IS_THE_STORED_SCORE
