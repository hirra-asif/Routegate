"""The run script's selection logic, without running it.

What is under test is what the command line decides *before* a model exists:
which states, which targets, which donors, which conditions, and how those
become a deterministic list of generations whose keys line up with the record
schema. Importing the script must not import torch or transformers, and these
tests would fail loudly if it did.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

from routegate.patchrecord import CONDITIONS, KEY_FIELDS, record_key
from routegate.validation import SealedSetError, StaleInputError, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO_ROOT / "experiments" / "causal_repair"


@pytest.fixture(scope="module")
def script():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        module = importlib.import_module("run_patching")
        yield importlib.reload(module)
    finally:
        sys.path.remove(str(EXPERIMENT))


@pytest.fixture
def pairs():
    return {
        "e:001": {"target_row_id": "e:001", "donor_row_id": "b:001",
                  "tier": "source_topic_task", "exact_match": True},
        "e:002": {"target_row_id": "e:002", "donor_row_id": "b:002",
                  "tier": "topic", "exact_match": False},
    }


def write_dataset(path: Path, rows):
    columns = ["row_id", "final_ebh", "prompt_text", "response_text",
               "source_family", "topic_standardized", "task_frame"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in rows:
            writer.writerow(record)
    return path


def dataset_row(row_id, label, prompt="a prompt", response="an answer"):
    return {"row_id": row_id, "final_ebh": label, "prompt_text": prompt,
            "response_text": response, "source_family": "orbench_hard",
            "topic_standardized": "CYBER", "task_frame": "INFORM_EXPLAIN"}


# ------------------------------------------------------- 1. selecting states
def test_states_parse_singly_and_in_ranges(script):
    assert script.parse_states("0", 49) == [0]
    assert script.parse_states("32,33,34", 49) == [32, 33, 34]
    assert script.parse_states("32-34", 49) == [32, 33, 34]
    assert script.parse_states("0,10-12,48", 49) == [0, 10, 11, 12, 48]


def test_states_are_deduplicated_and_sorted(script):
    assert script.parse_states("34,32,33,33", 49) == [32, 33, 34]


def test_a_state_outside_the_model_is_refused_on_the_command_line(script):
    for spec in ("49", "-1", "45-60"):
        with pytest.raises(ValidationError, match="outside"):
            script.parse_states(spec, 49)


def test_a_malformed_state_spec_is_a_refusal_not_a_traceback(script):
    """A mistyped --states must stop the command line, not raise ValueError.

    ``-1`` is the case that matters: read as a half-written range it is a bare
    ``ValueError`` from ``int('')``, and read as a number it is a refusal that
    names the states that do exist.
    """
    for spec in ("abc", "3-", "-", "1.5", "1-2-3"):
        with pytest.raises(ValidationError):
            script.parse_states(spec, 49)


def test_an_inverted_range_is_refused(script):
    with pytest.raises(ValidationError, match="ends before it starts"):
        script.parse_states("34-32", 49)


def test_the_whole_scan_is_forty_nine_states(script):
    assert len(script.parse_states("0-48", 49)) == 49


def test_a_comma_separated_list_ignores_blanks(script):
    assert script.parse_list(" a , ,b ") == ["a", "b"]
    assert script.parse_list(None) == []


# --------------------------------------------------------- 2. the work list
def test_the_baseline_runs_once_per_target_not_once_per_state(script, pairs):
    work = script.build_work_list(
        targets=["e:001", "e:002"], states=[32, 33, 34],
        conditions=["baseline", "donor_patch"], self_patch_states=[],
        pairs=pairs, strength=0.2)

    baselines = [w for w in work if w["condition"] == "baseline"]
    assert len(baselines) == 2
    assert all(w["state"] == "" for w in baselines)
    assert all(w["intervention_type"] == "none" for w in baselines)
    assert all(w["donor_row_id"] == "" for w in baselines)


def test_the_intervention_runs_once_per_state_per_target(script, pairs):
    work = script.build_work_list(
        targets=["e:001", "e:002"], states=[32, 33, 34],
        conditions=["donor_patch"], self_patch_states=[], pairs=pairs,
        strength=0.2)

    assert len(work) == 6
    assert {w["state"] for w in work} == {32, 33, 34}
    assert all(w["intervention_type"] == "donor_direction" for w in work)
    assert {w["donor_row_id"] for w in work} == {"b:001", "b:002"}


def test_the_frozen_donor_is_the_one_that_is_used(script, pairs):
    work = script.build_work_list(
        targets=["e:001"], states=[10], conditions=["donor_patch"],
        self_patch_states=[], pairs=pairs, strength=0.2)
    assert work[0]["donor_row_id"] == pairs["e:001"]["donor_row_id"]


def test_the_self_patch_control_uses_the_target_as_its_own_donor(script, pairs):
    work = script.build_work_list(
        targets=["e:001"], states=[32, 33], conditions=["self_patch"],
        self_patch_states=[32], pairs=pairs, strength=0.2)

    assert len(work) == 1
    assert work[0]["state"] == 32
    assert work[0]["donor_row_id"] == "e:001"
    assert work[0]["intervention_type"] == "self_patch"


def test_all_three_conditions_can_run_together(script, pairs):
    work = script.build_work_list(
        targets=["e:001"], states=[32, 33],
        conditions=["baseline", "self_patch", "donor_patch"],
        self_patch_states=[32, 33], pairs=pairs, strength=0.2)

    conditions = [w["condition"] for w in work]
    assert conditions.count("baseline") == 1
    assert conditions.count("self_patch") == 2
    assert conditions.count("donor_patch") == 2
    assert set(CONDITIONS) == set(script.CONDITION_INTERVENTION)


def test_the_work_list_is_deterministic(script, pairs):
    arguments = dict(targets=["e:002", "e:001"], states=[33, 32],
                     conditions=["baseline", "donor_patch"],
                     self_patch_states=[], pairs=pairs, strength=0.2)
    first = script.build_work_list(**arguments)
    second = script.build_work_list(**arguments)
    assert [script.work_key(w) for w in first] == [script.work_key(w)
                                                   for w in second]


def test_every_work_item_has_a_unique_key(script, pairs):
    work = script.build_work_list(
        targets=["e:001", "e:002"], states=[0, 32, 48],
        conditions=["baseline", "self_patch", "donor_patch"],
        self_patch_states=[0, 32, 48], pairs=pairs, strength=0.2)
    keys = [script.work_key(w) for w in work]
    assert len(set(keys)) == len(keys)


def test_the_work_key_matches_the_record_key(script, pairs):
    work = script.build_work_list(
        targets=["e:001"], states=[32], conditions=["donor_patch"],
        self_patch_states=[], pairs=pairs, strength=0.2)
    item = work[0]
    record_like = {field: value for field, value in zip(
        KEY_FIELDS, script.work_key(item))}
    assert record_key(record_like) == script.work_key(item)
    assert record_like["state"] == 32
    assert record_like["condition"] == "donor_patch"


def test_strength_travels_into_every_intervention(script, pairs):
    work = script.build_work_list(
        targets=["e:001"], states=[32], conditions=["donor_patch"],
        self_patch_states=[], pairs=pairs, strength=0.15)
    assert work[0]["intervention_strength"] == 0.15


# ------------------------------------------- 2b. which states get a hook
def test_the_hooked_states_cover_both_state_lists(script):
    """Every state some condition will plan for must have a hook installed.

    The controller refuses a plan for a state it never hooked, so a control
    state outside --states would fail per generation rather than on the command
    line. The union is resolved before the model is loaded.
    """
    assert script.hooked_states(states=[0, 1, 2], self_patch_states=[7],
                                conditions=["self_patch", "donor_patch"]) == [
        0, 1, 2, 7]


def test_the_hooked_states_are_sorted_and_deduplicated(script):
    assert script.hooked_states(states=[34, 32, 33],
                                self_patch_states=[33, 32],
                                conditions=["self_patch", "donor_patch"]) == [
        32, 33, 34]


def test_a_condition_that_is_not_running_hooks_nothing(script):
    """--self-patch-states is inert unless the control was actually asked for."""
    assert script.hooked_states(states=[32], self_patch_states=[7],
                                conditions=["donor_patch"]) == [32]
    assert script.hooked_states(states=[32], self_patch_states=[7],
                                conditions=["self_patch"]) == [7]


def test_a_baseline_only_run_installs_no_state_hook(script):
    assert script.hooked_states(states=[0, 32, 48], self_patch_states=[0, 32],
                                conditions=["baseline"]) == []


def test_every_planned_state_is_a_hooked_state(script, pairs):
    """The contract, stated over the work list rather than over the arguments."""
    states, self_patch_states = [32, 33], [0]
    conditions = ["baseline", "self_patch", "donor_patch"]
    work = script.build_work_list(
        targets=["e:001"], states=states, conditions=conditions,
        self_patch_states=self_patch_states, pairs=pairs, strength=0.2)

    hooked = set(script.hooked_states(states, self_patch_states, conditions))
    planned = {w["state"] for w in work if w["condition"] != "baseline"}
    assert planned <= hooked


# -------------------------------------- 3. what the run refuses to look at
def test_the_withheld_class_is_refused_as_a_target(script):
    table = {"e:001": {"final_ebh": "E"}, "h:001": {"final_ebh": "H"}}
    with pytest.raises(ValidationError, match="withheld"):
        script.assert_no_withheld(["e:001", "h:001"], table, "targets")
    script.assert_no_withheld(["e:001"], table, "targets")


def test_the_withheld_class_is_refused_as_a_donor(script):
    table = {"b:001": {"final_ebh": "B"}, "h:001": {"final_ebh": "H"}}
    with pytest.raises(ValidationError, match="withheld"):
        script.assert_no_withheld(["h:001"], table, "donors")


def test_a_sealed_looking_dataset_path_is_refused(script, tmp_path):
    path = tmp_path / "sealed_final_benchmark.csv"
    path.write_text("row_id\n", encoding="utf-8")
    with pytest.raises(SealedSetError):
        script.load_prompts_and_responses(path, "0" * 64)


def test_a_superseded_dataset_revision_is_refused(script, tmp_path):
    path = tmp_path / "rq3_dataset_final_before_e_axis2_reaudit.csv"
    path.write_text("row_id\n", encoding="utf-8")
    with pytest.raises(StaleInputError):
        script.load_prompts_and_responses(path, "0" * 64)


def test_the_dataset_hash_gates_the_prompt_table(script, tmp_path):
    path = write_dataset(tmp_path / "rq3_dataset_final.csv",
                         [dataset_row("e:001", "E"), dataset_row("b:001", "B")])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    table = script.load_prompts_and_responses(path, digest)
    assert sorted(table) == ["b:001", "e:001"]
    assert table["e:001"]["original_response"] == "an answer"

    with pytest.raises(ValidationError, match="hash mismatch"):
        script.load_prompts_and_responses(path, "0" * 64)


def test_the_prompt_table_excludes_the_withheld_class(script, tmp_path):
    path = write_dataset(tmp_path / "rq3_dataset_final.csv",
                         [dataset_row("e:001", "E"), dataset_row("b:001", "B"),
                          dataset_row("h:001", "H")])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    table = script.load_prompts_and_responses(path, digest)
    assert "h:001" not in table


# ----------------------------------------------- 4. the script stays offline
def test_importing_the_script_loads_no_deep_learning_runtime(script):
    """The planning half must be importable on a machine with no GPU stack."""
    assert "transformers" not in sys.modules
    assert script.RECORDS_FILENAME == "patch_records.jsonl"


def test_the_condition_map_covers_every_condition(script):
    assert script.CONDITION_INTERVENTION == {
        "baseline": "none",
        "self_patch": "self_patch",
        "donor_patch": "donor_direction",
    }
