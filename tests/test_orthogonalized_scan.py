"""What the scan decides before a model exists, and what the scheduler covers.

Under test: the state selection, the target selection, the work list those become,
the generation counts that work list implies, the record schema those generations
are written under, and the array job's partition of the block axis.

Importing the run script must not import torch or transformers -- it defers both
until after the plan is built and validated -- and these tests would fail loudly
if it did. Nothing here loads a model, downloads anything, or touches a GPU.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import re
import sys
from pathlib import Path

import pytest

from routegate.steerrecord import (
    CONDITION_BASELINE,
    CONDITION_REPAIR,
    CONDITIONS,
    KEY_FIELDS,
    RECORD_FIELDS,
    SteerRecordError,
    completed_keys,
    make_record,
    plan_work,
    record_key,
    transformer_block_of_state,
    verify_consistency,
)
from routegate.steering import CAUSAL_SCAN_STATES, FROZEN_SCAN_STRENGTH
from routegate.validation import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO_ROOT / "experiments" / "causal_repair"
SUBMIT_SCRIPT = EXPERIMENT / "submit_orthogonalized_population_scan.sh"

N_E = 89
N_H = 213
N_TARGETS = N_E + N_H                     # 302
N_BLOCK_STATES = 48
EXPECTED_INTERVENTIONS = N_TARGETS * N_BLOCK_STATES      # 14 496
EXPECTED_BASELINES = N_TARGETS                           # 302
EXPECTED_TOTAL = EXPECTED_INTERVENTIONS + EXPECTED_BASELINES   # 14 798


@pytest.fixture(scope="module")
def script():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        module = importlib.import_module("run_orthogonalized_population_scan")
        yield importlib.reload(module)
    finally:
        sys.path.remove(str(EXPERIMENT))


@pytest.fixture
def targets():
    return ([f"e_src:{i:03d}" for i in range(N_E)]
            + [f"h_src:{i:03d}" for i in range(N_H)])


def test_the_script_defers_every_deep_learning_import_until_after_validation():
    """Planning a 14 798-generation scan must not need a GPU node to do it.

    Checked on the source rather than on ``sys.modules``, which another test in
    the same session may already have populated: no module-level import of torch
    or transformers may exist, so ``--dry-run`` runs anywhere.
    """
    import ast

    source = (EXPERIMENT / "run_orthogonalized_population_scan.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "torch" not in top_level
    assert "transformers" not in top_level
    assert "routegate" in top_level


def test_importing_the_script_binds_no_tensor_library(script):
    assert not hasattr(script, "torch")
    assert not hasattr(script, "transformers")


# ------------------------------------------------------- 1. selecting states
def test_states_parse_singly_and_in_ranges(script):
    assert script.parse_states("7", 49) == [7]
    assert script.parse_states("1,8-10", 49) == [1, 8, 9, 10]
    assert script.parse_states("43-48", 49) == [43, 44, 45, 46, 47, 48]


def test_states_are_deduplicated_and_sorted(script):
    assert script.parse_states("14,8,8,9", 49) == [8, 9, 14]


def test_state_zero_is_refused_on_the_command_line(script):
    for spec in ("0", "0-7", "0,32"):
        with pytest.raises(ValidationError, match="state 0 is the embedding"):
            script.parse_states(spec, 49)


def test_a_state_outside_the_model_is_refused(script):
    for spec in ("49", "-1", "45-60"):
        with pytest.raises(ValidationError, match="outside"):
            script.parse_states(spec, 49)


def test_the_all_states_flag_is_the_forty_eight_block_outputs(script):
    assert list(CAUSAL_SCAN_STATES) == list(range(1, 49))
    assert len(CAUSAL_SCAN_STATES) == N_BLOCK_STATES
    assert 0 not in CAUSAL_SCAN_STATES


# -------------------------------------------------- 2. the work list it builds
def test_every_target_is_steered_exactly_once_at_every_scanned_state(script,
                                                                     targets):
    states = list(CAUSAL_SCAN_STATES)
    work = script.build_work_list(targets, states, [CONDITION_REPAIR],
                                  FROZEN_SCAN_STRENGTH)
    seen = {}
    for item in work:
        seen[(item["target_row_id"], item["state"])] = \
            seen.get((item["target_row_id"], item["state"]), 0) + 1
    assert len(seen) == len(targets) * len(states)
    assert set(seen.values()) == {1}
    assert {item["state"] for item in work} == set(states)


def test_the_baseline_is_generated_once_per_target_and_not_once_per_state(
        script, targets):
    work = script.build_work_list(targets, list(CAUSAL_SCAN_STATES),
                                  [CONDITION_BASELINE, CONDITION_REPAIR],
                                  FROZEN_SCAN_STRENGTH)
    baselines = [item for item in work if item["condition"] == CONDITION_BASELINE]
    assert len(baselines) == len(targets)
    assert sorted(item["target_row_id"] for item in baselines) == sorted(targets)
    assert {item["state"] for item in baselines} == {""}


def test_the_dry_run_counts_are_exactly_the_declared_budget(script, targets):
    work = script.build_work_list(targets, list(CAUSAL_SCAN_STATES),
                                  [CONDITION_BASELINE, CONDITION_REPAIR],
                                  FROZEN_SCAN_STRENGTH)
    counts = script.count_work(work)
    assert counts == {"intervention": EXPECTED_INTERVENTIONS,
                      "baseline": EXPECTED_BASELINES,
                      "total": EXPECTED_TOTAL}
    assert counts["intervention"] == 14496
    assert counts["baseline"] == 302
    assert counts["total"] == 14798


def test_state_zero_contributes_nothing_to_the_declared_budget(script, targets):
    work = script.build_work_list(targets, list(CAUSAL_SCAN_STATES),
                                  [CONDITION_BASELINE, CONDITION_REPAIR],
                                  FROZEN_SCAN_STRENGTH)
    assert all(item["state"] != 0 for item in work)
    assert script.count_work(work)["total"] == EXPECTED_TOTAL


def test_the_work_list_is_deterministic_and_target_major(script, targets):
    states = [1, 2, 3]
    first = script.build_work_list(targets, states,
                                   [CONDITION_BASELINE, CONDITION_REPAIR], 0.2)
    second = script.build_work_list(targets, states,
                                    [CONDITION_BASELINE, CONDITION_REPAIR], 0.2)
    assert first == second
    # Every target's generations are contiguous, so one capture forward serves
    # all of that target's states rather than one per generation.
    order = [item["target_row_id"] for item in first]
    runs = [row_id for row_id, _group in itertools.groupby(order)]
    assert len(runs) == len(set(runs)) == len(targets)
    assert [item["state"] for item in first[:4]] == ["", 1, 2, 3]


def test_the_work_key_matches_the_record_key(script, targets):
    work = script.build_work_list(targets[:2], [5], [CONDITION_REPAIR], 0.2)
    item = work[0]
    record = make_record(**base_record(target_row_id=item["target_row_id"],
                                       state=item["state"],
                                       condition=item["condition"],
                                       intervention_type=item["intervention_type"],
                                       intervention_strength=item[
                                           "intervention_strength"]))
    assert script.work_key(item) == record_key(record)
    assert KEY_FIELDS == ("target_row_id", "state", "condition",
                          "intervention_type", "intervention_strength")


# ------------------------------------------------------ 3. selecting targets
class FakeBank:
    """The parts of a frozen bank that target selection reads."""

    def __init__(self, labels):
        self.label_of_row = dict(labels)
        self.fold_of_row = {rid: i % 5 for i, rid in enumerate(sorted(labels))}

    def fold_of(self, row_id):
        return self.fold_of_row[row_id]


def table_for(labels):
    return {rid: {"prompt_text": "a prompt", "final_ebh": label,
                  "source_family": "a_family", "original_response": "an answer"}
            for rid, label in labels.items()}


def test_all_targets_is_every_e_and_h_row_and_no_b_row(script):
    labels = {"e:1": "E", "e:2": "E", "b:1": "B", "b:2": "B", "h:1": "H"}
    bank = FakeBank(labels)
    chosen = script.resolve_targets(bank, table_for(labels), [], True,
                                    script.TARGET_LABELS)
    assert chosen == ["e:1", "e:2", "h:1"]


def test_a_benign_row_offered_as_a_target_is_refused(script):
    labels = {"e:1": "E", "b:1": "B", "h:1": "H"}
    bank = FakeBank(labels)
    with pytest.raises(ValidationError, match="are class B"):
        script.resolve_targets(bank, table_for(labels), ["e:1", "b:1"], False,
                               script.TARGET_LABELS)


def test_a_target_outside_the_frozen_assignment_is_refused(script):
    labels = {"e:1": "E"}
    bank = FakeBank(labels)
    with pytest.raises(ValidationError, match="not in the frozen fold assignment"):
        script.resolve_targets(bank, table_for(labels), ["e:9"], False,
                               script.TARGET_LABELS)


def test_a_bank_labelling_a_row_differently_from_the_corpus_is_refused(script):
    labels = {"e:1": "E"}
    bank = FakeBank(labels)
    table = table_for({"e:1": "H"})
    with pytest.raises(ValidationError, match="labelled differently"):
        script.resolve_targets(bank, table, ["e:1"], False, script.TARGET_LABELS)


def test_no_target_at_all_is_refused(script):
    bank = FakeBank({"e:1": "E"})
    with pytest.raises(ValidationError, match="no target selected"):
        script.resolve_targets(bank, table_for({"e:1": "E"}), [], False,
                               script.TARGET_LABELS)


def test_the_target_classes_are_the_two_that_measure_repair_and_preservation(
        script):
    assert script.TARGET_LABELS == ("E", "H")


# ------------------------------------------------------------ 4. the record
def base_record(**overrides):
    record = {
        "run_id": "run", "created_utc": "2026-08-07T00:00:00Z",
        "target_row_id": "e:1", "target_label": "E", "target_fold": 2,
        "target_source_family": "a_family",
        "condition": CONDITION_REPAIR, "state": 17,
        "transformer_block_index": 16,
        "intervention_type": "orthogonalized_population_direction",
        "intervention_strength": 0.2, "direction_sha256": "a" * 64,
        "direction_fold": 2, "direction_status": "ok",
        "budget_reference": "block_input", "budget_reference_state": 16,
        "budget": 1.5, "reference_norm": 7.5, "raw_delta_norm": 1.0,
        "applied_delta_norm": 1.5,
        "model": "Qwen/Qwen3-30B-A3B", "model_revision": "ad44e777",
        "generation_settings": {}, "generation_config_sha256": "c" * 64,
        "prompt_token_count": 12, "completion_token_count": 40,
        "eos_reached": True, "hit_max_new_tokens": False,
        "elapsed_seconds": 1.0,
        "original_response": "the corpus response", "steered_response": "text",
        "output_status": "ok", "error_message": "", "reportable": True,
        "reportable_reason": "full run", "dataset_sha256": "d" * 64,
        "direction_bank_sha256": "b" * 64,
    }
    record.update(overrides)
    return record


def test_a_record_carries_the_target_fold_state_block_strength_and_direction():
    record = make_record(**base_record())
    assert list(record) == list(RECORD_FIELDS)
    for field in ("target_row_id", "target_fold", "state",
                  "transformer_block_index", "intervention_strength",
                  "direction_sha256"):
        assert field in record
    assert record["state"] == 17
    assert record["transformer_block_index"] == 16


def test_the_two_conditions_are_the_baseline_and_the_orthogonalised_repair():
    assert set(CONDITIONS) == {"baseline", "orthogonalized_population_repair"}


def test_an_intervention_record_must_name_a_transformer_block_state():
    for bad in ("", 0):
        with pytest.raises(SteerRecordError, match="transformer-block state"):
            make_record(**base_record(state=bad))


def test_an_intervention_record_must_name_its_direction():
    with pytest.raises(SteerRecordError, match="by hash"):
        make_record(**base_record(direction_sha256=""))


def test_a_failed_generation_may_have_no_direction_to_name():
    """Failing to resolve a direction is one of the ways a generation fails.

    The row still has to be written -- an unwritable failure would take the
    whole array task down with it -- so the hash is required of successes only.
    """
    record = make_record(**base_record(direction_sha256="", output_status="error",
                                       reportable=False,
                                       error_message="SteeringError: degenerate"))
    assert record["output_status"] == "error"
    assert record["direction_sha256"] == ""


def test_a_baseline_record_carries_no_state():
    record = make_record(**base_record(
        condition=CONDITION_BASELINE, state="", transformer_block_index="",
        intervention_type="none", intervention_strength=0.0,
        direction_sha256="", direction_fold="", direction_status="",
        budget_reference_state="", budget=0.0, reference_norm=0.0,
        raw_delta_norm=0.0, applied_delta_norm=0.0, output_status="baseline"))
    assert record["state"] == ""
    assert record["transformer_block_index"] == ""
    with pytest.raises(SteerRecordError, match="generated once per"):
        make_record(**base_record(condition=CONDITION_BASELINE, state=5,
                                  intervention_type="none",
                                  intervention_strength=0.0,
                                  output_status="baseline"))


def test_state_zero_is_not_a_transformer_block_in_the_record():
    assert transformer_block_of_state(0) == ""
    assert transformer_block_of_state("") == ""
    assert transformer_block_of_state(1) == 0
    assert transformer_block_of_state(48) == 47


def test_a_failed_generation_cannot_be_reported():
    with pytest.raises(SteerRecordError, match="cannot be marked reportable"):
        make_record(**base_record(output_status="error", reportable=True))


def test_an_unknown_field_is_refused():
    with pytest.raises(SteerRecordError, match="unknown record field"):
        make_record(**base_record(donor_row_id="b:1"))


def test_resuming_skips_exactly_what_is_done_and_retries_what_failed():
    done = make_record(**base_record(state=3))
    failed = make_record(**base_record(state=4, output_status="error",
                                       reportable=False,
                                       error_message="boom"))
    keys = completed_keys([done, failed])
    assert record_key(done) in keys
    assert record_key(failed) not in keys
    todo, skipped = plan_work([record_key(done), record_key(failed)], keys)
    assert todo == [record_key(failed)]
    assert skipped == [record_key(done)]


def test_a_file_mixing_two_banks_cannot_be_resumed_into():
    first = make_record(**base_record(state=3))
    second = make_record(**base_record(state=4,
                                       direction_bank_sha256="e" * 64))
    with pytest.raises(SteerRecordError, match="mixes 2 values"):
        verify_consistency([first, second])


def test_resuming_under_a_different_bank_is_refused():
    record = make_record(**base_record())
    with pytest.raises(SteerRecordError, match="direction_bank_sha256"):
        verify_consistency([record], direction_bank_sha256="f" * 64)


# --------------------------------------------------------- 5. the scheduler
@pytest.fixture(scope="module")
def submit_text():
    return SUBMIT_SCRIPT.read_text(encoding="utf-8")


def task_states(task, states_per_task=7, last_state=48):
    """The shell arithmetic, reproduced."""
    first = (task - 1) * states_per_task + 1
    last = min(first + states_per_task - 1, last_state)
    return list(range(first, last + 1))


def test_the_scheduler_declares_seven_array_tasks(submit_text):
    assert re.search(r"^#\$ -t 1-7$", submit_text, re.MULTILINE)


def test_the_scheduler_arithmetic_matches_the_declared_partition(submit_text):
    assert "STATES_PER_TASK=7" in submit_text
    assert "LAST_STATE=48" in submit_text
    assert "FIRST=$(( (TASK - 1) * STATES_PER_TASK + 1 ))" in submit_text
    expected = {1: (1, 7), 2: (8, 14), 3: (15, 21), 4: (22, 28), 5: (29, 35),
                6: (36, 42), 7: (43, 48)}
    for task, (first, last) in expected.items():
        states = task_states(task)
        assert states[0] == first and states[-1] == last


def test_the_seven_tasks_cover_states_one_to_forty_eight_exactly_once():
    covered = []
    for task in range(1, 8):
        covered.extend(task_states(task))
    assert sorted(covered) == list(range(1, 49))
    assert len(covered) == len(set(covered)) == 48
    assert 0 not in covered


def test_the_last_task_holds_six_states():
    assert len(task_states(7)) == 6
    assert all(len(task_states(t)) == 7 for t in range(1, 7))


def test_only_the_first_task_generates_baselines(submit_text):
    branch = re.search(r'if \[ "\$TASK" -eq 1 \]; then\s*\n\s*'
                       r'CONDITIONS="([^"]+)"\s*\n\s*else\s*\n\s*'
                       r'CONDITIONS="([^"]+)"', submit_text)
    assert branch, "the task-1 baseline branch is not in the expected shape"
    first_task, other_tasks = branch.group(1), branch.group(2)
    assert first_task == "baseline,orthogonalized_population_repair"
    assert other_tasks == "orthogonalized_population_repair"
    assert "baseline" not in other_tasks


def test_each_task_writes_to_its_own_directory(submit_text):
    assert 'OUTDIR="$SCAN_ROOT/scan/task_${TASK}"' in submit_text
    assert "--output-dir '${OUTDIR}'" in submit_text
    assert "--resume" in submit_text


def test_the_scheduler_pins_the_direction_bank_and_refuses_to_run_unpinned(
        submit_text):
    assert "--expect-bank-sha256 '${BANK_SHA256}'" in submit_text
    assert 'if [ -z "$BANK_SHA256" ]; then' in submit_text
    assert "Refusing to run unpinned" in submit_text


def test_the_scheduler_keeps_the_accepted_resource_request(submit_text):
    for line in ("#$ -l h_rt=36:00:00", "#$ -l mem=96G", "#$ -l gpu=4",
                 "#$ -l tmpfs=30G", "#$ -ac allow=L"):
        assert line in submit_text


def test_the_scheduler_runs_the_frozen_strength_and_budget_reference(submit_text):
    assert "--strength 0.2" in submit_text
    assert "--budget-reference block_input" in submit_text


def test_no_task_asks_for_state_zero(submit_text):
    assert "--states ${FIRST}-${LAST}" in submit_text
    assert all(0 not in task_states(task) for task in range(1, 8))


def test_the_scheduler_says_it_is_not_submitted(submit_text):
    """It is handed over, not launched: the script says so in its own header."""
    assert "NOT SUBMITTED" in submit_text


# --------------------------------------------------- 6. the frozen strength
def plan_args(**overrides):
    """The arguments ``run`` reads before it opens the direction bank."""
    args = {"config": REPO_ROOT / "configs" / "orthogonalized_population_scan.json",
            "max_new_tokens": None, "budget_reference": "block_input",
            "strength": FROZEN_SCAN_STRENGTH, "conditions": CONDITION_REPAIR,
            "all_block_states": True, "states": None}
    args.update(overrides)
    return argparse.Namespace(**args)


@pytest.mark.parametrize("strength", [0.15, 0.25, 1.0])
def test_a_strength_other_than_the_frozen_one_is_refused_during_the_scan(
        script, strength):
    with pytest.raises(ValidationError, match="not the frozen scan strength"):
        script.run(plan_args(strength=strength))


def test_the_refusal_says_why_the_strength_is_frozen_rather_than_optimal(script):
    with pytest.raises(ValidationError) as raised:
        script.run(plan_args(strength=0.3))
    message = str(raised.value)
    assert "pilot operating point" in message
    assert "not a measured optimum" in message
    assert "separate run" in message


def test_an_unknown_condition_is_refused(script):
    with pytest.raises(ValidationError, match="unknown condition"):
        script.run(plan_args(conditions="donor_patch"))


def test_the_declared_budget_matches_the_protocol_configuration():
    import json

    config = json.loads(
        (REPO_ROOT / "configs" / "orthogonalized_population_scan.json").read_text(
            encoding="utf-8"))
    budget = config["generation_budget"]
    assert budget["intervention"] == EXPECTED_INTERVENTIONS == 14496
    assert budget["baseline"] == EXPECTED_BASELINES == 302
    assert budget["total"] == EXPECTED_TOTAL == 14798
    assert budget["state_0_is_not_counted"] is True
    assert config["strength"]["value"] == FROZEN_SCAN_STRENGTH
    assert config["targets"]["per_state"] == N_TARGETS == 302
    assert config["states"]["primary_causal_layer_intervention_scan"] == \
        "states 1-48 only"
