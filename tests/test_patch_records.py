"""The output record and the resume rules.

A discovery scan over 49 states will be interrupted. These tests pin what
happens next: a completed generation is never repeated, a failed one is retried,
a half-written line is a failure rather than something to parse around, and a
file that was produced under different settings is never appended to.

Temporary files only. No model, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from routegate.patchrecord import (
    CONDITIONS,
    KEY_FIELDS,
    OUTPUT_STATUSES,
    RECORD_FIELDS,
    RECORD_SCHEMA,
    RecordError,
    append_record,
    completed_keys,
    make_record,
    plan_work,
    read_records,
    record_key,
    verify_consistency,
)
from routegate.runtime import GenerationConfig

BASE = {
    "run_id": "20260806T120000Z",
    "created_utc": "2026-08-06T12:00:00Z",
    "target_row_id": "e:001",
    "target_label": "E",
    "donor_row_id": "b:002",
    "donor_label": "B",
    "donor_match_tier": "source_topic_task",
    "donor_match_exact": True,
    "condition": "donor_patch",
    "state": 32,
    "historical_block_index": 31,
    "intervention_type": "donor_direction",
    "intervention_strength": 0.2,
    "budget_reference_state": 31,
    "budget": 12.5,
    "reference_norm": 62.5,
    "raw_delta_norm": 40.0,
    "applied_delta_norm": 12.5,
    "model": "Qwen/Qwen3-30B-A3B",
    "model_revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "generation_settings": {"do_sample": False, "max_new_tokens": 768},
    "generation_config_sha256": "cfg-1",
    "prompt_token_count": 41,
    "completion_token_count": 120,
    "eos_reached": True,
    "hit_max_new_tokens": False,
    "elapsed_seconds": 4.2,
    "original_response": "I can't help with that.",
    "patched_response": "Here is how you can do it safely: ...",
    "output_status": "ok",
    "error_message": "",
    "reportable": True,
    "reportable_reason": "full run under the frozen configuration and manifest",
    "dataset_sha256": "data-1",
    "donor_manifest_sha256": "manifest-1",
}


def make(**overrides):
    values = {k: v for k, v in BASE.items() if k != "schema"}
    values.update(overrides)
    return make_record(**values)


# --------------------------------------------------------- 1. output schema
def test_a_record_carries_every_required_field():
    record = make()
    assert record["schema"] == RECORD_SCHEMA
    for field in ("target_row_id", "donor_row_id", "target_label", "state",
                  "intervention_type", "intervention_strength",
                  "model_revision", "generation_settings", "original_response",
                  "patched_response", "output_status", "reportable"):
        assert field in record
    assert list(record) == list(RECORD_FIELDS)


def test_the_record_names_both_state_numberings():
    record = make(state=32, historical_block_index=31)
    assert record["state"] == 32
    assert record["historical_block_index"] == 31


def test_a_missing_field_is_refused():
    values = {k: v for k, v in BASE.items()}
    del values["patched_response"]
    with pytest.raises(RecordError, match="missing field"):
        make_record(**values)


def test_an_unknown_field_is_refused():
    with pytest.raises(RecordError, match="unknown record field"):
        make(judge_verdict="repaired")


def test_an_unknown_condition_is_refused():
    with pytest.raises(RecordError, match="unknown condition"):
        make(condition="mystery")
    assert set(CONDITIONS) == {"baseline", "self_patch", "donor_patch"}


def test_an_unknown_status_is_refused():
    with pytest.raises(RecordError, match="unknown output status"):
        make(output_status="probably_fine")
    assert set(OUTPUT_STATUSES) == {"ok", "zero_delta", "baseline", "error"}


def test_a_failed_generation_cannot_be_reportable():
    with pytest.raises(RecordError, match="cannot be marked reportable"):
        make(output_status="error", error_message="boom", reportable=True)
    assert make(output_status="error", error_message="boom",
                reportable=False)["reportable"] is False


def test_reportable_must_be_a_boolean():
    with pytest.raises(RecordError, match="boolean"):
        make(reportable="yes")


def test_the_key_is_what_makes_a_generation_the_same_generation():
    assert KEY_FIELDS == ("target_row_id", "donor_row_id", "state",
                          "intervention_type", "intervention_strength",
                          "condition")
    assert record_key(make()) == ("e:001", "b:002", 32, "donor_direction",
                                  0.2, "donor_patch")
    assert record_key(make(state=33)) != record_key(make(state=32))
    assert record_key(make(intervention_strength=0.15)) != record_key(make())


def test_a_baseline_record_has_no_state(tmp_path):
    record = make(condition="baseline", state="", historical_block_index="",
                  intervention_type="none", intervention_strength=0.0,
                  donor_row_id="", donor_label="", donor_match_tier="",
                  donor_match_exact=False, output_status="baseline",
                  budget_reference_state="")
    assert record["state"] == ""
    assert record_key(record)[2] == ""


# ------------------------------------------------------------- 2. round trip
def test_records_round_trip_through_the_file(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make())
    append_record(path, make(state=33, historical_block_index=32))

    records = read_records(path)
    assert len(records) == 2
    assert [r["state"] for r in records] == [32, 33]
    assert records[0]["generation_settings"]["max_new_tokens"] == 768


def test_reading_a_file_that_does_not_exist_yet_is_empty(tmp_path):
    assert read_records(tmp_path / "nothing.jsonl") == []


def test_blank_lines_are_tolerated(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n")
    assert len(read_records(path)) == 1


# ----------------------------------- 3. resuming without duplicating outputs
def test_a_completed_generation_is_not_repeated(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make(state=32))
    append_record(path, make(state=33, historical_block_index=32))

    done = completed_keys(read_records(path))
    requested = [record_key(make(state=s, historical_block_index=s - 1))
                 for s in (32, 33, 34)]
    todo, skipped = plan_work(requested, done)

    assert len(skipped) == 2
    assert len(todo) == 1
    assert todo[0][2] == 34


def test_a_failed_generation_is_retried_not_skipped(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make(state=32, output_status="error",
                             error_message="cuda hiccup", reportable=False,
                             patched_response=""))
    done = completed_keys(read_records(path))
    todo, skipped = plan_work([record_key(make(state=32))], done)
    assert len(todo) == 1
    assert skipped == []


def test_work_order_is_preserved(tmp_path):
    requested = [record_key(make(state=s, historical_block_index=s - 1))
                 for s in (1, 2, 3, 4)]
    todo, _ = plan_work(requested, {requested[1]})
    assert [key[2] for key in todo] == [1, 3, 4]


def test_nothing_left_to_do_is_an_empty_plan():
    key = record_key(make())
    todo, skipped = plan_work([key], {key})
    assert todo == []
    assert skipped == [key]


# --------------------------- 4. failure on incomplete or inconsistent output
def test_a_truncated_final_line_is_a_failure(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"schema": "routegate/patch_record/1", "target_row')

    with pytest.raises(RecordError, match="not valid JSON"):
        read_records(path)


def test_a_record_missing_fields_on_disk_is_a_failure(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    partial = {"schema": RECORD_SCHEMA, "target_row_id": "e:001"}
    path.write_text(json.dumps(partial) + "\n", encoding="utf-8")
    with pytest.raises(RecordError, match="missing field"):
        read_records(path)


def test_a_foreign_schema_on_disk_is_a_failure(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    record = dict(make())
    record["schema"] = "someone/else/1"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RecordError, match="declares schema"):
        read_records(path)


def test_a_file_that_mixes_two_configurations_is_a_failure(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make())
    append_record(path, make(state=33, historical_block_index=32,
                             generation_config_sha256="cfg-2"))
    with pytest.raises(RecordError, match="mixes"):
        verify_consistency(read_records(path))


def test_resuming_with_a_different_configuration_is_refused(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make())
    with pytest.raises(RecordError, match="not comparable"):
        verify_consistency(read_records(path),
                           generation_config_sha256="cfg-different")


def test_resuming_against_a_different_manifest_is_refused(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make())
    with pytest.raises(RecordError, match="not comparable"):
        verify_consistency(read_records(path),
                           donor_manifest_sha256="manifest-2")


def test_resuming_against_a_different_dataset_is_refused(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make())
    with pytest.raises(RecordError, match="not comparable"):
        verify_consistency(read_records(path), dataset_sha256="data-2")


def test_a_duplicated_generation_key_is_a_failure(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make())
    append_record(path, make())
    with pytest.raises(RecordError, match="duplicate"):
        verify_consistency(read_records(path))


def test_a_consistent_file_summarises_cleanly(tmp_path):
    path = tmp_path / "patch_records.jsonl"
    append_record(path, make(state=32))
    append_record(path, make(state=33, historical_block_index=32,
                             output_status="zero_delta"))
    summary = verify_consistency(read_records(path),
                                 generation_config_sha256="cfg-1",
                                 donor_manifest_sha256="manifest-1",
                                 dataset_sha256="data-1")
    assert summary["records"] == 2
    assert summary["duplicate_keys"] == 0
    assert summary["by_status"] == {"ok": 1, "zero_delta": 1}
    assert summary["reportable"] == 2


def test_an_empty_history_verifies(tmp_path):
    assert verify_consistency([])["records"] == 0


# ------------------------------------------- 5. the record pins the settings
def test_the_configuration_fingerprint_travels_with_every_record():
    config = GenerationConfig()
    record = make(generation_config_sha256=config.fingerprint(),
                  generation_settings=config.as_dict())
    assert record["generation_settings"]["enable_thinking"] is False
    assert record["generation_settings"]["do_sample"] is False
    assert record["generation_settings"]["batch_size"] == 1
    assert record["model_revision"] == config.revision
    assert record["generation_config_sha256"] == config.fingerprint()
