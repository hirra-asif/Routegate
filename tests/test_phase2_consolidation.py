"""Consolidating a phase-2 run: exactly 144, and nothing else accepted.

Every shard here is invented in ``tmp_path``. Nothing loads a model, generates
a token, calls a service or reads the real data root.

The size question this file settles: **the consolidated GENERATION output is
144 records.** The 36 inherited reference records are joined afterwards to make
the 180-row analysis table; requiring them inside the shards would demand that
phase 2 regenerate the very cells it inherits.
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

from routegate import dualsteer as D                                # noqa: E402
from routegate.dualsteer import (                                   # noqa: E402
    BANK_STATES,
    BUDGET_REFERENCE,
    HISTORICAL_LAYERS,
    INTERVENTION_TYPE,
    DualSteerError,
    protocol_for,
)
from routegate.validation import sha256_text                        # noqa: E402

STAGE = REPO_ROOT / "experiments" / "dual_direction_controller"
CONFIG = REPO_ROOT / "configs" / "dual_direction_controller.json"

BANK = "d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb"
GENCFG = "cd6838ca06ff260fec85529c4690dee76c36bd17517747d34c84631f06c3bb7f"
DATASET = "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08"


def _stage():
    key = "_consolidate_causal_screen"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        key, STAGE / "consolidate_causal_screen.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


LOCKED = {
    "direction_bank_sha256": BANK, "generation_config_sha256": GENCFG,
    "model": "Qwen/Qwen3-30B-A3B",
    "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "tokenizer_config_sha256":
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "chat_template_sha256":
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "intervention_type": INTERVENTION_TYPE,
    "budget_reference": BUDGET_REFERENCE,
    "states": list(BANK_STATES), "historical_layers": list(HISTORICAL_LAYERS),
    "dataset_sha256": DATASET, "smoke": False,
    "protocol": D.PHASE2_PROTOCOL,
}


def _record(row_id: str, arm_id: str) -> dict:
    """One well-formed phase-2 record. Invented text, honest digests."""
    arm = D.ARM_BY_ID[arm_id]
    text = f"synthetic response for {row_id} under {arm_id}"
    return {
        "schema": D.RECORD_SCHEMA,
        "row_id": row_id, "row_class": "H", "arm_id": arm_id,
        "alpha_R": arm.alpha_repair, "alpha_S": arm.alpha_safety,
        "prompt_sha256": sha256_text(f"prompt::{row_id}"),
        "response_text": text, "response_sha256": sha256_text(text),
        "prompt_token_count": 32, "completion_token_count": 64,
        "eos_reached": True, "hit_max_new_tokens": False,
        "elapsed_seconds": 1.0, "status": "ok", "error_message": None,
        "hooks_fired": 3, "hooks_expected": 3,
        "applied_by_state": {str(state): True for state in BANK_STATES},
        "per_state": {str(state): {
            "repair_direction_sha256": "1" * 64,
            "safety_direction_sha256": "2" * 64} for state in BANK_STATES},
        "generation": {"max_new_tokens": 768, "do_sample": False},
        "created_utc": "2026-08-26T00:00:00Z",
        "environment": {"package": "routegate"},
        **LOCKED,
    }


def complete_records() -> list:
    """The whole phase-2 generation: 18 rows x 8 arms, and nothing else."""
    return [_record(row, arm.arm_id)
            for row in D.PHASE2_SELECTED_ROWS
            for arm in D.PHASE2_NEW_ARMS]


def write_shard(root: Path, records, index: int = 1) -> Path:
    directory = root / f"shard_{index}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "screen_records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n"
                for record in records), encoding="utf-8")
    (directory / "run_manifest.json").write_text(
        json.dumps({"protocol": D.PHASE2_PROTOCOL,
                    "shard": {"index": index, "count": 1},
                    **LOCKED}, sort_keys=True), encoding="utf-8")
    return directory


def run(root: Path, out: Path):
    return _stage().consolidate(root, out, CONFIG, shard_count=1, write=False,
                                protocol_name=D.PHASE2_PROTOCOL)


# ============================================================ the happy path
def test_a_complete_144_record_phase2_run_consolidates(tmp_path):
    write_shard(tmp_path, complete_records())
    result = run(tmp_path, tmp_path / "out")
    validation = result["validation"]
    assert validation["counts"]["total"] == 144
    assert validation["counts"]["by_class"]["H"] == 144
    assert validation["counts"]["by_class"].get("E", 0) == 0
    assert validation["counts"]["rows_by_class"]["H"] == 18
    assert set(validation["counts"]["by_arm"]) == {
        arm.arm_id for arm in D.PHASE2_NEW_ARMS}
    assert all(count == 18 for count in validation["counts"]["by_arm"].values())
    assert validation["complete_protocol"] is True
    assert validation["checklist"]["all_passed"] is True


def test_the_consolidated_output_is_144_not_180(tmp_path):
    """The 36 inherited references are joined later, not consolidated here."""
    write_shard(tmp_path, complete_records())
    result = run(tmp_path, tmp_path / "out")
    assert len(result["records"]) == 144
    protocol = protocol_for(D.PHASE2_PROTOCOL)
    assert protocol.expected_generations == 144
    assert protocol.reference_records() == 36
    assert protocol.analysis_table_records() == 180
    assert result["validation"]["counts"]["total"] != 180


def test_the_checklist_names_the_phase2_gates(tmp_path):
    write_shard(tmp_path, complete_records())
    result = run(tmp_path, tmp_path / "out")
    names = {entry["check"] for entry in result["validation"]["checklist"]["checks"]}
    assert "no_reference_arm_was_regenerated" in names
    assert "H_rows_match_the_bound_selection" in names
    assert "no_duplicate_records" in names
    assert "no_unexpected_arms" in names
    assert "total_records" in names
    assert f"{D.PHASE2_PROTOCOL}_arms_H" in names


# ================================================================= refusals
def test_one_missing_cell_is_refused(tmp_path):
    records = complete_records()
    dropped = records.pop()
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError,
                       match="H_records|total_records|every_arm_is_complete"):
        run(tmp_path, tmp_path / "out")
    assert len(records) == 143 and dropped["row_id"]


def test_a_missing_row_entirely_is_refused(tmp_path):
    victim = D.PHASE2_SELECTED_ROWS[0]
    write_shard(tmp_path, [r for r in complete_records()
                           if r["row_id"] != victim])
    with pytest.raises(DualSteerError, match="H_records|H_rows"):
        run(tmp_path, tmp_path / "out")


def test_a_duplicate_record_is_refused(tmp_path):
    records = complete_records()
    write_shard(tmp_path, records + [dict(records[0])])
    with pytest.raises(DualSteerError, match="no_duplicate_records"):
        run(tmp_path, tmp_path / "out")


def test_a_duplicate_across_two_shards_is_refused(tmp_path):
    records = complete_records()
    write_shard(tmp_path, records, index=1)
    write_shard(tmp_path, records[:1], index=2)
    with pytest.raises(DualSteerError, match="no_duplicate_records"):
        _stage().consolidate(tmp_path, tmp_path / "out", CONFIG,
                             shard_count=2, write=False,
                             protocol_name=D.PHASE2_PROTOCOL)


def test_an_extra_row_beyond_the_frozen_eighteen_is_refused(tmp_path):
    records = complete_records()
    records.append(_record("harmbench_text_test:not_in_the_selection",
                           D.PHASE2_NEW_ARMS[0].arm_id))
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError, match="H_records"):
        run(tmp_path, tmp_path / "out")


def test_a_substituted_row_is_refused_even_though_every_count_agrees(tmp_path):
    """The gap counts alone leave open: swap one bound row for another and the
    run is still 144 records over 18 rows with every arm at 18. Only the row
    identities catch it."""
    intruder = "harmbench_text_test:not_in_the_selection"
    victim = D.PHASE2_SELECTED_ROWS[0]
    records = [record if record["row_id"] != victim
               else _record(intruder, record["arm_id"])
               for record in complete_records()]
    assert len(records) == 144
    assert len({record["row_id"] for record in records}) == 18
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError,
                       match="H_rows_match_the_bound_selection"):
        run(tmp_path, tmp_path / "out")


def test_the_bound_selection_gate_names_what_is_wrong(tmp_path):
    intruder = "harmbench_text_test:not_in_the_selection"
    victim = D.PHASE2_SELECTED_ROWS[0]
    records = [record if record["row_id"] != victim
               else _record(intruder, record["arm_id"])
               for record in complete_records()]
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError) as excinfo:
        run(tmp_path, tmp_path / "out")
    message = str(excinfo.value)
    assert intruder in message and victim in message


def test_an_arm_from_another_protocol_is_refused(tmp_path):
    records = complete_records()
    records.append(_record(D.PHASE2_SELECTED_ROWS[0], "repair_005"))
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError, match="no_unexpected_arms"):
        run(tmp_path, tmp_path / "out")


@pytest.mark.parametrize("arm_id", ["baseline", "repair_015"])
def test_an_inherited_reference_arm_in_a_phase2_shard_is_refused(
        tmp_path, arm_id):
    """The exact failure "reused without regeneration" exists to prevent: the
    immutable phase-1 record for that cell being shadowed by a fresh one."""
    records = complete_records()
    records.append(_record(D.PHASE2_SELECTED_ROWS[0], arm_id))
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError,
                       match="no_reference_arm_was_regenerated"):
        run(tmp_path, tmp_path / "out")


def test_the_reference_arm_refusal_names_both_inherited_arms(tmp_path):
    records = complete_records()
    records.append(_record(D.PHASE2_SELECTED_ROWS[0], "baseline"))
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError) as excinfo:
        run(tmp_path, tmp_path / "out")
    message = str(excinfo.value)
    assert "baseline" in message and "repair_015" in message
    assert "INHERITS" in message


def test_a_shard_recorded_under_another_protocol_is_refused(tmp_path):
    records = [dict(record, protocol="focused_direction_validation_v1")
               for record in complete_records()]
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError):
        run(tmp_path, tmp_path / "out")


def test_a_record_whose_digest_does_not_recompute_is_refused(tmp_path):
    records = complete_records()
    records[0] = dict(records[0], response_sha256="0" * 64)
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError, match="response_digests_recompute"):
        run(tmp_path, tmp_path / "out")


def test_a_record_with_the_wrong_coefficients_is_refused(tmp_path):
    records = complete_records()
    records[0] = dict(records[0], alpha_S=0.99)
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError, match="coefficients_match_their_arm"):
        run(tmp_path, tmp_path / "out")


def test_a_smoke_record_may_not_complete_the_protocol(tmp_path):
    records = [dict(record, smoke=True) for record in complete_records()]
    write_shard(tmp_path, records)
    with pytest.raises(DualSteerError, match="not_a_smoke_test|smoke"):
        run(tmp_path, tmp_path / "out")


# ================================================== phase 1 is not disturbed
def test_the_phase1_protocol_still_consolidates_at_its_own_size():
    """The protocol-aware counts must not have moved phase 1's expectations."""
    focused = protocol_for(D.FOCUSED_PROTOCOL)
    assert focused.expected_by_class() == {"E": 445, "H": 426}
    assert focused.expected_rows("E") == 89 and focused.expected_rows("H") == 213
    exploratory = protocol_for(D.EXPLORATORY_PROTOCOL)
    assert exploratory.expected_by_class() == {"E": 445, "H": 2769}
