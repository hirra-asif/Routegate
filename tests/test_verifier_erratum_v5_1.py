"""Stop-aware termination: the check that refused evidence it did not know.

``verify_technical_smoke.py`` required ``eos_reached OR hit_max_new_tokens``.
That rule predates stop sequences. When v5's tinyGSM8K stopping criterion fires
neither flag is set -- generation ended at the boundary, which is the intended
terminal mode -- so three technically correct rows were blocked at
``tinygsm8k__baseline_terminated_normally: eos False cap False``.

The v5 manifest pins the original verifier by digest, so it is untouched and
the correction lives in a new file pinned by its own erratum. Nothing about
generation or scoring moves.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from routegate.finaleval import (
    DATASET_SPEC, DS_GSM8K, DS_MMLU, GSM8K_STOP_STRINGS, PREDECLARATION,
    RECORD_SCHEMA)
from routegate.finalrunner import TELEMETRY_FIELDS
from routegate.validation import Checklist, ValidationError, sha256_file
from routegate.verifiererratum import (
    APPLIES_TO, CORRECTED_VERIFIER, DIGEST_FIELD, ERRATUM, ORIGINAL_VERIFIER,
    SCOPE, build_erratum, record_sha256, verify_erratum)

REPO = Path(__file__).resolve().parents[1]
NEW = REPO / CORRECTED_VERIFIER
OLD = REPO / ORIGINAL_VERIFIER


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixed = load(NEW, "_verifier_v51")
STOPS = list(GSM8K_STOP_STRINGS)


def rec(dataset=DS_GSM8K, **kw):
    base = {"stop_fired": True, "stop_reason": "stop_string",
            "hit_max_new_tokens": False, "eos_reached": False,
            "stop_strings": list(DATASET_SPEC[dataset].get("stop_strings")
                                 or [])}
    base.update(kw)
    return base


# ------------------------------------------------- the three terminal modes
def test_a_fired_declared_stop_is_a_valid_terminal_mode():
    """eos False, cap False, stop_fired True -- the exact blocked case."""
    assert fixed.terminal_mode(DS_GSM8K, rec()) == "stop_string"


def test_eos_termination_still_passes():
    assert fixed.terminal_mode(
        DS_MMLU, rec(DS_MMLU, stop_fired=False, stop_reason="eos",
                     eos_reached=True)) == "eos"


def test_token_cap_termination_still_passes():
    assert fixed.terminal_mode(
        DS_MMLU, rec(DS_MMLU, stop_fired=False, stop_reason="max_new_tokens",
                     hit_max_new_tokens=True)) == "max_new_tokens"


def test_no_terminal_mode_at_all_fails():
    with pytest.raises(ValidationError, match="no terminal mode"):
        fixed.terminal_mode(DS_GSM8K, rec(stop_fired=False, stop_reason=None))


# -------------------------------------------------- the ways it can be wrong
def test_an_undeclared_stop_reason_fails():
    with pytest.raises(ValidationError, match="not one of"):
        fixed.terminal_mode(DS_GSM8K, rec(stop_reason="something_else"))


def test_a_dataset_with_no_declared_stops_cannot_claim_one():
    """MMLU must not be able to pass by inventing a custom stop."""
    with pytest.raises(ValidationError, match="declares no stop sequences"):
        fixed.terminal_mode(DS_MMLU, rec(DS_MMLU, stop_fired=True,
                                         stop_reason="stop_string",
                                         stop_strings=["Question:"]))


def test_stop_reason_without_stop_fired_fails():
    with pytest.raises(ValidationError, match="stop_fired is false"):
        fixed.terminal_mode(DS_GSM8K, rec(stop_fired=False,
                                          stop_reason="stop_string"))


def test_stop_fired_with_a_mismatched_reason_fails():
    with pytest.raises(ValidationError, match="stop_reason is 'eos'"):
        fixed.terminal_mode(DS_GSM8K, rec(stop_reason="eos"))


def test_a_fired_stop_and_a_token_cap_cannot_both_be_true():
    with pytest.raises(ValidationError, match="cannot both be true"):
        fixed.terminal_mode(DS_GSM8K, rec(hit_max_new_tokens=True))


def test_stop_strings_must_equal_the_frozen_list():
    with pytest.raises(ValidationError, match="do not match the frozen list"):
        fixed.terminal_mode(DS_GSM8K, rec(stop_strings=["Question:"]))


# ------------------------------------ retained stop text is still blocked
def test_retained_stop_text_still_fails_the_boundary_check():
    with pytest.raises(ValidationError, match="stop_sequence_survives"):
        fixed.check_gsm8k_boundary(
            {"response_text": "a\n#### 4</s>", "stop_strings": STOPS,
             "stop_fired": True, "stop_reason": "stop_string",
             "hit_max_new_tokens": False}, Checklist("t"))


def test_text_after_a_stop_still_fails_the_boundary_check():
    with pytest.raises(
            ValidationError,
            match="stop_sequence_survives|later_question|exactly_one"):
        fixed.check_gsm8k_boundary(
            {"response_text": "a\n#### 4\n\nQuestion: b\n#### 5",
             "stop_strings": STOPS, "stop_fired": True,
             "stop_reason": "stop_string", "hit_max_new_tokens": False},
            Checklist("t"))


def test_a_bounded_row_passes_every_check():
    checks = Checklist("t")
    fixed.check_gsm8k_boundary(
        {"response_text": "work\n#### 42", "stop_strings": STOPS,
         "stop_fired": True, "stop_reason": "stop_string",
         "hit_max_new_tokens": False}, checks)
    assert checks.all_passed


# --------------------------------- check_common accepts the blocked shape
def _full(dataset, condition, **kw):
    base = {"schema": RECORD_SCHEMA, "condition": condition,
            "prompt_mode": DATASET_SPEC[dataset]["prompt_mode"],
            "status": "ok", "response_text": "x",
            "completion_token_count": 5,
            **{f: None for f in TELEMETRY_FIELDS}}
    base.update({"hooks_expected": 0, "hooks_fired": 0})
    base.update(rec(dataset))
    base.update(kw)
    return base


def test_check_common_accepts_the_previously_blocked_gsm8k_row():
    checks = Checklist("t")
    fixed.check_common(DS_GSM8K, "baseline",
                       _full(DS_GSM8K, "baseline"), checks)
    assert checks.all_passed


def test_the_original_verifier_would_have_blocked_that_same_row():
    """The defect, demonstrated against the pinned original."""
    old = load(OLD, "_verifier_v5_pinned")
    with pytest.raises(ValidationError, match="terminated_normally"):
        old.check_common(DS_GSM8K, "baseline",
                         _full(DS_GSM8K, "baseline"), Checklist("t"))


def test_mmlu_behaviour_is_unchanged_between_the_two_verifiers():
    old = load(OLD, "_verifier_v5_pinned2")
    row = _full(DS_MMLU, "baseline", stop_fired=False, stop_reason="eos",
                eos_reached=True, hit_max_new_tokens=False, stop_strings=[])
    for module in (old, fixed):
        checks = Checklist("t")
        module.check_common(DS_MMLU, "baseline", row, checks)
        assert checks.all_passed


# --------------------------------------------------- outcome blindness
def test_the_corrected_verifier_reads_no_gold_and_prints_no_value():
    source = NEW.read_text(encoding="utf-8")
    imports = "\n".join(l for l in source.splitlines()
                        if l.strip().startswith(("import ", "from ")))
    assert "capabilityscoring" not in imports
    for banned in ("mmlu_gold_letter", "gsm8k_gold_value", "score_arm",
                   "score_item", "answer_column", "paired_transitions"):
        assert banned not in source
    for line in source.splitlines():
        if "print(" in line:
            assert "{_prediction" not in line and "{prediction" not in line
    body = source.split("def terminal_mode(")[1].split("\ndef ")[0]
    assert "print(" not in body and "gold" not in body.lower()


# ------------------------------------------------------------ the erratum
def test_the_erratum_pins_both_verifiers_and_the_v5_manifest():
    committed = json.loads(ERRATUM.read_text(encoding="utf-8"))
    assert committed["applies_to"] == APPLIES_TO
    assert committed["applies_to_manifest_sha256"] == sha256_file(
        PREDECLARATION)
    assert committed["original_verifier_sha256"] == sha256_file(OLD)
    assert committed["corrected_verifier_sha256"] == sha256_file(NEW)
    assert committed["original_verifier_sha256"] != committed[
        "corrected_verifier_sha256"]
    assert record_sha256(committed) == committed[DIGEST_FIELD]


def test_the_erratum_pins_the_six_smoke_records():
    committed = json.loads(ERRATUM.read_text(encoding="utf-8"))
    digests = committed["smoke_record_sha256"]
    assert committed["smoke_records"] == 6 and len(digests) == 6
    assert set(digests) == {
        "tinymmlu__baseline", "tinymmlu__dynamic_raw",
        "tinymmlu__safety_first_routegate", "tinygsm8k__baseline",
        "tinygsm8k__dynamic_raw", "tinygsm8k__safety_first_routegate"}
    for arm, digest in digests.items():
        assert len(digest) == 64, arm


def test_the_erratum_changes_validation_only():
    assert SCOPE["changes_technical_validation_only"] is True
    for key in ("changes_generation", "changes_scoring",
                "changes_datasets_prompts_or_parsers",
                "changes_controllers_checkpoints_or_thresholds",
                "modifies_any_existing_record", "modifies_the_v5_manifest"):
        assert SCOPE[key] is False
    assert SCOPE["original_verifier_left_unchanged"] is True


def test_the_committed_erratum_verifies():
    checks = Checklist("t")
    verify_erratum(ERRATUM, checks)
    assert checks.all_passed


def test_the_v5_manifest_still_verifies_unchanged():
    from routegate.finaleval import verify_protocol
    checks = Checklist("t")
    verify_protocol(PREDECLARATION, checks)
    assert checks.all_passed, "the pinned v5 protocol must be untouched"


def test_the_smoke_records_on_disk_match_the_pinned_digests():
    from routegate.verifiererratum import smoke_record_digests
    committed = json.loads(ERRATUM.read_text(encoding="utf-8"))
    on_disk = smoke_record_digests()
    if not all(on_disk.values()):
        pytest.skip("the v5 smoke rows are not present on this machine")
    assert on_disk == committed["smoke_record_sha256"]


# ------------------------------------------- the scope of the pinned digest
def _records(tmp_path, rows):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b"".join(
        (json.dumps(r, sort_keys=True) + "\n").encode() for r in rows))
    return path


def test_the_pinned_digest_is_of_the_smoke_row_alone(tmp_path):
    """The erratum froze one-row files; the full run appended to them."""
    import hashlib
    from routegate.verifiererratum import smoke_row_sha256
    smoke = {"item_index": 0, "response_text": "x"}
    one_row = _records(tmp_path, [smoke])
    pinned = hashlib.sha256(one_row.read_bytes()).hexdigest()
    full = _records(tmp_path, [smoke] + [{"item_index": i}
                                         for i in range(1, 100)])
    assert smoke_row_sha256(full) == pinned
    assert hashlib.sha256(full.read_bytes()).hexdigest() != pinned


def test_an_altered_smoke_row_changes_the_digest(tmp_path):
    from routegate.verifiererratum import smoke_row_sha256
    a = smoke_row_sha256(_records(tmp_path, [{"item_index": 0, "t": "x"}]))
    b = smoke_row_sha256(_records(tmp_path, [{"item_index": 0, "t": "y"}]))
    assert a != b


def test_a_file_not_led_by_the_smoke_row_is_refused(tmp_path):
    from routegate.verifiererratum import (
        VerifierErratumError, smoke_row_sha256)
    with pytest.raises(VerifierErratumError):
        smoke_row_sha256(_records(tmp_path, [{"item_index": 1},
                                             {"item_index": 0}]))
    truncated = tmp_path / "records.jsonl"
    truncated.write_bytes(b'{"item_index": 0}')
    with pytest.raises(VerifierErratumError):
        smoke_row_sha256(truncated)
