"""The v5.1 verifier erratum: technical validation only, nothing scientific.

``verify_technical_smoke.py`` required ``eos_reached OR hit_max_new_tokens`` as
proof that generation terminated normally. That rule predates stop sequences.
v5 added a custom stopping criterion for tinyGSM8K, and when a declared stop
fires neither flag is set -- the boundary was reached, which is the intended
terminal mode -- so the verifier blocked three technically correct rows.

The v5 protocol pins the original verifier by digest, so it is left byte-for-
byte unchanged and the correction lives in ``verify_technical_smoke_v5_1.py``,
pinned here instead.

WHAT THIS ERRATUM DOES NOT DO
-----------------------------
It changes no generation source, dataset, prompt, parser, controller,
checkpoint, threshold or record. Not one token would be produced differently
under it, and not one answer would be scored differently. It corrects a check
that was refusing evidence it did not understand.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .finaleval import (
    ARMS, DS_SEALED, PREDECLARATION, RECORDS_FILE, arm_id, arm_output_dir)
from .triplets import sha256_text
from .validation import Checklist, ValidationError, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "routegate/verifier_erratum/1"
NAME = "routegate_verifier_erratum_v5_1"
VERSION = 1
DIGEST_FIELD = "record_sha256"
ERRATUM = REPO_ROOT / "manifests" / f"{NAME}.json"

APPLIES_TO = "routegate_final_evaluation_v5"
ORIGINAL_VERIFIER = "experiments/final_evaluation/verify_technical_smoke.py"
CORRECTED_VERIFIER = (
    "experiments/final_evaluation/verify_technical_smoke_v5_1.py")

REASON = (
    "verify_technical_smoke.py required eos_reached OR hit_max_new_tokens as "
    "proof of normal termination. That rule was written before stop sequences "
    "existed. v5 added a custom stopping criterion for tinyGSM8K; when a "
    "declared stop fires, neither flag is set, because generation ended at "
    "the boundary -- the intended terminal mode. The check therefore blocked "
    "three technically correct rows at "
    "tinygsm8k__baseline_terminated_normally: eos False cap False.")

CORRECTION = (
    "a record must evidence exactly one consistent terminal mode: EOS, "
    "max_new_tokens, or a declared custom stop. Custom-stop termination "
    "additionally requires stop_fired true, stop_reason 'stop_string', "
    "hit_max_new_tokens false, the record's stop_strings equal to the "
    "dataset's frozen list, and no fired stop or following continuation "
    "retained in the stored completion. A dataset that declares no stop "
    "sequences cannot claim custom-stop termination.")

SCOPE = {
    "changes_technical_validation_only": True,
    "changes_generation": False,
    "changes_scoring": False,
    "changes_datasets_prompts_or_parsers": False,
    "changes_controllers_checkpoints_or_thresholds": False,
    "modifies_any_existing_record": False,
    "modifies_the_v5_manifest": False,
    "original_verifier_left_unchanged": True,
}


class VerifierErratumError(ValidationError):
    """The erratum does not describe the files it claims to."""


#: The one official row each smoke invocation wrote (``--item-index 0``).
SMOKE_ITEM_INDEX = 0


def smoke_row_sha256(path: Path) -> str:
    """Digest of the smoke row: the first line of ``records.jsonl``, newline
    included.

    The erratum was frozen when each arm's records file held only its one-row
    smoke invocation, so the pinned digests are of that row. The full arm runs
    later appended the other 99 rows to the same append-only file, which
    changes the whole-file digest but not the smoke row's bytes. Whole-file
    integrity of a completed arm is pinned separately by ``arm_complete.json``.
    """
    with Path(path).open("rb") as handle:
        first = handle.readline()
    if not first.endswith(b"\n"):
        raise VerifierErratumError(f"{path}: no complete first record")
    if json.loads(first).get("item_index") != SMOKE_ITEM_INDEX:
        raise VerifierErratumError(
            f"{path}: first record is not the item_index "
            f"{SMOKE_ITEM_INDEX} smoke row")
    return hashlib.sha256(first).hexdigest()


def smoke_record_digests() -> Dict[str, str]:
    """The six v5 capability smoke records, by digest. Read, never written."""
    root = Path.home() / "routegate-data"
    out: Dict[str, str] = {}
    for dataset, condition in ARMS:
        if dataset == DS_SEALED:
            continue
        path = root / arm_output_dir(dataset, condition) / RECORDS_FILE
        out[arm_id(dataset, condition)] = (
            smoke_row_sha256(path) if path.is_file() else "")
    return out


def canonical(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def record_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_text(canonical(payload))


def build_erratum(record_digests: Optional[Mapping[str, str]] = None
                  ) -> Dict[str, Any]:
    """The committed payload. Deterministic given the files it pins."""
    digests = (dict(record_digests) if record_digests is not None
               else smoke_record_digests())
    record: Dict[str, Any] = {
        "schema": SCHEMA, "name": NAME, "version": VERSION,
        "applies_to": APPLIES_TO,
        "applies_to_manifest": f"manifests/{APPLIES_TO}.json",
        "applies_to_manifest_sha256": sha256_file(PREDECLARATION),
        "original_verifier": ORIGINAL_VERIFIER,
        "original_verifier_sha256": sha256_file(REPO_ROOT / ORIGINAL_VERIFIER),
        "corrected_verifier": CORRECTED_VERIFIER,
        "corrected_verifier_sha256": sha256_file(
            REPO_ROOT / CORRECTED_VERIFIER),
        "reason": REASON,
        "correction": CORRECTION,
        "smoke_record_sha256": dict(sorted(digests.items())),
        "smoke_records": len(digests),
        "scope": dict(SCOPE),
        "contains_prompt_response_gold_or_score": False,
    }
    record[DIGEST_FIELD] = record_sha256(record)
    return record


def verify_erratum(path: Path | str = ERRATUM,
                   checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The committed erratum still describes the files it pins."""
    checks = checks or Checklist(NAME)
    path = Path(path)
    checks.require("erratum_exists", path.is_file(), str(path))
    committed = json.loads(path.read_text(encoding="utf-8"))
    checks.require("schema", committed.get("schema") == SCHEMA,
                   f"{committed.get('schema')!r}")
    checks.require("record_digest",
                   record_sha256(committed) == committed.get(DIGEST_FIELD),
                   f"{committed.get(DIGEST_FIELD)}")
    checks.require("v5_manifest_unchanged",
                   committed.get("applies_to_manifest_sha256")
                   == sha256_file(PREDECLARATION),
                   "the frozen v5 manifest is the one this erratum applies to")
    checks.require("original_verifier_unchanged",
                   committed.get("original_verifier_sha256")
                   == sha256_file(REPO_ROOT / ORIGINAL_VERIFIER),
                   "the pinned verifier is byte-for-byte as v5 froze it")
    checks.require("corrected_verifier_matches",
                   committed.get("corrected_verifier_sha256")
                   == sha256_file(REPO_ROOT / CORRECTED_VERIFIER),
                   "the corrected verifier is the one pinned here")
    scope = committed.get("scope") or {}
    checks.require("scope_is_validation_only",
                   scope.get("changes_technical_validation_only") is True
                   and scope.get("changes_generation") is False
                   and scope.get("changes_scoring") is False
                   and scope.get("modifies_any_existing_record") is False
                   and scope.get("modifies_the_v5_manifest") is False,
                   "technical validation only")
    checks.require("six_smoke_records_pinned",
                   committed.get("smoke_records") == 6
                   and len(committed.get("smoke_record_sha256") or {}) == 6,
                   f"{committed.get('smoke_records')} record(s)")
    return {"path": str(path), "sha256": sha256_file(path),
            DIGEST_FIELD: record_sha256(committed)}


__all__ = [
    "APPLIES_TO", "CORRECTED_VERIFIER", "CORRECTION", "DIGEST_FIELD",
    "ERRATUM", "NAME", "ORIGINAL_VERIFIER", "REASON", "SCHEMA", "SCOPE",
    "SMOKE_ITEM_INDEX", "VERSION", "VerifierErratumError", "build_erratum",
    "canonical", "record_sha256", "smoke_record_digests", "smoke_row_sha256",
    "verify_erratum",
]
