"""The output record: one row per generation, and the rules for resuming.

Every generation this stage performs writes exactly one JSON line carrying what
it would take to argue about it later: which target, which donor, which state,
which intervention at which strength, which model revision and generation
settings, the original response, the patched response, how it ended, and whether
it may be reported at all.

Resuming is the reason the schema is strict. A discovery scan over 49 states is
long enough that it will be interrupted, and the wrong way to resume is to
append to whatever is there. So:

* a record is keyed by (target, donor, state, intervention, strength,
  condition), and a key already present is skipped rather than repeated;
* every record carries the configuration fingerprint, the donor-manifest hash
  and the dataset hash it was produced under, and a resume that disagrees with
  any of them stops rather than mixing two runs in one file;
* a trailing partial line -- what an interrupted write leaves behind -- is a
  failure, not something to parse around.

``reportable`` is carried per record rather than inferred from the directory
name. A smoke run, a dry run, or a generation whose intervention never applied
is not reportable, and it says so in its own row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .validation import ValidationError

RECORD_SCHEMA = "routegate/patch_record/1"

# The execution conditions. "baseline" is the untouched run, "self_patch" is the
# zero-delta control, "donor_patch" is the B-donor-to-E-target intervention.
CONDITIONS = ("baseline", "self_patch", "donor_patch")

OUTPUT_STATUSES = (
    "ok",              # the generation ran and the plan did what it declared
    "zero_delta",      # nothing was applied because the delta had no direction
    "baseline",        # no intervention was requested
    "error",           # the generation or the plan failed; see error_message
)

RECORD_FIELDS: Tuple[str, ...] = (
    "schema",
    "run_id",
    "created_utc",
    # what was patched, and into what
    "target_row_id",
    "target_label",
    "donor_row_id",
    "donor_label",
    "donor_match_tier",
    "donor_match_exact",
    "condition",
    "state",
    "historical_block_index",
    "intervention_type",
    "intervention_strength",
    "budget_reference_state",
    "budget",
    "reference_norm",
    "raw_delta_norm",
    "applied_delta_norm",
    # what produced the text
    "model",
    "model_revision",
    "generation_settings",
    "generation_config_sha256",
    "prompt_token_count",
    "completion_token_count",
    "eos_reached",
    "hit_max_new_tokens",
    "elapsed_seconds",
    # the text itself
    "original_response",
    "patched_response",
    # provenance and standing
    "output_status",
    "error_message",
    "reportable",
    "reportable_reason",
    "dataset_sha256",
    "donor_manifest_sha256",
)

KEY_FIELDS: Tuple[str, ...] = (
    "target_row_id", "donor_row_id", "state", "intervention_type",
    "intervention_strength", "condition",
)

# Fields that must agree across every record in one output file.
RUN_LOCKED_FIELDS: Tuple[str, ...] = (
    "generation_config_sha256", "donor_manifest_sha256", "dataset_sha256",
)


class RecordError(ValidationError):
    """An output record is missing, malformed or inconsistent."""


def make_record(**values: Any) -> Dict[str, Any]:
    """Build one record, refusing anything that is not the declared schema."""
    record: Dict[str, Any] = {"schema": RECORD_SCHEMA}
    record.update(values)
    record.setdefault("error_message", "")
    unknown = sorted(set(record) - set(RECORD_FIELDS))
    if unknown:
        raise RecordError(f"unknown record field(s): {unknown}")
    missing = [f for f in RECORD_FIELDS if f not in record]
    if missing:
        raise RecordError(f"record is missing field(s): {missing}")
    if record["condition"] not in CONDITIONS:
        raise RecordError(
            f"unknown condition {record['condition']!r}; valid: {list(CONDITIONS)}")
    if record["output_status"] not in OUTPUT_STATUSES:
        raise RecordError(
            f"unknown output status {record['output_status']!r}; valid: "
            f"{list(OUTPUT_STATUSES)}")
    if not isinstance(record["reportable"], bool):
        raise RecordError("reportable must be a boolean")
    if record["reportable"] and record["output_status"] == "error":
        raise RecordError("a failed generation cannot be marked reportable")
    return {field: record[field] for field in RECORD_FIELDS}


def record_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """What makes a generation the same generation, for resume purposes."""
    missing = [f for f in KEY_FIELDS if f not in record]
    if missing:
        raise RecordError(f"cannot key a record missing {missing}")
    return tuple(record[field] for field in KEY_FIELDS)


def append_record(path: Path | str, record: Dict[str, Any]) -> None:
    """Append one complete line, flushed, so an interruption truncates at most one."""
    line = json.dumps(record, ensure_ascii=False, sort_keys=False)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def read_records(path: Path | str, strict: bool = True) -> List[Dict[str, Any]]:
    """Every record in an output file.

    A line that does not parse, or that does not carry the schema, is a failure:
    it is the signature of an interrupted write, and silently discarding it would
    let a resumed run believe a generation completed when it did not.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                if not strict:
                    continue
                raise RecordError(
                    f"{path}:{number} is not valid JSON ({exc.msg}). An "
                    f"interrupted run leaves a partial final line; repair or "
                    f"remove it before resuming, rather than resuming over it."
                ) from exc
            if record.get("schema") != RECORD_SCHEMA:
                raise RecordError(
                    f"{path}:{number} declares schema {record.get('schema')!r}, "
                    f"expected {RECORD_SCHEMA!r}")
            missing = [f for f in RECORD_FIELDS if f not in record]
            if missing:
                raise RecordError(f"{path}:{number} is missing field(s) {missing}")
            records.append(record)
    return records


def completed_keys(records: Sequence[Dict[str, Any]]) -> set:
    """The keys already done, so a resumed run can skip exactly those."""
    keys = set()
    for record in records:
        if record.get("output_status") == "error":
            continue          # a failed generation is retried, not skipped
        keys.add(record_key(record))
    return keys


def verify_consistency(records: Sequence[Dict[str, Any]],
                       generation_config_sha256: Optional[str] = None,
                       donor_manifest_sha256: Optional[str] = None,
                       dataset_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Refuse to resume into a file produced under different conditions.

    Two runs with different settings in one file cannot be separated afterwards
    -- the rows look identical. This is the check that keeps that from happening
    quietly.
    """
    if not records:
        return {"records": 0, "duplicate_keys": 0}

    expected = {
        "generation_config_sha256": generation_config_sha256,
        "donor_manifest_sha256": donor_manifest_sha256,
        "dataset_sha256": dataset_sha256,
    }
    for field in RUN_LOCKED_FIELDS:
        found = sorted({str(r.get(field, "")) for r in records})
        if len(found) > 1:
            raise RecordError(
                f"the output file mixes {len(found)} values of {field}: {found}. "
                f"Two runs under different conditions cannot share one file.")
        want = expected.get(field)
        if want is not None and found and found[0] != want:
            raise RecordError(
                f"{field} in the existing output is {found[0]}, but this run has "
                f"{want}. Resuming would append rows that are not comparable to "
                f"the ones already there.")

    seen: Dict[Tuple[Any, ...], int] = {}
    duplicates = 0
    for record in records:
        key = record_key(record)
        if key in seen:
            duplicates += 1
        seen[key] = seen.get(key, 0) + 1
    if duplicates:
        raise RecordError(
            f"the output file already contains {duplicates} duplicate "
            f"generation key(s); a resumed run must never have written one")

    statuses: Dict[str, int] = {}
    for record in records:
        status = str(record.get("output_status"))
        statuses[status] = statuses.get(status, 0) + 1
    return {"records": len(records), "duplicate_keys": duplicates,
            "by_status": statuses,
            "reportable": sum(1 for r in records if r.get("reportable"))}


def plan_work(requested: Iterable[Tuple[Any, ...]],
              done: set) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
    """(still to do, already done) over requested generation keys, in order."""
    todo: List[Tuple[Any, ...]] = []
    skipped: List[Tuple[Any, ...]] = []
    for key in requested:
        (skipped if key in done else todo).append(key)
    return todo, skipped
