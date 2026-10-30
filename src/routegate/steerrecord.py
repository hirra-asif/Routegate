"""The per-generation record for the population steering scan, and its resume rules.

One JSON line per generation, carrying what an argument about it would need: the
target and its class, the fold whose direction was used, the state and the
transformer block that state is the output of, the strength, the SHA256 of the
exact direction that was added, how big the edit was against the target's own
norm, the response, and whether the row may be reported.

Why this is not ``patchrecord``. That module is the record of the donor-matched
patching stage: its condition vocabulary, its key and its locked fields are
fixed by that stage's tests, and its rows carry a donor and a donor-manifest
hash. A steering row has no donor and does have a fold and a direction hash.
Widening ``patchrecord`` to hold both would change the field list every donor
record is validated against, and reading an existing donor file would start
failing on fields that stage never wrote. So the two schemas are kept apart, and
the donor stage is left exactly as it was tested.

The resume discipline is the same one, for the same reason: a 14 798-generation
scan will be interrupted.

* a record is keyed by (target, state, condition, intervention type, strength),
  and a key already present is skipped rather than repeated;
* every record carries the generation-configuration fingerprint, the direction
  bank hash and the dataset hash it was produced under, and a resume that
  disagrees with any of them stops rather than mixing two runs in one file;
* a trailing partial line -- what an interrupted write leaves behind -- is a
  failure, not something to parse around;
* each array task writes its own file, so no two processes ever append to one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .validation import ValidationError

RECORD_SCHEMA = "routegate/steer_record/1"

# The two conditions of the discovery scan. There is no donor condition here and
# no self-patch: the zero-delta control belongs to the donor stage, and the
# controls this stage will run -- raw non-orthogonalised, equal-norm random,
# B preservation, strength sensitivity -- are the later, focused confirmation
# stage at a selected state, not extra arms multiplied across all 48 blocks.
CONDITION_BASELINE = "baseline"
CONDITION_REPAIR = "orthogonalized_population_repair"
CONDITIONS: Tuple[str, ...] = (CONDITION_BASELINE, CONDITION_REPAIR)

OUTPUT_STATUSES: Tuple[str, ...] = (
    "ok",              # the generation ran and the direction was applied
    "zero_delta",      # nothing was applied because the edit had no magnitude
    "baseline",        # no intervention was requested
    "error",           # the generation or the plan failed; see error_message
)

RECORD_FIELDS: Tuple[str, ...] = (
    "schema",
    "run_id",
    "created_utc",
    # who was steered
    "target_row_id",
    "target_label",
    "target_fold",
    "target_source_family",
    # what was done to them
    "condition",
    "state",
    "transformer_block_index",
    "intervention_type",
    "intervention_strength",
    "direction_sha256",
    "direction_fold",
    "direction_status",
    # how big the edit was
    "budget_reference",
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
    "steered_response",
    # provenance and standing
    "output_status",
    "error_message",
    "reportable",
    "reportable_reason",
    "dataset_sha256",
    "direction_bank_sha256",
)

KEY_FIELDS: Tuple[str, ...] = (
    "target_row_id", "state", "condition", "intervention_type",
    "intervention_strength",
)

# Fields that must agree across every record in one output file.
RUN_LOCKED_FIELDS: Tuple[str, ...] = (
    "generation_config_sha256", "direction_bank_sha256", "dataset_sha256",
)


class SteerRecordError(ValidationError):
    """An output record is missing, malformed or inconsistent."""


def transformer_block_of_state(state: Any) -> Any:
    """The zero-based transformer block whose output is ``state``.

    State 0 is the embedding output and is the output of no block, so it gets an
    empty value rather than -1: this stage writes "not applicable" as a blank,
    never as a number that could be sorted alongside real block indices. The
    baseline's blank state is treated the same way.
    """
    if state == "" or state is None:
        return ""
    state = int(state)
    if state < 0:
        raise SteerRecordError(f"state {state} is negative")
    return "" if state == 0 else state - 1


def make_record(**values: Any) -> Dict[str, Any]:
    """Build one record, refusing anything that is not the declared schema."""
    record: Dict[str, Any] = {"schema": RECORD_SCHEMA}
    record.update(values)
    record.setdefault("error_message", "")
    unknown = sorted(set(record) - set(RECORD_FIELDS))
    if unknown:
        raise SteerRecordError(f"unknown record field(s): {unknown}")
    missing = [f for f in RECORD_FIELDS if f not in record]
    if missing:
        raise SteerRecordError(f"record is missing field(s): {missing}")
    if record["condition"] not in CONDITIONS:
        raise SteerRecordError(
            f"unknown condition {record['condition']!r}; valid: {list(CONDITIONS)}")
    if record["output_status"] not in OUTPUT_STATUSES:
        raise SteerRecordError(
            f"unknown output status {record['output_status']!r}; valid: "
            f"{list(OUTPUT_STATUSES)}")
    if not isinstance(record["reportable"], bool):
        raise SteerRecordError("reportable must be a boolean")
    if record["reportable"] and record["output_status"] == "error":
        raise SteerRecordError("a failed generation cannot be marked reportable")
    if record["condition"] == CONDITION_REPAIR:
        if record["state"] in ("", 0, None):
            raise SteerRecordError(
                "an intervention record must name a transformer-block state; "
                "state 0 is the embedding output and is not scanned")
        # A generation that succeeded must say which direction it added. One
        # that failed may not have got as far as resolving one -- failing to
        # find a usable direction is itself a way for a generation to fail --
        # and the row still has to be written, so the failure is recorded rather
        # than crashing the task that hit it.
        if not record["direction_sha256"] and record["output_status"] != "error":
            raise SteerRecordError(
                "an intervention record must name the direction that was "
                "applied, by hash")
    if record["condition"] == CONDITION_BASELINE and record["state"] != "":
        raise SteerRecordError(
            "the baseline does not depend on a state and is generated once per "
            "target; its state field is blank")
    return {field: record[field] for field in RECORD_FIELDS}


def record_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """What makes a generation the same generation, for resume purposes."""
    missing = [f for f in KEY_FIELDS if f not in record]
    if missing:
        raise SteerRecordError(f"cannot key a record missing {missing}")
    return tuple(record[field] for field in KEY_FIELDS)


def append_record(path: Path | str, record: Dict[str, Any]) -> None:
    """Append one complete line, flushed, so an interruption truncates at most one."""
    line = json.dumps(record, ensure_ascii=False, sort_keys=False)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def read_records(path: Path | str, strict: bool = True) -> List[Dict[str, Any]]:
    """Every record in an output file, refusing a partial or foreign line."""
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
                raise SteerRecordError(
                    f"{path}:{number} is not valid JSON ({exc.msg}). An "
                    f"interrupted run leaves a partial final line; repair or "
                    f"remove it before resuming, rather than resuming over it."
                ) from exc
            if record.get("schema") != RECORD_SCHEMA:
                raise SteerRecordError(
                    f"{path}:{number} declares schema {record.get('schema')!r}, "
                    f"expected {RECORD_SCHEMA!r}")
            missing = [f for f in RECORD_FIELDS if f not in record]
            if missing:
                raise SteerRecordError(
                    f"{path}:{number} is missing field(s) {missing}")
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
                       direction_bank_sha256: Optional[str] = None,
                       dataset_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Refuse to resume into a file produced under different conditions."""
    if not records:
        return {"records": 0, "duplicate_keys": 0}

    expected = {
        "generation_config_sha256": generation_config_sha256,
        "direction_bank_sha256": direction_bank_sha256,
        "dataset_sha256": dataset_sha256,
    }
    for field in RUN_LOCKED_FIELDS:
        found = sorted({str(r.get(field, "")) for r in records})
        if len(found) > 1:
            raise SteerRecordError(
                f"the output file mixes {len(found)} values of {field}: {found}. "
                f"Two runs under different conditions cannot share one file.")
        want = expected.get(field)
        if want is not None and found and found[0] != want:
            raise SteerRecordError(
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
        raise SteerRecordError(
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
