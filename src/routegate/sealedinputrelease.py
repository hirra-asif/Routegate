"""A header-only compatibility adapter for the frozen v5 runner's held-out input.

The v5 runner reads its prompt column as ``prompt_text``. The authoritative
held-out manifest names that column ``canonical_prompt``. Both are frozen: the
runner is pinned by the v5 protocol digest, and the input is an archival file
that must not be touched. The gap is a column name and nothing else.

This module derives a compatibility adapter that differs from the authoritative
file in exactly one respect: the header token ``canonical_prompt`` is replaced
by ``prompt_text``. Every other byte -- every row, every value, every quote,
every line ending, and the order of all of it -- is copied through unchanged.

WHY A WHOLE-TOKEN SUBSTITUTION, NOT A TEXT REPLACE
--------------------------------------------------
The header also carries ``canonical_prompt_sha256``. ``canonical_prompt`` is
therefore present twice as a substring and once as a comma-delimited token. A
naive replace would rename the digest column too and silently change the
schema. The substitution here matches a whole token, locates its byte span, and
splices; the digest column keeps its name.

The file is CRLF-terminated at row boundaries while quoted prompt fields carry
bare line feeds, so "the first line" is defined by the first line feed and the
carriage return that precedes it is part of the header and is never touched.

WHAT THIS IS NOT
----------------
It is not a re-serialisation. Nothing is parsed and written back: the output is
the input's bytes with one span replaced. It introduces no column, no outcome,
no response and no score, and it reads no prompt semantically. Prompt values
are compared for equality and counted by digest; none is ever printed.

This module stores no directory and no absolute path for the held-out material.
It gates on a content digest supplied by the operator at launch.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .finaleval import DATA_ROOT_NAME, PREDECLARATION
from .triplets import sha256_text
from .validation import Checklist, ValidationError, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "routegate/sealed_input_release/1"
NAME = "routegate_final_evaluation_v5_sealed_input_release"
VERSION = 1
DIGEST_FIELD = "record_sha256"
RELEASE = REPO_ROOT / "manifests" / f"{NAME}.json"

APPLIES_TO = "routegate_final_evaluation_v5"
PREPARATION_SCRIPT = (
    "experiments/final_evaluation/prepare_sealed_input_release.py")

# ------------------------------------------------------------ the input gate
#: Only this exact content is accepted. Supplied by the operator; the file's
#: location is not stored here and cannot be inferred from this repository.
REQUIRED_SOURCE_SHA256 = (
    "6240887b290742da25c53016d96b2579878572fabdff6c3c49593b63ae1425f4")

#: The superseded predecessor, recorded so it can be refused by name of content.
#: It shares its basename with the authoritative file and differs only by its
#: containing directory, which is deliberately not stored. The digest is the
#: only identity this repository keeps, and it is the one that matters.
REJECTED_SOURCE_SHA256 = (
    "fef7d026c433992d698aa535f45a5530eb8cb08926608f7f4d578ac462cf2a0b")
REJECTED_ROWS = 379
REJECTED_SAFE = 180
REJECTED_HARMFUL = 199
REJECTED_REASON = (
    "379 rows, not 380: 180 safe_source and 199 harmful_source. It is the "
    "authoritative file minus one harmful row. Generating against it would "
    "silently drop a held-out item and make the confirmatory split a "
    "different split from the one committed to.")

# ------------------------------------------------------------- the transform
SOURCE_COLUMN = "canonical_prompt"
ADAPTER_COLUMN = "prompt_text"
ROW_ID_COLUMN = "row_id"
COMPOSITION_COLUMN = "source_side"
SAFE_VALUE = "safe_source"
HARMFUL_VALUE = "harmful_source"

EXPECTED_ROWS = 380
EXPECTED_SAFE = 180
EXPECTED_HARMFUL = 200
BYTE_DELTA = len(ADAPTER_COLUMN) - len(SOURCE_COLUMN)

TRANSFORMATION_RULE = (
    "the first line of the file is the CSV header. Among its comma-delimited "
    "tokens exactly one must equal 'canonical_prompt'; that token's byte span "
    "is replaced in place by 'prompt_text'. The match is on a whole token, so "
    "'canonical_prompt_sha256' keeps its name. No other byte of the header is "
    "touched, the carriage return ending the header is preserved, and every "
    "byte from the first line feed onward is copied through unchanged. The "
    "output is therefore the input with one 16-byte span replaced by an "
    "11-byte one and is exactly 5 bytes shorter. No record is parsed and "
    "re-serialised.")

#: Where the derived adapter lives: outside Git, beside the v5 output root.
RELEASE_DIR_NAME = "sealed_input_release"
RELEASE_RELATIVE = f"{DATA_ROOT_NAME}/{RELEASE_DIR_NAME}"
ADAPTER_SUFFIX = f"_{ADAPTER_COLUMN}.csv"

CHANGES = {
    "changes_scientific_protocol": False,
    "changes_prompts": False,
    "changes_row_order": False,
    "changes_generation": False,
    "changes_scoring": False,
}

POST_FREEZE_INSPECTION = {
    "occurred": True,
    "when": "after the v5 protocol was frozen and before capability "
            "generation completed",
    "what": "a structural, non-outcome inspection of the held-out input: its "
            "content digest, its header schema, its row count, its "
            "safe_source/harmful_source composition, and the uniqueness of "
            "its row identifiers and prompt values.",
    "prompt_text_printed": False,
    "prompt_text_semantically_reviewed": False,
    "outcomes_responses_scores_or_correctness_read": False,
    "changed_any_method_condition_or_parameter": False,
    "statement": "Prompt values were compared for equality and counted by "
                 "digest. None was printed, displayed or read for its "
                 "meaning. Nothing about the method, the conditions, the "
                 "parameters, the checkpoints or the thresholds was changed "
                 "as a result of this inspection, and nothing could have "
                 "been: the inspection produced a column name and a set of "
                 "counts.",
}


class SealedInputReleaseError(ValidationError):
    """The compatibility release was asked for something it must not do."""


# ------------------------------------------------------------------- helpers
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def adapter_name(source: Path | str) -> str:
    """The adapter's filename, derived from the input's own stem.

    Derived rather than written down: this repository does not name held-out
    material, and a derived name cannot drift from the file it came from.
    """
    return f"{Path(source).stem}{ADAPTER_SUFFIX}"


def release_dir(data_root: Optional[Path] = None) -> Path:
    root = Path(data_root) if data_root is not None else (
        Path.home() / "routegate-data")
    return root / DATA_ROOT_NAME / RELEASE_DIR_NAME


def default_adapter_path(source: Path | str,
                         data_root: Optional[Path] = None) -> Path:
    return release_dir(data_root) / adapter_name(source)


def split_first_line(data: bytes) -> Tuple[bytes, bytes]:
    """``(header, rest)`` where ``rest`` begins at the first line feed.

    The line feed itself belongs to ``rest``, so "every byte after the header"
    is a single slice and its preservation is a single comparison.
    """
    index = data.find(b"\n")
    if index < 0:
        raise SealedInputReleaseError(
            "the input has no line feed, so it has no header line")
    return data[:index], data[index:]


def header_tokens(header: bytes) -> List[bytes]:
    """The header's comma-delimited tokens, with any trailing CR removed."""
    line = header[:-1] if header.endswith(b"\r") else header
    if b'"' in line:
        raise SealedInputReleaseError(
            "the header line is quoted; a whole-token substitution is only "
            "safe on an unquoted header")
    return line.split(b",")


def prompt_column_span(header: bytes) -> Tuple[int, int]:
    """The byte span of the single header token equal to ``canonical_prompt``.

    Refuses a header that has no such token or more than one. The span is
    computed from the tokens' own lengths, so it can only ever point at a
    whole field.
    """
    target = SOURCE_COLUMN.encode()
    tokens = header_tokens(header)
    hits = [i for i, token in enumerate(tokens) if token == target]
    if len(hits) != 1:
        raise SealedInputReleaseError(
            f"the header must carry exactly one {SOURCE_COLUMN!r} column; it "
            f"carries {len(hits)}. A header that is ambiguous about which "
            f"column holds the prompt is not one this may rewrite.")
    index = hits[0]
    start = sum(len(t) + 1 for t in tokens[:index])
    return start, start + len(target)


def rewrite_header_bytes(data: bytes) -> bytes:
    """The input's bytes with the prompt column's header token replaced."""
    header, rest = split_first_line(data)
    start, end = prompt_column_span(header)
    spliced = header[:start] + ADAPTER_COLUMN.encode() + header[end:]
    # Cross-check the splice against an independent reconstruction from the
    # tokens. Two derivations, one answer, or this does not run.
    tokens = header_tokens(header)
    tail = b"\r" if header.endswith(b"\r") else b""
    rebuilt = b",".join(ADAPTER_COLUMN.encode() if t == SOURCE_COLUMN.encode()
                        else t for t in tokens) + tail
    if spliced != rebuilt:
        raise SealedInputReleaseError(
            "the byte splice and the token reconstruction of the header "
            "disagree; refusing to write an adapter neither derivation "
            "confirms")
    return spliced + rest


def read_rows(path: Path | str) -> Tuple[List[str], List[Dict[str, str]]]:
    """``(fieldnames, rows)``. Values are returned, never printed."""
    with open(str(path), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fields, rows


def expected_adapter_fields(source_fields: Sequence[str]) -> List[str]:
    return [ADAPTER_COLUMN if f == SOURCE_COLUMN else f for f in source_fields]


# --------------------------------------------------------------- preparation
def assert_source_is_authoritative(data: bytes, checks: Checklist) -> str:
    """Refuse any content that is not exactly the authoritative file."""
    digest = sha256_bytes(data)
    if digest == REJECTED_SOURCE_SHA256:
        raise SealedInputReleaseError(
            f"this is the superseded predecessor ({digest}). {REJECTED_REASON}")
    checks.require("source_sha256", digest == REQUIRED_SOURCE_SHA256,
                   f"{digest}, required {REQUIRED_SOURCE_SHA256}")
    return digest


def write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
        delete=False)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def prepare_adapter(source: Path | str, adapter: Optional[Path] = None,
                    checks: Optional[Checklist] = None,
                    data_root: Optional[Path] = None,
                    dry_run: bool = False) -> Dict[str, Any]:
    """Derive the adapter. Deterministic, idempotent, refuses to clobber."""
    checks = checks or Checklist(NAME)
    source = Path(source)
    checks.require("source_exists", source.is_file(), str(source.name))
    data = source.read_bytes()
    digest = assert_source_is_authoritative(data, checks)

    target = Path(adapter) if adapter is not None else default_adapter_path(
        source, data_root)
    derived = rewrite_header_bytes(data)
    derived_digest = sha256_bytes(derived)

    checks.require("byte_delta_is_the_header_substitution",
                   len(derived) - len(data) == BYTE_DELTA,
                   f"{len(derived) - len(data)} byte(s), expected {BYTE_DELTA}")
    checks.require("bytes_after_the_header_are_unchanged",
                   split_first_line(derived)[1] == split_first_line(data)[1],
                   "every byte from the first line feed onward is identical")

    written = False
    if dry_run:
        checks.record("dry_run", True, f"{target.name} was not written")
    elif target.exists():
        existing = target.read_bytes()
        if existing == derived:
            checks.record("already_present_and_identical", True,
                          f"{target.name} is already exactly this adapter")
        else:
            raise SealedInputReleaseError(
                f"{target} exists and is not this adapter "
                f"({sha256_bytes(existing)} on disk, {derived_digest} "
                f"derived). A derived input that can be silently replaced is "
                f"not one: remove it by hand, with a reason, and re-run.")
    else:
        write_atomically(target, derived)
        written = True
        checks.record("wrote_adapter", True, target.name)

    return {"source_sha256": digest, "adapter_sha256": derived_digest,
            "adapter_path": str(target), "adapter_bytes": len(derived),
            "source_bytes": len(data), "written": written}


# -------------------------------------------------------------- verification
def verify_adapter(source: Path | str, adapter: Path | str,
                   checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """Prove the adapter is the input with one header token renamed.

    Byte identity and semantic identity, both. No value is printed: prompts are
    compared for equality and counted by digest.
    """
    checks = checks or Checklist(NAME)
    source, adapter = Path(source), Path(adapter)
    checks.require("source_exists", source.is_file(), str(source.name))
    checks.require("adapter_exists", adapter.is_file(), str(adapter.name))

    original = source.read_bytes()
    derived = adapter.read_bytes()
    source_digest = assert_source_is_authoritative(original, checks)
    adapter_digest = sha256_bytes(derived)

    # ---- byte identity
    header, rest = split_first_line(original)
    start, end = prompt_column_span(header)
    checks.require(
        "the_only_transformation_is_the_header_substitution",
        derived == original[:start] + ADAPTER_COLUMN.encode() + original[end:],
        "the adapter equals the input with exactly one header token replaced")
    checks.require("bytes_after_the_header_are_identical",
                   split_first_line(derived)[1] == rest,
                   f"{len(rest)} byte(s) after the header, identical")
    checks.require("byte_length_delta",
                   len(derived) - len(original) == BYTE_DELTA,
                   f"{len(derived) - len(original)} byte(s)")
    derived_header = split_first_line(derived)[0]
    derived_tokens = header_tokens(derived_header)
    checks.require("the_digest_column_kept_its_name",
                   f"{SOURCE_COLUMN}_sha256".encode() in derived_tokens,
                   "the digest column was not renamed by a substring match")
    checks.require("the_header_declares_the_runner_column",
                   derived_tokens.count(ADAPTER_COLUMN.encode()) == 1
                   and SOURCE_COLUMN.encode() not in derived_tokens,
                   f"exactly one {ADAPTER_COLUMN} token, no {SOURCE_COLUMN} "
                   f"token")

    # ---- schemas
    source_fields, source_rows = read_rows(source)
    adapter_fields, adapter_rows = read_rows(adapter)
    checks.require("header_schema_is_the_rename_and_nothing_else",
                   adapter_fields == expected_adapter_fields(source_fields),
                   f"{len(adapter_fields)} column(s), one renamed")
    checks.require("no_column_was_added_or_removed",
                   len(adapter_fields) == len(source_fields),
                   f"{len(source_fields)} -> {len(adapter_fields)}")
    checks.require("no_outcome_or_response_column_is_introduced",
                   set(adapter_fields) - set(source_fields)
                   == {ADAPTER_COLUMN},
                   "the renamed prompt column is the only new name; no "
                   "outcome, response, score or correctness column exists")

    # ---- counts
    for label, rows in (("source", source_rows), ("adapter", adapter_rows)):
        checks.require(f"{label}_row_count", len(rows) == EXPECTED_ROWS,
                       f"{len(rows)} row(s), expected {EXPECTED_ROWS}")
    composition = {
        label: (sum(1 for r in rows if r.get(COMPOSITION_COLUMN) == SAFE_VALUE),
                sum(1 for r in rows
                    if r.get(COMPOSITION_COLUMN) == HARMFUL_VALUE))
        for label, rows in (("source", source_rows), ("adapter", adapter_rows))}
    for label, (safe, harmful) in composition.items():
        checks.require(f"{label}_composition",
                       safe == EXPECTED_SAFE and harmful == EXPECTED_HARMFUL,
                       f"{safe} {SAFE_VALUE}, {harmful} {HARMFUL_VALUE}")

    # ---- uniqueness, without printing a value
    ids = [r.get(ROW_ID_COLUMN, "") for r in adapter_rows]
    prompts = [r.get(ADAPTER_COLUMN, "") for r in adapter_rows]
    checks.require("no_blank_row_id", all(i.strip() for i in ids),
                   f"{sum(1 for i in ids if not i.strip())} blank")
    checks.require("no_blank_prompt", all(p.strip() for p in prompts),
                   f"{sum(1 for p in prompts if not p.strip())} blank")
    checks.require("unique_row_ids", len(set(ids)) == EXPECTED_ROWS,
                   f"{len(set(ids))} distinct identifier(s)")
    digests = {sha256_text(p) for p in prompts}
    checks.require("unique_prompts", len(digests) == EXPECTED_ROWS,
                   f"{len(digests)} distinct prompt value(s), counted by "
                   f"digest and never displayed")

    # ---- order and every value
    checks.require("row_order_is_identical",
                   [r.get(ROW_ID_COLUMN) for r in source_rows] == ids,
                   f"{len(ids)} identifier(s) in the same order")
    mismatched = 0
    for before, after in zip(source_rows, adapter_rows):
        remapped = {(ADAPTER_COLUMN if k == SOURCE_COLUMN else k): v
                    for k, v in before.items()}
        if remapped != after:
            mismatched += 1
    checks.require("every_value_in_every_row_is_identical", mismatched == 0,
                   f"{mismatched} differing row(s) of {len(source_rows)} "
                   f"under {SOURCE_COLUMN} -> {ADAPTER_COLUMN}")

    return {"source_sha256": source_digest, "adapter_sha256": adapter_digest,
            "rows": len(adapter_rows),
            "safe_source": composition["adapter"][0],
            "harmful_source": composition["adapter"][1],
            "unique_row_ids": len(set(ids)), "unique_prompts": len(digests),
            "source_fields": source_fields, "adapter_fields": adapter_fields,
            "bytes_after_header": len(rest),
            "source_bytes": len(original), "adapter_bytes": len(derived)}


# ------------------------------------------------------------ release record
def canonical(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def record_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_text(canonical(payload))


def protocol_record_digest() -> str:
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    return str(committed.get("record_sha256") or "")


def build_release(source: Path | str, adapter: Path | str,
                  checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The committed, metadata-only release record. No prompt, no outcome."""
    checks = checks or Checklist(NAME)
    result = verify_adapter(source, adapter, checks)
    record: Dict[str, Any] = {
        "schema": SCHEMA, "name": NAME, "version": VERSION,
        "applies_to": APPLIES_TO,
        "applies_to_manifest": f"manifests/{APPLIES_TO}.json",
        "applies_to_manifest_sha256": sha256_file(PREDECLARATION),
        "applies_to_manifest_record_sha256": protocol_record_digest(),
        "source": {
            "filename": Path(source).name,
            "sha256": result["source_sha256"],
            "required_sha256": REQUIRED_SOURCE_SHA256,
            "bytes": result["source_bytes"],
            "directory_stored_in_this_repository": False,
            "note": "the filename and the content digest identify the file; "
                    "its directory is deliberately not recorded, here or "
                    "anywhere in this repository.",
        },
        "rejected_predecessor": {
            "filename": Path(source).name,
            "sha256": REJECTED_SOURCE_SHA256,
            "rows": REJECTED_ROWS,
            "safe_source": REJECTED_SAFE,
            "harmful_source": REJECTED_HARMFUL,
            "reason": REJECTED_REASON,
            "note": "it shares its basename with the authoritative file and "
                    "differs only by its containing directory, which is not "
                    "stored. The content digest is the identity that "
                    "distinguishes them, and the preparation script refuses "
                    "this one by digest before it reads a row.",
        },
        "adapter": {
            "relative_path": f"{RELEASE_RELATIVE}/{Path(adapter).name}",
            "filename": Path(adapter).name,
            "sha256": result["adapter_sha256"],
            "bytes": result["adapter_bytes"],
            "committed_to_git": False,
            "location": "outside this repository, under the machine-local "
                        "data root",
        },
        "header_schema": {
            "original": list(result["source_fields"]),
            "adapter": list(result["adapter_fields"]),
            "renamed_from": SOURCE_COLUMN,
            "renamed_to": ADAPTER_COLUMN,
            "columns": len(result["adapter_fields"]),
            "digest_column_preserved": f"{SOURCE_COLUMN}_sha256",
        },
        "transformation_rule": TRANSFORMATION_RULE,
        "byte_delta": BYTE_DELTA,
        "bytes_after_header_identical": True,
        "bytes_after_header": result["bytes_after_header"],
        "counts": {
            "rows": result["rows"],
            "safe_source": result["safe_source"],
            "harmful_source": result["harmful_source"],
            "unique_row_ids": result["unique_row_ids"],
            "unique_prompts": result["unique_prompts"],
            "expected_rows": EXPECTED_ROWS,
            "expected_safe_source": EXPECTED_SAFE,
            "expected_harmful_source": EXPECTED_HARMFUL,
        },
        "preparation_script": PREPARATION_SCRIPT,
        "preparation_script_sha256": sha256_file(
            REPO_ROOT / PREPARATION_SCRIPT),
        "post_freeze_inspection": dict(POST_FREEZE_INSPECTION),
        "contains_prompt_response_gold_or_score": False,
        "row_order_preserved": True,
        "values_preserved": True,
    }
    record.update(CHANGES)
    record[DIGEST_FIELD] = record_sha256(record)
    return record


def verify_release(path: Path | str = RELEASE,
                   checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The committed release record still describes what it pins.

    Offline and input-free: it does not need the held-out file to be present,
    because what it verifies is the record's own consistency, the v5 manifest
    it applies to and the preparation script it names.
    """
    checks = checks or Checklist(NAME)
    path = Path(path)
    checks.require("release_exists", path.is_file(), str(path))
    committed = json.loads(path.read_text(encoding="utf-8"))
    checks.require("schema", committed.get("schema") == SCHEMA,
                   f"{committed.get('schema')!r}")
    checks.require("record_digest",
                   record_sha256(committed) == committed.get(DIGEST_FIELD),
                   f"{committed.get(DIGEST_FIELD)}")
    checks.require("v5_manifest_unchanged",
                   committed.get("applies_to_manifest_sha256")
                   == sha256_file(PREDECLARATION),
                   "the frozen v5 manifest is the one this release applies to")
    checks.require("v5_record_digest_unchanged",
                   committed.get("applies_to_manifest_record_sha256")
                   == protocol_record_digest(),
                   "the v5 record digest is the one this release applies to")
    checks.require("preparation_script_unchanged",
                   committed.get("preparation_script_sha256")
                   == sha256_file(REPO_ROOT / PREPARATION_SCRIPT),
                   "the script that derived the adapter is the one pinned")
    source = committed.get("source") or {}
    checks.require("source_is_the_authoritative_file",
                   source.get("sha256") == REQUIRED_SOURCE_SHA256
                   and source.get("required_sha256") == REQUIRED_SOURCE_SHA256,
                   "the pinned input digest is the required one")
    rejected = committed.get("rejected_predecessor") or {}
    checks.require("the_predecessor_is_recorded_as_rejected",
                   rejected.get("sha256") == REJECTED_SOURCE_SHA256
                   and rejected.get("rows") == REJECTED_ROWS,
                   f"{rejected.get('rows')} row(s), refused by digest")
    counts = committed.get("counts") or {}
    checks.require("counts",
                   counts.get("rows") == EXPECTED_ROWS
                   and counts.get("safe_source") == EXPECTED_SAFE
                   and counts.get("harmful_source") == EXPECTED_HARMFUL
                   and counts.get("unique_row_ids") == EXPECTED_ROWS
                   and counts.get("unique_prompts") == EXPECTED_ROWS,
                   f"{counts.get('rows')} rows, {counts.get('safe_source')} "
                   f"safe, {counts.get('harmful_source')} harmful, all unique")
    schema = committed.get("header_schema") or {}
    checks.require("header_rename_is_the_whole_change",
                   schema.get("renamed_from") == SOURCE_COLUMN
                   and schema.get("renamed_to") == ADAPTER_COLUMN
                   and schema.get("adapter")
                   == expected_adapter_fields(schema.get("original") or []),
                   f"{SOURCE_COLUMN} -> {ADAPTER_COLUMN}, nothing else")
    checks.require("scope_changes_nothing_scientific",
                   all(committed.get(k) is False for k in CHANGES),
                   "protocol, prompts, row order, generation and scoring all "
                   "unchanged")
    inspection = committed.get("post_freeze_inspection") or {}
    checks.require("the_inspection_is_recorded_and_was_non_semantic",
                   inspection.get("occurred") is True
                   and inspection.get("prompt_text_printed") is False
                   and inspection.get(
                       "prompt_text_semantically_reviewed") is False
                   and inspection.get(
                       "changed_any_method_condition_or_parameter") is False,
                   "a structural inspection is declared, and it changed "
                   "nothing")
    checks.require("the_adapter_is_not_committed",
                   (committed.get("adapter") or {}).get(
                       "committed_to_git") is False,
                   "the derived file stays outside Git")
    return {"path": str(path), "sha256": sha256_file(path),
            DIGEST_FIELD: record_sha256(committed)}


__all__ = [
    "ADAPTER_COLUMN", "ADAPTER_SUFFIX", "APPLIES_TO", "BYTE_DELTA", "CHANGES",
    "COMPOSITION_COLUMN", "DIGEST_FIELD", "EXPECTED_HARMFUL", "EXPECTED_ROWS",
    "EXPECTED_SAFE", "HARMFUL_VALUE", "NAME", "POST_FREEZE_INSPECTION",
    "PREPARATION_SCRIPT", "REJECTED_HARMFUL", "REJECTED_REASON",
    "REJECTED_ROWS", "REJECTED_SAFE", "REJECTED_SOURCE_SHA256", "RELEASE",
    "RELEASE_DIR_NAME", "RELEASE_RELATIVE", "REQUIRED_SOURCE_SHA256",
    "ROW_ID_COLUMN", "SAFE_VALUE", "SCHEMA", "SOURCE_COLUMN",
    "SealedInputReleaseError", "TRANSFORMATION_RULE", "VERSION",
    "adapter_name", "assert_source_is_authoritative", "build_release",
    "canonical", "default_adapter_path", "expected_adapter_fields",
    "header_tokens", "prepare_adapter", "prompt_column_span",
    "protocol_record_digest", "read_rows", "record_sha256", "release_dir",
    "rewrite_header_bytes", "sha256_bytes", "split_first_line",
    "verify_adapter", "verify_release", "write_atomically",
]
