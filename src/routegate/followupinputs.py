"""Prepare the existing development frame for the next steering comparison.

This module copies prompt and split metadata only.  It does not choose an edit,
read old responses as new outcomes, train a model, generate text or call judges.
The accepted corpus and the original five-fold assignment are fixed by hashes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Any


STUDY_ID = "repair_preservation_v1"
SCHEMA = "routegate/repair_preservation_inputs/1"
EXPECTED_CLASS_COUNTS = {"B": 178, "E": 89, "H": 213}
EXPECTED_FOLD_COUNTS = {"0": 97, "1": 97, "2": 96, "3": 95, "4": 95}
EXPECTED_ROWS_SHA256 = "d06ef80725a73a45a34dd5e78f4482e0a1d7d74e4f4c49416de96d1a1657016d"
EXPECTED_FOLD_MAP_DIGEST = "3692fd99cca4521e88176c5fa3e09253b27e9ae80251d92f4d8af8b65739fc6f"
ROW_FIELDS = frozenset(("row_id", "prompt_text", "prompt_sha256", "row_class",
                        "source_family", "fold_id"))
SOURCE_FILES = (
    {"role": "accepted_development_input_specification", "root": "repo",
     "path": "manifests/development_inputs.json",
     "sha256": "9d97bb2f89785e3edd6b5f982a8cb7eb6fbfdfef8f1436cb305342c224f599ca"},
    {"role": "authoritative_development_labels", "root": "data",
     "path": "frozen/corpus/rq3_dataset_final.csv",
     "sha256": "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08"},
    {"role": "eligible_prompt_inventory", "root": "data",
     "path": "frozen/residual_extraction/manifest.csv",
     "sha256": "a3b7444b1a2782b5bc04142f3545ac5d71ad57a5f546edc315d7d51d14947abd"},
    {"role": "existing_five_fold_assignment", "root": "repo",
     "path": "artifacts/gate_oof_v1/fold_assignments.json",
     "sha256": "f654c0bb6c1ccfcdaf924960ee92f67921c645169083fb586681f157972a2cd7"},
)


class FollowupInputError(ValueError):
    """The proposed inventory differs from the accepted development frame."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _row_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _read_verified_sources(repo_root: Path, data_root: Path) -> dict[str, bytes]:
    sources = {}
    for spec in SOURCE_FILES:
        root = repo_root if spec["root"] == "repo" else data_root
        path = root / spec["path"]
        try:
            content = path.read_bytes()
        except OSError as error:
            raise FollowupInputError(f"Required input is unavailable: {path}") from error
        observed = _sha(content)
        if observed != spec["sha256"]:
            raise FollowupInputError(
                f"Source hash mismatch for {path}: expected {spec['sha256']}, "
                f"found {observed}. The accepted input is not replaced automatically.")
        sources[spec["role"]] = content
    return sources


def _csv_rows(content: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


def _unique_index(rows: list[dict[str, Any]], description: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise FollowupInputError(f"Missing row_id in {description}.")
        if row_id in result:
            raise FollowupInputError(f"Duplicate row_id in {description}: {row_id}")
        result[row_id] = row
    return result


def _build_rows(sources: dict[str, bytes]) -> list[dict[str, Any]]:
    corpus = _csv_rows(sources["authoritative_development_labels"])
    corpus_by_id = _unique_index(corpus, "development corpus")
    if len(corpus_by_id) != 509:
        raise FollowupInputError("The accepted corpus must contain 509 rows before exclusions.")
    corpus_counts = Counter(row["final_ebh"] for row in corpus)
    if corpus_counts != {**EXPECTED_CLASS_COUNTS, "EXCLUDED_PARTIAL_OR_UNCLEAR": 29}:
        raise FollowupInputError("Development class counts differ from the accepted corpus.")
    eligible = {key: row for key, row in corpus_by_id.items()
                if row["final_ebh"] in EXPECTED_CLASS_COUNTS}
    extraction = _csv_rows(sources["eligible_prompt_inventory"])
    extraction_by_id = _unique_index(extraction, "eligible prompt inventory")
    if set(extraction_by_id) != set(eligible):
        raise FollowupInputError("Eligible prompt IDs differ between the corpus and inventory.")
    if [row["row_id"] for row in extraction] != sorted(eligible):
        raise FollowupInputError("The accepted prompt inventory is not sorted by row_id.")
    if [row.get("row_index") for row in extraction] != [str(i) for i in range(480)]:
        raise FollowupInputError("The accepted prompt inventory has inconsistent row indices.")
    folds = json.loads(sources["existing_five_fold_assignment"])
    if folds.get("n_splits") != 5 or folds.get("fold_assignment_sha256") != EXPECTED_FOLD_MAP_DIGEST:
        raise FollowupInputError("Unexpected existing fold specification.")
    fold_of_row = folds.get("fold_of_row", {})
    if set(fold_of_row) != set(eligible):
        raise FollowupInputError("Fold IDs do not cover exactly the 480 eligible development prompts.")
    rows = []
    for row_id in sorted(eligible):
        source, extracted = eligible[row_id], extraction_by_id[row_id]
        for field in ("prompt_text", "final_ebh", "source_family"):
            if source[field] != extracted[field]:
                raise FollowupInputError(f"Corpus and extraction disagree on {field} for {row_id}.")
        rows.append({"row_id": row_id, "prompt_text": source["prompt_text"],
                     "prompt_sha256": extracted["prompt_sha256"],
                     "row_class": source["final_ebh"],
                     "source_family": source["source_family"],
                     "fold_id": fold_of_row[row_id]})
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 480:
        raise FollowupInputError("The inventory must contain exactly 480 development prompts.")
    indexed = _unique_index(rows, "prepared inventory")
    if list(indexed) != sorted(indexed):
        raise FollowupInputError("Prepared rows must be ordered by row_id.")
    prompt_hashes = set()
    for row in rows:
        if set(row) != ROW_FIELDS:
            raise FollowupInputError("Prepared rows must contain only the declared prompt and fold fields.")
        if row["row_class"] not in EXPECTED_CLASS_COUNTS:
            raise FollowupInputError(f"Unexpected row class for {row['row_id']}.")
        if type(row["fold_id"]) is not int or row["fold_id"] not in range(5):
            raise FollowupInputError(f"Invalid fold ID for {row['row_id']}.")
        if not isinstance(row["prompt_text"], str) or not row["prompt_text"].strip():
            raise FollowupInputError(f"Empty prompt for {row['row_id']}.")
        if not isinstance(row["source_family"], str) or not row["source_family"]:
            raise FollowupInputError(f"Missing source family for {row['row_id']}.")
        expected_hash = _sha(row["prompt_text"].encode("utf-8"))
        if row["prompt_sha256"] != expected_hash:
            raise FollowupInputError(f"Prompt hash mismatch for {row['row_id']}.")
        if expected_hash in prompt_hashes:
            raise FollowupInputError("Duplicate prompt text in the prepared inventory.")
        prompt_hashes.add(expected_hash)
    if dict(Counter(row["row_class"] for row in rows)) != EXPECTED_CLASS_COUNTS:
        raise FollowupInputError("Prepared class counts differ from E=89, B=178, H=213.")
    if dict(Counter(str(row["fold_id"]) for row in rows)) != EXPECTED_FOLD_COUNTS:
        raise FollowupInputError("Prepared fold sizes differ from the original five-fold assignment.")
    if _sha(_row_bytes(rows)) != EXPECTED_ROWS_SHA256:
        raise FollowupInputError("Prepared prompts, classes or folds differ from the fixed original inventory.")


def _inventory_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: dict[str, Counter] = defaultdict(Counter)
    fold_class_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        source_counts[row["source_family"]][row["row_class"]] += 1
        fold_class_counts[str(row["fold_id"])][row["row_class"]] += 1
    return {
        "schema": SCHEMA, "study_id": STUDY_ID,
        "rows_file": "rows.jsonl", "rows_sha256": EXPECTED_ROWS_SHA256,
        "rows": 480, "class_counts": EXPECTED_CLASS_COUNTS,
        "fold_counts": EXPECTED_FOLD_COUNTS,
        "fold_class_counts": {key: dict(sorted(value.items()))
                              for key, value in sorted(fold_class_counts.items())},
        "source_class_counts": {key: dict(sorted(value.items()))
                                for key, value in sorted(source_counts.items())},
        "source_files": [dict(spec) for spec in SOURCE_FILES],
        "fold_assignment_sha256": EXPECTED_FOLD_MAP_DIGEST,
        "fold_interpretation": "The existing row-level five-fold assignment is preserved. "
                               "It is not an unseen-source or unseen-topic split.",
        "source_confound_note": "Within this development frame, H sources are distinct from "
                                "E/B sources. Classification performance alone cannot separate "
                                "source recognition from harmfulness recognition.",
        "evaluation_status": "DEVELOPMENT_INPUTS_PREPARED_ONLY",
        "selection_status": "PENDING_CANCELLATION_DIAGNOSTIC",
        "candidate_arms": [],
        "candidate_selection_note": "The edit and comparison conditions must be recorded in "
                                    "a separate run specification after reviewing the diagnostic. "
                                    "This inventory does not authorise training or choose an edit.",
        "external_benchmark_rows_loaded": 0,
        "old_responses_imported": 0, "old_verdicts_imported": 0,
        "new_generations": 0, "judge_calls": 0, "trained_models": 0,
    }


def prepare(repo_root: str | Path, data_root: str | Path,
            output_root: str | Path) -> dict[str, Any]:
    """Write a fixed input inventory; safely return it on an identical rerun."""
    repo_root, data_root = Path(repo_root).resolve(), Path(data_root).resolve()
    requested_output = Path(output_root).expanduser()
    if requested_output.is_symlink():
        raise FollowupInputError("The output directory must not be a symbolic link.")
    output_root = requested_output.resolve()
    for source_root in (repo_root, data_root):
        if output_root == source_root or output_root in source_root.parents:
            raise FollowupInputError("Choose a new study directory, not an existing source root.")
    frozen_root = data_root / "frozen"
    if output_root == frozen_root or frozen_root in output_root.parents:
        raise FollowupInputError("The new inventory cannot be written inside frozen input data.")
    sources = _read_verified_sources(repo_root, data_root)
    rows = _build_rows(sources)
    if output_root.exists():
        if not (output_root / "input_manifest.json").is_file():
            raise FollowupInputError(f"Output already exists without this study's manifest: {output_root}")
        manifest, existing_rows = verify(output_root)
        if existing_rows != rows:
            raise FollowupInputError("The existing study inventory differs from the accepted input.")
        return manifest
    fields = _inventory_fields(rows)
    manifest = {**fields, "inventory_sha256": _sha(_canonical(fields)),
                "created_utc": datetime.now(timezone.utc).isoformat()}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.prepare-",
                                      dir=output_root.parent))
    try:
        (temporary / "rows.jsonl").write_bytes(_row_bytes(rows))
        (temporary / "input_manifest.json").write_bytes(_canonical(manifest) + b"\n")
        verify(temporary)
        # rename publishes both files together; an already populated study is never replaced.
        os.rename(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def verify(output_root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify a prepared inventory without loading activations or a language model."""
    root = Path(output_root)
    try:
        manifest = json.loads((root / "input_manifest.json").read_bytes())
        content = (root / "rows.jsonl").read_bytes()
        rows = [json.loads(line) for line in content.splitlines()]
    except (OSError, ValueError) as error:
        raise FollowupInputError(f"Cannot read a complete prepared inventory at {root}: {error}") from error
    if not isinstance(manifest, dict) or not all(isinstance(row, dict) for row in rows):
        raise FollowupInputError("The prepared manifest and rows must be JSON objects.")
    _validate_rows(rows)
    if content != _row_bytes(rows):
        raise FollowupInputError("Prepared rows do not match their canonical saved representation.")
    expected_fields = _inventory_fields(rows)
    declared_fields = {key: value for key, value in manifest.items()
                       if key not in {"inventory_sha256", "created_utc"}}
    if declared_fields != expected_fields:
        raise FollowupInputError("Prepared input manifest metadata was changed.")
    if manifest.get("inventory_sha256") != _sha(_canonical(expected_fields)):
        raise FollowupInputError("Prepared input manifest digest does not match its contents.")
    try:
        created = datetime.fromisoformat(manifest["created_utc"])
        if created.tzinfo is None:
            raise ValueError("missing time zone")
    except (KeyError, TypeError, ValueError) as error:
        raise FollowupInputError("The inventory creation timestamp is missing or invalid.") from error
    return manifest, rows
