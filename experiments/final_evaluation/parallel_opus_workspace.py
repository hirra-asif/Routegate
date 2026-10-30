#!/usr/bin/env python3
"""Copy a frozen response packet for Opus; import its unchanged evidence later.

No network calls are made here. The original frozen judge runner does the calls,
with its original lock, against a separate data root. The active Sonnet runner,
original sources, release, packet identifiers and A/B order are not modified.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from routegate import finaljudging as F
from routegate import judgingprotocol as P
from routegate import judgingstages as S
from routegate.validation import ValidationError

SNAPSHOT = "parallel_opus_snapshot_v1.json"
RECEIPT = "parallel_opus_import_v1.json"
FORK_NAME = "sealed_opus_parallel_v1"


@contextlib.contextmanager
def locked(data_root):
    """Use exactly the same per-data-root lock as the frozen runner."""
    path = Path(data_root) / F.OUTPUT_NAME / ".judging.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise F.FinalJudgingError(
                "A judge or another stage is still running in " + str(data_root)
                + ". Wait for it to finish before importing results.")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def inventory(directory):
    """Hash ordinary files, rejecting links inside the copied evidence tree."""
    directory = Path(directory)
    F.require(not directory.is_symlink(), "Evidence directory is a link: " + str(directory))
    if not directory.exists():
        return {}
    out = {}
    for path in sorted(directory.rglob("*")):
        F.require(not path.is_symlink(), "Evidence contains a link: " + str(path))
        if path.is_file():
            out[path.relative_to(directory).as_posix()] = F.digest(path.read_bytes())
        else:
            F.require(path.is_dir(), "Evidence contains a special file: " + str(path))
    return out


def copy_exact(source, target, expected):
    """Atomic create-only copy. Identical existing bytes are accepted."""
    source, target = Path(source), Path(target)
    data = source.read_bytes()
    F.require(F.digest(data) == expected, "Source changed during copy: " + str(source))
    F.require(not target.is_symlink(), "Copy target is a link: " + str(target))
    if target.exists():
        F.require(target.is_file() and F.digest(target.read_bytes()) == expected,
                  "Existing destination differs; nothing is overwritten: " + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".opus-copy-", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(source, temporary)
        try:
            os.link(temporary, target)
        except FileExistsError:
            F.require(not target.is_symlink() and F.digest(target.read_bytes()) == expected,
                      "Destination changed during copy: " + str(target))
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepared(data_root, repo=REPO):
    release, root, manifest, files = F.context(data_root, "responses", repo=repo)
    F.verify_prepared(root, manifest, files)
    F.verify_packet(root, manifest)
    return release, root, manifest, files


def frozen_inventory(data_root, repo=REPO):
    release, root, manifest, files = prepared(data_root, repo)
    relative_paths = set(release["generation_files"]) | {F.ADAPTER_RELATIVE}
    base = F.phase_root(data_root, "baseline")
    for name in ("adjudication_binding.json", "judging/adjudicated/adjudicated_cells.jsonl",
                 "judging/adjudicated/adjudication_summary.json"):
        relative_paths.add((base / name).relative_to(data_root).as_posix())
    for name in ("protocol.json", "packet_binding.json"):
        relative_paths.add((root / name).relative_to(data_root).as_posix())
    for name in ("inputs", "frozen", "judging/packet"):
        for relative in inventory(root / name):
            relative_paths.add((root / name / relative).relative_to(data_root).as_posix())
    digests = {}
    for name in sorted(relative_paths):
        path = Path(data_root) / name
        F.require(path.is_file(), "Frozen input is missing: " + name)
        digests[name] = F.digest(path.read_bytes())
    return root, manifest, digests


def opus_inventory(root, manifest):
    out = {}
    for name in manifest.rubrics:
        folder = root / "judging/runs" / ("packet__" + name + "__opus")
        for relative, digest in inventory(folder).items():
            out[(folder / relative).relative_to(root).as_posix()] = digest
    return out


def before_review(root):
    for name in ("joined", "review", "adjudicated", "report"):
        F.require(not inventory(root / "judging" / name),
                  "Response " + name + " already exists. Do not change evidence after review starts.")


def expected_snapshot(source, parallel, digests, manifest):
    return {"schema": "routegate/parallel_opus_snapshot/1", "phase": "responses",
            "judge": "opus", "source_data_root": str(source),
            "parallel_data_root": str(parallel), "protocol_sha256": manifest.digest(),
            "frozen_files": digests,
            "note": "Operational isolation only. Original frozen runner and judgements are reused unchanged."}


def prepare(source, parallel, repo=REPO):
    source, parallel = Path(source).resolve(), Path(parallel).resolve()
    F.require(source != parallel, "The Opus workspace must differ from the main data root.")
    with locked(parallel):
        root, manifest, digests = frozen_inventory(source, repo)
        before_review(root)
        F.require(not (root / RECEIPT).exists(), "Opus was already imported. Use main progress now.")
        F.require(not opus_inventory(root, manifest),
                  "Main response Opus evidence already exists. Do not start a second Opus run; send progress first.")
        snapshot = expected_snapshot(source, parallel, digests, manifest)
        marker = parallel / SNAPSHOT
        if marker.exists():
            F.require(F.read_json(marker) == snapshot, "The existing Opus workspace belongs to different inputs.")
        else:
            fork_root = F.phase_root(parallel, "responses")
            F.require(not inventory(fork_root / "judging/runs"),
                      "Unregistered judge evidence exists in the separate workspace.")
        for name, digest in digests.items():
            copy_exact(source / name, parallel / name, digest)
        _, fork_root, fork_manifest, _ = prepared(parallel, repo)
        F.require(fork_manifest.raw == manifest.raw, "Copied protocol changed.")
        # Recheck immutable sources and absence of a second main Opus writer.
        _, _, final_digests = frozen_inventory(source, repo)
        F.require(final_digests == digests and not opus_inventory(root, manifest),
                  "The main inputs or Opus evidence changed during preparation.")
        S.dry_run(fork_manifest, S.resolved_dirs(fork_manifest, fork_root), fork_root, "opus")
        F.put_once(marker, F.json_bytes(snapshot))
        return {"stage": "prepare_parallel_opus", "api_calls": 0,
                "planned_opus_assessments": manifest.coverage["records_per_judge"],
                "data_root": str(parallel), "original_frozen_files_unchanged": True,
                "identical_packet_and_blinding": True}


def validate_opus(root, manifest):
    """Verify every attempt, its original request and its preserved raw body."""
    dirs = S.resolved_dirs(manifest, root)
    successful, failed, attempted = set(), set(), set()
    allowed_files = set()
    expected_model = manifest.judges["opus"]["model"]
    returned_model = manifest.judges["opus"].get("expected_returned", expected_model)
    for name, rubric in manifest.rubrics.items():
        folder = dirs["runs"] / ("packet__" + name + "__opus")
        rows = {r["row_id"]: r for r in S.load_judge_rows(dirs["packet"], name)}
        text = P.load_rubric_text(manifest, rubric, root=root)
        raw_seen = set()
        for filename, status in ((P.RESULTS_FILE, "OK"), (P.FAILURES_FILE, "FAILED")):
            path = folder / filename
            if not path.exists():
                continue
            allowed_files.add(path.relative_to(root).as_posix())
            for record in F.rows_bytes(path.read_bytes()):
                rid = record.get("row_id")
                F.require(rid in rows and record.get("judge") == "opus"
                          and record.get("rubric") == name and record.get("status") == status,
                          "Opus evidence has an unexpected row, judge, rubric or status.")
                F.require(record.get("model_requested") == expected_model
                          and record.get("temperature") == manifest.temperature
                          and record.get("max_tokens") == manifest.max_tokens,
                          "Opus evidence uses different model or sampling settings.")
                rendered = S.render_request(manifest, rubric, text, rows[rid], expected_model)
                F.require(record.get("request_sha256") == rendered["request_sha256"],
                          "An Opus request differs from the original frozen packet.")
                relative = Path(record["raw_response_path"])
                F.require(not relative.is_absolute() and ".." not in relative.parts,
                          "Unexpected raw-response path.")
                raw_path = folder / relative
                F.require(relative.as_posix() not in raw_seen, "Two attempts share one raw response.")
                raw_seen.add(relative.as_posix())
                allowed_files.add(raw_path.relative_to(root).as_posix())
                raw_bytes = raw_path.read_bytes()
                F.require(F.digest(raw_bytes) == record.get("raw_response_sha256"),
                          "A preserved Opus response has changed.")
                attempted.add(rid)
                if status == "OK":
                    F.require(rid not in successful, "Duplicate successful Opus verdict.")
                    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
                    F.require(record.get("record_sha256") == S.record_hash(unsigned),
                              "An Opus success record has changed.")
                    raw = json.loads(raw_bytes)
                    payload = json.loads(raw["raw_body"])
                    F.require(S.model_name_accepted(S.extract_returned_model(payload), returned_model),
                              "A successful response came from the wrong model.")
                    values = P.validate_reply(S.parse_judge_text(S.extract_text(payload)), rubric, rid)
                    F.require(values == record.get("parsed_result"),
                              "A stored verdict differs from its original response.")
                    successful.add(rid)
                else:
                    failed.add(rid)
        F.require(set(rows).issubset(attempted),
                  "Opus has not attempted every prepared case. Finish its pass before importing.")
    files = opus_inventory(root, manifest)
    F.require(set(files) == allowed_files, "Opus evidence contains unrecorded or unexpected files.")
    return files, {"successful": len(successful), "unresolved_failures": len(failed - successful),
                   "attempted_cases": len(attempted)}


def merge(source, parallel, repo=REPO):
    source, parallel = Path(source).resolve(), Path(parallel).resolve()
    F.require(source != parallel, "Main and separate roots must differ.")
    # Both original runner locks must be free before any result is imported.
    with locked(source), locked(parallel):
        root, manifest, digests = frozen_inventory(source, repo)
        fork_root, fork_manifest, fork_digests = frozen_inventory(parallel, repo)
        F.require(F.read_json(parallel / SNAPSHOT) == expected_snapshot(source, parallel, digests, manifest)
                  and fork_digests == digests and fork_manifest.raw == manifest.raw,
                  "The source, packet or baseline adjudication differs from the copied snapshot.")
        source_files, counts = validate_opus(fork_root, fork_manifest)
        target_files = opus_inventory(root, manifest)
        F.require(set(target_files).issubset(source_files)
                  and all(source_files[name] == sha for name, sha in target_files.items()),
                  "Main Opus evidence differs. No existing evidence will be overwritten.")
        receipt = {"schema": "routegate/parallel_opus_import/1", "judge": "opus", "phase": "responses",
                   "protocol_sha256": manifest.digest(), "source_data_root": str(parallel),
                   "destination_data_root": str(source), "files": source_files, **counts,
                   "raw_attempts_and_failures_preserved": True, "api_calls": 0}
        if (root / RECEIPT).exists():
            F.require(F.read_json(root / RECEIPT) == receipt and target_files == source_files,
                      "Previously imported Opus evidence has changed.")
            return {"stage": "import_parallel_opus", "already_imported": True, **counts}
        before_review(root)
        for name, sha in sorted(source_files.items()):
            copy_exact(fork_root / name, root / name, sha)
        F.require(opus_inventory(root, manifest) == source_files, "Imported files failed verification.")
        F.put_once(root / RECEIPT, F.json_bytes(receipt))
        return {"stage": "import_parallel_opus", "api_calls": 0, "copied_files": len(source_files),
                "all_original_bytes_preserved": True, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "merge"))
    parser.add_argument("--source-data-root", type=Path, default=Path.home() / "routegate-data")
    parser.add_argument("--parallel-data-root", type=Path)
    args = parser.parse_args()
    source = args.source_data_root.expanduser().resolve()
    parallel = (args.parallel_data_root or source / FORK_NAME).expanduser().resolve()
    try:
        result = (prepare if args.stage == "prepare" else merge)(source, parallel)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValidationError, OSError, ValueError, KeyError) as error:
        print("STOP: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
