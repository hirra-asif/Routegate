#!/usr/bin/env python3
"""Run one judge over one packet, resumably, with no sampling and no fallback.

**This is the only file in the retrospective validation that talks to an API.**
It is never executed during development; the user runs it in the terminal after
review. It loads no model, allocates no GPU and reads no credential from disk.

What it guarantees
------------------
*Exact protocol.* Every request is rendered from the frozen instruction and
template text, hash-gated before the first call, at temperature 0.0 with
max_tokens 300. The nine digests are written into the run manifest, so a run
whose protocol drifted is visible without re-reading the packet.

*Both judges, every row.* Sonnet and Opus each see every row. There is no
sampling, no cheaper-model tier and no early stop: the two judges are a
pre-registered pair, and dropping rows from one of them would turn a designed
comparison into an opportunistic one. ``--limit`` exists for a smoke test and
refuses to run without ``--smoke``, which writes to a separate directory.

*Resumability by opaque row key.* Results are appended to ``results.jsonl`` and
keyed by ``row_id``. On restart every row whose previous attempt succeeded is
skipped, so a successful billed call is never repeated. A row that previously
failed is retried, because a failure cost nothing and produced no verdict.

*No silent fallback.* A reply that does not parse, does not validate against the
frozen schema, or echoes the wrong ``item_id`` is written to ``failures.jsonl``
and **never becomes a verdict**. There is no regex rescue, no "closest match",
and no default label. A run with failures ends non-zero.

*No automatic retry.* A request that may have been billed is not repeated by
this script. Re-running it is a deliberate act by the operator.

*Append-only raw evidence.* Every reply is preserved before anything is parsed,
at a path that belongs to that attempt alone: ``raw/<row_id>.json`` for the
first, ``raw/<row_id>__attempt_000n.json`` for each retry. The number is
derived from the run's own history, then advanced past anything already on
disk. A retry therefore cannot overwrite the body of the attempt before it,
and every record -- successful or failed -- points at the exact file its own
call produced, whose ``raw_response_sha256`` recomputes from it.

Credentials
-----------
The key is read from ``MODEL_API_KEY`` -- the name the historical runner uses,
so one export serves both -- falling back to ``ROUTEGATE_JUDGE_API_KEY``. The
endpoint comes from ``ROUTEGATE_JUDGE_ENDPOINT``.

Neither is written into any record, printed, or committed, and neither is ever
placed on a command line: on a shared login node a command line is visible in
``ps`` to every other user. Under ``apptainer --cleanenv`` the host
environment is dropped, so the runbook propagates the key as
``APPTAINERENV_MODEL_API_KEY`` and proves it arrived by presence and length
only. The run manifest records presence, the variable name, and the length.

Usage
-----
    python experiments/triplet_validation/run_judge.py \\
        --packet <dir> --axis axis2 --judge sonnet --dry-run
    python experiments/triplet_validation/run_judge.py \\
        --packet <dir> --axis axis2 --judge sonnet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION      # noqa: E402
from routegate.runrecord import environment_record, utc_now          # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    ACCEPTED_MODEL_PREFIX,
    AXES,
    CONNECT_TIMEOUT_S,
    ENV_API_KEY,
    ENV_API_KEY_HISTORICAL,
    ENV_ENDPOINT,
    JUDGES,
    JudgeTransportError,
    MAX_TOKENS,
    READ_TIMEOUT_S,
    TEMPERATURE,
    TripletError,
    assert_opaque,
    assert_row_is_blind,
    extract_returned_model,
    extract_text,
    load_protocol,
    load_protocol_manifest,
    model_name_accepted,
    next_raw_response_path,
    parse_judge_text,
    read_jsonl,
    record_hash,
    redact_exception_message,
    request_body,
    request_hash,
    request_headers,
    validate_parsed,
    working_dir,
)

from routegate.validation import Checklist, sha256_text              # noqa: E402


def resolve_api_key() -> Tuple[Optional[str], Optional[str]]:
    """The credential, by environment variable only. Returns (value, name).

    ``MODEL_API_KEY`` is the historical contract -- it is the name the July
    2026 runner used against this gateway, so it is tried first. The
    RouteGate-specific name is accepted as an alias for an operator who
    prefers to keep the two separate.

    The value is never returned to a caller that prints it, never written to
    a file, and never placed on a command line.
    """
    for name in (ENV_API_KEY_HISTORICAL, ENV_API_KEY):
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


def judge_model(manifest: Dict[str, Any], judge: str) -> Dict[str, str]:
    for role in ("primary", "secondary"):
        spec = manifest["judges"][role]
        if spec["key"] == judge:
            return spec
    raise TripletError(f"unknown judge {judge!r}; valid: {list(JUDGES)}")


def already_done(results_path: Path) -> Dict[str, str]:
    """``row_id -> record hash`` for every previously SUCCESSFUL call."""
    if not results_path.exists():
        return {}
    done: Dict[str, str] = {}
    for record in read_jsonl(results_path):
        if record.get("status") == "OK" and record.get("row_id"):
            if record["row_id"] in done:
                raise TripletError(
                    f"{results_path} holds two successful records for "
                    f"{record['row_id']}. A duplicate successful call means the "
                    f"resume key is not unique; refusing to continue.")
            done[record["row_id"]] = record.get("record_sha256", "")
    return done


def call_api(endpoint: str, api_key: str, model: str,
             system: str, user: str) -> Dict[str, Any]:
    """One request, on the historical transport contract. No retry, ever.

    Ported from ``rq3_judge_axis_smoke.call_model_api``:

    * header ``X-Api-Key``, **not** ``Authorization: Bearer``;
    * the Anthropic Messages body in the historical key order;
    * timeouts: the historical runner used ``requests`` with a
      ``(connect, read)`` pair of 10 s / 120 s. ``urllib`` exposes only a
      single timeout, so the read bound is applied to the whole request and
      the connect bound is recorded but not separately enforceable. This is a
      documented divergence, and it is conservative in the safe direction:
      the effective bound is never tighter than the historical read bound;
    * any non-2xx is an error, and the body is preserved for inspection;
    * the parsed JSON payload is returned whole, under ``payload``.

    A timeout or transport error is reported as such and is **not** retried:
    the request may already have been billed.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps(request_body(model, system, user)).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, method="POST",
                                     headers=request_headers(api_key))
    timeout = max(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            code = response.getcode()
            headers = {k: v for k, v in response.headers.items()
                       if k.lower() not in ("x-api-key", "authorization",
                                            "set-cookie")}
    except urllib.error.HTTPError as exc:
        return {"http_status": exc.code, "raw_body": exc.read().decode(
            "utf-8", "replace"), "payload": None, "response_headers": {},
            "elapsed_seconds": time.time() - started,
            "error": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"http_status": None, "raw_body": "", "payload": None,
                "response_headers": {},
                "elapsed_seconds": time.time() - started,
                "error": redact_exception_message(
                    f"{type(exc).__name__}: {exc} — NOT retried; the "
                    f"request may already have been billed",
                    [api_key, endpoint])}
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        return {"http_status": code, "raw_body": body, "payload": None,
                "response_headers": headers,
                "elapsed_seconds": time.time() - started,
                "error": f"response body is not valid JSON: {exc}"}
    return {"http_status": code, "raw_body": body, "payload": parsed,
            "response_headers": headers,
            "elapsed_seconds": time.time() - started}


def preserve_raw(out_dir: Path, rel_path: str, reply: Dict[str, Any]) -> str:
    """Persist the exact raw body, mode 0600, machine-local. Returns its digest.

    The gateway envelope is undocumented, so an unrecognised shape must stay
    inspectable. The body may contain the judge's reasoning about a prompt, so
    it is written outside the repository, readable only by its owner, and is
    never committed.

    ``rel_path`` is the caller's already-chosen attempt path, from
    ``next_raw_response_path``, and it is **required not to exist**. An earlier
    attempt's body is evidence of a billed call: this function will refuse to
    stand on top of one rather than quietly replace it.
    """
    path = out_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise TripletError(
            f"{path} already holds a preserved response. A raw body is never "
            f"overwritten; the attempt path is chosen by "
            f"next_raw_response_path so that it cannot be.")
    payload = json.dumps({"http_status": reply.get("http_status"),
                          "response_headers": reply.get("response_headers"),
                          "raw_body": reply.get("raw_body", "")},
                         sort_keys=True, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    return sha256_text(payload)


def run(packet: Path, axis: str, judge: str, out_dir: Path,
        protocol_dir: Path, limit: Optional[int], dry_run: bool) -> int:
    if axis not in AXES:
        raise TripletError(f"unknown axis {axis!r}; valid: {list(AXES)}")
    manifest = load_protocol_manifest()
    spec = judge_model(manifest, judge)
    protocol = load_protocol(axis, directory=protocol_dir, manifest=manifest)

    rows = read_jsonl(packet / f"{axis}_judge_input.jsonl")
    if not rows:
        raise TripletError(f"{packet} holds no {axis} rows")
    for row in rows:
        assert_row_is_blind(row, axis)
        assert_opaque(row["row_id"], row["row_id"][:11])

    checks = Checklist(f"{axis}_{judge}")
    for key, value in protocol.hashes().items():
        checks.record(key, True, value)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    failures_path = out_dir / "failures.jsonl"
    done = already_done(results_path)
    # Every record this run has already written, successful or failed. It is
    # what the next attempt number is derived from, so a retry lands on its own
    # raw path instead of over the attempt before it.
    history: List[Dict[str, Any]] = []
    for path in (results_path, failures_path):
        if path.exists():
            history.extend(read_jsonl(path))
    todo = [r for r in rows if r["row_id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    print(f"[{judge}/{axis}] {len(rows)} row(s); {len(done)} already done; "
          f"{len(todo)} to call")
    print(f"[{judge}/{axis}] model {spec['model']} @ T={TEMPERATURE} "
          f"max_tokens={MAX_TOKENS}")

    api_key, api_key_env = resolve_api_key()
    endpoint = os.environ.get(ENV_ENDPOINT)
    run_manifest = {
        "axis": axis,
        "judge": judge,
        "model_requested": spec["model"],
        "model_expected_returned": spec["expected_returned"],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "schema_name": protocol.schema["name"],
        "protocol_hashes": protocol.hashes(),
        "axis1_version": manifest["axis1_version"]["name"],
        "packet": packet.name,
        "packet_rows": len(rows),
        "resumed_rows": len(done),
        # Presence only. The value is never recorded, printed, logged or
        # passed on a command line.
        "api_key_present": bool(api_key),
        "api_key_env_var": api_key_env,
        "api_key_length": len(api_key) if api_key else 0,
        "endpoint_present": bool(endpoint),
        "auto_retry": "disabled by design",
        "created_utc": utc_now(),
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }

    if dry_run:
        for row in todo[:3]:
            system, user = protocol.render(row)
            _ = request_hash(system, user, spec["model"])
        run_manifest["dry_run"] = True
        run_manifest["api_calls_attempted"] = 0
        (out_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[{judge}/{axis}] dry run: rendered and hashed, called nothing")
        return 0

    if not api_key or not endpoint:
        raise TripletError(
            f"set {ENV_API_KEY_HISTORICAL} (or {ENV_API_KEY}) and "
            f"{ENV_ENDPOINT} in the environment. They are never read from a "
            f"file and never committed. Under `apptainer --cleanenv` the "
            f"host environment is dropped, so export "
            f"APPTAINERENV_{ENV_API_KEY_HISTORICAL} before the exec — see "
            f"RUNBOOK.md.")

    attempted = failed = 0
    with open(results_path, "a", encoding="utf-8") as results, \
            open(failures_path, "a", encoding="utf-8") as failures:
        for index, row in enumerate(todo, start=1):
            system, user = protocol.render(row)
            sent_hash = request_hash(system, user, spec["model"])
            attempted += 1
            reply = call_api(endpoint, api_key, spec["model"], system, user)
            base = {
                "row_id": row["row_id"], "axis": axis, "judge": judge,
                "model_requested": spec["model"],
                "request_sha256": sent_hash,
                "response_sha256": sha256_text(reply.get("raw_body", "")),
                "http_status": reply.get("http_status"),
                "elapsed_seconds": round(reply.get("elapsed_seconds", 0.0), 3),
                "requested_at_utc": utc_now(),
                "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                "protocol_hashes": protocol.hashes(),
            }
            # The exact body is preserved before anything is parsed, so an
            # unrecognised envelope stays inspectable. Machine-local, 0600.
            rel_path, attempt_number = next_raw_response_path(
                out_dir, row["row_id"], history)
            base["raw_response_sha256"] = preserve_raw(out_dir, rel_path, reply)
            base["raw_response_path"] = rel_path
            base["attempt_number"] = attempt_number
            try:
                if reply.get("error"):
                    raise TripletError(reply["error"])
                if not (200 <= int(reply.get("http_status") or 0) < 300):
                    raise TripletError(
                        f"HTTP {reply.get('http_status')}; the body is "
                        f"preserved at {base['raw_response_path']}")
                envelope = reply["payload"]

                # Returned-model identity on the historical rule. An ABSENT
                # identifier is not accepted: model_name_accepted returns False
                # for None. The expected id is never written in as though the
                # server had confirmed it.
                returned = extract_returned_model(envelope)
                accepted = model_name_accepted(spec["model"], returned)
                base["returned_model"] = returned
                base["returned_model_accepted"] = accepted
                base["returned_model_evidence"] = (
                    "read from the response payload" if returned else
                    "the response carried no model/modelId/model_id field at "
                    "any level; the served model is UNESTABLISHED")
                if not accepted:
                    raise TripletError(
                        f"returned model {returned!r} does not match "
                        f"{spec['model']!r} (nor the accepted "
                        f"{ACCEPTED_MODEL_PREFIX!r}-prefixed form); recorded "
                        f"as a failure rather than assumed to be the "
                        f"requested model")

                text = extract_text(envelope)
                parsed = parse_judge_text(text)
                verdict = validate_parsed(parsed, protocol, row["row_id"])
            except (TripletError, JudgeTransportError, ValueError,
                    KeyError, TypeError) as exc:
                failed += 1
                record = dict(base, status="FAILED", error=f"{exc}",
                              error_type=type(exc).__name__,
                              retried=False,
                              billing_note="this request may already have been "
                                           "billed; it is recorded, not "
                                           "retried, and not silently dropped")
                record["record_sha256"] = record_hash(record)
                failures.write(json.dumps(record, sort_keys=True) + "\n")
                failures.flush()
                history.append(record)
                print(f"  [{index}/{len(todo)}] {row['row_id']} FAILED "
                      f"({type(exc).__name__})")
                continue

            record = dict(base, status="OK", parsed_result=verdict)
            record["record_sha256"] = record_hash(record)
            results.write(json.dumps(record, sort_keys=True) + "\n")
            results.flush()
            history.append(record)
            print(f"  [{index}/{len(todo)}] {row['row_id']} OK")

    run_manifest["api_calls_attempted"] = attempted
    run_manifest["failures"] = failed
    run_manifest["completed_utc"] = utc_now()
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[{judge}/{axis}] attempted {attempted}, failed {failed}")
    if failed:
        print(f"[{judge}/{axis}] failures are recorded and are NOT verdicts. "
              f"Re-run to retry them; successful rows are skipped.",
              file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--axis", required=True, choices=list(AXES))
    parser.add_argument("--judge", required=True, choices=list(JUDGES))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="permit --limit, into a separate directory")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.limit is not None and not args.smoke:
        print("[FATAL] --limit is sampling. Both judges see every row in a real "
              "run. Pass --smoke to use it, which writes elsewhere.",
              file=sys.stderr)
        return 1

    root = working_dir()
    out_dir = args.output_dir or (
        root / "judging" / ("smoke" if args.smoke else "")
        / f"{args.packet.name}__{args.axis}__{args.judge}")
    protocol_dir = args.protocol_dir or (root / "frozen" / "rq1_judge_protocol")
    try:
        return run(args.packet, args.axis, args.judge, out_dir, protocol_dir,
                   args.limit, args.dry_run)
    except TripletError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
