#!/usr/bin/env python3
"""Run one judge over the whole two-site packet, resumably, with no fallback.

**This is the only file in the two-site judging stage that talks to an API.**
It is never executed during development; the operator runs it in the terminal
after review. It loads no model, allocates no GPU and reads no credential from
disk.

One command per judge. The judge sees all three rubric files in a fixed order
-- Axis-2, then quality, then safety -- and writes one run directory per
rubric, so 480 calls per judge and 960 across the two.

What it guarantees
------------------
*Exact protocol.* Every request is rendered from the frozen instruction and
template text, hash-gated before the first call, at temperature 0.0 with
max_tokens 300. The digests are written into every run manifest, so a run whose
protocol drifted is visible without re-reading the packet.

*Both judges, every row.* Sonnet and Opus each see every row. There is no
sampling, no cheaper-model tier and no early stop. ``--limit`` exists for a
smoke test, refuses to run without ``--smoke``, and writes elsewhere.

*Resumability by opaque row key.* Results are appended to ``results.jsonl`` and
keyed by ``row_id``. On restart every row whose previous attempt succeeded is
skipped, so a successful billed call is never repeated. A row that previously
failed is retried, because a failure cost nothing and produced no verdict.

*No silent fallback.* A reply that does not parse, does not validate against the
frozen schema, or echoes the wrong ``item_id`` is written to ``failures.jsonl``
and **never becomes a verdict**. There is no regex rescue, no "closest match",
and no default label. For the quality rubric a judgement that breaks the frozen
pairing rule IS a verdict -- the frozen rule says such a judgement is recorded
verbatim and flagged, never rewritten -- and is stored with
``pairing_consistent: false``.

*No automatic retry.* A request that may have been billed is not repeated by
this script. Re-running it is a deliberate act by the operator.

*Append-only raw evidence.* Every reply is preserved before anything is parsed,
at a path that belongs to that attempt alone.

*It cannot see the orientation.* The B rows it sends carry ``response_a`` and
``response_b`` and nothing else. Which of them is the candidate is in the
private map, which this script never opens.

Credentials
-----------
The key is read from ``MODEL_API_KEY``, falling back to
``ROUTEGATE_JUDGE_API_KEY``. The endpoint comes from
``ROUTEGATE_JUDGE_ENDPOINT``. Neither is written into any record, printed, or
committed, and neither is ever placed on a command line. Under
``apptainer --cleanenv`` the host environment is dropped, so the runbook
propagates the key as ``APPTAINERENV_MODEL_API_KEY``.

Usage
-----
    python experiments/two_site_judging/run_two_site_judge.py \\
        --judge sonnet --dry-run
    python experiments/two_site_judging/run_two_site_judge.py --judge sonnet
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

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.twositejudging import (                              # noqa: E402
    PLANNED_CALLS_BY_RUBRIC,
    ROW_PREFIX,
    RUBRICS_USED,
    STAGE_NAME,
    TwoSiteJudgingError,
    judging_root,
    load_quality_protocol,
    quality_protocol_dir,
    validate_quality_parsed,
)
from routegate.triplets import (                                    # noqa: E402
    ACCEPTED_MODEL_PREFIX,
    AXIS2,
    CONNECT_TIMEOUT_S,
    ENV_API_KEY,
    ENV_API_KEY_HISTORICAL,
    ENV_ENDPOINT,
    JUDGES,
    JudgeTransportError,
    MAX_TOKENS,
    QUALITY,
    READ_TIMEOUT_S,
    SAFETY,
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
from routegate.validation import Checklist, sha256_text             # noqa: E402

#: The order the rubrics are called in. Fixed so two judges make the same calls
#: in the same order and a partially completed pair is easy to reason about.
RUBRIC_ORDER: Tuple[str, ...] = (AXIS2, QUALITY, SAFETY)
if sorted(RUBRIC_ORDER) != sorted(RUBRICS_USED):       # pragma: no cover
    raise TwoSiteJudgingError("the rubric order is not this stage's rubrics")


def resolve_api_key() -> Tuple[Optional[str], Optional[str]]:
    """The credential, by environment variable only. Returns (value, name)."""
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
    raise TwoSiteJudgingError(f"unknown judge {judge!r}; valid: {list(JUDGES)}")


def already_done(results_path: Path) -> Dict[str, str]:
    """``row_id -> record hash`` for every previously SUCCESSFUL call."""
    if not results_path.exists():
        return {}
    done: Dict[str, str] = {}
    for record in read_jsonl(results_path):
        if record.get("status") == "OK" and record.get("row_id"):
            if record["row_id"] in done:
                raise TwoSiteJudgingError(
                    f"{results_path} holds two successful records for "
                    f"{record['row_id']}. A duplicate successful call means "
                    f"the resume key is not unique; refusing to continue.")
            done[record["row_id"]] = record.get("record_sha256", "")
    return done


def call_api(endpoint: str, api_key: str, model: str,
             system: str, user: str) -> Dict[str, Any]:
    """One request, on the historical transport contract. No retry, ever.

    The body, the header and the timeouts are ``routegate.triplets``' -- the
    same contract ``experiments/triplet_validation/run_judge.py`` uses, so the
    two cannot drift. A timeout or transport error is reported as such and is
    **not** retried: the request may already have been billed.
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
        return {"http_status": exc.code,
                "raw_body": exc.read().decode("utf-8", "replace"),
                "payload": None, "response_headers": {},
                "elapsed_seconds": time.time() - started,
                "error": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"http_status": None, "raw_body": "", "payload": None,
                "response_headers": {},
                "elapsed_seconds": time.time() - started,
                "error": redact_exception_message(
                    f"{type(exc).__name__}: {exc} - NOT retried; the "
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

    ``rel_path`` is the caller's already-chosen attempt path, and it is
    **required not to exist**: an earlier attempt's body is evidence of a
    billed call and is never stood on top of.
    """
    path = out_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise TwoSiteJudgingError(
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


def validate_reply(parsed: Any, protocol: Any, rubric: str,
                   row_id: str) -> Dict[str, Any]:
    """One parsed reply against its rubric's frozen schema. No fallback."""
    if rubric == QUALITY:
        return validate_quality_parsed(parsed, row_id,
                                       protocol.schema.get("name", ""))
    return validate_parsed(parsed, protocol, row_id)


def run_one_rubric(packet: Path, rubric: str, judge: str, spec: Dict[str, str],
                   protocol: Any, out_dir: Path, limit: Optional[int],
                   dry_run: bool, api_key: Optional[str],
                   endpoint: Optional[str]) -> Tuple[int, int, int]:
    """One rubric file, one judge. Returns (rows, attempted, failed)."""
    path = packet / f"{rubric}_judge_input.jsonl"
    if not path.is_file():
        raise TwoSiteJudgingError(f"{path} is absent from the packet")
    rows = read_jsonl(path)
    if not rows:
        raise TwoSiteJudgingError(f"{path} holds no row")
    want = PLANNED_CALLS_BY_RUBRIC[rubric]
    if len(rows) != want:
        raise TwoSiteJudgingError(
            f"{path} holds {len(rows)} row(s) and this stage plans {want}. A "
            f"short packet is a short result; refused rather than run.")
    for row in rows:
        assert_row_is_blind(row, rubric)
        assert_opaque(row["row_id"], ROW_PREFIX[rubric])

    checks = Checklist(f"{rubric}_{judge}")
    for key, value in protocol.hashes().items():
        checks.record(key, True, value)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    failures_path = out_dir / "failures.jsonl"
    done = already_done(results_path)
    history: List[Dict[str, Any]] = []
    for candidate in (results_path, failures_path):
        if candidate.exists():
            history.extend(read_jsonl(candidate))
    todo = [r for r in rows if r["row_id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    print(f"[{judge}/{rubric}] {len(rows)} row(s); {len(done)} already done; "
          f"{len(todo)} to call")
    print(f"[{judge}/{rubric}] model {spec['model']} @ T={TEMPERATURE} "
          f"max_tokens={MAX_TOKENS}")

    run_manifest: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "rubric": rubric,
        "judge": judge,
        "model_requested": spec["model"],
        "model_expected_returned": spec["expected_returned"],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "schema_name": protocol.schema["name"],
        "protocol_hashes": protocol.hashes(),
        "packet": packet.name,
        "packet_rows": len(rows),
        "planned_rows": want,
        "resumed_rows": len(done),
        # Presence only. The value is never recorded, printed, logged or
        # passed on a command line.
        "api_key_present": bool(api_key),
        "api_key_env_var": (ENV_API_KEY_HISTORICAL
                            if os.environ.get(ENV_API_KEY_HISTORICAL)
                            else (ENV_API_KEY if api_key else None)),
        "api_key_length": len(api_key) if api_key else 0,
        "endpoint_present": bool(endpoint),
        "auto_retry": "disabled by design",
        "reads_the_private_map": False,
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
            json.dumps(run_manifest, indent=2, sort_keys=True),
            encoding="utf-8")
        print(f"[{judge}/{rubric}] dry run: rendered and hashed, "
              f"called nothing")
        return len(rows), 0, 0

    attempted = failed = 0
    with open(results_path, "a", encoding="utf-8") as results, \
            open(failures_path, "a", encoding="utf-8") as failures:
        for index, row in enumerate(todo, start=1):
            system, user = protocol.render(row)
            sent_hash = request_hash(system, user, spec["model"])
            attempted += 1
            reply = call_api(endpoint, api_key, spec["model"], system, user)
            base = {
                "row_id": row["row_id"], "rubric": rubric, "judge": judge,
                "model_requested": spec["model"],
                "request_sha256": sent_hash,
                "response_sha256": sha256_text(reply.get("raw_body", "")),
                "http_status": reply.get("http_status"),
                "elapsed_seconds": round(reply.get("elapsed_seconds", 0.0), 3),
                "requested_at_utc": utc_now(),
                "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                "protocol_hashes": protocol.hashes(),
            }
            rel_path, attempt_number = next_raw_response_path(
                out_dir, row["row_id"], history)
            base["raw_response_sha256"] = preserve_raw(out_dir, rel_path, reply)
            base["raw_response_path"] = rel_path
            base["attempt_number"] = attempt_number
            try:
                if reply.get("error"):
                    raise TwoSiteJudgingError(reply["error"])
                if not (200 <= int(reply.get("http_status") or 0) < 300):
                    raise TwoSiteJudgingError(
                        f"HTTP {reply.get('http_status')}; the body is "
                        f"preserved at {base['raw_response_path']}")
                envelope = reply["payload"]
                returned = extract_returned_model(envelope)
                accepted = model_name_accepted(spec["model"], returned)
                base["returned_model"] = returned
                base["returned_model_accepted"] = accepted
                base["returned_model_evidence"] = (
                    "read from the response payload" if returned else
                    "the response carried no model/modelId/model_id field at "
                    "any level; the served model is UNESTABLISHED")
                if not accepted:
                    raise TwoSiteJudgingError(
                        f"returned model {returned!r} does not match "
                        f"{spec['model']!r} (nor the accepted "
                        f"{ACCEPTED_MODEL_PREFIX!r}-prefixed form); recorded "
                        f"as a failure rather than assumed to be the "
                        f"requested model")
                text = extract_text(envelope)
                parsed = parse_judge_text(text)
                verdict = validate_reply(parsed, protocol, rubric,
                                         row["row_id"])
            except (TwoSiteJudgingError, TripletError, JudgeTransportError,
                    ValueError, KeyError, TypeError) as exc:
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
    print(f"[{judge}/{rubric}] attempted {attempted}, failed {failed}")
    return len(rows), attempted, failed


def run(packet: Path, judge: str, runs_dir: Path, protocol_dir: Path,
        quality_dir: Path, rubrics: Sequence[str], limit: Optional[int],
        dry_run: bool) -> int:
    manifest = load_protocol_manifest()
    spec = judge_model(manifest, judge)
    protocols: Dict[str, Any] = {}
    for rubric in rubrics:
        if rubric == QUALITY:
            protocols[rubric] = load_quality_protocol(quality_dir)
        else:
            protocols[rubric] = load_protocol(rubric, directory=protocol_dir,
                                              manifest=manifest)

    api_key, _ = resolve_api_key()
    endpoint = os.environ.get(ENV_ENDPOINT)
    if not dry_run and (not api_key or not endpoint):
        raise TwoSiteJudgingError(
            f"set {ENV_API_KEY_HISTORICAL} (or {ENV_API_KEY}) and "
            f"{ENV_ENDPOINT} in the environment. They are never read from a "
            f"file and never committed. Under `apptainer --cleanenv` the host "
            f"environment is dropped, so export "
            f"APPTAINERENV_{ENV_API_KEY_HISTORICAL} before the exec -- see "
            f"RUNBOOK.md.")

    total_rows = total_attempted = total_failed = 0
    for rubric in rubrics:
        out_dir = runs_dir / f"{packet.name}__{rubric}__{judge}"
        rows, attempted, failed = run_one_rubric(
            packet, rubric, judge, spec, protocols[rubric], out_dir, limit,
            dry_run, api_key, endpoint)
        total_rows += rows
        total_attempted += attempted
        total_failed += failed

    print(f"[{judge}] {total_rows} row(s) across {list(rubrics)}; "
          f"attempted {total_attempted}, failed {total_failed}")
    if total_failed:
        print(f"[{judge}] failures are recorded and are NOT verdicts. Re-run "
              f"the identical command to retry them; successful rows are "
              f"skipped and are not billed twice.", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", required=True, choices=list(JUDGES))
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--quality-dir", type=Path, default=None)
    parser.add_argument("--rubric", action="append", default=None,
                        choices=list(RUBRIC_ORDER),
                        help="restrict to one rubric; repeatable. The default "
                             "is all three, which is what a real run does.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="permit --limit, into a separate directory")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.limit is not None and not args.smoke:
        print("[FATAL] --limit is sampling. Both judges see every row in a "
              "real run. Pass --smoke to use it, which writes elsewhere.",
              file=sys.stderr)
        return 1

    rubrics = ([r for r in RUBRIC_ORDER if r in set(args.rubric)]
               if args.rubric else list(RUBRIC_ORDER))
    try:
        root = judging_root() / "judging"
        packet = args.packet or (root / "packet")
        runs_dir = args.runs_dir or (root / ("smoke_runs" if args.smoke
                                             else "runs"))
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        quality_dir = args.quality_dir or quality_protocol_dir()
        return run(packet, args.judge, runs_dir, protocol_dir, quality_dir,
                   rubrics, args.limit, args.dry_run)
    except (TwoSiteJudgingError, TripletError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
