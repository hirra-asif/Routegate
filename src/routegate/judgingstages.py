"""The nine common judging stages. One manifest in, one stage at a time.

Every stage takes the same two things -- a validated manifest and a set of
resolved directories -- and every stage takes ``write``. With ``write=False``
nothing is created, nothing is opened for writing and nothing is sent: a
dry-run is a full pass over the gates that happens to produce no artifact.

Only ``judge`` costs money. The other eight are offline and re-runnable.

The stages here own bookkeeping, not science. The echo rule, the adjudication
rule and the sealed-set guards are imported from where they were frozen and
called unchanged; see ``routegate.judgingprotocol``.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from .focusedjudging import adjudicate_cell
from .judgingprotocol import (
    ADJUDICATED_CELLS_FILE,
    FAILURES_FILE,
    JOINED_CELLS_FILE,
    MANIFEST_FILENAME,
    ORIGIN_CANDIDATE,
    ORIGIN_REFERENCE,
    POSITION_A,
    POSITION_B,
    REPORT_FILE,
    RESULTS_FILE,
    REVIEW_COMPLETED_FILE,
    REVIEW_INPUT_FILE,
    REVIEW_MAP_FILE,
    REVIEW_TEMPLATE_FILE,
    ROW_MAP_FILE,
    STATE_AGREE,
    STATE_DISAGREE,
    STATE_INCOMPLETE,
    STATE_MISSING,
    STATE_UNRESOLVED_TECHNICAL,
    Manifest,
    JudgingProtocolError,
    already_done,
    assert_no_secrets,
    assert_output_dir_is_new,
    assert_packet_new,
    assert_row_is_blind,
    cell_key,
    compose_verdict,
    decode_pairwise,
    draw_orientation,
    expected_coverage,
    index_cells,
    load_rubric_text,
    preserve_raw,
    project_row,
    register_rubric_vocabulary,
    validate_reply,
    verify_inputs,
)
from .runrecord import environment_record, utc_now
from .triplets import (
    ENV_API_KEY_HISTORICAL,
    ENV_ENDPOINT,
    EmptyJudgeResponse,
    JudgeTransportError,
    TripletError,
    assert_opaque,
    extract_returned_model,
    extract_text,
    model_name_accepted,
    new_opaque_ids,
    next_raw_response_path,
    parse_judge_text,
    read_jsonl,
    record_hash,
    redact_exception_message,
    request_body,
    request_hash,
    request_headers,
    result_join_key,
    sha256_text,
    write_jsonl,
)
from .validation import Checklist

from . import __version__ as PACKAGE_VERSION

#: A source record, as an experiment must produce it. Deliberately small: an
#: experiment's own generation code adapts into this and nothing else.
SOURCE_FIELDS: Tuple[str, ...] = (
    "screen_row_id", "arm_id", "row_class", "prompt_text", "response_text")



def resolved_dirs(manifest: Manifest, root: Path) -> Dict[str, Path]:
    """Every stage directory under one root. Used by tests and the runner."""
    return {name: Path(root) / str(sub)
            for name, sub in manifest.outputs.items()
            if name != "root_key"}


# ============================================================== 1. preflight
def preflight(manifest: Manifest, dirs: Mapping[str, Path],
              input_root: Path, protocol_root: Path,
              write: bool = False) -> Dict[str, Any]:
    """Every gate that can be checked before anything is built or billed."""
    checks = Checklist(f"{manifest.experiment_id}_preflight")
    checks.record("manifest_schema", True,
                  f"{manifest.path.name} @ {manifest.digest()[:12]}")
    checks.record("no_secret_in_manifest", True,
                  "walked by key name and value shape")
    if manifest.raw.get("data_scope") == "released_final_evaluation_v5":
        checks.record("released_final_evaluation_scope", True,
                      "the final entry point validates the released inputs before each stage")
    else:
        checks.record("sealed_set_unreachable", True,
                      "manifest and path file carry no sealed token")

    verify_inputs(manifest, checks, root=input_root)

    for name, rubric in sorted(manifest.rubrics.items()):
        load_rubric_text(manifest, rubric, root=protocol_root)
        checks.record(f"rubric_{name}_digests", True,
                      f"instructions {rubric.instructions_sha256[:12]}, "
                      f"template {rubric.user_template_sha256[:12]}, "
                      f"schema {rubric.schema_sha256[:12]}")

    expect = expected_coverage(manifest)
    checks.record("coverage_arithmetic", True,
                  f"{expect['cells']} cell(s), "
                  f"{expect['records_per_judge']} per judge, "
                  f"{expect['planned_calls']} planned call(s)")
    checks.record("judges_are_the_frozen_pair", True,
                  ", ".join(f"{j}={manifest.judges[j]['model']}"
                            for j in manifest.judge_order))
    checks.record("sampling", True,
                  f"T={manifest.temperature}, "
                  f"max_tokens={manifest.max_tokens}")
    checks.record("adjudication_procedure", True, manifest.procedure)
    checks.record("failure_policy", True,
                  "missing, malformed and transport failures are all "
                  "unresolved_technical; none can become a verdict")

    packet = Path(dirs["packet"])
    runs = Path(dirs["runs"])
    built = packet.exists() and any(packet.iterdir())
    judged = runs.exists() and any(runs.iterdir())
    checks.record("packet_state", True,
                  f"packet {'built' if built else 'not built'}; "
                  f"judging {'started' if judged else 'not started'}")
    if judged and not built:
        raise JudgingProtocolError(
            f"{runs} holds judging runs but {packet} is empty. The packet a "
            f"verdict was produced against is missing; nothing proceeds.")
    return {"stage": "preflight", "checks": checks.as_dict(),
            "coverage": expect, "manifest_sha256": manifest.digest()}


# =========================================================== 2. build packet
def build_packet(manifest: Manifest, dirs: Mapping[str, Path],
                 input_root: Path, write: bool = True) -> Dict[str, Any]:
    """Mint opaque ids and project blind rows. Built once, never again."""
    packet = Path(dirs["packet"])
    runs = Path(dirs["runs"])
    if write:
        assert_packet_new(packet, runs)

    inputs = manifest.inputs
    candidates = list(read_jsonl(
        Path(input_root) / str(inputs["candidate_records"]["path"])))
    references: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    if "reference_records" in inputs:
        for record in read_jsonl(
                Path(input_root) / str(inputs["reference_records"]["path"])):
            references[(str(record["screen_row_id"]),
                        str(record["row_class"]))] = record

    for record in candidates:
        row_class = str(record.get("row_class", ""))
        if row_class not in manifest.classes:
            raise JudgingProtocolError("source record has an undeclared class")
        rubric = manifest.rubric_for_class(row_class)
        required = tuple(f for f in SOURCE_FIELDS
                         if f != "response_text" or not rubric.prompt_only)
        missing = [f for f in required if f not in record]
        if missing:
            raise JudgingProtocolError(
                f"a source record is missing {missing}; every record carries "
                f"exactly {list(SOURCE_FIELDS)}")
        if str(record["arm_id"]) not in manifest.candidate_arms:
            raise JudgingProtocolError(
                f"record on arm {record['arm_id']!r}, which the manifest does "
                f"not declare as a candidate arm")

    corpus_ids = sorted({str(r["screen_row_id"]) for r in candidates})
    rows_by_rubric: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    private: List[Dict[str, Any]] = []

    for record in sorted(candidates,
                         key=lambda r: (str(r["row_class"]),
                                        str(r["screen_row_id"]))):
        row_class = str(record["row_class"])
        if row_class not in manifest.classes:
            raise JudgingProtocolError(
                f"class {row_class!r} is not declared by the manifest")
        rubric = manifest.rubric_for_class(row_class)
        row_id = new_opaque_ids(rubric.row_prefix, 1)[0]
        entry: Dict[str, Any] = {
            "row_id": row_id, "rubric": rubric.name,
            "screen_row_id": str(record["screen_row_id"]),
            "row_class": row_class, "arm_id": str(record["arm_id"]),
            "prompt_sha256": sha256_text(str(record["prompt_text"])),
        }
        if not rubric.prompt_only:
            entry["candidate_response_sha256"] = sha256_text(str(record["response_text"]))
        if rubric.prompt_only:
            row = {"row_id": row_id, "prompt_text": str(record["prompt_text"])}
        elif rubric.pairwise:
            key = (str(record["screen_row_id"]), row_class)
            if key not in references:
                raise JudgingProtocolError(
                    f"{key} has no reference response; a pairwise comparison "
                    f"with one side missing is not a comparison")
            reference = references[key]
            orientation = draw_orientation()
            texts = {ORIGIN_CANDIDATE: str(record["response_text"]),
                     ORIGIN_REFERENCE: str(reference["response_text"])}
            row = {"row_id": row_id,
                   "prompt_text": str(record["prompt_text"]),
                   "response_a": texts[orientation[POSITION_A]],
                   "response_b": texts[orientation[POSITION_B]]}
            entry["reference_response_sha256"] = sha256_text(
                str(reference["response_text"]))
            entry["response_a_origin"] = orientation[POSITION_A]
            entry["response_b_origin"] = orientation[POSITION_B]
        else:
            row = {"row_id": row_id,
                   "prompt_text": str(record["prompt_text"]),
                   "response_text": str(record["response_text"])}
        row = project_row(row, rubric)
        assert_row_is_blind(row, rubric, corpus_ids)
        entry["rendered_user_sha256"] = sha256_text(
            json.dumps(row, sort_keys=True))
        rows_by_rubric[rubric.name].append(row)
        private.append(entry)

    expect = expected_coverage(manifest)
    counted = {name: len(rows) for name, rows in rows_by_rubric.items()}
    if counted != expect["cells_by_rubric"]:
        raise JudgingProtocolError(
            f"the packet holds {counted} and the manifest declares "
            f"{expect['cells_by_rubric']}")
    ids = [e["row_id"] for e in private]
    if len(set(ids)) != len(ids):
        raise JudgingProtocolError("two rows share an opaque id")

    summary = {
        "stage": "build", "experiment_id": manifest.experiment_id,
        "packet_id": manifest.packet_id,
        "manifest_sha256": manifest.digest(),
        "rows_by_rubric": dict(sorted(counted.items())),
        "rows": len(private),
        "calls_per_judge": len(private),
        "planned_calls": len(private) * len(manifest.judge_order),
        "orientation_counts": dict(sorted(Counter(
            e["response_a_origin"] for e in private
            if "response_a_origin" in e).items())),
        "created_utc": utc_now(),
    }
    if write:
        (packet / "private").mkdir(parents=True, exist_ok=True)
        for name, rows in sorted(rows_by_rubric.items()):
            write_jsonl(packet / f"{name}_judge_input.jsonl",
                        sorted(rows, key=lambda r: r["row_id"]))
        write_jsonl(packet / "private" / ROW_MAP_FILE,
                    sorted(private, key=lambda r: r["row_id"]))
        (packet / MANIFEST_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


# ================================================================= row loading
def load_row_map(packet: Path) -> Dict[str, Mapping[str, Any]]:
    path = Path(packet) / "private" / ROW_MAP_FILE
    if not path.is_file():
        raise JudgingProtocolError(
            f"{path} is not a file. The private row map is the only record of "
            f"the A/B orientation; without it a pairwise verdict cannot be "
            f"decoded.")
    out: Dict[str, Mapping[str, Any]] = {}
    for entry in read_jsonl(path):
        out[str(entry["row_id"])] = entry
    return out


def load_judge_rows(packet: Path, rubric_name: str) -> List[Mapping[str, Any]]:
    path = Path(packet) / f"{rubric_name}_judge_input.jsonl"
    if not path.is_file():
        raise JudgingProtocolError(f"{path} is absent from the packet")
    return list(read_jsonl(path))


# ======================================================= 3./4. render, judge
def render_request(manifest: Manifest, rubric, text: Mapping[str, Any],
                   row: Mapping[str, Any], model: str) -> Dict[str, Any]:
    """One request, rendered from the frozen text and nothing else."""
    values = {"item_id": row["row_id"], "prompt_text": row["prompt_text"]}
    for field in ("response_text", "response_a", "response_b"):
        if field in rubric.row_fields:
            values[field] = row[field]
    try:
        user = text["user_template"].format(**values)
    except KeyError as exc:
        raise JudgingProtocolError(
            f"the frozen {rubric.name} template wants {exc}, which this row "
            f"does not carry") from exc
    body = request_body(model, text["instructions"], user)
    return {"system": text["instructions"], "user": user, "body": body,
            "request_sha256": request_hash(text["instructions"], user, model)}


def dry_run(manifest: Manifest, dirs: Mapping[str, Path],
            protocol_root: Path, judge: Optional[str] = None
            ) -> Dict[str, Any]:
    """Render and hash every request. Calls nothing, writes nothing."""
    packet = Path(dirs["packet"])
    row_map = load_row_map(packet)
    judges = [judge] if judge else list(manifest.judge_order)
    out: Dict[str, Any] = {"stage": "dryrun", "by_judge": {}, "calls": 0}
    for name in judges:
        model = str(manifest.judges[name]["model"])
        per_rubric: Dict[str, int] = {}
        for rubric_name, rubric in sorted(manifest.rubrics.items()):
            text = load_rubric_text(manifest, rubric, root=protocol_root)
            rows = load_judge_rows(packet, rubric_name)
            for row in rows:
                assert_row_is_blind(row, rubric)
                render_request(manifest, rubric, text, row, model)
            per_rubric[rubric_name] = len(rows)
        out["by_judge"][name] = {"model": model, "rows": per_rubric,
                                 "calls": sum(per_rubric.values())}
        out["calls"] += sum(per_rubric.values())
        if len(row_map) != sum(per_rubric.values()):
            raise JudgingProtocolError(
                f"the packet holds {len(row_map)} mapped row(s) and "
                f"{sum(per_rubric.values())} judge row(s)")
    out["called_nothing"] = True
    out["wrote_nothing"] = True
    return out


def call_judge(endpoint: str, api_key: str, model: str, system: str,
               user: str, manifest: Manifest, timeout: float = 120.0
               ) -> Dict[str, Any]:
    """One HTTP call. Never retried here: it may already have been billed.

    Every exception message is redacted before it is returned, because
    ``http.client`` puts an offending header *value* into its own message and
    a runner that records the raw text would write the key to disk.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps(request_body(model, system, user)).encode("utf-8")
    started = time.time()
    try:
        request = urllib.request.Request(
            endpoint, data=payload, method="POST",
            headers=request_headers(api_key))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            code = response.getcode()
            headers = {k: v for k, v in response.headers.items()
                       if k.lower() not in ("x-api-key", "authorization",
                                            "set-cookie")}
    except urllib.error.HTTPError as exc:
        return {"http_status": exc.code, "response_headers": {},
                "raw_body": exc.read().decode("utf-8", "replace"),
                "payload": None, "elapsed_seconds": time.time() - started,
                "error": redact_exception_message(f"HTTP {exc.code}",
                                                  [api_key, endpoint])}
    except Exception as exc:
        return {"http_status": None, "response_headers": {}, "raw_body": "",
                "payload": None, "elapsed_seconds": time.time() - started,
                "error": redact_exception_message(
                    f"{type(exc).__name__}: {exc} - NOT retried; the request "
                    f"may already have been billed", [api_key, endpoint])}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        return {"http_status": code, "response_headers": headers,
                "raw_body": raw, "payload": None,
                "elapsed_seconds": time.time() - started,
                "error": redact_exception_message(
                    f"response body is not valid JSON: {exc}",
                    [api_key, endpoint])}
    return {"http_status": code, "response_headers": headers, "raw_body": raw,
            "payload": parsed, "elapsed_seconds": time.time() - started}


def run_judge(manifest: Manifest, dirs: Mapping[str, Path],
              protocol_root: Path, judge: str, write: bool = True,
              transport=call_judge) -> Dict[str, Any]:
    """One judge over every row. Resumable; a successful row is never re-sent.

    ``transport`` is injected so tests drive the whole stage without a
    network. In production it is ``call_judge``.
    """
    if judge not in manifest.judge_order:
        raise JudgingProtocolError(
            f"{judge!r} is not one of {list(manifest.judge_order)}")
    endpoint = os.environ.get(ENV_ENDPOINT, "")
    api_key = os.environ.get(ENV_API_KEY_HISTORICAL, "")
    if write and (not endpoint or not api_key):
        raise JudgingProtocolError(
            f"set {ENV_ENDPOINT} and {ENV_API_KEY_HISTORICAL} in the "
            f"environment. Neither is ever read from a manifest or a file.")

    packet = Path(dirs["packet"])
    model = str(manifest.judges[judge]["model"])
    expected_returned = str(
        manifest.judges[judge].get("expected_returned", model))
    totals = {"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0}
    per_rubric: Dict[str, Dict[str, int]] = {}

    for rubric_name, rubric in sorted(manifest.rubrics.items()):
        text = load_rubric_text(manifest, rubric, root=protocol_root)
        rows = load_judge_rows(packet, rubric_name)
        out_dir = Path(dirs["runs"]) / f"packet__{rubric_name}__{judge}"
        results_path = out_dir / RESULTS_FILE
        failures_path = out_dir / FAILURES_FILE
        done = already_done(results_path)
        history = list(read_jsonl(failures_path)) if failures_path.is_file() \
            else []
        todo = [r for r in rows if r["row_id"] not in done]
        per_rubric[rubric_name] = {"rows": len(rows),
                                   "already_done": len(done),
                                   "to_call": len(todo)}
        totals["skipped"] += len(done)
        if not write:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in todo:
            assert_row_is_blind(row, rubric)
            rendered = render_request(manifest, rubric, text, row, model)
            reply = transport(endpoint, api_key, model, rendered["system"],
                              rendered["user"], manifest)
            totals["attempted"] += 1
            rel, attempt = next_raw_response_path(
                out_dir, str(row["row_id"]), history)
            digest = preserve_raw(out_dir, rel, reply)
            base = {
                "row_id": row["row_id"], "rubric": rubric_name,
                "judge": judge, "model_requested": model,
                "temperature": manifest.temperature,
                "max_tokens": manifest.max_tokens,
                "request_sha256": rendered["request_sha256"],
                "raw_response_path": rel, "raw_response_sha256": digest,
                "attempt_number": attempt,
                "http_status": reply.get("http_status"),
                "requested_at_utc": utc_now(),
                "elapsed_seconds": reply.get("elapsed_seconds"),
            }
            try:
                if reply.get("error"):
                    raise JudgeTransportError(str(reply["error"]))
                returned = extract_returned_model(reply["payload"])
                if not model_name_accepted(returned, expected_returned):
                    raise JudgingProtocolError(
                        f"{row['row_id']}: the gateway returned model "
                        f"{returned!r}, and this run pins "
                        f"{expected_returned!r}")
                answer = extract_text(reply["payload"])
                parsed = parse_judge_text(answer)
                values = validate_reply(parsed, rubric, str(row["row_id"]))
            except (JudgingProtocolError, TripletError, JudgeTransportError,
                    EmptyJudgeResponse) as exc:
                record = dict(base, status="FAILED",
                              error_type=type(exc).__name__,
                              error=redact_exception_message(
                                  str(exc), [api_key, endpoint]),
                              billing_note="a failed call may already have "
                                           "been billed and is never retried "
                                           "automatically")
                assert_no_secrets(record, "failure record")
                _append(failures_path, record)
                totals["failed"] += 1
                continue
            record = dict(base, status="OK", parsed_result=values,
                          returned_model=returned)
            record["record_sha256"] = record_hash(record)
            assert_no_secrets(record, "result record")
            _append(results_path, record)
            totals["succeeded"] += 1

    return {"stage": "judge", "judge": judge, "model": model,
            "by_rubric": per_rubric, **totals}


def _append(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True,
                                ensure_ascii=False) + "\n")


# ===================================================================== 5. join
def _failure_rows(directory: Path) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    path = Path(directory) / FAILURES_FILE
    if path.is_file():
        for record in read_jsonl(path):
            out[str(record.get("row_id", ""))].append(record)
    return out


def join_evidence(manifest: Manifest, dirs: Mapping[str, Path],
                  write: bool = True,
                  carried: Sequence[Mapping[str, Any]] = ()
                  ) -> Dict[str, Any]:
    """Bind both judges' verdicts onto cells. Decides nothing.

    The join key is the runner's own ``row_id``, never the model's echo. The
    echo was already required to match at validation time; binding on it here
    as well would make identity depend on the model twice over.
    """
    checks = Checklist(f"{manifest.experiment_id}_join")
    joined_dir = Path(dirs["joined"])
    if write:
        assert_output_dir_is_new(joined_dir,
                                 (Path(dirs["packet"]), Path(dirs["runs"])),
                                 "the join")
    row_map = load_row_map(Path(dirs["packet"]))
    carried_index = {(str(c["row_id"]), str(c["failed_judge"])): c
                     for c in carried}
    carried_rows = {str(c["row_id"]) for c in carried}

    runs: Dict[str, Dict[str, Mapping[str, Any]]] = {
        j: {} for j in manifest.judge_order}
    failures: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {
        j: {} for j in manifest.judge_order}
    for rubric_name in sorted(manifest.rubrics):
        for judge in manifest.judge_order:
            directory = Path(dirs["runs"]) / f"packet__{rubric_name}__{judge}"
            if not directory.is_dir():
                continue
            results = directory / RESULTS_FILE
            if results.is_file():
                for record in read_jsonl(results):
                    row_id = result_join_key(record)
                    if row_id in runs[judge]:
                        raise JudgingProtocolError(
                            f"{judge} has two successful records for {row_id}")
                    runs[judge][row_id] = record
            for row_id, records in _failure_rows(directory).items():
                failures[judge].setdefault(row_id, []).extend(records)

    for judge in manifest.judge_order:
        open_failures = sorted(
            row_id for row_id in failures[judge]
            if row_id not in runs[judge]
            and (row_id, judge) not in carried_index)
        checks.require(
            f"{judge}_no_uncarried_failure", not open_failures,
            f"{len(open_failures)} failed row(s) with no verdict and not "
            f"carried as technically unresolved"
            + (f": {open_failures[:3]}. A failure is not a verdict."
               if open_failures else ""))

    cells: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row_id, entry in sorted(row_map.items()):
        rubric = manifest.rubrics[str(entry["rubric"])]
        verdicts: Dict[str, Optional[str]] = {}
        digests: Dict[str, Optional[str]] = {}
        for judge in manifest.judge_order:
            record = runs[judge].get(row_id)
            if record is None:
                verdicts[judge] = None
                digests[judge] = None
                continue
            values = record["parsed_result"]
            verdict = compose_verdict(values, rubric)
            if rubric.pairwise:
                verdict = decode_pairwise(
                    verdict,
                    {POSITION_A: entry["response_a_origin"],
                     POSITION_B: entry["response_b_origin"]}, rubric)
            verdicts[judge] = verdict
            digests[judge] = str(record.get("record_sha256", ""))

        present = [j for j in manifest.judge_order if verdicts[j] is not None]
        if row_id in carried_rows and len(present) < len(manifest.judge_order):
            state = STATE_UNRESOLVED_TECHNICAL
        elif not present:
            state = STATE_MISSING
        elif len(present) < len(manifest.judge_order):
            state = STATE_INCOMPLETE
        else:
            first = verdicts[manifest.judge_order[0]]
            state = (STATE_AGREE
                     if all(verdicts[j] == first for j in manifest.judge_order)
                     else STATE_DISAGREE)

        key = cell_key(entry, "the private row map")
        if key in cells:
            raise JudgingProtocolError(
                f"cell {key} is claimed by two rows; one cell is one row")
        cell = {"screen_row_id": entry["screen_row_id"],
                "arm_id": entry["arm_id"], "rubric": entry["rubric"],
                "row_class": entry["row_class"], "row_id": row_id,
                "state": state, "verdicts": verdicts,
                "record_sha256": digests}
        if state == STATE_UNRESOLVED_TECHNICAL:
            entry_c = next(carried_index[(row_id, j)]
                           for j in manifest.judge_order
                           if (row_id, j) in carried_index)
            cell["technically_unresolved"] = True
            cell["missing_judge"] = str(entry_c["failed_judge"])
            cell["failure_type"] = str(entry_c.get("failure_type", "unknown"))
        cells[key] = cell

    expect = expected_coverage(manifest)
    checks.require("cells_total", len(cells) == expect["cells"],
                   f"{len(cells)} cell(s), expected {expect['cells']}")
    by_state = Counter(c["state"] for c in cells.values())
    stray = sorted(set(by_state) - set(expect["terminal_states"]))
    checks.require(
        "every_cell_is_terminal", not stray,
        f"non-terminal state(s) {stray}; a cell judged by one judge or by "
        f"neither is not a result unless it is carried as technically "
        f"unresolved" if stray else f"{dict(sorted(by_state.items()))}")
    for judge in manifest.judge_order:
        want = expect["records_per_judge"] - sum(
            1 for (_r, j) in carried_index if j == judge)
        checks.require(f"{judge}_record_count", len(runs[judge]) == want,
                       f"{len(runs[judge])} successful record(s), expected "
                       f"{want}")

    summary = {
        "stage": "join", "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.digest(),
        "cells": len(cells),
        "by_state": dict(sorted(by_state.items())),
        "records_by_judge": {j: len(runs[j]) for j in manifest.judge_order},
        "valid_verdicts": sum(len(runs[j]) for j in manifest.judge_order),
        "planned_verdicts": expect["planned_calls"],
        "technically_unresolved": sorted(carried_rows),
        "checks": checks.as_dict(), "created_utc": utc_now(),
    }
    if write:
        (joined_dir / "private").mkdir(parents=True, exist_ok=True)
        write_jsonl(joined_dir / "private" / JOINED_CELLS_FILE,
                    [cells[k] for k in sorted(cells)])
        (joined_dir / "join_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


# =================================================================== 6. export
def cells_needing_review(cells: Iterable[Mapping[str, Any]], procedure: str
                         ) -> List[Tuple[str, str, str]]:
    """Which cells a human sees. Under the corpus procedure: disagreements.

    A technically unresolved cell is never here. A human adjudicates a
    disagreement between two judges; it does not supply a verdict a judge
    never gave.
    """
    out: List[Tuple[str, str, str]] = []
    for cell in cells:
        if cell.get("state") == STATE_UNRESOLVED_TECHNICAL:
            continue
        if procedure == "triplet_independent_review":
            out.append(cell_key(cell, "joined cell"))
        elif cell.get("state") == STATE_DISAGREE:
            out.append(cell_key(cell, "joined cell"))
    return sorted(out)


def export_review(manifest: Manifest, dirs: Mapping[str, Path],
                  write: bool = True) -> Dict[str, Any]:
    """The blinded human-review packet: genuine disagreements only."""
    checks = Checklist(f"{manifest.experiment_id}_export")
    review_dir = Path(dirs["review"])
    joined = Path(dirs["joined"])
    if write:
        assert_output_dir_is_new(review_dir, (joined, Path(dirs["packet"])),
                                 "the review export")
    joined_path = joined / "private" / JOINED_CELLS_FILE
    if not joined_path.is_file():
        raise JudgingProtocolError(
            f"{joined_path} is not a file; run the join stage first.")
    cells = index_cells(read_jsonl(joined_path), str(joined_path))
    wanted = cells_needing_review(cells.values(), manifest.procedure)

    packet = Path(dirs["packet"])
    text_by_row: Dict[str, Mapping[str, Any]] = {}
    for rubric_name in sorted(manifest.rubrics):
        for row in load_judge_rows(packet, rubric_name):
            text_by_row[str(row["row_id"])] = row

    human_ids = new_opaque_ids(manifest.human_row_prefix, len(wanted))
    rows, template, mapping = [], [], []
    for human_id, key in zip(human_ids, wanted):
        cell = cells[key]
        rubric = manifest.rubrics[str(cell["rubric"])]
        source = text_by_row[str(cell["row_id"])]
        row = {"row_id": human_id, "rubric_task": rubric.name}
        for field in rubric.row_fields:
            if field != "row_id":
                row[field] = source[field]
        rows.append(row)
        blank = {"row_id": human_id}
        for field in rubric.verdict_fields:
            blank[field] = ""
        blank["note"] = ""
        template.append(blank)
        mapping.append({"row_id": human_id, "judged_row_id": cell["row_id"],
                        "screen_row_id": cell["screen_row_id"],
                        "arm_id": cell["arm_id"], "rubric": rubric.name})

    for row in rows:
        blob = json.dumps(row, sort_keys=True)
        for cell in cells.values():
            if str(cell["screen_row_id"]) in blob:
                raise JudgingProtocolError(
                    f"{row['row_id']}: the review row names a corpus row")

    summary = {"stage": "export", "procedure": manifest.procedure,
               "rows": len(rows), "cells": len(cells),
               "by_rubric": dict(sorted(Counter(
                   r["rubric_task"] for r in rows).items())),
               "technically_unresolved_excluded": sum(
                   1 for c in cells.values()
                   if c.get("state") == STATE_UNRESOLVED_TECHNICAL),
               "checks": checks.as_dict(), "created_utc": utc_now()}
    if write:
        (review_dir / "private").mkdir(parents=True, exist_ok=True)
        write_jsonl(review_dir / REVIEW_INPUT_FILE, rows)
        write_jsonl(review_dir / REVIEW_TEMPLATE_FILE, template)
        write_jsonl(review_dir / "private" / REVIEW_MAP_FILE, mapping)
    return summary


# ================================================== 7./8. import, adjudicate
#: What a completed review row may carry: its opaque id, the verdict fields of
#: its own rubric, and a free-text note. A whitelist, so a reviewer's file
#: cannot carry a field -- an origin, an arm, a corpus row id -- that a public
#: review file must never hold, and cannot answer a rubric it was not shown.
REVIEW_ANSWER_FIELDS: Tuple[str, ...] = ("row_id", "note")


def review_answer_values(answer: Mapping[str, Any], rubric: Any, row_id: str
                         ) -> Dict[str, str]:
    """One reviewer answer against its own rubric's schema. Fail closed.

    The rubric comes from the private review map, never from the shape of the
    answer: a row carrying another rubric's verdict field is a mismatched row,
    and this engine has no path that decides which rubric it meant.
    """
    allowed = set(REVIEW_ANSWER_FIELDS) | set(rubric.verdict_fields)
    extra = sorted(set(answer) - allowed)
    if extra:
        raise JudgingProtocolError(
            f"{row_id}: the review row carries {extra}, and a "
            f"{rubric.name} review row carries exactly {sorted(allowed)}")
    missing = [f for f in rubric.verdict_fields if f not in answer]
    if missing:
        raise JudgingProtocolError(
            f"{row_id}: the {rubric.name} review row has no {missing} "
            f"field(s). A field the rubric asks for is answered or left "
            f"blank; it is never simply absent.")
    return {f: str(answer[f]).strip() for f in rubric.verdict_fields}


def review_orientation(row_map: Mapping[str, Mapping[str, Any]],
                       entry: Mapping[str, Any], row_id: str
                       ) -> Dict[str, str]:
    """The private A/B orientation behind one reviewed pairwise row.

    A reviewer answers about presentation positions -- reply A, reply B -- and
    is told nothing about which arm sits where. Which position held the
    candidate is recorded in the packet's private row map and nowhere else, so
    an A/B answer has no candidate/reference reading until that map is loaded.
    Absent or ambiguous, this refuses: the orientation was drawn per row, and
    a global rename would mislabel every row it was drawn the other way for.
    """
    judged = str(entry["judged_row_id"])
    source = row_map.get(judged)
    if source is None:
        raise JudgingProtocolError(
            f"{row_id}: the private row map holds no entry for {judged}, so "
            f"the A/B orientation of this pairwise row is unknown. The "
            f"answer is refused; an A/B label is never renamed by any other "
            f"means.")
    origins = {POSITION_A: str(source.get("response_a_origin", "")),
               POSITION_B: str(source.get("response_b_origin", ""))}
    if set(origins.values()) != {ORIGIN_CANDIDATE, ORIGIN_REFERENCE}:
        raise JudgingProtocolError(
            f"{row_id}: {judged} records origins "
            f"{[origins[POSITION_A], origins[POSITION_B]]}, which do not name "
            f"one {ORIGIN_CANDIDATE} and one {ORIGIN_REFERENCE}. An ambiguous "
            f"orientation is refused, never resolved by a default.")
    return origins


def import_review(manifest: Manifest, dirs: Mapping[str, Path],
                  completed: Optional[Path] = None, write: bool = True
                  ) -> Dict[str, Any]:
    """Read the reviewer's answers and adjudicate every cell.

    The frozen rule decides: ``focusedjudging.adjudicate_cell``, imported and
    called unchanged. This function only decides *what to hand it*.
    """
    checks = Checklist(f"{manifest.experiment_id}_adjudicate")
    register_rubric_vocabulary(manifest)
    out_dir = Path(dirs["adjudicated"])
    joined = Path(dirs["joined"])
    review_dir = Path(dirs["review"])
    if write:
        assert_output_dir_is_new(out_dir, (joined, review_dir), "adjudication")

    joined_path = joined / "private" / JOINED_CELLS_FILE
    if not joined_path.is_file():
        raise JudgingProtocolError(f"{joined_path} is not a file")
    cells = index_cells(read_jsonl(joined_path), str(joined_path))

    human: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    map_path = review_dir / "private" / REVIEW_MAP_FILE
    completed = Path(completed) if completed else (
        review_dir / REVIEW_COMPLETED_FILE)
    exported_ids: set = set()
    answered_ids: set = set()
    blank_ids: set = set()
    shown: Counter = Counter()
    translated: Dict[str, Counter] = defaultdict(Counter)
    if map_path.is_file() and completed.is_file():
        mapping = {str(m["row_id"]): m for m in read_jsonl(map_path)}
        exported_ids = set(mapping)
        # The private map first, and only then a translation. A pairwise
        # answer is in A/B presentation space and stays there until the map
        # that drew the orientation says which position held the candidate.
        row_map = load_row_map(Path(dirs["packet"]))
        for answer in read_jsonl(completed):
            row_id = assert_opaque(str(answer["row_id"]),
                                   manifest.human_row_prefix)
            if row_id in answered_ids:
                raise JudgingProtocolError(
                    f"{row_id} is answered twice in {completed.name}. One "
                    f"reviewed row is one answer; a duplicate is refused, "
                    f"never resolved by taking the last one.")
            answered_ids.add(row_id)
            entry = mapping.get(row_id)
            if entry is None:
                raise JudgingProtocolError(
                    f"{row_id} is not a row this export carried")
            rubric = manifest.rubrics[str(entry["rubric"])]
            values = review_answer_values(answer, rubric, row_id)
            if not all(values.values()):
                blank_ids.add(row_id)
                continue                       # blank stays UNRESOLVED
            verdict = compose_verdict(values, rubric)
            if verdict not in rubric.accepted_values:
                raise JudgingProtocolError(
                    f"{row_id}: {verdict!r} is not one of "
                    f"{list(rubric.accepted_values)}. A review cannot "
                    f"introduce a label the frozen schema does not define."
                    + (f" A pairwise review answers about presentation "
                       f"positions, so a {ORIGIN_CANDIDATE}/"
                       f"{ORIGIN_REFERENCE} label in a public review file "
                       f"would mean the blinding had already been broken."
                       if rubric.pairwise else ""))
            if rubric.pairwise:
                orientation = review_orientation(row_map, entry, row_id)
                shown[orientation[POSITION_A]] += 1
                verdict = decode_pairwise(verdict, orientation, rubric)
            translated[rubric.name][verdict] += 1
            key = cell_key(entry, "review map")
            if key in human:
                raise JudgingProtocolError(
                    f"{row_id} and {human[key]['human_row_id']} both answer "
                    f"cell {key}. One cell is one reviewed row.")
            human[key] = {"verdict": verdict, "human_row_id": row_id}

    needed = set(cells_needing_review(cells.values(), manifest.procedure))
    stray = sorted(set(human) - needed)
    checks.require(
        "no_review_of_a_cell_this_procedure_does_not_send_to_a_human",
        not stray,
        f"{len(stray)} answered cell(s) outside the {manifest.procedure} "
        f"selection, first {stray[:3]}")

    technical = sorted(k for k, c in cells.items()
                       if c.get("state") == STATE_UNRESOLVED_TECHNICAL)
    answered_technical = sorted(set(human) & set(technical))
    checks.require(
        "no_human_verdict_replaces_a_missing_judge_verdict",
        not answered_technical,
        f"{len(answered_technical)} technically unresolved cell(s) carry a "
        f"human answer, first {answered_technical[:3]}. A human adjudicates a "
        f"disagreement; it does not supply a verdict a judge never gave."
        if answered_technical else
        f"{len(technical)} technically unresolved cell(s), none answered")

    unanswered = sorted(exported_ids - answered_ids)
    checks.require(
        "every_exported_review_row_is_answered_exactly_once",
        not unanswered,
        f"{len(unanswered)} exported review row(s) carry no answer, first "
        f"{unanswered[:3]}. A partial review is not imported: a row that was "
        f"never seen and a row deliberately left blank are different findings."
        if unanswered else
        (f"{len(answered_ids)} of {len(exported_ids)} exported review row(s) "
         f"answered, {len(blank_ids)} left blank"
         if exported_ids else
         "no completed review file was supplied; no cell takes a human label"))

    adjudicated: List[Dict[str, Any]] = []
    for key in sorted(cells):
        cell = cells[key]
        payload = {"axis": str(cell["rubric"]), "verdicts": cell["verdicts"],
                   "screen_row_id": cell["screen_row_id"],
                   "arm_id": cell["arm_id"], "rubric": cell["rubric"]}
        outcome = adjudicate_cell(payload, human.get(key), manifest.procedure)
        adjudicated.append({
            "screen_row_id": cell["screen_row_id"], "arm_id": cell["arm_id"],
            "rubric": cell["rubric"], "row_class": cell["row_class"],
            "judged_row_id": cell["row_id"],
            "technically_unresolved": bool(cell.get("technically_unresolved")),
            "missing_judge": cell.get("missing_judge"),
            "failure_type": cell.get("failure_type"),
            **{f"{j}_verdict": cell["verdicts"].get(j)
               for j in manifest.judge_order},
            **outcome})

    undecided = sorted(r["judged_row_id"] for r in adjudicated
                       if r["technically_unresolved"]
                       and r.get("verdict") is not None)
    checks.require("no_technical_cell_carries_a_verdict", not undecided,
                   f"{len(undecided)} carried cell(s) hold a verdict"
                   if undecided else "no carried cell holds a verdict")

    summary = {"stage": "adjudicate", "procedure": manifest.procedure,
               "cells": len(adjudicated),
               "human_rows_exported": len(exported_ids),
               "human_rows_answered": len(answered_ids),
               "human_rows_left_blank": len(blank_ids),
               "pairwise_candidate_shown_as": {
                   POSITION_A: shown[ORIGIN_CANDIDATE],
                   POSITION_B: shown[ORIGIN_REFERENCE]},
               "human_verdicts_after_translation": {
                   name: dict(sorted(counts.items()))
                   for name, counts in sorted(translated.items())},
               "by_label_source": dict(sorted(Counter(
                   str(r["label_source"]) for r in adjudicated).items())),
               "checks": checks.as_dict(), "created_utc": utc_now()}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / ADJUDICATED_CELLS_FILE, adjudicated)
        (out_dir / "adjudication_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


# =================================================================== 9. report
def report(manifest: Manifest, dirs: Mapping[str, Path], write: bool = True
           ) -> Dict[str, Any]:
    """Verdict tallies with explicit denominators. No experiment-specific rate.

    Every planned cell is retained. Every rate names the denominator it used,
    and that denominator is the cells actually decided -- never the class
    total when fewer were decided. A rule such as "repair rate" belongs to the
    experiment and is computed from these tallies, elsewhere.
    """
    checks = Checklist(f"{manifest.experiment_id}_report")
    out_dir = Path(dirs["report"])
    adjudicated_dir = Path(dirs["adjudicated"])
    if write:
        assert_output_dir_is_new(out_dir, (adjudicated_dir,), "the report")
    path = adjudicated_dir / ADJUDICATED_CELLS_FILE
    if not path.is_file():
        raise JudgingProtocolError(
            f"{path} is not a file; run the adjudicate stage first.")
    records = list(read_jsonl(path))
    expect = expected_coverage(manifest)
    checks.require("cells", len(records) == expect["cells"],
                   f"{len(records)} cell(s), expected {expect['cells']}")

    readings: Dict[str, Any] = {}
    for row_class, spec in sorted(manifest.classes.items()):
        rubric = manifest.rubric_for_class(row_class)
        rows = [r for r in records if str(r["row_class"]) == row_class]
        decided = [r for r in rows if r.get("verdict") is not None]
        unresolved = [r for r in rows if r.get("verdict") is None]
        tally = Counter(str(r["verdict"]) for r in decided)
        readings[row_class] = {
            "class": row_class, "rubric": rubric.name,
            "planned_rows": int(spec["rows"]), "rows": len(rows),
            "decided_denominator": len(decided),
            "decided_denominator_is": (
                f"{row_class} cells carrying an adjudicated {rubric.name} "
                f"verdict; excludes any cell a judge never answered"),
            "unresolved": len(unresolved),
            "by_verdict": {v: tally.get(v, 0) for v in rubric.values},
            "rates": {v: (tally.get(v, 0) / len(decided) if decided else None)
                      for v in rubric.values},
            "complete": not unresolved and len(rows) == int(spec["rows"]),
            "unresolved_cells": sorted(
                ({"row_id": str(r["judged_row_id"]), "rubric": rubric.name,
                  "row_class": row_class,
                  "missing_judge": r.get("missing_judge"),
                  "failure_type": r.get("failure_type")}
                 for r in unresolved), key=lambda d: d["row_id"]),
        }

    unresolved_total = sum(r["unresolved"] for r in readings.values())
    summary = {
        "stage": "report", "experiment_id": manifest.experiment_id,
        "packet_id": manifest.packet_id,
        "manifest_sha256": manifest.digest(),
        "adjudication_procedure": manifest.procedure,
        "candidate_arms": list(manifest.candidate_arms),
        "reference_arms": list(manifest.reference_arms),
        "cells": len(records),
        "planned_cells": expect["cells"],
        "planned_verdicts": expect["planned_calls"],
        "decided_cells": sum(r["decided_denominator"]
                             for r in readings.values()),
        "unresolved_cells": unresolved_total,
        "unresolved_by_class": {k: v["unresolved"]
                                for k, v in sorted(readings.items())},
        "unresolved_cell_ids": sorted(
            c["row_id"] for r in readings.values()
            for c in r["unresolved_cells"]),
        "original_protocol_complete": unresolved_total == 0,
        "denominator_note": (
            "every rate divides by decided_denominator, which is the "
            "number of cells actually adjudicated for that class. No rate "
            "divides by the class total when fewer were decided."),
        "readings": readings,
        "by_label_source": dict(sorted(Counter(
            str(r["label_source"]) for r in records).items())),
        "reporting_module": manifest.reporting_module,
        "reporting_note": (
            "experiment-specific readings are computed outside this engine, "
            "from these tallies and denominators"),
        "checks": checks.as_dict(),
        "environment": environment_record(PACKAGE_VERSION),
        "created_utc": utc_now(),
    }
    assert_no_secrets(summary, "report")
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / REPORT_FILE).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


__all__ = [
    "REVIEW_ANSWER_FIELDS", "SOURCE_FIELDS", "build_packet", "call_judge",
    "cells_needing_review", "dry_run", "export_review", "import_review",
    "join_evidence", "load_judge_rows", "load_row_map", "preflight",
    "render_request", "report", "resolved_dirs", "review_answer_values",
    "review_orientation", "run_judge",
]
