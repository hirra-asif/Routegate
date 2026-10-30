#!/usr/bin/env python3
"""Gold-free technical verification of one smoke item across three conditions.

    python experiments/final_evaluation/verify_technical_smoke.py \
        --dataset tinymmlu --item-index 0

Checks that the machinery ran, not that the model was right. It reads the three
arms' records, recomputes the gate from what was recorded, compares hook counts
against the condition, re-verifies every frozen digest, confirms the three
conditions saw the identical prompt, and executes the frozen parser to confirm
it runs and yields a status.

WHAT IT CANNOT SEE
------------------
It imports ``routegate.capabilityparsers``, which carries no gold map, no
comparison, no accuracy and no denominator. It never opens the gold column,
never imports ``capabilityscoring``, and never prints a parsed prediction. A
technical PASS is therefore independent of whether any answer was right.

THE ONE PLACE PARSING IS A GATE
-------------------------------
The BASELINE row is the predeclared format-compatibility check, and it turns on
PARSING ALONE: if the baseline completion does not parse, the prompt mode is
wrong and the result is ``BLOCKED_FORMAT_COMPATIBILITY``. The row is preserved
and its correctness is not inspected.

TRUNCATION IS NOT A PARSE FAILURE
---------------------------------
A completion that parsed and then ran into the token cap has already produced
its answer; the cap fell after it. Under raw completion that is the expected
shape -- the model emits the option letter and carries on into the next
few-shot question until the budget ends. Such a row is reported as
``PARSE_OK_TRUNCATED_AFTER_ANSWER``, is a technical PASS in every condition,
and its parsed answer is scored normally. The cap is diagnostic metadata and
never changes correctness.

Only an ACTUAL parse failure is a finding: at baseline it is
``BLOCKED_FORMAT_COMPATIBILITY``; under ``dynamic_raw`` or
``safety_first_routegate`` it is ``BEHAVIOURAL_PARSE_FAILURE`` -- possible
intervention-induced degradation, which is a result rather than a fault. That
row is retained unchanged and scored incorrect later by the frozen rule.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.capabilityparsers import (                        # noqa: E402
    PARSE_OK, parse_gsm8k, parse_mmlu)
from routegate.finaleval import (                                # noqa: E402
    CONDITIONS, COND_BASELINE, COND_DYNAMIC, COND_ROUTEGATE, DATASET_SPEC,
    GSM8K_STOP_STRINGS,
    DS_GSM8K, DS_MMLU, PREDECLARATION, RECORDS_FILE, RECORD_SCHEMA,
    SENSOR_WEIGHTS_SHA256, CONTROLLER_WEIGHTS_SHA256, arm_id, arm_output_dir,
    expected_rows, source_digests, verify_protocol)
from routegate.finalrunner import (                              # noqa: E402
    DYNAMIC_HOOKS, DYNAMIC_READ_STATE, GATE_READ_STATE,
    NOOP_GATE_DID_NOT_FIRE, STATIC_HOOKS, TELEMETRY_FIELDS)
from routegate.validation import (                               # noqa: E402
    Checklist, ValidationError, sha256_file)

STATUS_PASS = "PASS"
STATUS_BLOCKED = "BLOCKED"
STATUS_BOUNDARY = "BLOCKED_GSM8K_BOUNDARY"
STRICT_MARKER = re.compile(r"#### (\-?[0-9\.\,]+)")
STATUS_FORMAT = "BLOCKED_FORMAT_COMPATIBILITY"
BEHAVIOURAL = "BEHAVIOURAL_PARSE_FAILURE"
#: Parsed successfully, then hit the cap. Diagnostic only, never a fault and
#: never a reason to change correctness.
TRUNCATED_AFTER_ANSWER = "PARSE_OK_TRUNCATED_AFTER_ANSWER"

PARSERS = {DS_MMLU: parse_mmlu, DS_GSM8K: parse_gsm8k}
EXPECTED_STATES = {COND_DYNAMIC: [31, 32], COND_ROUTEGATE: [31, 32, 33]}


class SmokeVerificationError(ValidationError):
    """A smoke row does not evidence the machinery it claims to have used."""


def read_one(path: Path, index: int, checks: Checklist) -> Dict[str, Any]:
    """The single record for this item, refusing duplicates and version mixing."""
    if not path.is_file():
        raise SmokeVerificationError(f"{path} is absent; that arm has no row")
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    schemas = {str(r.get("schema")) for r in rows}
    stray = sorted(s for s in schemas if s != RECORD_SCHEMA)
    checks.require(f"{path.parent.name}_no_version_mixing", not stray,
                   f"{stray} beside {RECORD_SCHEMA}" if stray
                   else f"every record is {RECORD_SCHEMA}")
    ids = [str(r["item_id"]) for r in rows]
    checks.require(f"{path.parent.name}_no_duplicate_rows",
                   len(ids) == len(set(ids)), f"{len(ids)} row(s), "
                   f"{len(set(ids))} unique")
    matching = [r for r in rows if int(r["item_index"]) == index]
    checks.require(f"{path.parent.name}_exactly_one_record_for_the_item",
                   len(matching) == 1,
                   f"{len(matching)} record(s) with item_index {index}")
    return matching[0]


def check_common(dataset: str, condition: str, record: Mapping[str, Any],
                 checks: Checklist) -> None:
    name = arm_id(dataset, condition)
    checks.require(f"{name}_schema", record.get("schema") == RECORD_SCHEMA,
                   f"{record.get('schema')!r}")
    checks.require(f"{name}_condition", record.get("condition") == condition,
                   f"{record.get('condition')!r}")
    checks.require(f"{name}_prompt_mode",
                   record.get("prompt_mode")
                   == DATASET_SPEC[dataset]["prompt_mode"],
                   f"{record.get('prompt_mode')!r}")
    for field in TELEMETRY_FIELDS:
        checks.require(f"{name}_has_{field}", field in record,
                       f"{field} present")
    checks.require(f"{name}_structurally_complete",
                   bool(record.get("response_text") is not None)
                   and record.get("status") == "ok"
                   and record.get("completion_token_count") is not None,
                   f"status {record.get('status')!r}")
    checks.require(f"{name}_terminated_normally",
                   record.get("eos_reached") is True
                   or record.get("hit_max_new_tokens") is True,
                   f"eos {record.get('eos_reached')} cap "
                   f"{record.get('hit_max_new_tokens')}")


def check_baseline(record: Mapping[str, Any], checks: Checklist) -> None:
    checks.require("baseline_no_controller", record.get("mode") == "off",
                   f"mode {record.get('mode')!r}")
    for field in ("gate_p_E", "gate_p_H", "gate_fires", "gate_read_state",
                  "threshold_E", "threshold_H", "g_R",
                  "controller_read_state"):
        checks.require(f"baseline_{field}_is_null",
                       record.get(field) is None, f"{record.get(field)!r}")
    alpha = record.get("applied_alpha_R")
    checks.require("baseline_applied_alpha_null_or_zero",
                   alpha is None or float(alpha) == 0.0, f"{alpha!r}")
    checks.require("baseline_hooks", int(record["hooks_expected"]) == 0
                   and int(record["hooks_fired"]) == 0,
                   f"{record['hooks_expected']}/{record['hooks_fired']}")


def check_dynamic(record: Mapping[str, Any], checks: Checklist) -> None:
    import math
    checks.require("dynamic_controller_present",
                   record.get("mode") == "patch",
                   f"mode {record.get('mode')!r}")
    checks.require("dynamic_read_state",
                   record.get("controller_read_state") == DYNAMIC_READ_STATE,
                   f"{record.get('controller_read_state')!r}")
    g_r = record.get("g_R")
    checks.require("dynamic_g_R_is_finite",
                   g_r is not None and math.isfinite(float(g_r)),
                   f"g_R {g_r!r}")
    alpha = float(record["applied_alpha_R"])
    checks.require("dynamic_applied_alpha_relationship",
                   abs(alpha - 0.15 * float(g_r)) <= 1e-12 and alpha > 0.0,
                   f"applied {alpha!r} against 0.15 * g_R")
    checks.require("dynamic_states",
                   list(record.get("applied_at_historical_blocks") or [])
                   == EXPECTED_STATES[COND_DYNAMIC],
                   f"{record.get('applied_at_historical_blocks')}")
    checks.require("dynamic_hooks",
                   int(record["hooks_expected"]) == DYNAMIC_HOOKS
                   and int(record["hooks_fired"]) == DYNAMIC_HOOKS,
                   f"{record['hooks_expected']}/{record['hooks_fired']}")


def check_routegate(record: Mapping[str, Any], checks: Checklist) -> None:
    import math
    p_e, p_h = record.get("gate_p_E"), record.get("gate_p_H")
    checks.require("gate_probabilities_finite",
                   p_e is not None and p_h is not None
                   and math.isfinite(float(p_e)) and math.isfinite(float(p_h)),
                   f"p_E {p_e!r} p_H {p_h!r}")
    checks.require("gate_read_state",
                   record.get("gate_read_state") == GATE_READ_STATE,
                   f"{record.get('gate_read_state')!r}")
    from routegate.gatesensors import load_thresholds
    frozen = load_thresholds()
    checks.require("gate_thresholds_match_the_frozen_ones",
                   float(record["threshold_E"]) == frozen.threshold_E
                   and float(record["threshold_H"]) == frozen.threshold_H,
                   f"E {record['threshold_E']!r} H {record['threshold_H']!r}")
    recomputed = (float(p_e) >= float(record["threshold_E"])
                  and float(p_h) < float(record["threshold_H"]))
    checks.require("gate_decision_recomputes",
                   bool(record["gate_fires"]) == recomputed,
                   f"recorded {record['gate_fires']}, recomputed {recomputed}")
    if record["gate_fires"]:
        checks.require("gate_fired_states",
                       list(record.get("applied_at_historical_blocks") or [])
                       == EXPECTED_STATES[COND_ROUTEGATE],
                       f"{record.get('applied_at_historical_blocks')}")
        checks.require("gate_fired_hooks",
                       int(record["hooks_expected"]) == STATIC_HOOKS
                       and int(record["hooks_fired"]) == STATIC_HOOKS,
                       f"{record['hooks_expected']}/{record['hooks_fired']}")
    else:
        checks.require("gate_noop_hooks",
                       int(record["hooks_expected"]) == 0
                       and int(record["hooks_fired"]) == 0,
                       f"{record['hooks_expected']}/{record['hooks_fired']}")
        checks.require("gate_noop_reason_is_explicit",
                       record.get("noop_reason") == NOOP_GATE_DID_NOT_FIRE,
                       f"{record.get('noop_reason')!r}")


def check_gsm8k_boundary(record: Mapping[str, Any], checks: Checklist) -> None:
    """The stored completion must end at the current question, not run past it.

    This is what blocked v4: an unbounded completion carried later few-shot
    questions and their answers, and the last-number rule reached into them.
    Nothing here reads a value; it counts markers and checks the boundary.
    """
    text = str(record.get("response_text") or "")
    declared = list(record.get("stop_strings") or [])
    checks.require("gsm8k_stop_strings_declared",
                   declared == list(GSM8K_STOP_STRINGS),
                   f"{declared}")
    present = [s for s in GSM8K_STOP_STRINGS if s in text]
    checks.require("no_stop_sequence_survives_in_the_stored_completion",
                   not present,
                   f"{present} still present" if present
                   else "no stop sequence remains")
    checks.require("no_later_question_in_the_stored_completion",
                   "Question:" not in text,
                   "a later 'Question:' is present" if "Question:" in text
                   else "the completion ends at the current question")
    markers = STRICT_MARKER.findall(text)
    checks.require("exactly_one_current_question_answer_marker",
                   len(markers) == 1,
                   f"{len(markers)} strict '####' marker(s); the frozen rule "
                   f"expects the current question's answer and no other")
    reason = record.get("stop_reason")
    fired = bool(record.get("stop_fired"))
    checks.require("stop_telemetry_is_consistent",
                   reason in ("stop_string", "eos", "max_new_tokens")
                   and (reason == "stop_string") == fired
                   and not (fired and record.get("hit_max_new_tokens")),
                   f"stop_reason {reason!r} stop_fired {fired} cap "
                   f"{record.get('hit_max_new_tokens')}")


def check_digests(checks: Checklist) -> None:
    verify_protocol(PREDECLARATION, checks)
    for dataset in (DS_MMLU, DS_GSM8K):
        spec = DATASET_SPEC[dataset]
        local = Path.home() / spec["local"]
        checks.require(f"{dataset}_dataset_digest",
                       sha256_file(local) == spec["sha256"],
                       spec["sha256"][:16])
    sensors = REPO_ROOT / "outputs" / "routegate_gate_deployment_v1"
    for key, filename in (("E_sensor",
                           "gate_sensor_E_all_development_refit.json"),
                          ("H_veto",
                           "gate_sensor_H_all_development_refit.json")):
        blob = json.loads((sensors / filename).read_text(encoding="utf-8"))
        checks.require(f"{key}_weights_digest",
                       blob["weights_sha256"] == SENSOR_WEIGHTS_SHA256[key],
                       SENSOR_WEIGHTS_SHA256[key][:16])
    controller = json.loads(
        (Path.home() / "routegate-data/gate_oof_v1_repro"
         / "controller_all_development_refit.json").read_text(encoding="utf-8"))
    checks.require("controller_weights_digest",
                   controller["weights_sha256"] == CONTROLLER_WEIGHTS_SHA256,
                   CONTROLLER_WEIGHTS_SHA256[:16])
    checks.require("source_digests", source_digests()
                   == json.loads(PREDECLARATION.read_text(
                       encoding="utf-8"))["source_sha256"],
                   "every v3 source file matches the frozen protocol")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[DS_MMLU, DS_GSM8K], required=True)
    parser.add_argument("--item-index", type=int, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "routegate-data")
    args = parser.parse_args(argv)

    checks = Checklist(f"technical_smoke_{args.dataset}_{args.item_index}")
    verdict = STATUS_PASS
    notes: List[str] = []
    try:
        want = expected_rows(args.dataset)
        if not 0 <= args.item_index < want:
            raise SmokeVerificationError(
                f"--item-index {args.item_index} is outside [0, {want})")
        check_digests(checks)

        records: Dict[str, Dict[str, Any]] = {}
        for condition in CONDITIONS:
            path = (args.data_root
                    / arm_output_dir(args.dataset, condition) / RECORDS_FILE)
            records[condition] = read_one(path, args.item_index, checks)

        ids = {r["item_id"] for r in records.values()}
        indexes = {int(r["item_index"]) for r in records.values()}
        digests = {r["prompt_sha256"] for r in records.values()}
        modes = {r["prompt_mode"] for r in records.values()}
        checks.require("identical_item_id_across_conditions", len(ids) == 1,
                       f"{sorted(ids)}")
        checks.require("identical_item_index_across_conditions",
                       indexes == {args.item_index}, f"{sorted(indexes)}")
        checks.require("identical_prompt_digest_across_conditions",
                       len(digests) == 1, f"{len(digests)} distinct digest(s)")
        checks.require("identical_prompt_mode_across_conditions",
                       len(modes) == 1, f"{sorted(modes)}")

        for condition, record in records.items():
            check_common(args.dataset, condition, record, checks)
        if args.dataset == DS_GSM8K:
            for condition, record in records.items():
                check_gsm8k_boundary(record, checks)
        check_baseline(records[COND_BASELINE], checks)
        check_dynamic(records[COND_DYNAMIC], checks)
        check_routegate(records[COND_ROUTEGATE], checks)

        # ---- parser execution. Status only; the prediction is never shown.
        parse = PARSERS[args.dataset]
        for condition, record in records.items():
            _prediction, status = parse(record.get("response_text", ""))
            capped = bool(record.get("hit_max_new_tokens"))
            # Parse status and token-cap status are reported SEPARATELY. A cap
            # after a successful parse is not a parse failure.
            checks.record(f"{condition}_parse_status", True, str(status))
            checks.record(f"{condition}_hit_max_new_tokens", True, str(capped))
            if status == PARSE_OK:
                if capped:
                    notes.append(
                        f"{TRUNCATED_AFTER_ANSWER} under {condition}: the "
                        f"answer parsed and the cap fell after it. Diagnostic "
                        f"only; the parsed answer is scored normally and "
                        f"correctness is unaffected.")
                continue
            if condition == COND_BASELINE:
                verdict = STATUS_FORMAT
                notes.append(
                    f"baseline {args.dataset}: parse status {status}. The "
                    f"prompt mode is wrong for this benchmark. The row is "
                    f"preserved and its correctness is not inspected.")
            else:
                notes.append(
                    f"{BEHAVIOURAL} under {condition}: parse status {status}. "
                    f"Retained unchanged; scored as incorrect by the frozen "
                    f"rule in the scoring stage. Not a technical fault and "
                    f"not a reason to invalidate the arm.")

        if not checks.all_passed:
            verdict = STATUS_BLOCKED
    except ValidationError as exc:
        print(f"[smoke] {STATUS_BLOCKED}: {exc}", file=sys.stderr)
        return 1

    print(f"[smoke] dataset {args.dataset} item_index {args.item_index}")
    for condition in CONDITIONS:
        print(f"[smoke]   {arm_id(args.dataset, condition)}")
    for note in notes:
        print(f"[smoke] note: {note}")
    print(f"[smoke] {checks.as_dict()['checks_run']} technical check(s), all "
          f"passed: {checks.all_passed}")
    print(f"[smoke] TECHNICAL VERDICT: {verdict}")
    print(f"[smoke] no gold label, correctness, score or parsed prediction is "
          f"read or printed by this command")
    if args.dataset == DS_GSM8K:
        print(f"[smoke] gsm8k boundary: stop sequences "
              f"{list(GSM8K_STOP_STRINGS)}; the stored completion is checked "
              f"to end at the current question")
    return 0 if verdict == STATUS_PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
