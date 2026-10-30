#!/usr/bin/env python3
"""Gold-free technical verification, with stop-aware termination. v5.1.

    python experiments/final_evaluation/verify_technical_smoke_v5_1.py \
        --dataset tinygsm8k --item-index 0

WHY THIS FILE EXISTS SEPARATELY
-------------------------------
``verify_technical_smoke.py`` is pinned by digest in
``manifests/routegate_final_evaluation_v5.json``. Editing it in place would
invalidate that frozen protocol, so it is left byte-for-byte unchanged and this
corrected verifier sits beside it, pinned in its own erratum manifest.

WHAT WAS WRONG
--------------
The original ``check_common`` was written before stop sequences existed and
required ``eos_reached OR hit_max_new_tokens`` as proof that generation
terminated normally. v5 added a custom stopping criterion for tinyGSM8K, and
when a declared stop fires neither flag is set -- generation ended because the
boundary was reached, which is the *intended* terminal mode. The verifier
therefore blocked three technically correct rows.

Nothing about generation or scoring changes here. Every other check, including
every GSM8K boundary check, is imported unchanged from the original module so
the delta is exactly one function.

THE CORRECTED RULE
------------------
A record must evidence exactly one consistent terminal mode:

* **EOS** -- the model emitted end-of-sequence;
* **max_new_tokens** -- the budget ran out;
* **declared custom stop** -- a stop sequence from the dataset's FROZEN list
  fired, ``hit_max_new_tokens`` is false, and neither the stop nor anything
  after it survives in the stored completion.

A dataset that declares no stop sequences -- tinyMMLU and sealed -- cannot
claim custom-stop termination. That keeps the escape hatch closed: an
undeclared stop can never be used to make a row look terminated.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

_ORIGINAL = Path(__file__).with_name("verify_technical_smoke.py")


def _load_original() -> Any:
    """The pinned verifier, imported unchanged. Its digest must not move."""
    spec = importlib.util.spec_from_file_location(
        "_verify_technical_smoke_pinned", _ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_original()

# Everything except check_common is reused verbatim.
read_one = base.read_one
check_baseline = base.check_baseline
check_dynamic = base.check_dynamic
check_routegate = base.check_routegate
check_gsm8k_boundary = base.check_gsm8k_boundary
check_digests = base.check_digests
PARSERS = base.PARSERS
STATUS_PASS = base.STATUS_PASS
STATUS_BLOCKED = base.STATUS_BLOCKED
STATUS_FORMAT = base.STATUS_FORMAT
BEHAVIOURAL = base.BEHAVIOURAL
TRUNCATED_AFTER_ANSWER = base.TRUNCATED_AFTER_ANSWER
SmokeVerificationError = base.SmokeVerificationError

from routegate.capabilityparsers import PARSE_OK                 # noqa: E402
from routegate.finaleval import (                                # noqa: E402
    CONDITIONS, COND_BASELINE, COND_DYNAMIC, COND_ROUTEGATE, DATASET_SPEC,
    DS_GSM8K, DS_MMLU, PREDECLARATION, RECORDS_FILE, RECORD_SCHEMA, arm_id,
    arm_output_dir, expected_rows)
from routegate.finalrunner import TELEMETRY_FIELDS               # noqa: E402
from routegate.validation import Checklist, ValidationError       # noqa: E402

ERRATUM = "routegate_verifier_erratum_v5_1"
TERMINAL_EOS = "eos"
TERMINAL_MAX = "max_new_tokens"
TERMINAL_STOP = "stop_string"
TERMINAL_MODES = (TERMINAL_EOS, TERMINAL_MAX, TERMINAL_STOP)


def terminal_mode(dataset: str, record: Mapping[str, Any]) -> str:
    """Which terminal mode this record evidences, or a refusal.

    Exactly one, and it must be consistent with the dataset's frozen stop
    list. A row that evidences none did not terminate in a way this stage can
    account for and is refused rather than accepted on the strength of a flag.
    """
    declared = list(DATASET_SPEC[dataset].get("stop_strings") or [])
    fired = bool(record.get("stop_fired"))
    reason = record.get("stop_reason")
    capped = bool(record.get("hit_max_new_tokens"))
    eos = bool(record.get("eos_reached"))

    if reason is not None and reason not in TERMINAL_MODES:
        raise SmokeVerificationError(
            f"stop_reason {reason!r} is not one of {list(TERMINAL_MODES)}")

    if fired:
        if not declared:
            raise SmokeVerificationError(
                f"{dataset} declares no stop sequences, so it cannot claim "
                f"custom-stop termination. An undeclared stop is never a way "
                f"to make a row look terminated.")
        if reason != TERMINAL_STOP:
            raise SmokeVerificationError(
                f"stop_fired is true but stop_reason is {reason!r}")
        if capped:
            raise SmokeVerificationError(
                "a fired stop sequence and hit_max_new_tokens cannot both be "
                "true; the boundary was reached before the budget")
        if list(record.get("stop_strings") or []) != declared:
            raise SmokeVerificationError(
                f"the record's stop_strings do not match the frozen list for "
                f"{dataset}")
        return TERMINAL_STOP
    if reason == TERMINAL_STOP:
        raise SmokeVerificationError(
            "stop_reason is 'stop_string' but stop_fired is false")
    if eos:
        return TERMINAL_EOS
    if capped:
        return TERMINAL_MAX
    raise SmokeVerificationError(
        "the record evidences no terminal mode: no EOS, no token cap and no "
        "declared stop sequence fired")


def check_common(dataset: str, condition: str, record: Mapping[str, Any],
                 checks: Checklist) -> None:
    """The original checks, with a stop-aware terminal-mode rule."""
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
    mode = terminal_mode(dataset, record)
    checks.require(f"{name}_terminated_normally", mode in TERMINAL_MODES,
                   f"terminal mode {mode}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[DS_MMLU, DS_GSM8K], required=True)
    parser.add_argument("--item-index", type=int, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "routegate-data")
    args = parser.parse_args(argv)

    checks = Checklist(f"{ERRATUM}_{args.dataset}_{args.item_index}")
    verdict = STATUS_PASS
    notes = []
    try:
        want = expected_rows(args.dataset)
        if not 0 <= args.item_index < want:
            raise SmokeVerificationError(
                f"--item-index {args.item_index} is outside [0, {want})")
        check_digests(checks)

        records = {}
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
            checks.record(f"{condition}_terminal_mode", True,
                          terminal_mode(args.dataset, record))
        if args.dataset == DS_GSM8K:
            for condition, record in records.items():
                check_gsm8k_boundary(record, checks)
        check_baseline(records[COND_BASELINE], checks)
        check_dynamic(records[COND_DYNAMIC], checks)
        check_routegate(records[COND_ROUTEGATE], checks)

        parse = PARSERS[args.dataset]
        for condition, record in records.items():
            _prediction, status = parse(record.get("response_text", ""))
            capped = bool(record.get("hit_max_new_tokens"))
            checks.record(f"{condition}_parse_status", True, str(status))
            checks.record(f"{condition}_hit_max_new_tokens", True, str(capped))
            if status == PARSE_OK:
                if capped:
                    notes.append(
                        f"{TRUNCATED_AFTER_ANSWER} under {condition}: the "
                        f"answer parsed and the cap fell after it. Diagnostic "
                        f"only; scored normally.")
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
                    f"rule. Not a technical fault.")

        if not checks.all_passed:
            verdict = STATUS_BLOCKED
    except ValidationError as exc:
        print(f"[smoke] {STATUS_BLOCKED}: {exc}", file=sys.stderr)
        return 1

    print(f"[smoke] {ERRATUM}: stop-aware termination")
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
    return 0 if verdict == STATUS_PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
