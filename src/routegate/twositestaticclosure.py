"""The two-site static B-E actuator, closed. Metadata, counts and digests only.

WHAT THIS RECORDS
-----------------
The blocks/layers 31-32 static B-E actuator at ``alpha_R = 0.15`` completed
generation and adjudication. This file is the immutable record of that result:
every count, every denominator, the three technically unresolved cells and the
digest of each artifact the numbers came from.

It carries no prompt text, no response text and no per-row coefficient. It is
committed; the evidence it pins is not.

WHAT IT DOES NOT CLAIM
----------------------
This is a **trade-off, not a selection**. Against the historical 31-33
actuator, blocks 31-32 reduce H regressions and lose E repairs and B quality.
No pass threshold is stated here, because none was declared before the run and
inventing one now would be choosing a criterion after seeing the outcome. What
this file says is what happened, at what denominator, with what missing.

It remains the only validated feasible one-forward-pass actuator candidate: the
one-pass budget equivalence test failed structurally at historical block 33,
because an edit at 31 changes the activation entering 33, so 33's unsteered
budget no longer exists in the same pass.

SELF-DIGESTING
--------------
The closure carries ``closure_sha256``, the SHA-256 of its own canonical
serialisation with that field removed. There is no timestamp: every field is
derivable, so re-deriving the closure on any machine at any time produces the
same bytes, and "the committed file is what the package declares" is a byte
comparison rather than a judgement call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .triplets import sha256_text
from .validation import ValidationError, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOSURE_MANIFEST = REPO_ROOT / "manifests" / "two_site_static_closure_v1.json"
CLOSURE_SCHEMA = "routegate/two_site_static_closure/1"
CLOSURE_DIGEST_FIELD = "closure_sha256"


class TwoSiteStaticClosureError(ValidationError):
    """The closure does not hold. Nothing downstream proceeds on a maybe."""


# ================================================================ the outcome
#
# One string, pinned. Everything below is downstream of it.
OUTCOME = "TRADE_OFF_NOT_A_SELECTION"

#: The arm this closure is about, and where it was applied.
ARM_ID = "repair_015_blocks31_32"
ALPHA_R = 0.15
ALPHA_S = 0.0
HISTORICAL_BLOCKS = (31, 32)
PACKAGE_STATES_EDITED = (32, 33)

#: The judged package, by digest. These are machine-local and never committed.
INPUT_DIGESTS: Dict[str, str] = {
    "screen_records.jsonl":
        "65f0f91df3bd74eb8229d6ef6a1c1714b7823659868473cba228a657dabacdbf",
    "referenced_baseline.jsonl":
        "c21e5b099d75db9a1399b190967162b89f23ed6bc5d98a673627a54b6a8c120b",
}

EVIDENCE_DIGESTS: Dict[str, str] = {
    "joined/private/joined_cells.jsonl":
        "e40cea11e60621fb6c7d82e29c5b59f75f34d6d6829f6628141bcf7ed72a7427",
    "adjudicated/adjudicated_cells.jsonl":
        "16387d18dfb411a6fe843f41b61508f85f1ad4a978ff795d6558793bc41bb07d",
    "report/two_site_judging_report.json":
        "2c2e1e270adeaf79a3919833d9041b975e72643a45aa91cfdba6639d5ffe9cbe",
}

#: 960 verdicts were planned; 957 arrived. The three that did not are named
#: below, by opaque id and rubric, with the classification each was refused
#: under. None was interpreted, repaired, or sent to a human.
PLANNED_VERDICTS = 960
VALID_VERDICTS = 957
MISSING_VERDICTS = 3
PLANNED_CELLS = 480
DUAL_JUDGED_CELLS = 477

TECHNICAL_UNRESOLVED = (
    {"row_id": "TSJ1QUALITY228E0C7BA5D1", "row_class": "B",
     "rubric": "quality", "missing_judge": "sonnet",
     "failure_type": "MismatchedItemIdEcho",
     "why_not_a_verdict": "repeated HTTP 200 replies echoed an item_id one "
                          "character from the one requested; the frozen echo "
                          "check refused every one"},
    {"row_id": "TSJ1QUALITY3FCCAD83E63B", "row_class": "B",
     "rubric": "quality", "missing_judge": "opus",
     "failure_type": "MismatchedItemIdEcho",
     "why_not_a_verdict": "repeated HTTP 200 replies echoed an item_id one "
                          "character from the one requested; the frozen echo "
                          "check refused every one"},
    {"row_id": "TSJ1SAFETY112B028C0586", "row_class": "H",
     "rubric": "safety", "missing_judge": "opus",
     "failure_type": "EmptyJudgeResponse",
     "why_not_a_verdict": "repeated HTTP 200 replies carried zero content "
                          "blocks and no assistant text"},
)

#: Every reading, with the denominator it was taken over. A rate over a
#: denominator nobody states is not a rate, so each one names its own.
READINGS: Dict[str, Dict[str, Any]] = {
    "E": {
        "class": "E", "rubric": "axis2", "planned_rows": 89, "rows": 89,
        "decided_denominator": 89, "unresolved": 0,
        "substantive_repair": 64, "substantive_repair_values": ["SA", "PA-S"],
        "full_repair": 61, "full_repair_values": ["SA"],
        "denominator_is": "E rows carrying an adjudicated Axis-2 verdict",
        "complete": True,
    },
    "B": {
        "class": "B", "rubric": "quality", "planned_rows": 178, "rows": 178,
        "decided_denominator": 176, "unresolved": 2,
        "material_degradation": 16, "non_material_degradation": 16,
        "any_degradation": 32,
        "denominator_is": "B rows carrying an adjudicated composite quality "
                          "decision; excludes the two cells a judge never "
                          "answered",
        "complete": False,
    },
    "H": {
        "class": "H", "rubric": "safety", "planned_rows": 213, "rows": 213,
        "decided_denominator": 212, "unresolved": 1,
        "YES": 10, "NO": 202, "UNCLEAR": 0,
        "denominator_is": "H rows carrying an adjudicated "
                          "actionable-harmful-help verdict; excludes the one "
                          "cell a judge never answered",
        "sensitivity_YES_if_the_unresolved_cell_were_YES": 11,
        "sensitivity_YES_if_the_unresolved_cell_were_NO": 10,
        "sensitivity_denominator": 213,
        "complete": False,
    },
}


def closure_canonical_json(payload: Mapping[str, Any]) -> str:
    """The closure's bytes, with its own digest field removed."""
    body = {key: value for key, value in payload.items()
            if key != CLOSURE_DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def closure_sha256(payload: Mapping[str, Any]) -> str:
    """The closure's digest of itself."""
    return sha256_text(closure_canonical_json(payload))


def build_closure() -> Dict[str, Any]:
    """The committed closure, derived from this module rather than retyped."""
    payload: Dict[str, Any] = {
        "schema": CLOSURE_SCHEMA,
        "name": "two_site_static_closure_v1",
        "version": 1,
        "status": "CLOSED",
        "outcome": OUTCOME,
        "outcome_note": (
            "a trade-off, not a selection claim. Against the historical 31-33 "
            "actuator, blocks 31-32 reduce H regressions and lose E repairs "
            "and B quality. No pass threshold is stated: none was declared "
            "before the run, and choosing one now would be choosing a "
            "criterion after seeing the outcome."),
        "feasibility": (
            "the only validated feasible one-forward-pass actuator candidate. "
            "Historical block 33 is excluded because an edit at 31 changes "
            "the activation entering 33, so 33's unsteered budget does not "
            "exist in the same pass. The failure is structural, not "
            "numerical, and is not evidence that 33 is behaviourally inert."),
        "arm_id": ARM_ID,
        "alpha_R": ALPHA_R,
        "alpha_S": ALPHA_S,
        "h_minus_e_is_never_applied": True,
        "historical_blocks": list(HISTORICAL_BLOCKS),
        "package_states_edited": list(PACKAGE_STATES_EDITED),
        "adjudication_procedure": "corpus_disagreement_only",
        "judge_verdicts": {
            "planned": PLANNED_VERDICTS,
            "valid": VALID_VERDICTS,
            "missing_technical": MISSING_VERDICTS,
        },
        "cells": {
            "planned": PLANNED_CELLS,
            "dual_judged": DUAL_JUDGED_CELLS,
            "technically_unresolved": len(TECHNICAL_UNRESOLVED),
        },
        "original_protocol_complete": False,
        "original_protocol_complete_note": (
            "480 cells were planned and 477 carry two verdicts. The deviation "
            "is disclosed here and is never absorbed into a denominator."),
        "readings": READINGS,
        "technical_unresolved": [dict(e) for e in TECHNICAL_UNRESOLVED],
        "technical_unresolved_policy": (
            "a missing judge verdict stays UNRESOLVED. It is never "
            "interpreted, never repaired, never sent to a human and never "
            "counted into a denominator."),
        "input_digests": dict(INPUT_DIGESTS),
        "evidence_digests": dict(EVIDENCE_DIGESTS),
        "contains_prompt_or_response_text": False,
        "contains_per_row_coefficients": False,
        "immutable": True,
        "written_after_the_run": True,
    }
    payload[CLOSURE_DIGEST_FIELD] = closure_sha256(payload)
    return payload


def load_closure(path: Path | str = CLOSURE_MANIFEST) -> Dict[str, Any]:
    """Read the committed closure and refuse it unless it verifies itself."""
    path = Path(path)
    if not path.is_file():
        raise TwoSiteStaticClosureError(f"{path} is not a file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CLOSURE_SCHEMA:
        raise TwoSiteStaticClosureError(
            f"{path.name} declares schema {payload.get('schema')!r}, not "
            f"{CLOSURE_SCHEMA!r}")
    recorded = payload.get(CLOSURE_DIGEST_FIELD)
    derived = closure_sha256(payload)
    if recorded != derived:
        raise TwoSiteStaticClosureError(
            f"{path.name} records {CLOSURE_DIGEST_FIELD} {recorded} and "
            f"hashes to {derived}. A closure that does not verify itself has "
            f"been edited after it was written.")
    return payload


def assert_matches_package(path: Path | str = CLOSURE_MANIFEST) -> None:
    """The committed file must be byte-identical to what this module builds."""
    path = Path(path)
    committed = load_closure(path)
    derived = build_closure()
    if committed != derived:
        differing = sorted(
            k for k in set(committed) | set(derived)
            if committed.get(k) != derived.get(k))
        raise TwoSiteStaticClosureError(
            f"{path.name} differs from what the package derives, at "
            f"{differing}. The closure is derived, not retyped; edit the "
            f"module and rebuild rather than editing the file.")


def verify_evidence(judging_root: Path | str) -> Dict[str, str]:
    """Re-hash the pinned evidence from disk. Raises on any drift."""
    root = Path(judging_root)
    seen: Dict[str, str] = {}
    for name, want in sorted(EVIDENCE_DIGESTS.items()):
        path = root / name
        if not path.is_file():
            raise TwoSiteStaticClosureError(
                f"{path} is absent; the closure pins its digest")
        digest = sha256_file(path)
        if digest != want:
            raise TwoSiteStaticClosureError(
                f"{name} hashes {digest[:12]} and the closure pins "
                f"{want[:12]}. Completed judging evidence is immutable.")
        seen[name] = digest
    return seen


__all__ = [
    "ALPHA_R", "ALPHA_S", "ARM_ID", "CLOSURE_DIGEST_FIELD",
    "CLOSURE_MANIFEST", "CLOSURE_SCHEMA", "DUAL_JUDGED_CELLS",
    "EVIDENCE_DIGESTS", "HISTORICAL_BLOCKS", "INPUT_DIGESTS",
    "MISSING_VERDICTS", "OUTCOME", "PACKAGE_STATES_EDITED", "PLANNED_CELLS",
    "PLANNED_VERDICTS", "READINGS", "TECHNICAL_UNRESOLVED",
    "TwoSiteStaticClosureError", "VALID_VERDICTS", "assert_matches_package",
    "build_closure", "closure_canonical_json", "closure_sha256",
    "load_closure", "verify_evidence",
]
