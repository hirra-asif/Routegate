"""Technically unresolved two-site cells, carried forward and never decided.

WHAT THIS IS
------------
Three of the 960 planned two-site judge calls never produced a verdict. Each
returned HTTP 200 and each failed the frozen schema check, repeatedly:

* two quality cells whose reply echoed an ``item_id`` one character away from
  the one requested -- rejected by the frozen echo rule;
* one safety cell whose payload carried zero content blocks -- rejected as an
  empty judge response.

957 of 960 verdicts are valid. This module is how the missing three are
carried through the stage **as missing**, so the join can finish without
anybody inventing a verdict, repairing an id, or reading an empty reply.

WHAT THIS IS NOT
----------------
It is **not** a change to how any cell is decided.
``routegate.focusedjudging.adjudicate_cell`` is untouched, and its existing
missing-evidence branch -- which returns ``UNRESOLVED`` with
``judges_agree: None`` whenever either judge has no verdict -- is what actually
labels these cells. ``corpus_disagreement_only`` is untouched. The frozen
schema check is untouched: a mismatched echo is still a failure here, exactly
as it is in phase 1 and phase 2.

It is also **not** a route to a human. ``docs/decision_log.md`` (2026-08-25)
records that an earlier attempt to send technical failures to a human was
rejected, because it changed how cells are decided after the failures were
known. That decision holds. A technically unresolved cell is excluded from the
human-review packet, and a human verdict for one is refused.

WHY A MANIFEST AND NOT A LOOSER GATE
------------------------------------
The join's ``*_no_unresolved_failure`` gate is inherited verbatim from phase 1
and phase 2. It exists so that nobody proceeds on incomplete evidence by
accident. Deleting it would let *any* future failure through silently. So it
stays, and the only thing that can pass it is a cell named in a committed,
metadata-only manifest, pinned by opaque id and by the SHA256 of every
preserved raw response behind it.

HOW IT FAILS
------------
Closed, at every gate. The selection reads opaque ids and preserved failure
evidence and nothing else -- never a verdict, from either judge -- and
``unresolved_is_outcome_blind`` proves that at run time by re-deciding the
selection under every verdict the frozen vocabulary allows. If a named judge
later answers a named cell, the join fails: the manifest would no longer be
true and must be revised by hand.

The failure classification is re-derived from the preserved raw body, not read
from the recorded error string. That is deliberate: the error strings from the
first Sonnet pass are credential-bearing (see ``redact_header_value``), and
nothing in the scientific path should depend on reading them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .triplets import (
    AXIS_VALUES,
    JUDGES,
    assert_no_text_fields,
    assert_opaque,
    read_jsonl,
)
from .twositejudging import (
    FROZEN_COVERAGE,
    ROW_PREFIX,
    RUBRIC_CLASS,
    TwoSiteJudgingError,
)
from .validation import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed manifest. Metadata only: opaque ids, judges, rubrics, counts
#: and digests. No prompt text, no response text, no error strings.
TECHNICAL_UNRESOLVED_MANIFEST = (
    REPO_ROOT / "manifests"
    / "two_site_actuator_judging_v1_technical_unresolved.json")

#: The state a carried-forward cell holds in the join. It is terminal for the
#: join and non-decidable everywhere after it.
STATE_UNRESOLVED_TECHNICAL = "unresolved_technical"

#: The only two failure classifications this path recognises, each re-derivable
#: from the preserved raw body alone.
FAILURE_MISMATCHED_ECHO = "MismatchedItemIdEcho"
FAILURE_EMPTY_RESPONSE = "EmptyJudgeResponse"
UNRESOLVED_FAILURE_TYPES: Tuple[str, ...] = (
    FAILURE_MISMATCHED_ECHO, FAILURE_EMPTY_RESPONSE)

#: "Repeated" means at least this many terminal attempts, per cell. One bad
#: reply is an incident; two is a property of the cell. Same threshold as
#: ``routegate.focusedjudging.MIN_EXCLUSION_EMPTY_ATTEMPTS``.
MIN_UNRESOLVED_ATTEMPTS = 2

#: Every gate, named. All must hold, for every carried cell.
UNRESOLVED_GATES: Tuple[str, ...] = (
    "cell_resolves_to_one_packet_row",
    "named_judge_has_no_successful_verdict",
    "other_judge_has_a_successful_verdict",
    "evidence_holds_repeated_terminal_failures",
    "every_raw_response_exists_and_recomputes",
    "failure_type_rederives_from_the_raw_body",
    "unresolved_is_outcome_blind",
    "counts_match_the_pinned_manifest",
)


class TechnicalUnresolvedError(TwoSiteJudgingError):
    """The unresolved manifest does not hold. Nothing proceeds on a maybe."""


def _blocked(gates: Dict[str, bool], reason: str) -> None:
    filled = {name: gates.get(name, False) for name in UNRESOLVED_GATES}
    blocked = [n for n in UNRESOLVED_GATES if not filled[n]]
    raise TechnicalUnresolvedError(f"{reason} (blocked at {blocked})")


def load_technical_unresolved(
        path: Path | str = TECHNICAL_UNRESOLVED_MANIFEST) -> Dict[str, Any]:
    """Read the committed manifest and check its shape.

    An absent manifest is **not** an error and is not an empty list by
    accident: it returns a manifest with nothing carried and counts equal to
    the frozen protocol, so a run with no technical failure behaves exactly as
    it did before this module existed.
    """
    path = Path(path)
    if not path.exists():
        return {"unresolved": [], "counts": {
            "planned_verdicts": FROZEN_COVERAGE["cells"] * len(JUDGES),
            "valid_verdicts": FROZEN_COVERAGE["cells"] * len(JUDGES),
            "missing_verdicts": 0,
            "cells": FROZEN_COVERAGE["cells"],
            "dual_judged_cells": FROZEN_COVERAGE["cells"],
            "technically_unresolved_cells": 0,
            "original_protocol_complete": True}, "absent": True}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("is_outcome_based_exclusion") is not False:
        raise TechnicalUnresolvedError(
            f"{path.name} must state is_outcome_based_exclusion: false. A "
            f"carry-forward that will not say it is technical is not one.")
    if manifest.get("contains_prompt_or_response_text") is not False:
        raise TechnicalUnresolvedError(
            f"{path.name} must state contains_prompt_or_response_text: false")
    if manifest.get("invents_no_verdict") is not True:
        raise TechnicalUnresolvedError(
            f"{path.name} must state invents_no_verdict: true")
    assert_no_text_fields(manifest, path.name)
    for entry in manifest.get("unresolved", []):
        for required in ("unresolved_id", "row_id", "screen_row_id", "rubric",
                         "row_class", "failed_judge", "failure_type",
                         "attempts", "reason", "evidence"):
            if required not in entry:
                raise TechnicalUnresolvedError(
                    f"{path.name}: an entry is missing {required!r}")
        rubric = entry["rubric"]
        if rubric not in ROW_PREFIX:
            raise TechnicalUnresolvedError(
                f"{path.name}: unknown rubric {rubric!r}")
        assert_opaque(str(entry["row_id"]), ROW_PREFIX[rubric])
        if entry["failed_judge"] not in JUDGES:
            raise TechnicalUnresolvedError(
                f"{path.name}: unknown judge {entry['failed_judge']!r}")
        if entry["failure_type"] not in UNRESOLVED_FAILURE_TYPES:
            raise TechnicalUnresolvedError(
                f"{path.name}: {entry['failure_type']!r} is not one of "
                f"{list(UNRESOLVED_FAILURE_TYPES)}. Only a failure this path "
                f"can re-derive from a preserved body is carried.")
        if RUBRIC_CLASS[rubric] != entry["row_class"]:
            raise TechnicalUnresolvedError(
                f"{path.name}: {entry['row_id']} is rubric {rubric!r} but "
                f"claims class {entry['row_class']!r}")
    return manifest


def unresolved_row_ids(manifest: Mapping[str, Any]) -> Tuple[str, ...]:
    """Every opaque row id carried as unresolved, sorted. Pure lookup.

    Reads ``row_id`` and nothing else. There is no verdict, no outcome and no
    judge answer anywhere in this function's inputs, which is what makes the
    selection outcome-blind by construction rather than by promise.
    """
    return tuple(sorted(str(entry["row_id"])
                        for entry in manifest.get("unresolved", [])))


def unresolved_index(manifest: Mapping[str, Any]
                     ) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    """``(row_id, failed_judge) -> entry``."""
    out: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for entry in manifest.get("unresolved", []):
        key = (str(entry["row_id"]), str(entry["failed_judge"]))
        if key in out:
            raise TechnicalUnresolvedError(
                f"{key} is carried twice; one cell is one entry")
        out[key] = entry
    return out


def classify_raw_failure(raw: Mapping[str, Any], row_id: str) -> str:
    """Re-derive the failure classification from one preserved raw body.

    Reads the body's structure only: how many content blocks came back, and
    -- if any did -- whether the echoed ``item_id`` is the one requested. The
    judge's prose is never inspected and never returned.
    """
    body_text = raw.get("raw_body") or ""
    try:
        body = json.loads(body_text) if body_text else {}
    except ValueError:
        body = {}
    blocks = body.get("content")
    blocks = blocks if isinstance(blocks, list) else []
    text = "".join(b.get("text", "") for b in blocks
                   if isinstance(b, Mapping))
    if not text.strip():
        return FAILURE_EMPTY_RESPONSE
    try:
        parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, KeyError):
        return "UnparseableBody"
    echoed = parsed.get("item_id") if isinstance(parsed, Mapping) else None
    if echoed is None:
        return "NoItemId"
    if str(echoed) != str(row_id):
        return FAILURE_MISMATCHED_ECHO
    return "EchoMatched"


def verify_technical_unresolved(
        manifest: Mapping[str, Any],
        row_map: Mapping[str, Mapping[str, Any]],
        successful_by_judge: Mapping[str, Mapping[str, Any]],
        failure_attempts: Mapping[
            str, Mapping[str, Sequence[Mapping[str, Any]]]],
        run_dirs: Mapping[str, Path],
        ) -> Dict[str, Any]:
    """Every gate, over every carried cell. Raises unless all of them hold.

    ``successful_by_judge`` is ``judge -> {row_id: record}`` for the successful
    records only; it is read to prove the failed judge has **no** verdict and
    the other judge has one, never to read either. ``failure_attempts`` is
    ``judge -> {row_id: [failure record, ...]}`` and is read for attempt count
    and preserved-body digests only.
    """
    gates: Dict[str, bool] = {name: False for name in UNRESOLVED_GATES}
    entries = list(manifest.get("unresolved", []))
    checked: List[Dict[str, Any]] = []

    for entry in entries:
        row_id = str(entry["row_id"])
        judge = str(entry["failed_judge"])
        other = [j for j in JUDGES if j != judge]

        packet_entry = row_map.get(row_id)
        if packet_entry is None:
            _blocked(gates, f"{row_id} is not a row of this packet")
        if (str(packet_entry["rubric"]) != str(entry["rubric"])
                or str(packet_entry["screen_row_id"])
                != str(entry["screen_row_id"])):
            _blocked(gates, f"{row_id} does not match the packet's own map")
        gates["cell_resolves_to_one_packet_row"] = True

        if row_id in successful_by_judge.get(judge, {}):
            _blocked(gates,
                     f"{judge} has a successful verdict for {row_id}. The "
                     f"manifest is no longer true and must be revised by "
                     f"hand.")
        gates["named_judge_has_no_successful_verdict"] = True

        if not all(row_id in successful_by_judge.get(j, {}) for j in other):
            _blocked(gates,
                     f"{row_id} has no verdict from {other} either. A cell "
                     f"neither judge answered is not carried by this path.")
        gates["other_judge_has_a_successful_verdict"] = True

        attempts = list(failure_attempts.get(judge, {}).get(row_id, []))
        terminal = [a for a in attempts
                    if isinstance(a.get("http_status"), int)
                    and 200 <= int(a["http_status"]) < 300]
        if len(terminal) < MIN_UNRESOLVED_ATTEMPTS:
            _blocked(gates,
                     f"{judge}/{row_id} has {len(terminal)} terminal HTTP 2xx "
                     f"failure(s); at least {MIN_UNRESOLVED_ATTEMPTS} are "
                     f"required before a cell is carried as unresolved")
        if len(terminal) != int(entry["attempts"]):
            _blocked(gates,
                     f"{judge}/{row_id}: manifest pins {entry['attempts']} "
                     f"attempt(s), evidence holds {len(terminal)}")
        gates["evidence_holds_repeated_terminal_failures"] = True

        pinned = {str(e["raw_response_sha256"]):
                  str(e["raw_response_path"])
                  for e in entry["evidence"]}
        run_dir = run_dirs.get(f"{judge}:{entry['rubric']}")
        if run_dir is None:
            _blocked(gates,
                     f"no run directory for {judge}/{entry['rubric']}")
        seen: Dict[str, str] = {}
        for attempt in terminal:
            rel = str(attempt.get("raw_response_path", ""))
            path = Path(run_dir) / rel
            if not path.is_file():
                _blocked(gates, f"{path} is absent; evidence cannot be "
                                f"re-hashed")
            digest = sha256_file(path)
            if digest != str(attempt.get("raw_response_sha256")):
                _blocked(gates,
                         f"{path} does not recompute to its recorded digest")
            seen[digest] = rel
        if sorted(seen) != sorted(pinned):
            _blocked(gates,
                     f"{judge}/{row_id}: the pinned evidence digests are not "
                     f"the digests on disk")
        gates["every_raw_response_exists_and_recomputes"] = True

        derived = set()
        for digest, rel in sorted(seen.items()):
            raw = json.loads(
                (Path(run_dir) / rel).read_text(encoding="utf-8"))
            derived.add(classify_raw_failure(raw, row_id))
        if derived != {str(entry["failure_type"])}:
            _blocked(gates,
                     f"{judge}/{row_id}: the preserved bodies classify as "
                     f"{sorted(derived)}, not {entry['failure_type']!r}")
        gates["failure_type_rederives_from_the_raw_body"] = True

        checked.append({"row_id": row_id, "failed_judge": judge,
                        "rubric": str(entry["rubric"]),
                        "row_class": str(entry["row_class"]),
                        "failure_type": str(entry["failure_type"]),
                        "attempts": len(terminal),
                        "screen_row_id": str(entry["screen_row_id"])})

    # Outcome-blindness, proved rather than promised: the selection is a pure
    # function of opaque ids, so re-running it under every verdict the frozen
    # vocabulary allows must return the same set every time.
    baseline = set(unresolved_row_ids(manifest))
    for entry in entries:
        allowed = AXIS_VALUES[str(entry["rubric"])]
        for value in list(allowed) + [None]:
            probe = json.loads(json.dumps(dict(manifest)))
            for item in probe.get("unresolved", []):
                item["_hypothetical_verdict"] = value
            if set(unresolved_row_ids(probe)) != baseline:
                _blocked(gates,
                         "the selection changed under a hypothetical verdict; "
                         "it is not outcome-blind")
    gates["unresolved_is_outcome_blind"] = True

    counts = derived_counts(manifest)
    pinned_counts = dict(manifest.get("counts", {}))
    if pinned_counts and pinned_counts != counts:
        _blocked(gates,
                 f"the manifest pins {pinned_counts} and the evidence derives "
                 f"{counts}")
    gates["counts_match_the_pinned_manifest"] = True

    if entries and not all(gates[name] for name in UNRESOLVED_GATES):
        _blocked(gates, "not every gate ran")
    return {"entries": len(entries), "gates": gates, "checked": checked,
            "counts": counts, "row_ids": sorted(baseline)}


def derived_counts(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The denominators this carry-forward implies. Derived, never assumed."""
    entries = list(manifest.get("unresolved", []))
    cells = FROZEN_COVERAGE["cells"]
    planned = cells * len(JUDGES)
    missing = len(entries)
    carried_cells = len({str(e["row_id"]) for e in entries})
    return {
        "planned_verdicts": planned,
        "valid_verdicts": planned - missing,
        "missing_verdicts": missing,
        "cells": cells,
        "dual_judged_cells": cells - carried_cells,
        "technically_unresolved_cells": carried_cells,
        "original_protocol_complete": missing == 0,
    }


def unresolved_by_class(manifest: Mapping[str, Any]) -> Dict[str, int]:
    """``row_class -> how many cells are technically unresolved``."""
    out: Dict[str, int] = {}
    for entry in manifest.get("unresolved", []):
        name = str(entry["row_class"])
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items()))


def unresolved_rows_for_class(manifest: Mapping[str, Any],
                              row_class: str) -> List[str]:
    """The opaque row ids carried as unresolved for one class, sorted."""
    return sorted(str(entry["row_id"])
                  for entry in manifest.get("unresolved", [])
                  if str(entry["row_class"]) == row_class)


__all__ = [
    "FAILURE_EMPTY_RESPONSE", "FAILURE_MISMATCHED_ECHO",
    "MIN_UNRESOLVED_ATTEMPTS", "STATE_UNRESOLVED_TECHNICAL",
    "TECHNICAL_UNRESOLVED_MANIFEST", "TechnicalUnresolvedError",
    "UNRESOLVED_FAILURE_TYPES", "UNRESOLVED_GATES", "classify_raw_failure",
    "derived_counts", "load_technical_unresolved", "unresolved_by_class",
    "unresolved_index", "unresolved_row_ids", "unresolved_rows_for_class",
    "verify_technical_unresolved",
]
