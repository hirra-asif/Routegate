"""Deterministic offline scoring for the two capability benchmarks.

Nothing here loads a model, generates text, calls a judge or reads held-out
material. It reads a completed arm's records and the pinned gold column, and
decides -- by a rule fixed before any generation -- whether each completion is
correct.

THE GOLD LABEL IS NEVER VISIBLE TO GENERATION
---------------------------------------------
Both parsers take a completion and nothing else. The gold value enters only in
:func:`score_item`, after parsing has already produced its answer, and the
runner that generates never imports this module. ``assert_generation_cannot_
read_gold`` makes that structural rather than a convention.

tinyMMLU
--------
The prompt is a few-shot COMPLETION ending at ``Answer:``, so a correct
continuation is a single option letter. The parser accepts exactly that: the
first non-whitespace run must be one of ``A B C D``, optionally followed by
harmless punctuation or whitespace and nothing else that looks like a second
answer. Prose before the letter, an absent letter, or two different letters is
``INVALID`` -- never a guess. The gold column is a private integer and is
mapped explicitly, ``0=A 1=B 2=C 3=D``.

tinyGSM8K
---------
The rule is the established lm-evaluation-harness ``gsm8k`` behaviour: take the
LAST number appearing in the completion. That revision is pinned in
:data:`GSM8K_HARNESS_REVISION` and the regex is recorded verbatim.

Comparison is exact decimal, never binary float: ``Decimal("0.1") * 3`` is not
``0.3`` in binary, and a benchmark that marked a correct answer wrong for that
reason would be silently under-reporting. Commas are stripped, a leading sign
is kept, trailing full stops are not consumed as decimal points, and
``42``/``42.0``/``+42``/``42,000`` compare as numbers rather than strings.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .validation import ValidationError

#: The gold integer -> option letter map. Deliberately NOT in
#: capabilityparsers: a technical verifier executes a parser and must have no
#: route to a gold label.
MMLU_GOLD_LETTERS: Dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "D"}
#: The gold column's own format: GSM8K answers end '#### <number>'.
GSM8K_GOLD_SEPARATOR = "####"


class CapabilityScoringError(ValidationError):
    """A capability arm could not be scored from what it recorded."""


from .capabilityparsers import (            # noqa: F401,E402  (re-exported)
    GSM8K_HARNESS,
    GSM8K_HARNESS_REVISION,
    GSM8K_HARNESS_TASK,
    GSM8K_NUMBER_REGEX,
    GSM8K_RULE,
    MMLU_OPTIONS,
    MMLU_RULE,
    MMLU_TRAILING_PUNCTUATION,
    PARSE_AMBIGUOUS,
    PARSE_NO_ANSWER,
    PARSE_OK,
    PARSE_PROSE_BEFORE_ANSWER,
    PARSE_STATUSES,
    _canonical_number,
    parse_gsm8k,
    parse_mmlu,
)


def mmlu_gold_letter(gold):
    """The private integer gold label as its option letter. Explicit map."""
    try:
        index = int(gold)
    except (TypeError, ValueError) as exc:
        raise CapabilityScoringError(
            f"the tinyMMLU gold label {gold!r} is not an integer; this "
            f"benchmark stores 0..3 and nothing else") from exc
    if index not in MMLU_GOLD_LETTERS:
        raise CapabilityScoringError(
            f"the tinyMMLU gold label {index} is outside 0..3")
    return MMLU_GOLD_LETTERS[index]


def gsm8k_gold_value(gold):
    """The gold final number, after the '#### ' separator."""
    text = str(gold)
    if GSM8K_GOLD_SEPARATOR not in text:
        raise CapabilityScoringError(
            f"a tinyGSM8K gold answer carries no {GSM8K_GOLD_SEPARATOR!r} "
            f"separator; the final number cannot be located")
    value = _canonical_number(text.split(GSM8K_GOLD_SEPARATOR)[-1])
    if value is None:
        raise CapabilityScoringError(
            "a tinyGSM8K gold answer has no numeric final value")
    return value


# ================================================================== scoring
#: Reported when a completion parsed and THEN ran into the token cap. It is
#: diagnostic metadata: the answer was already produced, so correctness is
#: decided by the parsed answer and the cap changes nothing.
TRUNCATED_AFTER_ANSWER = "PARSE_OK_TRUNCATED_AFTER_ANSWER"

TRUNCATION_RULE = (
    "a completion that parsed and then hit the token cap has already produced "
    "its answer. The parsed answer is scored normally, in every condition, "
    "and hit_max_new_tokens never changes correctness. It is recorded as "
    f"{TRUNCATED_AFTER_ANSWER!r} for diagnosis only. Only an actual parse "
    "failure scores incorrect.")


@dataclass(frozen=True)
class ItemScore:
    """One scored item: what was parsed, whether it parsed, whether it hit."""

    item_index: int
    item_id: str
    prediction: Optional[str]
    parse_status: str
    correct: bool
    #: Diagnostic only. Never consulted when deciding ``correct``.
    truncated: bool = False

    @property
    def diagnostic(self) -> Optional[str]:
        if self.parse_status == PARSE_OK and self.truncated:
            return TRUNCATED_AFTER_ANSWER
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {"item_index": self.item_index, "item_id": self.item_id,
                "prediction": self.prediction,
                "parse_status": self.parse_status,
                "correct": bool(self.correct),
                "hit_max_new_tokens": bool(self.truncated),
                "diagnostic": self.diagnostic}


def score_item(dataset: str, index: int, item_id: str, completion: str,
               gold: Any, truncated: bool = False) -> ItemScore:
    """Parse first, then compare. A failed parse is incorrect, never a guess.

    ``truncated`` is recorded and never consulted: a completion that parsed
    and then hit the cap is scored on its parsed answer like any other.
    """
    if dataset == "tinymmlu":
        prediction, status = parse_mmlu(completion)
        correct = status == PARSE_OK and prediction == mmlu_gold_letter(gold)
        shown: Optional[str] = prediction
    elif dataset == "tinygsm8k":
        value, status = parse_gsm8k(completion)
        correct = status == PARSE_OK and value == gsm8k_gold_value(gold)
        shown = None if value is None else str(value)
    else:
        raise CapabilityScoringError(
            f"{dataset!r} is not a capability dataset")
    return ItemScore(item_index=int(index), item_id=str(item_id),
                     prediction=shown, parse_status=status,
                     correct=bool(correct), truncated=bool(truncated))


def score_arm(dataset: str, records: Sequence[Mapping[str, Any]],
              golds: Sequence[Any]) -> Dict[str, Any]:
    """One arm's ordered correctness vector, accuracy and parse tally."""
    expected = len(golds)
    by_index: Dict[int, Mapping[str, Any]] = {}
    for record in records:
        index = int(record["item_index"])
        if index in by_index:
            raise CapabilityScoringError(
                f"item_index {index} appears twice in this arm; a resumed run "
                f"must never duplicate an item")
        by_index[index] = record
    missing = [i for i in range(expected) if i not in by_index]
    if missing:
        raise CapabilityScoringError(
            f"{len(missing)} item(s) are absent from this arm "
            f"(first {missing[:3]}); an arm is scored only when complete")

    scores = [score_item(dataset, i, str(by_index[i]["item_id"]),
                         by_index[i].get("response_text", ""), golds[i],
                         truncated=bool(by_index[i].get("hit_max_new_tokens")))
              for i in range(expected)]
    correctness = [int(s.correct) for s in scores]
    parsed = {status: sum(1 for s in scores if s.parse_status == status)
              for status in PARSE_STATUSES}
    return {
        "dataset": dataset,
        "denominator": expected,
        "correct": sum(correctness),
        "accuracy": sum(correctness) / expected if expected else None,
        "correctness_vector": correctness,
        "parse_status_counts": parsed,
        "parse_rule": MMLU_RULE if dataset == "tinymmlu" else GSM8K_RULE,
        "truncation_rule": TRUNCATION_RULE,
        "parsed_then_truncated": sum(
            1 for s in scores if s.diagnostic == TRUNCATED_AFTER_ANSWER),
        "harness": (None if dataset == "tinymmlu" else
                    {"repo": GSM8K_HARNESS, "revision": GSM8K_HARNESS_REVISION,
                     "task": GSM8K_HARNESS_TASK,
                     "regex": GSM8K_NUMBER_REGEX}),
        "items": [s.as_dict() for s in scores],
    }


# ============================================== paired analysis vs baseline
def paired_transitions(baseline: Sequence[int],
                       candidate: Sequence[int]) -> Dict[str, Any]:
    """The 2x2 item-level transition table against baseline."""
    if len(baseline) != len(candidate):
        raise CapabilityScoringError(
            f"paired analysis needs equal-length vectors, got "
            f"{len(baseline)} and {len(candidate)}")
    pairs = list(zip(baseline, candidate))
    counts = {
        "correct_to_correct": sum(1 for b, c in pairs if b and c),
        "correct_to_incorrect": sum(1 for b, c in pairs if b and not c),
        "incorrect_to_correct": sum(1 for b, c in pairs if not b and c),
        "incorrect_to_incorrect": sum(1 for b, c in pairs if not b and not c),
    }
    discordant = (counts["correct_to_incorrect"]
                  + counts["incorrect_to_correct"])
    delta = (sum(candidate) - sum(baseline)) / len(pairs) if pairs else None
    return {
        "n": len(pairs), **counts,
        "discordant": discordant,
        "accuracy_delta": delta,
        "delta_confidence_interval": wilson_paired_interval(
            counts["incorrect_to_correct"], counts["correct_to_incorrect"]),
        "note": "paired at the item level: both conditions answered the same "
                "prompt in the same order, so the comparison is within-item.",
    }


def wilson_paired_interval(gains: int, losses: int,
                           z: float = 1.959963984540054
                           ) -> Optional[Dict[str, float]]:
    """A Wilson interval on the discordant split, or None if none discordant."""
    n = int(gains) + int(losses)
    if n == 0:
        return None
    p = gains / n
    denominator = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denominator
    half = (z * ((p * (1 - p) / n + (z * z) / (4 * n * n)) ** 0.5)
            / denominator)
    return {"discordant": n, "gains": int(gains), "losses": int(losses),
            "proportion_gains": p,
            "low": max(0.0, centre - half), "high": min(1.0, centre + half),
            "method": "Wilson score interval on the discordant pairs"}


# ===================================== tinyBenchmarks IRT estimate (secondary)
TINYBENCHMARKS_NOTE = (
    "raw accuracy and the paired item transitions are the primary readings. "
    "The tinyBenchmarks IRT / pIRT / gpIRT estimate is reported beside them as "
    "a secondary calibration of the 100-item subset onto the full benchmark, "
    "and no conclusion rests on it alone.")


def tinybenchmarks_estimate(benchmark: str, correctness: Sequence[int]
                            ) -> Dict[str, Any]:
    """The official tinyBenchmarks estimate, when the package is installed."""
    if len(correctness) != 100:
        raise CapabilityScoringError(
            f"tinyBenchmarks expects the ordered 100-item correctness vector, "
            f"got {len(correctness)}")
    try:
        import numpy as np
        import tinyBenchmarks as tb                       # noqa: N813
    except Exception as exc:                              # noqa: BLE001
        return {"available": False, "standing": "secondary",
                "reason": f"{type(exc).__name__}: {exc}",
                "note": TINYBENCHMARKS_NOTE}
    estimate = tb.evaluate(np.asarray(correctness, dtype=float), benchmark)
    return {"available": True, "standing": "secondary",
            "benchmark": benchmark, "estimate": estimate,
            "note": TINYBENCHMARKS_NOTE}


# ================================================================== hygiene
#: Field names a generating runner must never read from a dataset row.
PRIVATE_GOLD_FIELDS: Tuple[str, ...] = ("answer", "gold", "label", "target")


def assert_generation_cannot_read_gold(source: str, label: str) -> None:
    """The generating runner must not reference a gold column at all."""
    for field in PRIVATE_GOLD_FIELDS:
        for pattern in (f'"{field}"', f"'{field}'", f"[{field!r}]",
                        f'answer_column'):
            if pattern in source and field in ("answer", "gold"):
                raise CapabilityScoringError(
                    f"{label} references {pattern}; the generating path must "
                    f"never read a gold column, and scoring is a separate "
                    f"offline stage")


__all__ = [
    "CapabilityScoringError", "GSM8K_GOLD_SEPARATOR", "GSM8K_HARNESS",
    "GSM8K_HARNESS_REVISION", "GSM8K_HARNESS_TASK", "GSM8K_NUMBER_REGEX",
    "GSM8K_RULE", "ItemScore", "MMLU_GOLD_LETTERS", "MMLU_OPTIONS",
    "MMLU_RULE", "MMLU_TRAILING_PUNCTUATION", "PARSE_AMBIGUOUS",
    "PARSE_NO_ANSWER", "PARSE_OK", "PARSE_PROSE_BEFORE_ANSWER",
    "PARSE_STATUSES", "PRIVATE_GOLD_FIELDS", "TINYBENCHMARKS_NOTE",
    "TRUNCATED_AFTER_ANSWER", "TRUNCATION_RULE",
    "assert_generation_cannot_read_gold", "gsm8k_gold_value",
    "mmlu_gold_letter", "paired_transitions", "parse_gsm8k", "parse_mmlu",
    "score_arm", "score_item", "tinybenchmarks_estimate",
    "wilson_paired_interval",
]
