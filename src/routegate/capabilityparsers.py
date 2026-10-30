"""The frozen capability parsers, and nothing else.

Split out of :mod:`routegate.capabilityscoring` so a technical verifier can
execute a parser WITHOUT importing anything that can reach a gold label, a
correctness vector or a score. This module has no gold map, no comparison, no
accuracy and no denominator: it turns a completion into either an answer or a
named parse failure, and stops there.

The rules themselves are unchanged and are the ones the frozen protocol pins.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional, Tuple

from .validation import ValidationError


class CapabilityParseError(ValidationError):
    """A completion could not be parsed under the frozen rule."""


# ------------------------------------------------------------- parse status
PARSE_OK = "ok"
PARSE_NO_ANSWER = "no_answer"
PARSE_AMBIGUOUS = "ambiguous"
PARSE_PROSE_BEFORE_ANSWER = "prose_before_answer"
PARSE_STATUSES: Tuple[str, ...] = (
    PARSE_OK, PARSE_NO_ANSWER, PARSE_AMBIGUOUS, PARSE_PROSE_BEFORE_ANSWER)

MMLU_OPTIONS: Tuple[str, ...] = ("A", "B", "C", "D")

#: Characters permitted after the letter without making the answer ambiguous.
MMLU_TRAILING_PUNCTUATION = ".),:;-–—\"'`*"

MMLU_RULE = (
    "the ANSWER LINE is the completion up to its first newline. It must be "
    "exactly one of A B C D, optionally followed by punctuation from "
    f"{MMLU_TRAILING_PUNCTUATION!r} or whitespace, and by nothing else. Prose "
    "before the letter, an absent letter, and a second different option "
    "letter on the answer line are all INVALID and score as incorrect; "
    "nothing is guessed. Text after the first newline is the model continuing "
    "the few-shot pattern into the next question -- under raw-completion mode "
    "that continuation is expected -- and is not part of the answer.")

#: The pinned lm-evaluation-harness revision whose gsm8k behaviour is
#: reproduced: take the last number in the completion.
GSM8K_HARNESS = "EleutherAI/lm-evaluation-harness"
GSM8K_HARNESS_REVISION = "v0.4.3"
GSM8K_HARNESS_TASK = "lm_eval/tasks/gsm8k/gsm8k.yaml"
#: The harness regex, recorded verbatim. ``findall`` then take the last match.
GSM8K_NUMBER_REGEX = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
GSM8K_RULE = (
    f"lm-evaluation-harness {GSM8K_HARNESS_REVISION} gsm8k: apply "
    f"{GSM8K_NUMBER_REGEX!r} to the completion and take the LAST match, then "
    "strip '$' and ',' and any trailing '.'. Comparison is exact Decimal, so "
    "42, 42.0, +42 and 42,000 compare as numbers and never as strings. No "
    "match, or a match that is not a number, is INVALID.")

# ================================================================== tinyMMLU
def parse_mmlu(completion: str) -> Tuple[Optional[str], str]:
    """``(letter or None, parse status)`` from one raw completion."""
    if completion is None:
        return None, PARSE_NO_ANSWER
    text = str(completion)
    # The answer line only. Under raw completion the model goes on to the next
    # few-shot question; that continuation is expected and is not the answer.
    stripped = text.lstrip().split("\n", 1)[0].strip()
    if not stripped:
        return None, PARSE_NO_ANSWER

    head = stripped[0].upper()
    if head not in MMLU_OPTIONS:
        # Something that is not an option letter comes first. If an option
        # letter appears later it is prose-before-answer, not an answer.
        return None, (PARSE_PROSE_BEFORE_ANSWER
                      if re.search(r"\b[ABCD]\b", stripped.upper())
                      else PARSE_NO_ANSWER)

    rest = stripped[1:]
    # A letter immediately glued to more word characters is not an option.
    if rest[:1].isalnum():
        return None, PARSE_PROSE_BEFORE_ANSWER

    # A second, DIFFERENT option letter anywhere makes the answer ambiguous.
    others = {m for m in re.findall(r"\b([ABCD])\b", rest.upper())}
    if others - {head}:
        return None, PARSE_AMBIGUOUS

    tail = rest.strip()
    for character in tail:
        if character in MMLU_TRAILING_PUNCTUATION or character.isspace():
            continue
        if character.upper() == head:
            continue
        return None, PARSE_AMBIGUOUS
    return head, PARSE_OK


# ================================================================= tinyGSM8K
def _canonical_number(token: str) -> Optional[Decimal]:
    """One matched token as an exact Decimal, or None if it is not a number."""
    text = str(token).strip().replace(",", "").replace("$", "")
    while text.endswith("."):
        text = text[:-1]
    if text in ("", "+", "-", "."):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_gsm8k(completion: str) -> Tuple[Optional[Decimal], str]:
    """``(value or None, parse status)`` -- the LAST number, harness rule."""
    if completion is None:
        return None, PARSE_NO_ANSWER
    matches = re.findall(GSM8K_NUMBER_REGEX, str(completion))
    if not matches:
        return None, PARSE_NO_ANSWER
    for groups in reversed(matches):
        token = next((g for g in groups if g), "")
        value = _canonical_number(token)
        if value is not None:
            return value, PARSE_OK
    return None, PARSE_NO_ANSWER


__all__ = [
    "CapabilityParseError", "GSM8K_HARNESS", "GSM8K_HARNESS_REVISION",
    "GSM8K_HARNESS_TASK", "GSM8K_NUMBER_REGEX", "GSM8K_RULE",
    "MMLU_OPTIONS", "MMLU_RULE",
    "MMLU_TRAILING_PUNCTUATION", "PARSE_AMBIGUOUS", "PARSE_NO_ANSWER",
    "PARSE_OK", "PARSE_PROSE_BEFORE_ANSWER", "PARSE_STATUSES",
    "_canonical_number", "parse_gsm8k", "parse_mmlu",
]
