"""Choosing a donor for every target, and freezing the choice before any model runs.

The historical RQ1 donor was not a corpus prompt at all: it was the same refused
prompt re-rendered under a disambiguating system instruction, or -- where that
donor was itself refused -- a hand-written benign rephrasing, checked by a human
and recorded per case. That is fine for ten cases and impossible for 89, and it
used two things this stage must not use: a human read of the prompt, and the
donor's own *generated* answer.

So the pairing here is new, and it is deliberately narrow. A donor is chosen for
each E target from the B class by metadata alone:

    source_family        which benchmark the prompt came from
    topic_standardized   the adjudicated topic
    task_frame           the adjudicated task frame
    prompt length        character count, as a proximity tiebreak

Nothing else is legible to it. It never sees an activation, a generated
response, a patching result, or anything outside the accepted development
corpus, and it is a pure function of the dataset -- run it twice and it returns
the same pairs.

Matching degrades through a fixed ladder rather than failing. Tier 0 is the
exact match on all three columns; each later tier drops one key, in a fixed
order chosen so that *what is being asked about* survives longer than *where the
prompt came from*. Every pair records the tier it was satisfied at, so a
downstream reader can drop the fallbacks entirely, and a run whose pairs are
mostly fallbacks can be recognised as such instead of being read as matched.

The manifest is written, hashed and frozen **before** the model is loaded. The
run refuses to execute against a manifest whose hash does not match the one it
was told to expect, so a pairing cannot quietly change between a run and its
re-run.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .data import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    SOURCE_COLUMN,
    TASK_FRAME_COLUMN,
    TOPIC_COLUMN,
    WITHHELD_LABEL,
)
from .validation import (
    ValidationError,
    assert_hash,
    assert_not_sealed,
    assert_not_stale,
    sha256_text,
)

MANIFEST_SCHEMA = "routegate/donor_manifest/1"

# The only columns donor selection may read. Anything not on this list is not
# available to it, which is enforced by projecting the rows before matching.
PAIRING_COLUMNS = ("row_id", "final_ebh", "prompt_text",
                   SOURCE_COLUMN, TOPIC_COLUMN, TASK_FRAME_COLUMN)

# Fixed ladder, most specific first. Dropping source_family before topic is
# deliberate: two prompts about the same topic in the same task frame are a
# closer pair than two prompts from the same benchmark about different things.
MATCH_TIERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("source_topic_task", (SOURCE_COLUMN, TOPIC_COLUMN, TASK_FRAME_COLUMN)),
    ("topic_task", (TOPIC_COLUMN, TASK_FRAME_COLUMN)),
    ("topic", (TOPIC_COLUMN,)),
    ("task_frame", (TASK_FRAME_COLUMN,)),
    ("length_only", ()),
)
EXACT_TIER = MATCH_TIERS[0][0]


class DonorError(ValidationError):
    """Donor selection was asked for something it must not do."""


@dataclass(frozen=True)
class DonorMatch:
    """One frozen (target, donor) pair and the evidence for it."""

    target_row_id: str
    target_label: str
    donor_row_id: str
    donor_label: str
    tier: str
    tier_index: int
    exact_match: bool
    relaxed_keys: Tuple[str, ...]
    matched_values: Dict[str, str]
    target_values: Dict[str, str]
    donor_values: Dict[str, str]
    target_prompt_chars: int
    donor_prompt_chars: int
    length_delta_chars: int
    candidates_in_tier: int

    def as_dict(self) -> Dict[str, Any]:
        record = asdict(self)
        record["relaxed_keys"] = list(self.relaxed_keys)
        return record


# ------------------------------------------------------------------- loading
def load_pairing_rows(dataset_path: Path | str,
                      expected_sha256: Optional[str] = None
                      ) -> Tuple[List[Dict[str, str]], str]:
    """The E and B rows, projected to the columns pairing may read.

    A text-only view of the accepted corpus. It exists separately from
    ``data.load_development_inputs`` because the manifest is built before any
    activation is needed: memory-mapping a 192 MB array to decide which prompt
    pairs with which would be theatre, not a check.

    The withheld class is refused here rather than filtered. This stage analyses
    E against B; a code path that quietly dropped H would be a code path that
    could quietly keep it.
    """
    path = assert_not_stale(Path(dataset_path))
    assert_not_sealed(path, "donor pairing dataset")
    digest = (assert_hash(path, expected_sha256, "pairing dataset")
              if expected_sha256 else None)
    with open(path, newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise DonorError(f"{path} contains no rows")
    missing = [c for c in PAIRING_COLUMNS if c not in raw_rows[0]]
    if missing:
        raise DonorError(
            f"{path} does not carry the columns donor pairing needs: {missing}")

    rows = [{column: row[column] for column in PAIRING_COLUMNS}
            for row in raw_rows
            if row["final_ebh"] in (POSITIVE_LABEL, NEGATIVE_LABEL)]
    assert_no_withheld_rows(rows)
    for row in rows:
        for column in (SOURCE_COLUMN, TOPIC_COLUMN, TASK_FRAME_COLUMN):
            if not row[column].strip():
                raise DonorError(
                    f"{row['row_id']} has no {column}; donor matching may not "
                    f"invent one")
    return rows, (digest or "")


def assert_no_withheld_rows(rows: Sequence[Dict[str, str]]) -> None:
    """Refuse the withheld class outright at this stage."""
    withheld = [r["row_id"] for r in rows if r.get("final_ebh") == WITHHELD_LABEL]
    if withheld:
        raise DonorError(
            f"{len(withheld)} row(s) of the withheld class {WITHHELD_LABEL} "
            f"reached donor pairing (first: {withheld[0]}). This stage runs E "
            f"targets against B donors only; harmful-preservation is a separate "
            f"stage with its own protocol.")


def split_targets_and_donors(rows: Sequence[Dict[str, str]]
                             ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """(E targets, B donors), each in ascending row_id order."""
    targets = sorted((r for r in rows if r["final_ebh"] == POSITIVE_LABEL),
                     key=lambda r: r["row_id"])
    donors = sorted((r for r in rows if r["final_ebh"] == NEGATIVE_LABEL),
                    key=lambda r: r["row_id"])
    if not targets:
        raise DonorError(f"no {POSITIVE_LABEL} targets in the corpus")
    if not donors:
        raise DonorError(f"no {NEGATIVE_LABEL} donors in the corpus")
    return targets, donors


# ------------------------------------------------------------------ matching
def prompt_chars(row: Dict[str, str]) -> int:
    """Prompt length in characters.

    Characters, not tokens: the extraction saved no token count and this package
    never loads a tokeniser, so a "token length" here would be a number invented
    after the fact. Length is only ever a tiebreak within a metadata tier.
    """
    return len(row["prompt_text"])


def match_donor(target: Dict[str, str],
                donors: Sequence[Dict[str, str]]) -> DonorMatch:
    """The donor for one target: first satisfied tier, then nearest length.

    Deterministic twice over -- the tier ladder is fixed, and within a tier the
    ordering is (absolute length difference, row_id), which is a total order
    because row_id is unique.
    """
    target_values = {column: target[column]
                     for column in (SOURCE_COLUMN, TOPIC_COLUMN, TASK_FRAME_COLUMN)}
    target_chars = prompt_chars(target)

    for tier_index, (tier_name, keys) in enumerate(MATCH_TIERS):
        candidates = [d for d in donors
                      if all(d[key] == target[key] for key in keys)
                      and d["row_id"] != target["row_id"]]
        if not candidates:
            continue
        best = min(candidates,
                   key=lambda d: (abs(prompt_chars(d) - target_chars), d["row_id"]))
        relaxed = tuple(k for k in MATCH_TIERS[0][1] if k not in keys)
        return DonorMatch(
            target_row_id=target["row_id"],
            target_label=target["final_ebh"],
            donor_row_id=best["row_id"],
            donor_label=best["final_ebh"],
            tier=tier_name,
            tier_index=tier_index,
            exact_match=(tier_index == 0),
            relaxed_keys=relaxed,
            matched_values={key: target[key] for key in keys},
            target_values=target_values,
            donor_values={column: best[column]
                          for column in (SOURCE_COLUMN, TOPIC_COLUMN,
                                         TASK_FRAME_COLUMN)},
            target_prompt_chars=target_chars,
            donor_prompt_chars=prompt_chars(best),
            length_delta_chars=abs(prompt_chars(best) - target_chars),
            candidates_in_tier=len(candidates),
        )

    raise DonorError(
        f"no donor at all for {target['row_id']}: even the length-only tier was "
        f"empty, which means the donor pool was empty")


def build_pairs(targets: Sequence[Dict[str, str]],
                donors: Sequence[Dict[str, str]]) -> List[DonorMatch]:
    """One donor per target, in ascending target row_id order."""
    return [match_donor(target, donors)
            for target in sorted(targets, key=lambda r: r["row_id"])]


# ------------------------------------------------------------------ manifest
def build_manifest(pairs: Sequence[DonorMatch], dataset_path: Path | str,
                   dataset_sha256: str, created_utc: str,
                   n_targets: int, n_donors: int) -> Dict[str, Any]:
    """The frozen pairing, ready to hash and write."""
    tier_counts: Dict[str, int] = {name: 0 for name, _ in MATCH_TIERS}
    for pair in pairs:
        tier_counts[pair.tier] += 1
    donor_uses: Dict[str, int] = {}
    for pair in pairs:
        donor_uses[pair.donor_row_id] = donor_uses.get(pair.donor_row_id, 0) + 1

    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": created_utc,
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha256},
        "rule": {
            "targets": f"every {POSITIVE_LABEL} row of the accepted corpus",
            "donors": f"{NEGATIVE_LABEL} rows only",
            "permitted_fields": list(PAIRING_COLUMNS),
            "forbidden_inputs": [
                "activation values", "generated responses", "patching results",
                "anything from the held-out benchmark split",
            ],
            "tiers": [{"index": i, "name": name, "keys": list(keys)}
                      for i, (name, keys) in enumerate(MATCH_TIERS)],
            "within_tier": "smallest absolute prompt-character difference, ties "
                           "broken by ascending row_id",
            "length_unit": "characters; no tokeniser is loaded and no token "
                           "count was saved by the extraction",
            "self_pairing": "excluded: a donor is never the target itself",
            "donor_reuse": "permitted and counted; donors are not a scarce "
                           "resource and forcing a bijection would degrade the "
                           "metadata match for no scientific gain",
        },
        "counts": {
            "targets": n_targets,
            "donor_pool": n_donors,
            "pairs": len(pairs),
            "exact_matches": tier_counts[EXACT_TIER],
            "fallback_matches": len(pairs) - tier_counts[EXACT_TIER],
            "by_tier": tier_counts,
            "distinct_donors_used": len(donor_uses),
            "max_uses_of_one_donor": max(donor_uses.values()) if donor_uses else 0,
        },
        "pairs": [pair.as_dict() for pair in pairs],
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def canonical_json(manifest: Dict[str, Any]) -> str:
    """The manifest as hashed: sorted keys, no whitespace, hash field removed."""
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def manifest_sha256(manifest: Dict[str, Any]) -> str:
    return sha256_text(canonical_json(manifest))


def write_manifest(path: Path | str, manifest: Dict[str, Any]) -> str:
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    return manifest["manifest_sha256"]


def load_manifest(path: Path | str,
                  expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Read a frozen manifest and refuse it if it has moved.

    The recomputed hash is checked against the one the file carries, and against
    the caller's expectation if it has one. A manifest that was edited after it
    was frozen fails both.
    """
    path = Path(path)
    if not path.exists():
        raise DonorError(
            f"{path} does not exist. The donor manifest is built and frozen "
            f"before the model is loaded; there is no run-time pairing.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise DonorError(
            f"{path} declares schema {manifest.get('schema')!r}, expected "
            f"{MANIFEST_SCHEMA!r}")
    recorded = manifest.get("manifest_sha256")
    recomputed = manifest_sha256(manifest)
    if recorded != recomputed:
        raise DonorError(
            f"{path} has been modified since it was frozen:\n"
            f"  recorded   {recorded}\n"
            f"  recomputed {recomputed}")
    if expected_sha256 and recorded != expected_sha256:
        raise DonorError(
            f"{path} is not the manifest this run was told to use:\n"
            f"  expected {expected_sha256}\n"
            f"  actual   {recorded}")
    return manifest


def pairs_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The frozen pairs, keyed by target row_id."""
    pairs: Dict[str, Dict[str, Any]] = {}
    for entry in manifest.get("pairs", []):
        target = entry["target_row_id"]
        if target in pairs:
            raise DonorError(f"the manifest pairs {target} more than once")
        pairs[target] = entry
    if not pairs:
        raise DonorError("the manifest carries no pairs")
    return pairs
