#!/usr/bin/env python3
"""Compare a fresh routing smoke against the historical smoke reference.

Phase 2 step 10 replaced the extraction's ``rq3_dev_generation`` import with
``routegate.runtime``. Step 9 proved the two render and tokenise all 480
prompts identically; this proves the swap changed nothing the GPU actually
produced, by re-running the two-row smoke and comparing artifact for artifact.

**The inventory comes first.** Each directory must hold exactly eleven regular
files -- the seven artifacts below plus the four derived records -- and nothing
else. No extra file, no subdirectory, no symlink. Comparing only the named
artifacts left anything else in the directory unexamined, so a candidate
carrying an extra array passed while nothing had ever looked at it.

**What must be byte-identical.** Every numerical artifact and the two records
derived only from them::

    router_logits.npy  selected_experts.npy  selected_probs.npy
    routing_entropy.npy  routing_margin.npy  row_ids.json  router_metadata.json

A single differing byte in any of these is a hard failure. There is no
tolerance option, and none should be added: a float tolerance here would hide
exactly the drift this comparison exists to catch. When the logits do differ the
maximum absolute difference is reported, as a diagnosis aid only -- it never
softens the verdict.

**What is expected to differ, and why.** Three files cannot be byte-identical
even when the extraction is perfect:

``completed_rows.jsonl``
    carries ``completed_utc`` per row -- wall-clock, different on every run.
    Every other ledger field, ``logits_sha256`` and ``seq_len`` included, must
    match exactly.

``config.json``
    carries ``prompt_config``, which names the module that renders the prompt.
    That string is *supposed* to change at step 10, from
    ``rq3_dev_generation.build_chat_strict`` to
    ``routegate.runtime.build_chat_strict``. It is the visible record of the
    cutover and must not be disguised. It also carries ``packages`` (interpreter
    and library versions) and ``source_manifest`` (recorded relative to the
    historical workspace before step 10, canonically after it).

``artifact_hashes.json``
    hashes every other file in the directory, so it inherits the two above and
    cannot match as a whole. Its *stable* entries -- the seven artifacts listed
    above -- are compared individually instead.

``extraction_validation.txt`` carries a timestamp and an elapsed time; the
candidate is required to report zero failures rather than to match it.

Usage::

    python tools/migration/compare_smoke_extractions.py \\
        --reference outputs/routing_extraction_v1_smoke \\
        --candidate <fresh smoke directory>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Byte-identical or the comparison fails. Nothing is added to this list without
# a reason recorded in the module docstring.
STABLE_ARTIFACTS: Tuple[str, ...] = (
    "router_logits.npy",
    "selected_experts.npy",
    "selected_probs.npy",
    "routing_entropy.npy",
    "routing_margin.npy",
    "row_ids.json",
    "router_metadata.json",
)

NUMERIC_ARRAYS: Tuple[str, ...] = (
    "router_logits.npy", "selected_experts.npy", "selected_probs.npy",
    "routing_entropy.npy", "routing_margin.npy",
)

# The only ledger field allowed to differ.
LEDGER_VOLATILE_FIELDS: Tuple[str, ...] = ("completed_utc",)

# The one config value allowed to differ, stated in full on both sides. Complete
# strings, never substrings: a substring test accepts
# "...enable_thinking=False (fail-closed)  AND ALSO enable_thinking=True", which
# is exactly the drift this comparison exists to catch.
PROMPT_CONFIG_BEFORE = (
    "rq3_dev_generation.build_chat_strict: single user message, no system "
    "prompt, add_generation_prompt=True, enable_thinking=False (fail-closed)")
PROMPT_CONFIG_AFTER = (
    "routegate.runtime.build_chat_strict: single user message, no system "
    "prompt, add_generation_prompt=True, enable_thinking=False (fail-closed)")

# There is no environment-tolerance list. `packages` and `source_manifest` are
# compared exactly: the job runs in the same pinned Apptainer image, and
# CANON_MANIFEST reproduces the historical source_manifest value verbatim. A
# drift in either is a real difference in what produced the artifacts, not noise.
CONFIG_EXACT_CHANGE_KEY = "prompt_config"

DERIVED_FILES: Tuple[str, ...] = (
    "completed_rows.jsonl", "config.json", "artifact_hashes.json",
    "extraction_validation.txt",
)

# The complete contents of a smoke run: eleven regular files, nothing else.
# Comparing only the named artifacts left an extra file entirely unexamined --
# a candidate carrying, say, an unexpected_scientific_output.npy passed, because
# nothing ever asked what else was in the directory. The inventory is therefore
# checked as a set: no missing file, no additional file, no subdirectory, and no
# symlink (which could point anywhere at all).
EXPECTED_FILES: Tuple[str, ...] = STABLE_ARTIFACTS + DERIVED_FILES


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Comparison:
    checks: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)

    @property
    def failures(self) -> List[str]:
        return [c["check"] for c in self.checks if not c["ok"]]


# ---------------------------------------------------------------- inventory
def inventory_problems(directory: Path) -> List[str]:
    """Everything wrong with one directory's top-level contents.

    Returned as a list of human-readable problems rather than a bool, so a
    failing check can say which file it objected to and why.
    """
    expected = set(EXPECTED_FILES)
    problems: List[str] = []
    seen = set()
    for entry in sorted(Path(directory).iterdir()):
        if entry.is_symlink():
            problems.append(f"symlink: {entry.name}")
            continue
        if entry.is_dir():
            problems.append(f"unexpected directory: {entry.name}")
            continue
        if not entry.is_file():
            problems.append(f"not a regular file: {entry.name}")
            continue
        seen.add(entry.name)
        if entry.name not in expected:
            problems.append(f"unexpected file: {entry.name}")
    for missing in sorted(expected - seen):
        problems.append(f"missing file: {missing}")
    return problems


def compare_inventory(reference: Path, candidate: Path, result: Comparison) -> None:
    """Both directories must hold exactly the eleven expected files.

    Recorded as two separate checks, because "the reference is not what this
    tool was written against" and "the candidate grew a file" are different
    problems needing different responses.
    """
    for label, directory in (("reference", reference), ("candidate", candidate)):
        problems = inventory_problems(directory)
        result.record(
            f"{label}_inventory_is_exactly_the_expected_{len(EXPECTED_FILES)}_files",
            not problems,
            "; ".join(problems) if problems
            else f"{len(EXPECTED_FILES)} regular files, nothing else")


# ------------------------------------------------------- numerical artifacts
def max_abs_difference(reference: Path, candidate: Path) -> Optional[float]:
    """Max |a - b| between two .npy arrays, for diagnosis only.

    Never consulted to decide pass or fail -- the verdict is byte equality.
    Returns ``None`` when the arrays cannot be compared elementwise at all: a
    different shape, a different dtype, or a file too damaged to load. That is a
    more serious difference than any magnitude, and it must not raise: this runs
    only on a path that has *already* failed, and a diagnostic that crashes
    would replace a clear FAIL with a traceback.

    ``allow_pickle`` is left at its default of False. A .npy carrying pickled
    objects is not an artifact this stage produces, and unpickling an unexpected
    file to print a number would be a poor trade.
    """
    try:
        import numpy as np
    except ImportError:                                   # pragma: no cover
        return None
    try:
        a = np.load(reference, mmap_mode="r")
        b = np.load(candidate, mmap_mode="r")
        if a.shape != b.shape:
            return None
        return float(np.abs(np.asarray(a, dtype="float64")
                            - np.asarray(b, dtype="float64")).max())
    except Exception:
        return None


def compare_stable_artifacts(reference: Path, candidate: Path,
                             result: Comparison) -> Dict[str, str]:
    """Byte equality on every artifact that must not move."""
    reference_hashes: Dict[str, str] = {}
    for name in STABLE_ARTIFACTS:
        ref, cand = reference / name, candidate / name
        if not ref.is_file():
            result.record(f"reference_has:{name}", False, "missing from the reference")
            continue
        reference_hashes[name] = sha256_file(ref)
        if not cand.is_file():
            result.record(f"identical:{name}", False, "missing from the candidate")
            continue
        cand_hash = sha256_file(cand)
        identical = cand_hash == reference_hashes[name]
        detail = "byte-identical" if identical else f"SHA256 differs ({cand_hash[:16]}…)"
        if not identical and name in NUMERIC_ARRAYS:
            delta = max_abs_difference(ref, cand)
            note = (f"{name}: max |difference| = {delta:.6g}" if delta is not None
                    else f"{name}: shape or dtype differs; not comparable elementwise")
            result.diagnostics.append(note)
            # Diagnosis only. The check still fails.
            detail = f"{detail}; {note}"
        result.record(f"identical:{name}", identical, detail)
    return reference_hashes


# ------------------------------------------------------------------- ledger
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_volatile(row: Dict[str, Any],
                   volatile: Sequence[str] = LEDGER_VOLATILE_FIELDS) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in volatile}


def compare_ledger(reference: Path, candidate: Path, result: Comparison) -> None:
    """Every ledger field but ``completed_utc``, including logits_sha256."""
    ref_path, cand_path = reference / "completed_rows.jsonl", candidate / "completed_rows.jsonl"
    if not (ref_path.is_file() and cand_path.is_file()):
        result.record("ledger_present", False, "completed_rows.jsonl missing on one side")
        return
    ref_rows, cand_rows = read_jsonl(ref_path), read_jsonl(cand_path)

    result.record("ledger_row_count", len(ref_rows) == len(cand_rows),
                  f"{len(cand_rows)} rows (reference {len(ref_rows)})")
    if len(ref_rows) != len(cand_rows):
        return

    ref_fields = {k for row in ref_rows for k in row}
    cand_fields = {k for row in cand_rows for k in row}
    result.record("ledger_field_set", ref_fields == cand_fields,
                  f"{sorted(cand_fields)}")

    # Named explicitly rather than relied on implicitly: these two are the
    # scientific content of the ledger.
    for key in ("logits_sha256", "seq_len"):
        present = all(key in row for row in cand_rows)
        result.record(f"ledger_carries:{key}", present,
                      "present on every row" if present else "missing on some row")

    mismatched = [i for i, (a, b) in enumerate(zip(ref_rows, cand_rows))
                  if strip_volatile(a) != strip_volatile(b)]
    detail = (f"{len(cand_rows)}/{len(ref_rows)} rows identical once "
              f"{list(LEDGER_VOLATILE_FIELDS)} is excluded")
    if mismatched:
        differing_keys = sorted({
            k for i in mismatched
            for k in set(strip_volatile(ref_rows[i])) | set(strip_volatile(cand_rows[i]))
            if strip_volatile(ref_rows[i]).get(k) != strip_volatile(cand_rows[i]).get(k)})
        detail = (f"rows {mismatched[:5]} differ in fields {differing_keys}")
    result.record("ledger_rows_match_excluding_only_completed_utc",
                  not mismatched, detail)


# ------------------------------------------------------------------- config
def compare_config(reference: Path, candidate: Path, result: Comparison) -> None:
    """Exactly one value may differ, and it must be exactly the expected one.

    Every other key and value must be identical, including ``packages`` and
    ``source_manifest``. Nothing here is matched by substring: the two
    ``prompt_config`` values are compared in full against the complete strings
    recorded above, so neither an unchanged value nor an appended suffix can
    pass.
    """
    ref_path, cand_path = reference / "config.json", candidate / "config.json"
    if not (ref_path.is_file() and cand_path.is_file()):
        result.record("config_present", False, "config.json missing on one side")
        return
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    cand = json.loads(cand_path.read_text(encoding="utf-8"))

    result.record("config_key_set_identical", set(ref) == set(cand),
                  f"added={sorted(set(cand) - set(ref))} "
                  f"removed={sorted(set(ref) - set(cand))}"
                  if set(ref) != set(cand) else f"{len(ref)} keys, identical")

    key = CONFIG_EXACT_CHANGE_KEY
    result.record(f"reference_{key}_is_exactly_the_pre_cutover_string",
                  ref.get(key) == PROMPT_CONFIG_BEFORE,
                  "exact match" if ref.get(key) == PROMPT_CONFIG_BEFORE
                  else f"reference carries {ref.get(key)!r}")
    result.record(f"candidate_{key}_is_exactly_the_post_cutover_string",
                  cand.get(key) == PROMPT_CONFIG_AFTER,
                  "exact match" if cand.get(key) == PROMPT_CONFIG_AFTER
                  else f"candidate carries {cand.get(key)!r}")

    # Every key except the one permitted change must be byte-equal. Listed by
    # difference rather than by allowance, so a key nobody thought of is caught.
    differing = sorted(k for k in set(ref) | set(cand)
                       if k != key and ref.get(k) != cand.get(k))
    result.record("every_other_config_value_is_identical", not differing,
                  f"these differ and none may: {differing}" if differing
                  else f"{len(set(ref) | set(cand)) - 1} keys identical, "
                       f"including packages and source_manifest")


# --------------------------------------------------------- artifact hashes
def compare_artifact_hashes(reference: Path, candidate: Path,
                            reference_hashes: Dict[str, str],
                            result: Comparison) -> None:
    """The whole file cannot match; its stable entries must.

    ``artifact_hashes.json`` hashes every other file in the directory, so it
    inherits the ledger's timestamps and the config's provenance change. Only
    the entries for the artifacts that must not move are checked.
    """
    cand_path = candidate / "artifact_hashes.json"
    if not cand_path.is_file():
        result.record("artifact_hashes_present", False, "missing from the candidate")
        return
    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    result.record("artifact_hashes_covers_every_stable_artifact",
                  all(name in cand for name in STABLE_ARTIFACTS),
                  f"{sorted(n for n in STABLE_ARTIFACTS if n not in cand)} missing"
                  if not all(n in cand for n in STABLE_ARTIFACTS) else "all present")
    wrong = [name for name in STABLE_ARTIFACTS
             if name in cand and name in reference_hashes
             and cand[name] != reference_hashes[name]]
    result.record("artifact_hashes_stable_entries_match", not wrong,
                  f"entries differ: {wrong}" if wrong
                  else f"{len(reference_hashes)} stable entries agree with the reference")
    # Self-consistency: the recorded hash must be the file's actual hash.
    inconsistent = [name for name in STABLE_ARTIFACTS
                    if name in cand and (candidate / name).is_file()
                    and sha256_file(candidate / name) != cand[name]]
    result.record("artifact_hashes_are_self_consistent", not inconsistent,
                  f"recorded hash wrong for: {inconsistent}" if inconsistent
                  else "each recorded hash re-computes from its own file")


# ----------------------------------------------------------------- validation
# Anchored and complete: the whole line must be the summary, so a summary
# embedded in prose is not mistaken for one, and -- with findall rather than
# search -- a second summary cannot hide behind the first.
SUMMARY_LINE = re.compile(r"^CHECKS:\s+(\d+)\s+FAILURES:\s+(\d+)\s*$", re.M)


def validation_summaries(directory: Path) -> Optional[List[Tuple[int, int]]]:
    """Every anchored summary line in a run's report, or ``None`` if absent.

    All of them, not the first. A report ending

        CHECKS: 21   FAILURES: 0
        CHECKS: 21   FAILURES: 7

    is not a passing report, and a parser that stops at the first match reads it
    as one. Which summary is the "real" one is unanswerable, so the correct
    response to more than one is to refuse the report rather than to choose.
    """
    path = directory / "extraction_validation.txt"
    if not path.is_file():
        return None
    return [(int(c), int(f))
            for c, f in SUMMARY_LINE.findall(path.read_text(encoding="utf-8"))]


def compare_validation(reference: Path, candidate: Path, result: Comparison) -> None:
    """Both runs must validate, and to the same extent.

    "Zero failures" alone is not enough: a report with no checks in it also has
    no failures, and a report with fewer checks than the reference is a weaker
    claim wearing the same words. Both are refused.
    """
    ref_summaries = validation_summaries(reference)
    cand_summaries = validation_summaries(candidate)

    if not result.record("both_validation_reports_present",
                         ref_summaries is not None and cand_summaries is not None,
                         f"reference={'present' if ref_summaries is not None else 'MISSING'} "
                         f"candidate={'present' if cand_summaries is not None else 'MISSING'}"):
        return

    exactly_one = True
    for label, summaries in (("reference", ref_summaries), ("candidate", cand_summaries)):
        ok = len(summaries) == 1
        exactly_one = exactly_one and ok
        result.record(f"{label}_validation_has_exactly_one_summary_line", ok,
                      f"{len(summaries)} summary lines: {summaries}" if not ok
                      else f"one summary line: CHECKS {summaries[0][0]}, "
                           f"FAILURES {summaries[0][1]}")
    if not exactly_one:
        # Which summary counts is unanswerable; do not guess by using one.
        return

    ref_checks, ref_failures = ref_summaries[0]
    cand_checks, cand_failures = cand_summaries[0]

    result.record("reference_validation_zero_failures", ref_failures == 0,
                  f"CHECKS: {ref_checks}   FAILURES: {ref_failures}")
    result.record("candidate_validation_zero_failures", cand_failures == 0,
                  f"CHECKS: {cand_checks}   FAILURES: {cand_failures}")
    # A report that checked nothing trivially has no failures.
    result.record("reference_validation_is_not_empty", ref_checks > 0,
                  f"{ref_checks} checks were run")
    result.record("candidate_validation_is_not_empty", cand_checks > 0,
                  f"{cand_checks} checks were run")
    result.record("candidate_validation_check_count_matches_reference",
                  cand_checks == ref_checks,
                  f"{cand_checks} vs reference {ref_checks}")

    # The summary line is a count; an explicit [FAIL] anywhere contradicts it.
    for label, directory in (("reference", reference), ("candidate", candidate)):
        text = (directory / "extraction_validation.txt").read_text(encoding="utf-8")
        fail_lines = [l.strip() for l in text.splitlines() if "[FAIL]" in l]
        result.record(f"{label}_validation_has_no_fail_line", not fail_lines,
                      f"{len(fail_lines)} [FAIL] line(s): {fail_lines[:3]}"
                      if fail_lines else f"no [FAIL] line in the {label} report")


# ------------------------------------------------------------------- driver
def compare(reference: Path, candidate: Path) -> Comparison:
    result = Comparison()
    reference, candidate = Path(reference), Path(candidate)
    for label, path in (("reference", reference), ("candidate", candidate)):
        if not path.is_dir():
            result.record(f"{label}_directory_exists", False, str(path))
    if not result.ok:
        return result
    result.record("reference_and_candidate_differ", reference.resolve() != candidate.resolve(),
                  "a directory compared against itself proves nothing")
    if not result.ok:
        return result

    compare_inventory(reference, candidate, result)
    reference_hashes = compare_stable_artifacts(reference, candidate, result)
    compare_ledger(reference, candidate, result)
    compare_config(reference, candidate, result)
    compare_artifact_hashes(reference, candidate, reference_hashes, result)
    compare_validation(reference, candidate, result)
    return result


def render(result: Comparison) -> str:
    lines = ["=" * 78,
             "FRESH SMOKE vs HISTORICAL SMOKE",
             "=" * 78, "",
             f"Inventory: each directory must hold exactly these "
             f"{len(EXPECTED_FILES)} regular files and nothing else --",
             "  " + ", ".join(EXPECTED_FILES),
             "  no extra file, no subdirectory, no symlink.", "",
             "Byte-identical or fail:",
             "  " + ", ".join(STABLE_ARTIFACTS), "",
             "Not expected to be byte-identical, and why:",
             "  completed_rows.jsonl  -- completed_utc is wall-clock, per row",
             "  config.json           -- prompt_config names the rendering module,",
             "                           which step 10 deliberately changes",
             "  artifact_hashes.json  -- hashes the two above, so it inherits both;",
             "                           its stable entries are checked individually",
             "  extraction_validation.txt -- timestamp and elapsed time", "",
             "Validation reports: exactly one anchored CHECKS/FAILURES summary",
             "  line each, both with more than zero checks and zero failures,",
             "  matching check counts, and no [FAIL] line in either.", ""]
    for check in result.checks:
        lines.append(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']:<52} {check['detail']}")
    if result.diagnostics:
        lines += ["", "Diagnostics (for investigation only -- no tolerance is applied):"]
        lines += [f"  {d}" for d in result.diagnostics]
    lines += ["", "=" * 78,
              f"VERDICT: {'PASS' if result.ok else 'FAIL'}   "
              f"{len(result.checks) - len(result.failures)}/{len(result.checks)} checks",
              "=" * 78]
    if result.failures:
        lines.append(f"failed: {result.failures}")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path,
                        help="the historical smoke directory (read-only)")
    parser.add_argument("--candidate", required=True, type=Path,
                        help="the freshly produced smoke directory")
    parser.add_argument("--report", type=Path, default=None,
                        help="optional path to write the report to")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    result = compare(args.reference, args.candidate)
    report = render(result)
    print(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n", encoding="utf-8")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
