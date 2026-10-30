#!/usr/bin/env python3
"""Freeze the 20 LEGACY H regressions as a metadata-only stress set.

Metadata only. This loads no model, calls nothing, judges nothing, reads no
sealed material and copies no prompt or response text. It reads three frozen
files under the RouteGate data root, checks them against their recorded
digests, and writes a list of twenty row ids.

What the twenty are
-------------------
They are the H rows that the **legacy** static actuator turned from a refusal
into actionable harmful help:

    variant   consensus23_global_frozen   (the RQ2 *primary* actuator)
    donors    B_d2 and B_d3
    strength  0.15
    layers    historical 31, 32, 33, final prompt token, prefill only

Not inferred, not re-derived and not keyword-matched: the label is read from
``final_actionable_harmful_help == "YES"`` in the frozen static-steering
master, where all twenty carry ``final_safety_label_source == "human"``. The
condition is read field by field from the run configuration that produced them.

What produced them, and what did not
------------------------------------
The treatment that produced these regressions is ``consensus23_global_frozen``,
built from donors **d2 and d3**. It is **not** ``historical_d1_global_frozen``
(``B_d1 - E``), which is the comparator and the direction the validated
direction bank reproduces. The two are different vectors from different
donors, and ``primary_actuator_source`` is validated for 3 of 10 cases where
``comparator_actuator_source`` is validated for 10 of 10.

So this file records, and the manifest states, three things it does **not**
claim:

* not that ``B_d1 - E`` caused these regressions -- it did not; a different
  direction did;
* not that these are failures of the validated direction bank -- the bank did
  not exist when they were produced and was not the treatment;
* not that testing ``H - E`` on them demonstrates rescue of current
  ``B_d1 - E`` failures -- a rescue of legacy regressions is evidence about
  legacy regressions.

What it is for
--------------
A **legacy stress set**: twenty rows already known to be steerable into harmful
compliance by *a* B-minus-E direction at this strength and depth. That makes
them a demanding place to ask whether ``H - E`` restores refusal. It is a
diagnostic, and a restricted one.

Why it is not a population number
---------------------------------
Twenty rows chosen because they regressed are twenty rows selected on the
outcome. Anything measured on them is conditioned on that selection, so the
manifest records ``targeted_diagnostic: true`` and
``population_reportable: false``. The population claim, if there is one, comes
from the full 213-row H arm of the causal screen.

Usage
-----
    python experiments/dual_direction_controller/build_targeted_h_legacy_regressions.py \\
        --static-master <data-root>/frozen/static_steering/<master>.csv \\
        --corpus <data-root>/frozen/corpus/rq3_dataset_final.csv \\
        --static-run-config <data-root>/frozen/static_steering/rq3_bh_rq2_static_20260810/run_config.json \\
        --output manifests/targeted_h_legacy_regressions_v1.json [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    assert_not_sealed,
    sha256_file,
)

SCHEMA = "routegate/targeted_h_legacy_regressions/1"
DEFAULT_OUTPUT = (REPO_ROOT / "manifests"
                  / "targeted_h_legacy_regressions_v1.json")

#: How many rows the recorded regression set holds. Frozen: this file exists to
#: pin that number, and a different one is a different finding.
EXPECTED_ROWS = 20
SCREEN_CLASS = "H"

#: The columns read from the static-steering master. Deliberately narrow: no
#: prompt_text, no before_response_text, no after_response_text.
MASTER_COLUMNS = ("row_id", "frozen_class", "source", "before_axis2",
                  "final_actionable_harmful_help", "final_safety_label_source",
                  "intervention_hit_count")

#: Columns that carry prompt or response text and are never read or copied.
TEXT_COLUMNS = ("prompt_text", "before_response_text", "after_response_text",
                "human_reason")

#: The recorded digests of the two frozen inputs, from manifests/frozen_inputs.json.
STATIC_MASTER_LOGICAL = "static_steering_ebh_master"
CORPUS_LOGICAL = "development_corpus"

#: The static condition these regressions were produced under, checked field by
#: field against the run configuration rather than asserted here.
EXPECTED_CONDITION: Dict[str, Any] = {
    "layers": [31, 32, 33],
    "window": 1,
    "slot": 0,
    "strength": 0.15,
    "site": "residual_stream",
    "prefill_only": True,
    "expected_patch_hits_per_row": 3,
}
EXPECTED_MODEL = "Qwen/Qwen3-30B-A3B"
EXPECTED_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

#: The variant actually used, and the donors it was built from. Recorded, not
#: assumed: see the module docstring.
EXPECTED_VARIANT = "consensus23_global_frozen"
SOURCE_DONORS: tuple = ("B_d2", "B_d3")

#: The direction the validated bank reproduces, named here only so the manifest
#: can state that it was NOT the treatment.
CURRENT_VALIDATED_REPAIR_VARIANT = "historical_d1_global_frozen"

#: The migrated condition file, and its digest. It is not in
#: manifests/frozen_inputs.json, so the digest is pinned here and checked the
#: same way: a different file is refused, never adapted to.
STATIC_RUN_CONFIG_RELATIVE = (
    "frozen/static_steering/rq3_bh_rq2_static_20260810/run_config.json")
STATIC_RUN_CONFIG_SHA256 = (
    "e8b1b46f0923ba1bef1a770ed1bc95d4e92c49e6457a52bcdb0ef3a0d1e31777")

#: Path fragments that mean the historical workspace. This stage reads only the
#: migrated data root; a run that reached back into SteerMoE-main would leave
#: the selection unreproducible from RouteGate and routegate-data alone.
HISTORICAL_WORKSPACE_TOKENS: tuple = ("steermoe",)


def assert_not_historical_workspace(path: Path, label: str) -> None:
    """Refuse an input that still resolves into the historical workspace."""
    text = str(Path(path).resolve()).lower()
    hit = [token for token in HISTORICAL_WORKSPACE_TOKENS if token in text]
    if hit:
        raise LegacyRegressionSelectionError(
            f"{label} resolves to {path}, which is in the historical "
            f"workspace ({hit[0]}). This selection reads only the migrated "
            f"data root; a runtime dependency on that workspace would leave "
            f"it unreproducible from RouteGate and routegate-data alone.")


class LegacyRegressionSelectionError(ValidationError):
    """The legacy stress set could not be established from the evidence."""


def gate(checks: Checklist, name: str, passed: bool, detail: str) -> None:
    checks.record(name, passed, detail)
    if not passed:
        raise LegacyRegressionSelectionError(f"{name}: {detail}")


def recorded_digest(logical_name: str) -> Mapping[str, Any]:
    """What manifests/frozen_inputs.json records for one frozen input."""
    manifest = json.loads(
        (REPO_ROOT / "manifests" / "frozen_inputs.json").read_text("utf-8"))
    for entry in manifest["inputs"]:
        if entry["logical_name"] == logical_name:
            return entry
    raise LegacyRegressionSelectionError(
        f"manifests/frozen_inputs.json records no input named "
        f"{logical_name!r}; the selection is not built from an unrecorded file")


def verify_input(path: Path, logical_name: str, checks: Checklist) -> str:
    """Re-hash a frozen input and refuse it unless it is the recorded one."""
    path = Path(path)
    if not path.is_file():
        raise LegacyRegressionSelectionError(f"{path} is not a file")
    assert_not_sealed(path, f"legacy H regression input {logical_name}")
    assert_not_historical_workspace(path, logical_name)
    entry = recorded_digest(logical_name)
    digest = sha256_file(path)
    gate(checks, f"{logical_name}_sha256", digest == entry["sha256"],
         f"{path.name} is {digest}, recorded {entry['sha256']}")
    return digest


def verify_condition(run_config: Mapping[str, Any], checks: Checklist
                     ) -> Dict[str, Any]:
    """The static condition, read from the run that produced the regressions."""
    actuator = run_config.get("actuator")
    generation = run_config.get("generation")
    if not isinstance(actuator, Mapping) or not isinstance(generation, Mapping):
        raise LegacyRegressionSelectionError(
            "the run configuration carries no actuator or generation block; "
            "the static condition cannot be established from it")

    for field, expected in EXPECTED_CONDITION.items():
        actual = actuator.get(field)
        gate(checks, f"condition_{field}", actual == expected,
             f"{field}={actual!r}, expected {expected!r}")
    gate(checks, "condition_direction_variant",
         actuator.get("direction_variant") == EXPECTED_VARIANT,
         f"{actuator.get('direction_variant')!r}")
    gate(checks, "condition_model", generation.get("model") == EXPECTED_MODEL,
         f"{generation.get('model')!r}")
    for field in ("revision_resolved", "tokenizer_revision"):
        gate(checks, f"condition_{field}",
             generation.get(field) == EXPECTED_REVISION,
             f"{field}={generation.get(field)!r}")
    return {
        "direction_variant": actuator["direction_variant"],
        "direction_role": actuator.get("role"),
        "site": actuator["site"],
        "historical_layers": list(actuator["layers"]),
        "window": actuator["window"],
        "slot": actuator["slot"],
        "slot_semantics": actuator.get("slot_semantics"),
        "prefill_only": actuator["prefill_only"],
        "strength_alpha_repair": actuator["strength"],
        "alpha_safety": 0.0,
        "alpha_safety_note": "no safety direction existed when this ran; the "
                             "static actuator applies one direction only",
        "expected_patch_hits_per_row": actuator["expected_patch_hits_per_row"],
        "application_scaling": actuator.get("application_scaling"),
        "model": generation["model"],
        "revision": generation["revision_resolved"],
        "tokenizer_revision": generation["tokenizer_revision"],
        "tokenizer_config_sha256": generation.get("tokenizer_config_sha256"),
        "chat_template_sha256": generation.get("chat_template_sha256"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "decoding": generation.get("decoding"),
    }


def read_master_rows(path: Path) -> List[Dict[str, str]]:
    """The master, projected to the metadata columns. Text is never read."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in MASTER_COLUMNS
                   if column not in (reader.fieldnames or [])]
        if missing:
            raise LegacyRegressionSelectionError(
                f"{path.name} is missing column(s) {missing}")
        return [{column: row[column] for column in MASTER_COLUMNS}
                for row in reader]


def select(master_rows: Sequence[Mapping[str, str]],
           corpus_class: Mapping[str, str],
           checks: Checklist) -> List[Dict[str, Any]]:
    """The recorded regressions, by the one rule, checked every way it can be.

    The rule is a field lookup, not a search: ``frozen_class == "H"`` and
    ``final_actionable_harmful_help == "YES"``. Nothing is matched on text.
    """
    selected = [row for row in master_rows
                if row["frozen_class"] == SCREEN_CLASS
                and row["final_actionable_harmful_help"].strip().upper() == "YES"]

    ids = [row["row_id"] for row in selected]
    gate(checks, "row_count", len(selected) == EXPECTED_ROWS,
         f"{len(selected)} row(s) match the rule, expected {EXPECTED_ROWS}")
    gate(checks, "row_ids_unique", len(set(ids)) == len(ids),
         f"{len(set(ids))} unique of {len(ids)}")

    not_h = [row["row_id"] for row in selected
             if row["frozen_class"] != SCREEN_CLASS]
    gate(checks, "all_rows_are_H_in_the_master", not not_h,
         f"{len(not_h)} non-{SCREEN_CLASS} row(s)")

    absent = [i for i in ids if i not in corpus_class]
    gate(checks, "all_rows_in_the_accepted_corpus", not absent,
         f"{len(absent)} row(s) absent from the corpus"
         + (f", first {absent[0]}" if absent else ""))
    wrong = [i for i in ids if corpus_class.get(i) != SCREEN_CLASS]
    gate(checks, "all_rows_are_H_in_the_corpus", not wrong,
         f"{len(wrong)} row(s) are not {SCREEN_CLASS} in the corpus"
         + (f", first {wrong[0]}" if wrong else ""))

    hits = {row["intervention_hit_count"] for row in selected}
    gate(checks, "three_patch_hits_per_row", hits == {"3"},
         f"intervention_hit_count values {sorted(hits)}; the condition patches "
         f"three layers and every selected row must record three hits")

    sources = {row["final_safety_label_source"] for row in selected}
    gate(checks, "safety_labels_are_adjudicated", sources == {"human"},
         f"final_safety_label_source values {sorted(sources)}")

    before = {row["before_axis2"] for row in selected}
    gate(checks, "every_row_refused_before", before <= {"DR", "IR"},
         f"before_axis2 values {sorted(before)}; a regression is a row that "
         f"refused before the intervention")

    return [{"row_id": row["row_id"],
             "row_class": row["frozen_class"],
             "source_family": row["source"],
             "before_axis2": row["before_axis2"],
             "final_actionable_harmful_help": row["final_actionable_harmful_help"],
             "final_safety_label_source": row["final_safety_label_source"],
             "intervention_hit_count": int(row["intervention_hit_count"])}
            for row in sorted(selected, key=lambda r: r["row_id"])]


#: Keys whose values are this manifest's own explanation of itself. They are
#: allowed to be long; everything else in the file is an identifier, a digest,
#: a count or a label, and a long string under any other key is text that does
#: not belong in a committed artifact.
PROSE_KEYS: tuple = (
    "statement", "note", "purpose", "description", "how_established",
    "slot_semantics", "application_scaling",
)
PROSE_SUFFIXES: tuple = ("_note", "_caveat")


def _is_prose_key(key: str) -> bool:
    return key in PROSE_KEYS or key.endswith(PROSE_SUFFIXES)


def assert_no_text(payload: Any, label: str = "manifest") -> None:
    """No prompt or response text anywhere in what is about to be written.

    Two rules, because one is not enough. A value under a column name that
    carries text is refused however short it is; a long value under any key
    that is not this manifest's own prose is refused however it is named. The
    prose exemption is by key, not by length, so it cannot be used to smuggle
    a response in under a plausible name.
    """
    def walk(node: Any, trail: str, key: Optional[str] = None) -> None:
        if isinstance(node, dict):
            for child_key, value in node.items():
                if child_key in TEXT_COLUMNS and isinstance(value, str) and value:
                    raise LegacyRegressionSelectionError(
                        f"{label}{trail}.{child_key} would put prompt or "
                        f"response text into a committed manifest")
                walk(value, f"{trail}.{child_key}", child_key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]", key)
        elif isinstance(node, str) and len(node) > 400:
            if key is not None and _is_prose_key(key):
                return
            raise LegacyRegressionSelectionError(
                f"{label}{trail} is a {len(node)}-character string under a "
                f"non-prose key; that is text, not an identifier or a digest")
    walk(payload, "")


def build(static_master: Path, corpus: Path, static_run_config: Path,
          output: Path, write: bool = True) -> Dict[str, Any]:
    checks = Checklist("targeted_h_legacy_regressions_v1")

    master_sha = verify_input(static_master, STATIC_MASTER_LOGICAL, checks)
    corpus_sha = verify_input(corpus, CORPUS_LOGICAL, checks)
    if not Path(static_run_config).is_file():
        raise LegacyRegressionSelectionError(
            f"{static_run_config} is not a file. The static condition is read "
            f"from the run configuration that produced the regressions; "
            f"without it the condition cannot be established and no selection "
            f"is written. It must be the migrated copy under the data root at "
            f"{STATIC_RUN_CONFIG_RELATIVE}.")
    assert_not_sealed(static_run_config, "legacy H regression run config")
    assert_not_historical_workspace(static_run_config, "--static-run-config")
    run_config_sha = sha256_file(static_run_config)
    gate(checks, "static_run_config_sha256",
         run_config_sha == STATIC_RUN_CONFIG_SHA256,
         f"{Path(static_run_config).name} is {run_config_sha}, pinned "
         f"{STATIC_RUN_CONFIG_SHA256}")
    run_config = json.loads(Path(static_run_config).read_text("utf-8"))
    condition = verify_condition(run_config, checks)

    master_rows = read_master_rows(Path(static_master))
    with open(corpus, newline="", encoding="utf-8") as handle:
        corpus_class = {row["row_id"]: row["final_ebh"]
                        for row in csv.DictReader(handle)}
    rows = select(master_rows, corpus_class, checks)

    # The sealed split stores no row ids anywhere in this repository, so
    # non-overlap is established structurally rather than by comparison: every
    # selected row is a row of the accepted development corpus, and the sealed
    # split is a separate held-out set this repository cannot name or reach.
    sealed = json.loads(
        (REPO_ROOT / "manifests" / "sealed_set_commitment.json").read_text("utf-8"))
    gate(checks, "sealed_set_unreachable",
         sealed["availability"][
             "sealed_prompt_text_available_to_development_code"] is False,
         "every selected row is a development-corpus row; the sealed split is "
         "a separate held-out set and this repository stores no sealed row id, "
         "prompt, hash or path")

    manifest = {
        "schema": SCHEMA,
        "name": "targeted_h_legacy_regressions_v1",
        "recorded_utc": utc_now(),
        "targeted_diagnostic": True,
        "population_reportable": False,
        "legacy_stress_set": True,
        "current_validated_repair_direction_was_not_the_selection_treatment": True,
        "source_actuator": {
            "variant": EXPECTED_VARIANT,
            "role": "primary",
            "donors": list(SOURCE_DONORS),
            "strength": 0.15,
            "historical_layers": [31, 32, 33],
            "routegate_states": [32, 33, 34],
            "token_position": "final prompt token",
            "prefill_only": True,
            "note": "the treatment that produced these twenty regressions. "
                    "Built from donors d2 and d3, not from d1.",
        },
        "population_reportable_note":
            "these twenty rows were selected because they regressed. Anything "
            "measured on them is conditioned on that selection and is a "
            "diagnostic, never a population rate. The population claim comes "
            "from the full 213-row H arm of the causal screen.",
        "contains_prompt_or_response_text": False,
        "selection_rule": {
            "statement": "rows of the frozen static-steering master with "
                         "frozen_class == 'H' and "
                         "final_actionable_harmful_help == 'YES'",
            "field_lookup_only": True,
            "no_keyword_matching": True,
            "no_relabelling": True,
            "expected_rows": EXPECTED_ROWS,
            "corroborating_checks": [
                "every row is H in the accepted development corpus",
                "every row records three patch hits, one per intervention layer",
                "every row's safety label was human-adjudicated",
                "every row refused before the intervention (before_axis2 in DR, IR)",
            ],
        },
        "static_condition": condition,
        "does_not_claim": {
            "b_d1_minus_e_caused_these_regressions": False,
            "these_are_failures_of_the_validated_direction_bank": False,
            "testing_h_minus_e_here_proves_rescue_of_current_b_d1_failures": False,
            "statement": "the treatment that produced these rows was "
                         "consensus23_global_frozen, built from donors d2 and "
                         "d3. It was not historical_d1_global_frozen "
                         "(B_d1 - E), the direction the validated bank "
                         "reproduces; that bank did not exist when these rows "
                         "were produced and was not the treatment. A rescue "
                         "measured here is evidence about legacy regressions "
                         "under a legacy direction, and is not evidence that "
                         "the current validated repair direction fails on "
                         "these rows or that H - E repairs failures it "
                         "causes.",
        },
        "what_it_is_for": {
            "role": "legacy stress set",
            "statement": "twenty rows already known to be steerable into "
                         "harmful compliance by a B-minus-E direction at this "
                         "strength and depth. That makes them a demanding "
                         "place to ask whether H - E restores refusal. It is a "
                         "diagnostic, and a restricted one.",
            "current_validated_repair_direction": CURRENT_VALIDATED_REPAIR_VARIANT,
            "relation_to_the_treatment": "different donors, different vector; "
                                         "close but not the same direction",
        },
        "sources": {
            "static_steering_master": {
                "path": "frozen/static_steering/"
                        + Path(static_master).name,
                "root": "routegate-data",
                "sha256": master_sha,
                "logical_name": STATIC_MASTER_LOGICAL,
                "recorded_in": "manifests/frozen_inputs.json",
            },
            "development_corpus": {
                "path": "frozen/corpus/" + Path(corpus).name,
                "root": "routegate-data",
                "sha256": corpus_sha,
                "logical_name": CORPUS_LOGICAL,
                "recorded_in": "manifests/frozen_inputs.json",
            },
            "static_run_config": {
                "path": STATIC_RUN_CONFIG_RELATIVE,
                "root": "routegate-data",
                "sha256": run_config_sha,
                "pinned_sha256": STATIC_RUN_CONFIG_SHA256,
                "migrated_to_routegate_data": True,
                "note": "the evidence of the static condition, migrated with "
                        "its digest preserved. This selection is rebuildable "
                        "from RouteGate and routegate-data alone; no runtime "
                        "path reaches the historical workspace.",
            },
        },
        "judgement_provenance": {
            "read_at_runtime": False,
            "note": "the safety labels are carried by the static-steering "
                    "master, which the master's own validation report traces "
                    "to these two files. Their digests are reproduced here as "
                    "a provenance record only. Neither is opened by this "
                    "builder and neither is a runtime dependency.",
            "h_before_after_audit": {
                "path": "results/rq3_bh_rq2_static_20260810/"
                        "h_before_after_audit.csv",
                "root": "SteerMoE-main",
                "sha256": "90831f165a283de436658afd6ddc13a71c9444aafb7f07e6b0"
                          "bc83b40d5a7fb8",
            },
            "final_h_safety": {
                "path": "results/rq3_bh_rq2_static_20260810/h_judging/"
                        "safety_human_adjudication/final_h_safety.csv",
                "root": "SteerMoE-main",
                "sha256": "51bf090088009a4f349299f2bc6ac6de002b050db488240eb70"
                          "77cbf8a2684ad",
            },
        },
        "sealed_set": {
            "overlap": False,
            "how_established": "structurally: every selected row is a row of "
                               "the 480-row development corpus. This "
                               "repository stores no sealed row id, prompt, "
                               "hash or path, so no id comparison is possible "
                               "and none is claimed.",
            "commitment": "manifests/sealed_set_commitment.json",
        },
        "row_count": len(rows),
        "row_ids": [row["row_id"] for row in rows],
        "rows": rows,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text(manifest)

    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    print(f"[targeted] {len(rows)} row(s), all {SCREEN_CLASS}, "
          f"human-adjudicated")
    print(f"[targeted] condition {condition['direction_variant']} "
          f"alpha_R={condition['strength_alpha_repair']} "
          f"alpha_S={condition['alpha_safety']} "
          f"layers {condition['historical_layers']}")
    print(f"[targeted] legacy_stress_set=True targeted_diagnostic=True "
          f"population_reportable=False")
    print(f"[targeted] treatment was {condition['direction_variant']} "
          f"(donors {'/'.join(SOURCE_DONORS)}), NOT "
          f"{CURRENT_VALIDATED_REPAIR_VARIANT}")
    if write:
        Path(output).write_text(text, encoding="utf-8")
        print(f"[targeted] wrote {output}")
    else:
        print("[targeted] dry run: nothing written")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-master", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--static-run-config", type=Path, required=True,
                        help="the run configuration of the static transfer "
                             "that produced these regressions")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        build(args.static_master, args.corpus, args.static_run_config,
              args.output, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
