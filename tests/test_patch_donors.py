"""Donor pairing: determinism, fallbacks, hashes, and what it may not see.

Every test here runs on a synthetic corpus in a temporary directory. Nothing
reads the real dataset, loads a model, or touches a GPU -- the point is the
rule, not the data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from routegate.data import SOURCE_COLUMN, TASK_FRAME_COLUMN, TOPIC_COLUMN
from routegate.donors import (
    EXACT_TIER,
    MATCH_TIERS,
    PAIRING_COLUMNS,
    DonorError,
    assert_no_withheld_rows,
    build_manifest,
    build_pairs,
    canonical_json,
    load_manifest,
    load_pairing_rows,
    manifest_sha256,
    match_donor,
    pairs_from_manifest,
    prompt_chars,
    split_targets_and_donors,
    write_manifest,
)
from routegate.validation import SealedSetError, StaleInputError, ValidationError

COLUMNS = ["row_id", "item_id", "final_ebh", "prompt_text", "response_text",
           SOURCE_COLUMN, TOPIC_COLUMN, TASK_FRAME_COLUMN]


def row(row_id, label, source, topic, frame, prompt, response="an answer"):
    return {"row_id": row_id, "item_id": row_id, "final_ebh": label,
            "prompt_text": prompt, "response_text": response,
            SOURCE_COLUMN: source, TOPIC_COLUMN: topic,
            TASK_FRAME_COLUMN: frame}


def write_dataset(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for record in rows:
            writer.writerow(record)
    return path


@pytest.fixture
def corpus(tmp_path):
    rows = [
        row("e:001", "E", "orbench_hard", "CYBER", "INFORM_EXPLAIN", "x" * 100),
        row("e:002", "E", "false_reject_test", "SELF_HARM", "ADVISE_RECOMMEND",
            "y" * 40),
        row("e:003", "E", "orbench_hard", "DRUGS", "CODE", "z" * 60),
        # donors: two exact matches for e:001 at different lengths
        row("b:001", "B", "orbench_hard", "CYBER", "INFORM_EXPLAIN", "a" * 130),
        row("b:002", "B", "orbench_hard", "CYBER", "INFORM_EXPLAIN", "b" * 104),
        # topic+task match for e:002 but from the other source
        row("b:003", "B", "orbench_hard", "SELF_HARM", "ADVISE_RECOMMEND",
            "c" * 45),
        # topic-only match for e:003
        row("b:004", "B", "false_reject_test", "DRUGS", "INFORM_EXPLAIN",
            "d" * 55),
        # H is present in the corpus and must never be paired
        row("h:001", "H", "orbench_hard", "CYBER", "INFORM_EXPLAIN", "h" * 100),
    ]
    return write_dataset(tmp_path / "rq3_dataset_final.csv", rows)


# ----------------------------------------------------- 1. what pairing sees
def test_pairing_reads_only_the_permitted_columns(corpus):
    rows, _ = load_pairing_rows(corpus)
    assert set(rows[0]) == set(PAIRING_COLUMNS)
    assert "response_text" not in rows[0]
    assert "item_id" not in rows[0]


def test_the_permitted_columns_are_exactly_the_four_allowed_signals():
    assert set(PAIRING_COLUMNS) == {
        "row_id", "final_ebh", "prompt_text",
        SOURCE_COLUMN, TOPIC_COLUMN, TASK_FRAME_COLUMN}


def test_the_withheld_class_never_becomes_a_target_or_a_donor(corpus):
    rows, _ = load_pairing_rows(corpus)
    targets, donors = split_targets_and_donors(rows)
    labels = {r["final_ebh"] for r in rows}
    assert labels == {"E", "B"}
    assert all(t["final_ebh"] == "E" for t in targets)
    assert all(d["final_ebh"] == "B" for d in donors)
    assert "h:001" not in {r["row_id"] for r in rows}


def test_a_withheld_row_reaching_pairing_is_refused():
    with pytest.raises(DonorError, match="withheld"):
        assert_no_withheld_rows([
            {"row_id": "h:001", "final_ebh": "H"},
            {"row_id": "e:001", "final_ebh": "E"},
        ])
    assert_no_withheld_rows([{"row_id": "e:001", "final_ebh": "E"}])


def test_a_row_with_a_missing_covariate_is_refused(tmp_path):
    path = write_dataset(tmp_path / "rq3_dataset_final.csv", [
        row("e:001", "E", "orbench_hard", "", "INFORM_EXPLAIN", "x" * 10),
        row("b:001", "B", "orbench_hard", "CYBER", "INFORM_EXPLAIN", "y" * 10),
    ])
    with pytest.raises(DonorError, match="may not invent"):
        load_pairing_rows(path)


# ------------------------------------------------- 2. sealed and stale input
def test_a_sealed_looking_path_is_refused(tmp_path):
    path = tmp_path / "sealed_benchmark_prompts.csv"
    path.write_text("row_id\n", encoding="utf-8")
    with pytest.raises(SealedSetError):
        load_pairing_rows(path)


def test_a_superseded_dataset_revision_is_refused(tmp_path):
    path = tmp_path / "rq3_ebh_final_analysis_dataset.csv"
    path.write_text("row_id\n", encoding="utf-8")
    with pytest.raises(StaleInputError):
        load_pairing_rows(path)


# ------------------------------------------------------- 3. dataset hashing
def test_the_dataset_hash_is_verified(corpus):
    import hashlib

    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    rows, recorded = load_pairing_rows(corpus, digest)
    assert recorded == digest
    assert rows

    with pytest.raises(ValidationError, match="hash mismatch"):
        load_pairing_rows(corpus, "0" * 64)


# -------------------------------------------------- 4. deterministic matching
def test_matching_is_deterministic_over_repeated_runs(corpus):
    rows, _ = load_pairing_rows(corpus)
    targets, donors = split_targets_and_donors(rows)
    first = [(p.target_row_id, p.donor_row_id, p.tier)
             for p in build_pairs(targets, donors)]
    second = [(p.target_row_id, p.donor_row_id, p.tier)
              for p in build_pairs(list(reversed(targets)),
                                   list(reversed(donors)))]
    assert first == second


def test_the_nearest_length_wins_inside_a_tier(corpus):
    rows, _ = load_pairing_rows(corpus)
    targets, donors = split_targets_and_donors(rows)
    target = next(t for t in targets if t["row_id"] == "e:001")

    match = match_donor(target, donors)
    assert match.donor_row_id == "b:002"          # 104 chars against a 100-char target
    assert match.tier == EXACT_TIER
    assert match.exact_match is True
    assert match.length_delta_chars == 4
    assert match.candidates_in_tier == 2


def test_a_length_tie_is_broken_by_row_id():
    target = row("e:001", "E", "s", "t", "f", "x" * 10)
    donors = [row("b:002", "B", "s", "t", "f", "y" * 20),
              row("b:001", "B", "s", "t", "f", "z" * 20)]
    assert match_donor(target, donors).donor_row_id == "b:001"
    assert match_donor(target, list(reversed(donors))).donor_row_id == "b:001"


def test_a_donor_is_never_the_target_itself():
    target = row("x:001", "E", "s", "t", "f", "p" * 10)
    donors = [dict(target), row("b:001", "B", "s", "t", "f", "q" * 400)]
    assert match_donor(target, donors).donor_row_id == "b:001"


def test_length_is_measured_in_characters():
    assert prompt_chars({"prompt_text": "abc def"}) == 7


# ------------------------------------------------------ 5. fallback recording
def test_a_fallback_records_which_keys_were_relaxed(corpus):
    rows, _ = load_pairing_rows(corpus)
    targets, donors = split_targets_and_donors(rows)

    second = match_donor(next(t for t in targets if t["row_id"] == "e:002"),
                         donors)
    assert second.tier == "topic_task"
    assert second.exact_match is False
    assert second.relaxed_keys == (SOURCE_COLUMN,)
    assert second.matched_values == {TOPIC_COLUMN: "SELF_HARM",
                                     TASK_FRAME_COLUMN: "ADVISE_RECOMMEND"}

    third = match_donor(next(t for t in targets if t["row_id"] == "e:003"),
                        donors)
    assert third.tier == "topic"
    assert third.exact_match is False
    assert set(third.relaxed_keys) == {SOURCE_COLUMN, TASK_FRAME_COLUMN}


def test_the_tier_ladder_degrades_one_key_at_a_time():
    names = [name for name, _ in MATCH_TIERS]
    assert names == ["source_topic_task", "topic_task", "topic", "task_frame",
                     "length_only"]
    assert MATCH_TIERS[0][1] == (SOURCE_COLUMN, TOPIC_COLUMN, TASK_FRAME_COLUMN)
    assert MATCH_TIERS[-1][1] == ()


def test_the_last_tier_matches_on_length_alone():
    target = row("e:001", "E", "s1", "t1", "f1", "x" * 10)
    donors = [row("b:001", "B", "s2", "t2", "f2", "y" * 12)]
    match = match_donor(target, donors)
    assert match.tier == "length_only"
    assert match.exact_match is False
    assert match.matched_values == {}


def test_an_empty_donor_pool_is_refused():
    with pytest.raises(DonorError, match="donor pool was empty"):
        match_donor(row("e:001", "E", "s", "t", "f", "x"), [])


def test_a_corpus_without_donors_is_refused(tmp_path):
    path = write_dataset(tmp_path / "rq3_dataset_final.csv",
                         [row("e:001", "E", "s", "t", "f", "x")])
    rows, _ = load_pairing_rows(path)
    with pytest.raises(DonorError, match="no B donors"):
        split_targets_and_donors(rows)


# ---------------------------------------------------------- 6. the manifest
@pytest.fixture
def manifest(corpus):
    rows, _ = load_pairing_rows(corpus)
    targets, donors = split_targets_and_donors(rows)
    pairs = build_pairs(targets, donors)
    return build_manifest(pairs, corpus, "dataset-sha", "2026-08-06T00:00:00Z",
                          n_targets=len(targets), n_donors=len(donors))


def test_the_manifest_counts_exact_matches_and_fallbacks(manifest):
    counts = manifest["counts"]
    assert counts["pairs"] == 3
    assert counts["exact_matches"] == 1
    assert counts["fallback_matches"] == 2
    assert counts["by_tier"]["source_topic_task"] == 1
    assert counts["by_tier"]["topic_task"] == 1
    assert counts["by_tier"]["topic"] == 1


def test_the_manifest_declares_what_pairing_may_not_use(manifest):
    forbidden = manifest["rule"]["forbidden_inputs"]
    assert any("activation" in f for f in forbidden)
    assert any("response" in f for f in forbidden)
    assert any("patching" in f for f in forbidden)
    assert any("held-out" in f for f in forbidden)
    assert manifest["rule"]["permitted_fields"] == list(PAIRING_COLUMNS)


def test_the_manifest_hash_covers_the_pairs(manifest):
    digest = manifest["manifest_sha256"]
    assert digest == manifest_sha256(manifest)
    assert "manifest_sha256" not in canonical_json(manifest)

    tampered = json.loads(json.dumps(manifest))
    tampered["pairs"][0]["donor_row_id"] = "b:999"
    assert manifest_sha256(tampered) != digest


def test_the_hash_is_stable_across_key_order(manifest):
    reordered = {k: manifest[k] for k in reversed(list(manifest))}
    assert manifest_sha256(reordered) == manifest["manifest_sha256"]


def test_a_written_manifest_round_trips(tmp_path, manifest):
    path = tmp_path / "donor_manifest.json"
    digest = write_manifest(path, manifest)
    loaded = load_manifest(path, digest)
    assert loaded["manifest_sha256"] == digest
    assert len(pairs_from_manifest(loaded)) == 3


def test_a_manifest_edited_after_freezing_is_refused(tmp_path, manifest):
    path = tmp_path / "donor_manifest.json"
    write_manifest(path, manifest)
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["pairs"][0]["donor_row_id"] = "b:999"
    path.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(DonorError, match="modified since it was frozen"):
        load_manifest(path)


def test_a_manifest_that_is_not_the_expected_one_is_refused(tmp_path, manifest):
    path = tmp_path / "donor_manifest.json"
    write_manifest(path, manifest)
    with pytest.raises(DonorError, match="not the manifest this run"):
        load_manifest(path, "0" * 64)


def test_a_missing_manifest_says_it_must_be_built_first(tmp_path):
    with pytest.raises(DonorError, match="frozen before the model is loaded"):
        load_manifest(tmp_path / "absent.json")


def test_a_manifest_of_the_wrong_schema_is_refused(tmp_path, manifest):
    path = tmp_path / "donor_manifest.json"
    manifest["schema"] = "something/else/1"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DonorError, match="schema"):
        load_manifest(path)


def test_pairs_are_keyed_by_target_and_never_duplicated(manifest):
    pairs = pairs_from_manifest(manifest)
    assert sorted(pairs) == ["e:001", "e:002", "e:003"]

    doubled = json.loads(json.dumps(manifest))
    doubled["pairs"].append(dict(doubled["pairs"][0]))
    with pytest.raises(DonorError, match="more than once"):
        pairs_from_manifest(doubled)
