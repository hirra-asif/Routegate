"""The frozen final evaluation: three conditions, nine arms, nothing else.

Two hybrids were composed offline and rejected under a fixed eligibility rule
before this protocol was frozen. The strongest test here is a negative one:
neither name may appear as a condition, in an arm, or anywhere in the runner.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from routegate.finaleval import (
    ARMS,
    CAPABILITY_ROWS,
    CONDITIONS,
    COND_BASELINE,
    COND_DYNAMIC,
    COND_ROUTEGATE,
    DATASETS,
    DATASET_SPEC,
    DIGEST_FIELD,
    DS_GSM8K,
    DS_MMLU,
    DS_SEALED,
    GENERATIONS_PER_CONDITION,
    NAME,
    PREDECLARATION,
    RECORD_FIELDS,
    REJECTED_CANDIDATES,
    SCHEMA,
    SEALED_ROWS,
    TOTAL_GENERATIONS,
    arm_id,
    arm_output_dir,
    build_protocol,
    expected_rows,
    record_sha256,
    verify_protocol,
)
from routegate.validation import Checklist, looks_sealed

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/final_evaluation/run_final_evaluation.py"


# ------------------------------------------------- the rejected hybrids
@pytest.mark.parametrize("name", ["dynamic_safety_constrained",
                                  "dynamic_fullgate"])
def test_a_rejected_hybrid_is_never_a_condition(name):
    assert name not in CONDITIONS
    assert name in REJECTED_CANDIDATES
    assert not any(name == c for _d, c in ARMS)


@pytest.mark.parametrize("name", ["dynamic_safety_constrained",
                                  "dynamic_fullgate"])
def test_a_rejected_hybrid_appears_in_no_arm_id(name):
    assert not any(name in arm_id(d, c) for d, c in ARMS)
    protocol = build_protocol()
    assert not any(name in a["arm_id"] for a in protocol["arms"])
    assert name in protocol["rejected_candidates"]


def test_the_runner_offers_only_the_nine_frozen_arms():
    source = RUNNER.read_text(encoding="utf-8")
    assert "dynamic_safety_constrained" not in source
    assert "dynamic_fullgate" not in source


def test_v5_supersedes_v4_and_retains_the_v1_invalidation_record():
    from routegate.finaleval import (INVALIDATED_PROTOCOL,
                                     INVALIDATION_REASON, SUPERSEDES)
    protocol = build_protocol()
    assert SUPERSEDES == "routegate_final_evaluation_v4"
    assert INVALIDATED_PROTOCOL == "routegate_final_evaluation_v1"
    assert protocol["v1_generated_rows"] == 0
    assert protocol["v2_generated_rows"] == 0
    assert len(INVALIDATION_REASON) >= 4
    joined = " ".join(INVALIDATION_REASON)
    for defect in ("discarded", "mode='off'", "parser", "chat"):
        assert defect in joined


def test_the_v1_manifest_is_preserved_on_disk_unedited():
    v1 = REPO / "manifests" / "routegate_final_evaluation_v1.json"
    assert v1.is_file(), "v1 is retained as the record of what was invalid"
    blob = json.loads(v1.read_text())
    assert blob["name"] == "routegate_final_evaluation_v1"
    assert blob["output"]["data_root"] == "final_evaluation_v1"


def test_neither_v1_nor_v2_output_root_was_ever_created():
    for name in ("final_evaluation_v1", "final_evaluation_v2"):
        assert not (Path.home() / "routegate-data" / name).exists(), \
            f"{name} generated zero rows"


# ------------------------------------------------------ conditions and arms
def test_exactly_three_conditions():
    assert CONDITIONS == (COND_BASELINE, COND_DYNAMIC, COND_ROUTEGATE)
    assert len(CONDITIONS) == 3


def test_exactly_nine_arms_three_datasets_by_three_conditions():
    assert len(ARMS) == 9
    assert {d for d, _ in ARMS} == set(DATASETS) == {DS_SEALED, DS_MMLU,
                                                     DS_GSM8K}
    assert {c for _, c in ARMS} == set(CONDITIONS)
    assert len({arm_id(d, c) for d, c in ARMS}) == 9


def test_capability_arms_are_ordered_before_the_sealed_arms():
    order = [d for d, _ in ARMS]
    assert order.index(DS_SEALED) > max(order.index(DS_MMLU),
                                        order.index(DS_GSM8K))
    assert order[-3:] == [DS_SEALED] * 3


def test_each_arm_has_its_own_output_directory():
    dirs = [arm_output_dir(d, c) for d, c in ARMS]
    assert len(set(dirs)) == 9
    assert all(p.startswith("final_evaluation_v5/") for p in dirs)


# ------------------------------------------------------------------ counts
def test_the_generation_counts_are_exactly_the_approved_ones():
    assert SEALED_ROWS == 380 and CAPABILITY_ROWS == 100
    assert SEALED_ROWS * 3 == 1140
    assert CAPABILITY_ROWS * 3 == 300
    assert GENERATIONS_PER_CONDITION == 580
    assert TOTAL_GENERATIONS == 1740
    g = build_protocol()["generations"]
    assert g == {"sealed": 1140, "tinymmlu": 300, "tinygsm8k": 300,
                 "per_condition": 580, "total": 1740,
                 "inherited_or_reused": 0}


def test_the_arm_rows_sum_to_the_total():
    assert sum(expected_rows(d) for d, _ in ARMS) == TOTAL_GENERATIONS


# ---------------------------------------------------------------- datasets
@pytest.mark.parametrize("dataset,repo,revision,digest", [
    (DS_MMLU, "tinyBenchmarks/tinyMMLU",
     "cd6a8ee9e1e6f3f18f683535f6c096c386a85da3",
     "256f81ffdd6e64699df096c02afafba33626bfca08baed484c275e351993f317"),
    (DS_GSM8K, "tinyBenchmarks/tinyGSM8k",
     "b870449ed96ee3e308748b550908332450ec4098",
     "07c2857ce7f4f4cf123066fede0f52f2aab7458c0c9bae3293c3f656a9eac53c"),
])
def test_each_capability_dataset_is_pinned_at_exactly_100_items(
        dataset, repo, revision, digest):
    spec = DATASET_SPEC[dataset]
    assert spec["rows"] == 100
    assert spec["repo"] == repo
    assert spec["revision"] == revision
    assert spec["sha256"] == digest
    assert len(revision) == 40 and len(digest) == 64


def test_the_pinned_parquet_files_are_on_disk_with_those_digests():
    from routegate.validation import sha256_file
    for dataset in (DS_MMLU, DS_GSM8K):
        spec = DATASET_SPEC[dataset]
        local = Path.home() / spec["local"]
        if not local.is_file():
            pytest.skip(f"{local} is not present on this machine")
        assert sha256_file(local) == spec["sha256"]


def test_no_excluded_benchmark_is_referenced():
    protocol = build_protocol()
    excluded = protocol["excluded"]["benchmarks"]
    assert "HellaSwag" in excluded and "ARC-Challenge" in excluded
    for name in ("hellaswag", "arc_challenge", "truthfulqa"):
        assert name not in DATASETS
        assert not any(name in a["arm_id"] for a in protocol["arms"])


# ------------------------------------------------------------ sealed guards
def test_the_protocol_stores_no_sealed_path():
    protocol = build_protocol()
    assert protocol["sealed_set"]["path_stored_in_this_repository"] is False
    assert protocol["sealed_set"][
        "prompt_content_read_by_this_module"] is False
    assert protocol["contains_sealed_material"] is False
    blob = json.dumps(protocol)
    assert "sealed_final" not in blob and "prompt_splits" not in blob


def test_the_sealed_split_is_the_frozen_380():
    spec = DATASET_SPEC[DS_SEALED]
    assert spec["rows"] == 380
    assert spec["expected_safe_source_count"] == 180
    assert spec["expected_harmful_source_count"] == 200
    commitment = json.loads(
        (REPO / "manifests" / "sealed_set_commitment.json").read_text())
    assert commitment["expected_row_count"] == 380


def test_no_dataset_local_path_reads_like_held_out_material():
    for dataset in (DS_MMLU, DS_GSM8K):
        assert not looks_sealed(DATASET_SPEC[dataset]["local"])


def test_the_protocol_was_frozen_before_sealed_access_and_generation():
    protocol = build_protocol()
    assert protocol["frozen_before_any_sealed_prompt_was_read"] is True
    assert protocol["frozen_before_any_generation"] is True
    assert protocol["judging"]["performed_by_this_stage"] is False


# ------------------------------------------------------------ the committed
def test_the_committed_protocol_verifies():
    checks = Checklist("test")
    verify_protocol(PREDECLARATION, checks)
    assert checks.all_passed


def test_the_committed_protocol_is_deterministic_and_self_describing():
    a, b = build_protocol(), build_protocol()
    assert a == b
    assert a[DIGEST_FIELD] == record_sha256(a)
    assert a["schema"] == SCHEMA and a["name"] == NAME


def test_an_edited_count_is_caught_by_verification(tmp_path):
    committed = json.loads(PREDECLARATION.read_text(encoding="utf-8"))
    committed["generations"]["total"] = 9999
    committed[DIGEST_FIELD] = record_sha256(committed)
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(committed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(Exception):
        verify_protocol(path, Checklist("test"))


def test_the_record_schema_carries_what_a_reader_needs():
    for field in ("arm_id", "dataset", "condition", "item_id",
                  "prompt_sha256", "gate_fires", "g_R", "applied_alpha_R",
                  "hooks_fired", "response_text", "model", "revision",
                  "decoding", "max_new_tokens"):
        assert field in RECORD_FIELDS


# ------------------------------------------------------------- the runner
def test_the_runner_exposes_progress_resume_and_dry_run():
    source = RUNNER.read_text(encoding="utf-8")
    for flag in ("--dry-run", "--progress", "--sealed-manifest", "--arm"):
        assert flag in source
    assert "completed_items" in source
    assert "assert_capability_complete" in source


def test_the_runner_rejects_an_unknown_arm():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.ARM_IDS) == 9
    with pytest.raises(Exception):
        module.split_arm("sealed__dynamic_fullgate")
    for name in module.ARM_IDS:
        dataset, condition = module.split_arm(name)
        assert condition in CONDITIONS
