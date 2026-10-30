"""The two-site continuation: its pins, its partition, and every gate.

Nothing here loads the frozen model, generates a token, judges a response or
reads held-out material. Tests that need the private artifacts skip when they
are unreachable; everything else runs on synthetic records.

What is under test is what a restart gets wrong quietly: a shard that overlaps
another, a record inherited from a crash without being checked, a baseline
described as generated when it was referenced, and a "completed" arm that is
short by one row.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import twositecontinuation as C                      # noqa: E402
from routegate.twosite import (                                     # noqa: E402
    CANDIDATE_ARM_ID,
    CANDIDATE_BLOCKS,
    CANDIDATE_STATES,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    SealedSetError,
    ValidationError,
)

RUNNER = (REPO_ROOT / "experiments" / "single_direction_controller"
          / "run_two_site_continuation.py")
CONSOLIDATOR = (REPO_ROOT / "experiments" / "single_direction_controller"
                / "consolidate_two_site.py")
OLD_RUNNER = (REPO_ROOT / "experiments" / "single_direction_controller"
              / "run_two_site_validation.py")


def _artifacts_reachable() -> bool:
    try:
        pinned = C.pinned_inputs()
    except Exception:
        return False
    return all(Path(e["path"]).is_file() for e in pinned.values())


needs_artifacts = pytest.mark.skipif(
    not _artifacts_reachable(),
    reason="the pinned private artifacts are not reachable on this machine")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LOCKED = {
    "protocol": C.PROTOCOL, "arm_id": CANDIDATE_ARM_ID,
    "alpha_R": 0.15, "alpha_S": 0.0,
    "intervention_type": "validated_two_site_repair",
    "budget_reference": "block_input",
    "states": list(CANDIDATE_STATES),
    "historical_layers": list(CANDIDATE_BLOCKS),
    "model": "Qwen/Qwen3-30B-A3B", "revision": "rev",
    "generation_config_sha256": "g" * 64,
    "tokenizer_config_sha256": "t" * 64,
    "chat_template_sha256": "c" * 64,
    "direction_bank_sha256": "d" * 64,
    "dataset_sha256": "s" * 64, "smoke": False,
}
CORPUS = {f"row:{i:03d}": {"row_id": f"row:{i:03d}", "final_ebh": "E",
                           "prompt_sha256": f"{i:064d}",
                           "prompt_text": f"prompt {i}"}
          for i in range(10)}


def _record(row_id, **override):
    record = {
        **LOCKED, "row_id": row_id, "row_class": "E",
        "prompt_sha256": CORPUS[row_id]["prompt_sha256"],
        "status": "ok", "error_message": "",
        "hooks_expected": 2, "hooks_fired": 2,
        "per_state": {str(s): {"repair_direction_sha256": "a" * 64}
                      for s in CANDIDATE_STATES},
    }
    record.update(override)
    return record


def _write(path, records):
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                            for r in records), encoding="utf-8")
    return path


# ================================================== 1. the audit's two defects
def test_the_old_runner_declared_shard_arguments_it_never_used():
    """The defect this continuation exists to correct, pinned as history."""
    import ast
    tree = ast.parse(OLD_RUNNER.read_text(encoding="utf-8"))
    source = ast.unparse(tree)
    assert "--shard-index" in source and "--shard-count" in source
    assert "args.shard_index" not in source
    assert "args.shard_count" not in source


def test_the_old_runner_checked_its_output_directory_before_resume():
    source = OLD_RUNNER.read_text(encoding="utf-8")
    first_claim = source.index("claim_output_dir(args.output_dir")
    resume = source.index("if args.resume else set()")
    assert first_claim < resume, (
        "the old runner claimed its output directory before reading --resume, "
        "which is why --resume was unreachable")


def test_the_continuation_reads_resume_before_claiming_the_directory():
    source = RUNNER.read_text(encoding="utf-8")
    assert "prepare_output_dir(args.output_dir, records_name," in source
    prepare = source.index("output_dir = prepare_output_dir(")
    validate = source.index("validate_existing_shard(")
    assert prepare < validate
    assert "bool(args.resume)" in source


def test_the_continuation_actually_uses_its_shard_arguments():
    source = RUNNER.read_text(encoding="utf-8")
    assert "shard_rows(ordered, int(args.shard_index), int(args.shard_count))" \
        in source
    assert "assert_shards_partition(ordered, int(args.shard_count), checks)" \
        in source


# ============================================== 2. the partition is genuine
@pytest.mark.parametrize("count", [1, 2, 3, 5, 7, 8, 16, 480])
def test_shards_partition_the_order_with_no_overlap_and_full_coverage(count):
    ordered = [f"row:{i:03d}" for i in range(480)]
    shards = [C.shard_rows(ordered, i, count) for i in range(1, count + 1)]
    flat = [r for s in shards for r in s]
    assert sorted(flat) == sorted(ordered)          # full coverage
    assert len(flat) == len(set(flat)) == 480       # no overlap
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1
    C.assert_shards_partition(ordered, count, Checklist("t"))


def test_a_shard_index_outside_its_count_is_refused():
    ordered = [f"row:{i}" for i in range(10)]
    for index, count in ((0, 4), (5, 4), (-1, 4)):
        with pytest.raises(C.ContinuationError, match="shard-index"):
            C.shard_rows(ordered, index, count)
    with pytest.raises(C.ContinuationError, match="shard-count"):
        C.shard_rows(ordered, 1, 0)


def test_the_row_order_is_deterministic_and_refuses_duplicates():
    rows = [{"row_id": "b"}, {"row_id": "a"}, {"row_id": "c"}]
    assert C.deterministic_row_order(rows) == ["a", "b", "c"]
    assert C.deterministic_row_order(rows) == C.deterministic_row_order(
        list(reversed(rows)))
    with pytest.raises(C.ContinuationError, match="duplicated"):
        C.deterministic_row_order([{"row_id": "a"}, {"row_id": "a"}])


# ================================================= 3. the counts are declared
def test_the_counts_are_477_3_480_and_0():
    counts = C.continuation_counts()
    assert counts == {"generated_candidate_records": 477,
                      "reused_candidate_records": 3,
                      "referenced_baselines": 480,
                      "generated_baselines": 0}
    assert C.EXPECTED_NEW_CANDIDATES + C.EXPECTED_REUSABLE_CANDIDATES == 480


def test_the_manifests_state_the_counts():
    for path in (RUNNER, CONSOLIDATOR):
        source = path.read_text(encoding="utf-8")
        assert "**counts," in source, path.name
        assert "continuation_counts" in source, path.name


# ======================================== 4. record validation, field by field
def test_a_fully_valid_record_is_accepted():
    C.validate_candidate_record(_record("row:000"), LOCKED, CORPUS, "t")


@pytest.mark.parametrize("field,bad", [
    ("alpha_R", 0.2), ("alpha_S", 0.05), ("direction_bank_sha256", "x" * 64),
    ("dataset_sha256", "x" * 64), ("model", "other"), ("revision", "moved"),
    ("generation_config_sha256", "x" * 64), ("smoke", True),
    ("budget_reference", "block_output"),
])
def test_a_record_with_any_moved_locked_field_is_refused(field, bad):
    with pytest.raises(C.ContinuationError, match=field):
        C.validate_candidate_record(_record("row:000", **{field: bad}),
                                    LOCKED, CORPUS, "t")


def test_a_record_at_the_wrong_sites_is_refused():
    with pytest.raises(C.ContinuationError):
        C.validate_candidate_record(_record("row:000", states=[32, 33, 34]),
                                    LOCKED, CORPUS, "t")


def test_a_record_touching_block_33_is_refused_by_name():
    record = _record("row:000")
    record["states"] = list(CANDIDATE_STATES)
    record["historical_layers"] = [31, 32, 33]
    with pytest.raises(C.ContinuationError, match="historical_layers"):
        C.validate_candidate_record(record, LOCKED, CORPUS, "t")


def test_an_error_record_is_never_inherited():
    with pytest.raises(C.ContinuationError, match="status"):
        C.validate_candidate_record(_record("row:000", status="error"),
                                    LOCKED, CORPUS, "t")
    with pytest.raises(C.ContinuationError, match="error_message"):
        C.validate_candidate_record(
            _record("row:000", error_message="boom"), LOCKED, CORPUS, "t")


def test_a_record_whose_edit_did_not_fire_is_refused():
    with pytest.raises(C.ContinuationError, match="hook"):
        C.validate_candidate_record(_record("row:000", hooks_fired=1),
                                    LOCKED, CORPUS, "t")
    with pytest.raises(C.ContinuationError, match="hook"):
        C.validate_candidate_record(_record("row:000", hooks_expected=3,
                                            hooks_fired=3),
                                    LOCKED, CORPUS, "t")


def test_a_record_with_a_mismatched_prompt_digest_is_refused():
    with pytest.raises(C.ContinuationError, match="prompt digest"):
        C.validate_candidate_record(_record("row:000", prompt_sha256="9" * 64),
                                    LOCKED, CORPUS, "t")


def test_a_record_for_an_unknown_row_is_refused():
    record = _record("row:000")
    record["row_id"] = "row:999"
    with pytest.raises(C.ContinuationError, match="not a corpus row"):
        C.validate_candidate_record(record, LOCKED, CORPUS, "t")


def test_a_record_with_the_wrong_arm_is_refused():
    with pytest.raises(C.ContinuationError, match="arm"):
        C.validate_candidate_record(_record("row:000", arm_id="baseline"),
                                    LOCKED, CORPUS, "t")


def test_a_record_missing_its_direction_digest_is_refused():
    record = _record("row:000")
    record["per_state"] = {str(s): {} for s in CANDIDATE_STATES}
    with pytest.raises(C.ContinuationError, match="direction digest"):
        C.validate_candidate_record(record, LOCKED, CORPUS, "t")


# ================================================ 5. restart safety
def test_existing_records_are_validated_before_being_skipped(tmp_path):
    path = _write(tmp_path / "s.jsonl", [_record("row:000"), _record("row:001")])
    done = C.validate_existing_shard(path, ["row:000", "row:001", "row:002"],
                                     LOCKED, CORPUS, Checklist("t"))
    assert sorted(done) == ["row:000", "row:001"]


def test_a_duplicate_in_the_resume_file_is_refused(tmp_path):
    path = _write(tmp_path / "s.jsonl", [_record("row:000"), _record("row:000")])
    with pytest.raises(C.ContinuationError, match="more than once"):
        C.validate_existing_shard(path, ["row:000"], LOCKED, CORPUS,
                                  Checklist("t"))


def test_a_row_from_another_shard_in_this_file_is_refused(tmp_path):
    path = _write(tmp_path / "s.jsonl", [_record("row:005")])
    with pytest.raises(C.ContinuationError, match="not one of this shard"):
        C.validate_existing_shard(path, ["row:000"], LOCKED, CORPUS,
                                  Checklist("t"))


def test_an_invalid_record_in_the_resume_file_is_refused(tmp_path):
    path = _write(tmp_path / "s.jsonl", [_record("row:000", status="error")])
    with pytest.raises(C.ContinuationError, match="status"):
        C.validate_existing_shard(path, ["row:000"], LOCKED, CORPUS,
                                  Checklist("t"))


def test_a_truncated_jsonl_line_is_refused_not_skipped(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(_record("row:000")) + "\n{\"row_id\": \"row:0",
                    encoding="utf-8")
    with pytest.raises(C.ContinuationError, match="not valid JSON"):
        C.read_jsonl_strict(path)


def test_a_completed_shard_is_not_re_entered():
    with pytest.raises(C.ContinuationError, match="already complete"):
        C.assert_shard_is_not_complete({"a": 1, "b": 2}, ["a", "b"])
    C.assert_shard_is_not_complete({"a": 1}, ["a", "b"])      # still work left


def test_an_empty_resume_file_is_a_fresh_shard(tmp_path):
    assert C.validate_existing_shard(tmp_path / "absent.jsonl", ["row:000"],
                                     LOCKED, CORPUS, Checklist("t")) == {}


# ============================================ 6. the output directory
def test_a_non_empty_directory_without_resume_is_refused(tmp_path):
    module = _load(RUNNER, "_cont_runner")
    out = tmp_path / "run"
    out.mkdir()
    (out / "screen_records.shard01of08.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValidationError, match="--resume was not given"):
        module.prepare_output_dir(out, "screen_records.shard01of08.jsonl",
                                  resume=False)


def test_a_non_empty_directory_with_resume_is_accepted(tmp_path):
    """The case the old runner made unreachable."""
    module = _load(RUNNER, "_cont_runner")
    out = tmp_path / "run"
    out.mkdir()
    name = "screen_records.shard01of08.jsonl"
    (out / name).write_text("", encoding="utf-8")
    assert module.prepare_output_dir(out, name, resume=True) == out


def test_resuming_into_another_runs_directory_is_refused(tmp_path):
    module = _load(RUNNER, "_cont_runner")
    out = tmp_path / "run"
    out.mkdir()
    (out / "somebody_elses_output.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValidationError, match="did not write"):
        module.prepare_output_dir(out, "screen_records.shard01of08.jsonl",
                                  resume=True)


def test_a_sealed_output_directory_is_refused(tmp_path):
    module = _load(RUNNER, "_cont_runner")
    with pytest.raises(SealedSetError):
        module.prepare_output_dir(tmp_path / "sealed_run", "x.jsonl", False)
    consolidator = _load(CONSOLIDATOR, "_cont_consolidator")
    with pytest.raises(SealedSetError):
        consolidator.claim_output_dir(tmp_path / "sealed_out", create=False)


def test_neither_entry_point_creates_a_directory_while_checking(tmp_path):
    module = _load(RUNNER, "_cont_runner")
    consolidator = _load(CONSOLIDATOR, "_cont_consolidator")
    a, b = tmp_path / "a", tmp_path / "b"
    module.prepare_output_dir(a, "x.jsonl", resume=False)
    consolidator.claim_output_dir(b, create=False)
    assert not a.exists() and not b.exists()


def test_shard_files_are_named_per_shard(tmp_path):
    module = _load(RUNNER, "_cont_runner")
    names = {module.shard_records_name(i, 8) for i in range(1, 9)}
    assert len(names) == 8
    assert "screen_records.shard01of08.jsonl" in names


# ================================== 7. the pinned artifacts and the baseline
def test_the_four_pinned_digests_are_recorded():
    for digest in (C.MASTER_SHA256, C.MASTER_RUN_CONFIG_SHA256,
                   C.PHASE1_RECORDS_SHA256, C.V1_RECORDS_SHA256):
        assert len(digest) == 64
    assert C.MASTER_SHA256.startswith("525e62e0")
    assert C.V1_RECORDS_SHA256.startswith("6da77402")
    assert C.REUSABLE_CANDIDATE_ROWS == (
        "false_reject_test:00005:23e9d7db72f9",
        "false_reject_test:00053:050d1511e6c5",
        "false_reject_test:00071:d73ec27c1782")


@needs_artifacts
def test_the_pinned_artifacts_hash_as_recorded():
    outcome = C.verify_pinned_inputs(checks=Checklist("t"))
    assert set(outcome) == {"master", "master_run_config", "phase1_records",
                            "v1_records"}


@needs_artifacts
def test_an_altered_pinned_artifact_is_refused(tmp_path):
    fake = tmp_path / "root"
    for relative in (C.MASTER_PATH, C.MASTER_RUN_CONFIG_PATH,
                     C.PHASE1_RECORDS_PATH, C.V1_RECORDS_PATH):
        target = fake / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not the pinned artifact", encoding="utf-8")
    with pytest.raises(ValidationError, match="sha256"):
        C.verify_pinned_inputs(fake, Checklist("t"))


@needs_artifacts
def test_the_master_revalidates_against_the_corpus_and_reproduces_302_and_3():
    from routegate.data import load_development_inputs
    pinned = C.pinned_inputs()
    master = C.read_master(pinned["master"]["path"])
    corpus_rows = list(load_development_inputs().manifest_rows)
    phase1 = C.read_jsonl_strict(pinned["phase1_records"]["path"])
    v1 = C.read_jsonl_strict(pinned["v1_records"]["path"])
    checks = Checklist("t")
    outcome = C.verify_referenced_baseline(master, corpus_rows, phase1, v1,
                                           checks)
    assert checks.all_passed
    assert outcome["phase1_reproduced"] == 302
    assert outcome["interrupted_v1_reproduced"] == 3
    assert outcome["independently_reproduced_count"] == 305
    assert outcome["referenced_on_provenance_only"] == 175
    assert outcome["generated_by_this_run"] == 0


def _synthetic_frame(n=480):
    """A 480-row master and matching corpus, so the byte-comparison is reached
    rather than short-circuited by the row-count gate."""
    import hashlib
    master, corpus = {}, []
    for i in range(n):
        rid = f"m:{i:04d}"
        text = f"prompt {i}"
        master[rid] = {"row_id": rid, "frozen_class": "E",
                       "prompt_text": text,
                       "before_response_text": f"answer {i}"}
        corpus.append({"row_id": rid, "final_ebh": "E", "prompt_text": text,
                       "prompt_sha256": hashlib.sha256(
                           text.encode()).hexdigest()})
    return master, corpus


def test_a_baseline_that_does_not_reproduce_is_refused():
    master, corpus = _synthetic_frame()
    phase1 = [{"arm_id": "baseline", "row_id": "m:0000",
               "response_text": "a DIFFERENT answer",
               "prompt_sha256": corpus[0]["prompt_sha256"]}]
    with pytest.raises(ValidationError, match="byte_identical"):
        C.verify_referenced_baseline(master, corpus, phase1, [], Checklist("t"),
                                     expected_phase1=1, expected_v1=0)


def test_a_master_that_is_not_the_corpus_is_refused():
    master, corpus = _synthetic_frame()
    corpus[0] = {**corpus[0], "final_ebh": "H"}
    with pytest.raises(ValidationError, match="classes_match"):
        C.verify_referenced_baseline(master, corpus, [], [], Checklist("t"), expected_phase1=0, expected_v1=0)


def test_a_master_prompt_that_differs_byte_wise_is_refused():
    master, corpus = _synthetic_frame()
    master["m:0000"] = {**master["m:0000"], "prompt_text": "moved"}
    with pytest.raises(ValidationError, match="prompt_bytes_match"):
        C.verify_referenced_baseline(master, corpus, [], [], Checklist("t"), expected_phase1=0, expected_v1=0)


def test_a_master_row_with_no_baseline_response_is_refused():
    master, corpus = _synthetic_frame()
    master["m:0000"] = {**master["m:0000"], "before_response_text": ""}
    with pytest.raises(ValidationError, match="carries_a_baseline_response"):
        C.verify_referenced_baseline(master, corpus, [], [], Checklist("t"), expected_phase1=0, expected_v1=0)


def test_the_baseline_is_never_described_as_generated_here():
    for path in (RUNNER, CONSOLIDATOR):
        source = path.read_text(encoding="utf-8")
        assert "generated_baselines" in source, path.name
    assert C.continuation_counts()["generated_baselines"] == 0
    assert "REFERENCED" in C.__doc__ or "referenced" in C.__doc__


def test_the_masters_after_column_is_named_only_to_be_refused():
    assert C.MASTER_BASELINE_COLUMN == "before_response_text"
    assert "after_response_text" in C.MASTER_FORBIDDEN_COLUMNS
    consolidator = CONSOLIDATOR.read_text(encoding="utf-8")
    assert "after_response_text" not in consolidator


# ============================================ 8. no text reaches git
def test_no_prompt_or_response_text_is_committed():
    """Row ids and column names are metadata. A prompt or response BODY is not,
    and none is embedded in any committed file on this path."""
    import csv as _csv
    pinned = None
    try:
        pinned = C.pinned_inputs()
    except Exception:
        pass
    if pinned is None or not Path(pinned["master"]["path"]).is_file():
        pytest.skip("the master is not reachable to compare against")

    with open(pinned["master"]["path"], newline="", encoding="utf-8") as handle:
        sample = [next(_csv.DictReader(handle)) for _ in range(1)]
    prompt, response = sample[0]["prompt_text"], sample[0]["before_response_text"]

    for path in (RUNNER, CONSOLIDATOR,
                 REPO_ROOT / "src" / "routegate" / "twositecontinuation.py",
                 REPO_ROOT / "tests" / "test_two_site_continuation.py"):
        source = path.read_text(encoding="utf-8")
        assert prompt[:60] not in source, path.name
        assert response[:60] not in source, path.name


def test_the_entry_points_judge_nothing_and_train_nothing():
    import ast
    for path in (RUNNER, CONSOLIDATOR):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:]
        code = ast.unparse(tree).lower()
        for banned in ("adjudicat", "rubric", "optimizer", "backward()"):
            assert banned not in code, (path.name, banned)
        assert "judges_responses" in code and "trains_a_controller" in code


def test_the_runner_exposes_no_option_reaching_held_out_material():
    from routegate.validation import looks_sealed
    for path, name in ((RUNNER, "_cont_runner2"),
                       (CONSOLIDATOR, "_cont_consolidator2")):
        module = _load(path, name)
        options = [o for a in module.build_parser()._actions
                   for o in a.option_strings]
        assert options
        for option in options:
            assert looks_sealed(option) is None, option
