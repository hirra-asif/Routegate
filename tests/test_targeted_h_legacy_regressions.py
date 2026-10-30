"""The frozen 20-row LEGACY H regression stress set, on synthetic tables.

Nothing here loads a model, allocates a GPU, calls an API, judges anything or
reads sealed material. Every CSV is invented in ``tmp_path`` and every digest
is recomputed for it, so the frozen inputs are never needed to test the gates
that guard them.

Organised by the failure each group is there to prevent:

    1. selection   the rule picks other than the twenty recorded rows
    2. identity    a row is not H, is duplicated, or is not in the corpus
    3. condition   the rows came from some other static condition
    4. hashes      an input that is not the recorded artifact
    5. hygiene     prompt or response text reaches a committed manifest
    6. committed   the manifest on disk says what it is
    7. attribution the record credits these regressions to the wrong direction
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_script(name: str):
    path = REPO_ROOT / "experiments" / "dual_direction_controller" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ddc_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load_script("build_targeted_h_legacy_regressions")
Error = BUILD.LegacyRegressionSelectionError
MANIFEST = (REPO_ROOT / "manifests"
            / "targeted_h_legacy_regressions_v1.json")


# ---------------------------------------------------------------- fixtures
def _master_rows(n_yes=20, n_no=193, n_e=89, n_b=178):
    """A static-steering master in the real one's shape, without any text."""
    rows = []
    for index in range(n_yes):
        rows.append({"row_id": f"harmbench_text_test:h{index:03d}",
                     "frozen_class": "H", "source": "harmbench_text_test",
                     "before_axis2": "DR" if index % 4 else "IR",
                     "final_actionable_harmful_help": "YES",
                     "final_safety_label_source": "human",
                     "intervention_hit_count": "3"})
    for index in range(n_no):
        rows.append({"row_id": f"orbench_toxic:n{index:03d}",
                     "frozen_class": "H", "source": "orbench_toxic",
                     "before_axis2": "DR",
                     "final_actionable_harmful_help": "NO",
                     "final_safety_label_source": "agreed_llm",
                     "intervention_hit_count": "3"})
    for index in range(n_e):
        rows.append({"row_id": f"false_reject_test:e{index:03d}",
                     "frozen_class": "E", "source": "false_reject_test",
                     "before_axis2": "DR", "final_actionable_harmful_help": "",
                     "final_safety_label_source": "",
                     "intervention_hit_count": "3"})
    for index in range(n_b):
        rows.append({"row_id": f"orbench_hard:b{index:03d}",
                     "frozen_class": "B", "source": "orbench_hard",
                     "before_axis2": "SA", "final_actionable_harmful_help": "",
                     "final_safety_label_source": "",
                     "intervention_hit_count": "3"})
    return rows


def _run_config(**overrides):
    config = {
        "actuator": {
            "direction_variant": "consensus23_global_frozen", "role": "primary",
            "site": "residual_stream", "layers": [31, 32, 33], "window": 1,
            "slot": 0, "strength": 0.15, "prefill_only": True,
            "expected_patch_hits_per_row": 3,
            "slot_semantics": "the final prefill row",
            "application_scaling": "unit_direction * strength * norm",
        },
        "generation": {
            "model": "Qwen/Qwen3-30B-A3B",
            "revision_resolved": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "tokenizer_revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "tokenizer_config_sha256": "d" * 64,
            "chat_template_sha256": "e" * 64,
            "max_new_tokens": 768, "decoding": "greedy",
        },
    }
    for path, value in overrides.items():
        block, field = path.split(".", 1)
        config[block][field] = value
    return config


def _write_world(tmp_path, master=None, corpus=None, run_config=None,
                 monkeypatch=None):
    """A complete synthetic world, with frozen_inputs pinned to it."""
    master = master if master is not None else _master_rows()
    columns = list(BUILD.MASTER_COLUMNS)
    master_path = tmp_path / "master.csv"
    with open(master_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(master)

    if corpus is None:
        corpus = [{"row_id": row["row_id"], "final_ebh": row["frozen_class"]}
                  for row in master]
    corpus_path = tmp_path / "corpus.csv"
    with open(corpus_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "final_ebh"])
        writer.writeheader()
        writer.writerows(corpus)

    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(run_config or _run_config()),
                           encoding="utf-8")
    if monkeypatch is not None:
        # the real builder pins the migrated condition file by digest; the
        # synthetic one is pinned to its own so the other gates can be reached
        monkeypatch.setattr(
            BUILD, "STATIC_RUN_CONFIG_SHA256",
            hashlib.sha256(config_path.read_bytes()).hexdigest())

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    pinned = {BUILD.STATIC_MASTER_LOGICAL: {"sha256": sha(master_path)},
              BUILD.CORPUS_LOGICAL: {"sha256": sha(corpus_path)}}
    if monkeypatch is not None:
        monkeypatch.setattr(BUILD, "recorded_digest",
                            lambda name: pinned[name])
    return {"master": master_path, "corpus": corpus_path,
            "run_config": config_path, "out": tmp_path / "manifest.json",
            "pinned": pinned}


def _build(world, write=False):
    return BUILD.build(world["master"], world["corpus"], world["run_config"],
                       world["out"], write=write)


# ============================================================ 1. selection
def test_the_rule_selects_exactly_the_twenty_recorded_rows(tmp_path,
                                                           monkeypatch):
    world = _write_world(tmp_path, monkeypatch=monkeypatch)
    manifest = _build(world)
    assert manifest["row_count"] == BUILD.EXPECTED_ROWS == 20
    assert len(manifest["row_ids"]) == 20 == len(set(manifest["row_ids"]))
    assert all(row["row_class"] == "H" for row in manifest["rows"])
    assert all(row["final_actionable_harmful_help"] == "YES"
               for row in manifest["rows"])
    assert manifest["targeted_diagnostic"] is True
    assert manifest["population_reportable"] is False


def test_the_rule_is_a_field_lookup_and_not_a_keyword_search(tmp_path,
                                                             monkeypatch):
    world = _write_world(tmp_path, monkeypatch=monkeypatch)
    manifest = _build(world)
    rule = manifest["selection_rule"]
    assert rule["field_lookup_only"] is True
    assert rule["no_keyword_matching"] is True
    assert rule["no_relabelling"] is True
    assert rule["expected_rows"] == 20
    # the reader never opens a text column
    assert "prompt_text" not in BUILD.MASTER_COLUMNS
    assert "after_response_text" not in BUILD.MASTER_COLUMNS


@pytest.mark.parametrize("n_yes", [19, 21, 0])
def test_a_different_number_of_regressions_is_refused(tmp_path, monkeypatch,
                                                      n_yes):
    world = _write_world(tmp_path, master=_master_rows(n_yes=n_yes),
                         monkeypatch=monkeypatch)
    with pytest.raises(Error, match="row_count"):
        _build(world)


def test_the_rows_are_ordered_deterministically(tmp_path, monkeypatch):
    world = _write_world(tmp_path, monkeypatch=monkeypatch)
    first = _build(world)
    shuffled = list(reversed(_master_rows()))
    other = tmp_path / "b"
    other.mkdir()
    second_world = _write_world(other, master=shuffled, monkeypatch=monkeypatch)
    second = _build(second_world)
    assert first["row_ids"] == second["row_ids"] == sorted(first["row_ids"])


# ============================================================= 2. identity
def test_a_duplicate_row_id_is_refused(tmp_path, monkeypatch):
    rows = _master_rows()
    rows.append(dict(rows[0]))
    world = _write_world(tmp_path, master=rows, monkeypatch=monkeypatch)
    with pytest.raises(Error, match="row_count|row_ids_unique"):
        _build(world)


def test_a_selected_row_that_is_not_h_in_the_corpus_is_refused(tmp_path,
                                                               monkeypatch):
    master = _master_rows()
    corpus = [{"row_id": row["row_id"],
               "final_ebh": "B" if row["row_id"] == master[0]["row_id"]
               else row["frozen_class"]} for row in master]
    world = _write_world(tmp_path, corpus=corpus, monkeypatch=monkeypatch)
    with pytest.raises(Error, match="all_rows_are_H_in_the_corpus"):
        _build(world)


def test_a_selected_row_absent_from_the_corpus_is_refused(tmp_path,
                                                          monkeypatch):
    master = _master_rows()
    corpus = [{"row_id": row["row_id"], "final_ebh": row["frozen_class"]}
              for row in master if row["row_id"] != master[0]["row_id"]]
    world = _write_world(tmp_path, corpus=corpus, monkeypatch=monkeypatch)
    with pytest.raises(Error, match="all_rows_in_the_accepted_corpus"):
        _build(world)


def test_a_row_that_did_not_refuse_before_is_not_a_regression(tmp_path,
                                                              monkeypatch):
    rows = _master_rows()
    rows[0]["before_axis2"] = "SA"
    world = _write_world(tmp_path, master=rows, monkeypatch=monkeypatch)
    with pytest.raises(Error, match="every_row_refused_before"):
        _build(world)


def test_a_row_whose_label_was_not_adjudicated_is_refused(tmp_path,
                                                          monkeypatch):
    rows = _master_rows()
    rows[0]["final_safety_label_source"] = "agreed_llm"
    world = _write_world(tmp_path, master=rows, monkeypatch=monkeypatch)
    with pytest.raises(Error, match="safety_labels_are_adjudicated"):
        _build(world)


def test_a_row_with_the_wrong_patch_hit_count_is_refused(tmp_path, monkeypatch):
    rows = _master_rows()
    rows[0]["intervention_hit_count"] = "2"
    world = _write_world(tmp_path, master=rows, monkeypatch=monkeypatch)
    with pytest.raises(Error, match="three_patch_hits_per_row"):
        _build(world)


# ============================================================ 3. condition
@pytest.mark.parametrize("field,value", [
    ("actuator.strength", 0.2),
    ("actuator.layers", [30, 31, 32]),
    ("actuator.slot", 1),
    ("actuator.window", 4),
    ("actuator.prefill_only", False),
    ("actuator.expected_patch_hits_per_row", 1),
    ("actuator.site", "mlp_output"),
])
def test_a_different_static_condition_is_refused(tmp_path, monkeypatch, field,
                                                 value):
    world = _write_world(tmp_path, run_config=_run_config(**{field: value}),
                         monkeypatch=monkeypatch)
    with pytest.raises(Error, match="condition_"):
        _build(world)


def test_a_different_direction_variant_is_refused(tmp_path, monkeypatch):
    world = _write_world(
        tmp_path,
        run_config=_run_config(**{"actuator.direction_variant":
                                  "historical_d1_global_frozen"}),
        monkeypatch=monkeypatch)
    with pytest.raises(Error, match="condition_direction_variant"):
        _build(world)


@pytest.mark.parametrize("field,value", [
    ("generation.model", "Qwen/Qwen2-7B"),
    ("generation.revision_resolved", "0" * 40),
    ("generation.tokenizer_revision", "1" * 40),
])
def test_a_different_model_or_revision_is_refused(tmp_path, monkeypatch, field,
                                                  value):
    world = _write_world(tmp_path, run_config=_run_config(**{field: value}),
                         monkeypatch=monkeypatch)
    with pytest.raises(Error, match="condition_"):
        _build(world)


def test_a_run_config_with_no_actuator_block_is_refused(tmp_path, monkeypatch):
    world = _write_world(tmp_path, run_config={"generation": {}},
                         monkeypatch=monkeypatch)
    with pytest.raises(Error, match="carries no actuator"):
        _build(world)


def test_an_absent_run_config_stops_the_build(tmp_path, monkeypatch):
    world = _write_world(tmp_path, monkeypatch=monkeypatch)
    world["run_config"].unlink()
    with pytest.raises(Error, match="static condition is read"):
        _build(world)


def test_the_recorded_condition_names_alpha_s_zero(tmp_path, monkeypatch):
    world = _write_world(tmp_path, monkeypatch=monkeypatch)
    condition = _build(world)["static_condition"]
    assert condition["strength_alpha_repair"] == 0.15
    assert condition["alpha_safety"] == 0.0
    assert condition["historical_layers"] == [31, 32, 33]
    assert condition["prefill_only"] is True
    assert condition["direction_variant"] == "consensus23_global_frozen"


# =============================================================== 4. hashes
@pytest.mark.parametrize("which", ["master", "corpus"])
def test_an_input_whose_digest_moved_is_refused(tmp_path, monkeypatch, which):
    world = _write_world(tmp_path, monkeypatch=monkeypatch)
    path = world[which]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Error, match="_sha256"):
        _build(world)


def test_a_file_that_is_not_the_recorded_artifact_is_refused(tmp_path):
    """No monkeypatch: the real frozen_inputs digests are the gate, and a
    synthetic table is not the artifact they record."""
    world = _write_world(tmp_path)          # note: recorded_digest not patched
    with pytest.raises(Error, match="_sha256"):
        _build(world)


def test_the_real_frozen_inputs_are_recorded_by_logical_name():
    for logical in (BUILD.STATIC_MASTER_LOGICAL, BUILD.CORPUS_LOGICAL):
        entry = BUILD.recorded_digest(logical)
        assert len(entry["sha256"]) == 64, logical
    with pytest.raises(Error, match="records no input"):
        BUILD.recorded_digest("not_a_frozen_input")


# ============================================================== 5. hygiene
def test_prompt_or_response_text_cannot_reach_the_manifest():
    for column in ("prompt_text", "before_response_text",
                   "after_response_text", "human_reason"):
        with pytest.raises(Error, match="prompt or response text"):
            BUILD.assert_no_text({"rows": [{column: "some text"}]})


def test_a_long_string_under_a_data_key_is_refused_but_prose_is_allowed():
    long = "x" * 500
    with pytest.raises(Error, match="non-prose key"):
        BUILD.assert_no_text({"row_ids": [long]})
    with pytest.raises(Error, match="non-prose key"):
        BUILD.assert_no_text({"something": long})
    # the manifest's own explanation of itself is allowed to be long
    BUILD.assert_no_text({"population_reportable_note": long})
    BUILD.assert_no_text({"statement": long})
    BUILD.assert_no_text({"direction_variant_caveat": long})


def test_the_builder_reads_no_text_column_and_loads_no_model():
    source = (REPO_ROOT / "experiments" / "dual_direction_controller"
              / "build_targeted_h_legacy_regressions.py"
              ).read_text(encoding="utf-8")
    for banned in ("load_model_and_tokenizer", "from_pretrained", "cuda",
                   "generate(", "/home/"):
        assert banned not in source, banned
    assert "assert_not_sealed" in source


def test_the_held_out_split_is_unreachable(tmp_path, monkeypatch):
    """Named without the word the guard scans for: pytest derives tmp_path
    from the test name, and assert_not_sealed would refuse the fixture's own
    directory. That is the guard working, not a problem to route around."""
    world = _write_world(tmp_path, monkeypatch=monkeypatch)
    sealed = _build(world)["sealed_set"]
    assert sealed["overlap"] is False
    commitment = json.loads(
        (REPO_ROOT / "manifests" / "sealed_set_commitment.json").read_text("utf-8"))
    assert commitment["availability"][
        "sealed_prompt_text_available_to_development_code"] is False
    # the commitment stores counts, never row ids
    assert "row_ids" not in commitment


# ============================================================ 6. committed
def test_the_committed_manifest_is_the_frozen_twenty():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == BUILD.SCHEMA
    assert manifest["row_count"] == 20
    assert len(manifest["row_ids"]) == 20 == len(set(manifest["row_ids"]))
    assert manifest["targeted_diagnostic"] is True
    assert manifest["population_reportable"] is False
    assert manifest["contains_prompt_or_response_text"] is False
    assert all(row["row_class"] == "H" for row in manifest["rows"])
    assert manifest["checklist"]["all_passed"] is True
    BUILD.assert_no_text(manifest, "committed manifest")


def test_the_committed_manifest_records_its_sources():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    for name in ("static_steering_master", "development_corpus",
                 "static_run_config"):
        assert len(sources[name]["sha256"]) == 64, name
    # every runtime input now resolves under the data root
    for name in ("static_steering_master", "development_corpus",
                 "static_run_config"):
        assert sources[name]["root"] == "routegate-data", name
    assert sources["static_run_config"]["migrated_to_routegate_data"] is True
    assert (sources["static_run_config"]["sha256"]
            == BUILD.STATIC_RUN_CONFIG_SHA256)
    # provenance references are records, not runtime reads
    assert manifest["judgement_provenance"]["read_at_runtime"] is False


# =========================================================== 7. attribution
def test_the_manifest_credits_the_legacy_actuator_and_its_donors():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = manifest["source_actuator"]
    assert source["variant"] == "consensus23_global_frozen"
    assert source["donors"] == ["B_d2", "B_d3"]
    assert source["strength"] == 0.15
    assert source["historical_layers"] == [31, 32, 33]
    assert source["routegate_states"] == [32, 33, 34]
    assert source["prefill_only"] is True
    assert manifest["static_condition"]["direction_variant"] == (
        "consensus23_global_frozen")


def test_the_manifest_carries_the_four_required_flags():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["targeted_diagnostic"] is True
    assert manifest["population_reportable"] is False
    assert manifest["legacy_stress_set"] is True
    assert manifest[
        "current_validated_repair_direction_was_not_the_selection_treatment"
    ] is True
    assert manifest["row_count"] == 20


def test_the_manifest_makes_none_of_the_three_forbidden_claims():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    claims = manifest["does_not_claim"]
    assert claims["b_d1_minus_e_caused_these_regressions"] is False
    assert claims["these_are_failures_of_the_validated_direction_bank"] is False
    assert claims[
        "testing_h_minus_e_here_proves_rescue_of_current_b_d1_failures"] is False
    statement = claims["statement"]
    assert "consensus23_global_frozen" in statement
    assert "historical_d1_global_frozen" in statement
    assert manifest["what_it_is_for"]["current_validated_repair_direction"] == (
        "historical_d1_global_frozen")


def test_the_record_never_attributes_the_regressions_to_b_d1():
    """The one sentence that must not appear, in any file of this selection."""
    manifest = MANIFEST.read_text(encoding="utf-8")
    builder = (REPO_ROOT / "experiments" / "dual_direction_controller"
               / "build_targeted_h_legacy_regressions.py").read_text("utf-8")
    for text in (manifest, builder):
        # B_d1 may only ever appear beside a denial or as the *current* bank's
        # direction, never as the treatment that produced these rows
        for sentence in text.replace("\n", " ").split("."):
            if "B_d1" not in sentence:
                continue
            lowered = sentence.lower()
            assert any(marker in lowered for marker in
                       ("not ", "never", "does_not", "did not", "was not",
                        "reproduces", "comparator", "validated")), sentence


@pytest.mark.parametrize("variant", ["historical_d1_global_frozen",
                                     "some_other_variant"])
def test_a_run_config_naming_another_variant_is_refused(tmp_path, monkeypatch,
                                                        variant):
    world = _write_world(
        tmp_path,
        run_config=_run_config(**{"actuator.direction_variant": variant}),
        monkeypatch=monkeypatch)
    with pytest.raises(Error, match="condition_direction_variant"):
        _build(world)


def test_the_condition_file_must_be_the_pinned_migrated_one(tmp_path,
                                                            monkeypatch):
    world = _write_world(tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(BUILD, "STATIC_RUN_CONFIG_SHA256", "0" * 64)
    with pytest.raises(Error, match="static_run_config_sha256"):
        _build(world)


def test_an_input_in_the_historical_workspace_is_refused(tmp_path,
                                                         monkeypatch):
    """No runtime dependency on SteerMoE-main may remain."""
    workspace = tmp_path / "SteerMoE-main"
    workspace.mkdir()
    world = _write_world(workspace, monkeypatch=monkeypatch)
    with pytest.raises(Error, match="historical workspace"):
        _build(world)


def test_the_pinned_condition_digest_is_the_real_migrated_file():
    """The constant in the builder is the digest of the file on disk."""
    assert len(BUILD.STATIC_RUN_CONFIG_SHA256) == 64
    assert BUILD.STATIC_RUN_CONFIG_RELATIVE.endswith(
        "rq3_bh_rq2_static_20260810/run_config.json")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert (manifest["sources"]["static_run_config"]["pinned_sha256"]
            == BUILD.STATIC_RUN_CONFIG_SHA256)
