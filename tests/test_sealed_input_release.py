"""The header-only compatibility adapter for the v5 runner's held-out input.

Everything here is offline and synthetic. No model is loaded, no row is
generated, and no test needs the archival input to exist: the fixtures below
reproduce its shape -- CRLF row terminators, quoted fields carrying bare line
feeds, and a header that contains ``canonical_prompt`` twice as a substring but
once as a token.

The digest gate is exercised by monkeypatching the module's required hash, so
the production path keeps no test bypass at all. A test below asserts that
neither command-line entry point can override the gate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from routegate import sealedinputrelease as sir                    # noqa: E402
from routegate.finaleval import PREDECLARATION, verify_protocol    # noqa: E402
from routegate.sealedinputrelease import (                         # noqa: E402
    ADAPTER_COLUMN,
    BYTE_DELTA,
    EXPECTED_HARMFUL,
    EXPECTED_ROWS,
    EXPECTED_SAFE,
    REJECTED_SOURCE_SHA256,
    RELEASE,
    REQUIRED_SOURCE_SHA256,
    SOURCE_COLUMN,
    SealedInputReleaseError,
    adapter_name,
    build_release,
    default_adapter_path,
    expected_adapter_fields,
    header_tokens,
    prepare_adapter,
    prompt_column_span,
    read_rows,
    rewrite_header_bytes,
    sha256_bytes,
    split_first_line,
    verify_adapter,
    verify_release,
)
from routegate.triplets import sha256_text                         # noqa: E402
from routegate.validation import Checklist, ValidationError        # noqa: E402

PREPARE = REPO / "experiments/final_evaluation/prepare_sealed_input_release.py"
FREEZE = REPO / "experiments/final_evaluation/freeze_sealed_input_release.py"

#: Present in every synthetic prompt, so "no prompt was printed" is a search.
SENTINEL = "ZEBRAFISH-SENTINEL"

HEADER = (b"row_id,origin,canonical_prompt,canonical_prompt_sha256,"
          b"source_side,axis1_label")


def field(value: str) -> bytes:
    if any(c in value for c in ',"\n\r'):
        return b'"' + value.replace('"', '""').encode() + b'"'
    return value.encode()


def prompt_for(index: int) -> str:
    """Unique, and shaped like the real thing: quotes, commas, line feeds."""
    if index % 37 == 0:
        return (f"{SENTINEL} {index}: a prompt that spans\nmore than one "
                f'line, carries a comma and a "quotation".')
    return f"{SENTINEL} {index}: a single-line prompt."


def make_csv(rows: int = EXPECTED_ROWS, safe: int = EXPECTED_SAFE,
             header: bytes = HEADER, unique_prompts: bool = True,
             blank_id_at: int = -1) -> bytes:
    out = [header]
    for i in range(rows):
        text = prompt_for(i if unique_prompts else 0)
        row_id = "" if i == blank_id_at else f"r{i:04d}"
        out.append(b",".join([
            field(row_id), field("external"), field(text),
            field(sha256_text(text)),
            field("safe_source" if i < safe else "harmful_source"),
            field("a1")]))
    return b"\r\n".join(out) + b"\r\n"


@pytest.fixture()
def authoritative(monkeypatch):
    """A synthetic file that the module accepts as authoritative."""
    data = make_csv()
    monkeypatch.setattr(sir, "REQUIRED_SOURCE_SHA256", sha256_bytes(data))
    return data


@pytest.fixture()
def source(tmp_path, authoritative):
    path = tmp_path / "input_manifest.csv"
    path.write_bytes(authoritative)
    return path


# --------------------------------------------------------- the header itself
def test_the_header_carries_the_prompt_column_twice_as_a_substring():
    """The trap this module exists to avoid, asserted on the real shape."""
    assert HEADER.count(SOURCE_COLUMN.encode()) == 2
    assert sum(1 for t in header_tokens(HEADER)
               if t == SOURCE_COLUMN.encode()) == 1


def test_only_the_whole_token_is_replaced_not_the_digest_column():
    rewritten = rewrite_header_bytes(make_csv())
    tokens = header_tokens(split_first_line(rewritten)[0])
    assert tokens.count(ADAPTER_COLUMN.encode()) == 1
    assert SOURCE_COLUMN.encode() not in tokens
    assert f"{SOURCE_COLUMN}_sha256".encode() in tokens


def test_a_naive_text_replace_would_have_corrupted_the_digest_column():
    """Stated as a test so the reason for the token match cannot be lost."""
    naive = HEADER.replace(SOURCE_COLUMN.encode(), ADAPTER_COLUMN.encode())
    assert f"{ADAPTER_COLUMN}_sha256".encode() in naive, "the trap is real"
    assert f"{SOURCE_COLUMN}_sha256".encode() not in naive
    correct = split_first_line(rewrite_header_bytes(make_csv()))[0]
    assert correct != naive


def test_the_carriage_return_ending_the_header_is_preserved():
    data = make_csv()
    assert split_first_line(data)[0].endswith(b"\r")
    assert split_first_line(rewrite_header_bytes(data))[0].endswith(b"\r")


def test_a_header_without_the_prompt_column_is_refused():
    data = make_csv(header=b"row_id,origin,text,source_side,axis1_label")
    with pytest.raises(SealedInputReleaseError, match="exactly one"):
        rewrite_header_bytes(data)


def test_a_header_with_two_prompt_columns_is_refused():
    data = make_csv(
        header=b"row_id,canonical_prompt,canonical_prompt,source_side")
    with pytest.raises(SealedInputReleaseError, match="exactly one|carries 2"):
        rewrite_header_bytes(data)


def test_a_quoted_header_is_refused():
    with pytest.raises(SealedInputReleaseError, match="quoted"):
        header_tokens(b'row_id,"canonical_prompt",source_side')


def test_a_file_with_no_line_feed_is_refused():
    with pytest.raises(SealedInputReleaseError, match="no line feed|no header"):
        split_first_line(b"row_id,canonical_prompt")


def test_the_span_points_at_the_first_token_not_the_digest_column():
    start, end = prompt_column_span(HEADER)
    assert HEADER[start:end] == SOURCE_COLUMN.encode()
    assert HEADER[end:end + 1] == b","


# --------------------------------------------------------------- the gate
def test_a_wrong_input_digest_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sir, "REQUIRED_SOURCE_SHA256", "0" * 64)
    path = tmp_path / "input_manifest.csv"
    path.write_bytes(make_csv())
    with pytest.raises(ValidationError, match="source_sha256"):
        prepare_adapter(path, tmp_path / "out.csv", Checklist("t"))
    assert not (tmp_path / "out.csv").exists()


def test_the_superseded_predecessor_is_refused_by_digest(tmp_path,
                                                         monkeypatch):
    """379 rows: the authoritative file minus one harmful row."""
    predecessor = make_csv(rows=379, safe=180)
    monkeypatch.setattr(sir, "REQUIRED_SOURCE_SHA256",
                        sha256_bytes(make_csv()))
    monkeypatch.setattr(sir, "REJECTED_SOURCE_SHA256",
                        sha256_bytes(predecessor))
    path = tmp_path / "input_manifest.csv"
    path.write_bytes(predecessor)
    with pytest.raises(SealedInputReleaseError, match="superseded|379"):
        prepare_adapter(path, tmp_path / "out.csv", Checklist("t"))
    assert not (tmp_path / "out.csv").exists()


def test_the_predecessor_rejection_reason_states_the_row_deficit():
    assert sir.REJECTED_ROWS == 379
    assert sir.REJECTED_SAFE == 180 and sir.REJECTED_HARMFUL == 199
    assert "379" in sir.REJECTED_REASON
    assert "harmful" in sir.REJECTED_REASON


def test_the_two_pinned_digests_are_distinct_and_well_formed():
    for digest in (REQUIRED_SOURCE_SHA256, REJECTED_SOURCE_SHA256):
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    assert REQUIRED_SOURCE_SHA256 != REJECTED_SOURCE_SHA256


def test_a_short_row_count_alone_cannot_pass_the_gate(tmp_path, monkeypatch):
    """Even a 379-row file that is not the known predecessor is refused."""
    other = make_csv(rows=379, safe=179)
    monkeypatch.setattr(sir, "REQUIRED_SOURCE_SHA256",
                        sha256_bytes(make_csv()))
    path = tmp_path / "input_manifest.csv"
    path.write_bytes(other)
    with pytest.raises(ValidationError, match="source_sha256"):
        prepare_adapter(path, tmp_path / "out.csv", Checklist("t"))


# ------------------------------------------------------------- preparation
def test_the_adapter_is_the_input_with_five_fewer_bytes(source, tmp_path):
    out = tmp_path / "adapter.csv"
    result = prepare_adapter(source, out, Checklist("t"))
    assert BYTE_DELTA == -5
    assert result["adapter_bytes"] - result["source_bytes"] == BYTE_DELTA
    assert out.read_bytes() == rewrite_header_bytes(source.read_bytes())


def test_every_byte_after_the_header_is_identical(source, tmp_path):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    before = split_first_line(source.read_bytes())[1]
    after = split_first_line(out.read_bytes())[1]
    assert after == before
    assert len(before) > 0


def test_preparation_is_idempotent(source, tmp_path):
    out = tmp_path / "adapter.csv"
    first = prepare_adapter(source, out, Checklist("t"))
    digest = sha256_bytes(out.read_bytes())
    second = prepare_adapter(source, out, Checklist("t"))
    assert first["adapter_sha256"] == second["adapter_sha256"] == digest
    assert first["written"] is True and second["written"] is False
    assert sha256_bytes(out.read_bytes()) == digest


def test_an_existing_different_adapter_is_never_overwritten(source, tmp_path):
    out = tmp_path / "adapter.csv"
    out.write_bytes(b"row_id,prompt_text\r\nr0000,something else\r\n")
    stale = out.read_bytes()
    with pytest.raises(SealedInputReleaseError, match="exists and is not"):
        prepare_adapter(source, out, Checklist("t"))
    assert out.read_bytes() == stale


def test_a_dry_run_writes_nothing(source, tmp_path):
    out = tmp_path / "adapter.csv"
    result = prepare_adapter(source, out, Checklist("t"), dry_run=True)
    assert not out.exists()
    assert result["written"] is False
    assert len(result["adapter_sha256"]) == 64


def test_the_input_is_not_modified(source, tmp_path):
    before = source.read_bytes()
    prepare_adapter(source, tmp_path / "adapter.csv", Checklist("t"))
    assert source.read_bytes() == before


def test_the_adapter_name_is_derived_from_the_input_stem():
    assert adapter_name(Path("/a/b/some_input.csv")) == (
        f"some_input_{ADAPTER_COLUMN}.csv")
    assert default_adapter_path(
        Path("/a/b/some_input.csv"), Path("/root")).parent == (
        Path("/root") / sir.RELEASE_RELATIVE)


# ------------------------------------------------------------ verification
def test_the_adapter_verifies_against_its_input(source, tmp_path):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    checks = Checklist("t")
    result = verify_adapter(source, out, checks)
    assert checks.all_passed
    assert result["rows"] == EXPECTED_ROWS
    assert result["safe_source"] == EXPECTED_SAFE
    assert result["harmful_source"] == EXPECTED_HARMFUL
    assert result["unique_row_ids"] == EXPECTED_ROWS
    assert result["unique_prompts"] == EXPECTED_ROWS
    assert result["adapter_fields"] == expected_adapter_fields(
        result["source_fields"])


@pytest.mark.parametrize("offset_from_end", [1, 40, 400])
def test_a_change_outside_the_header_line_is_detected(source, tmp_path,
                                                      offset_from_end):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    data = bytearray(out.read_bytes())
    index = len(data) - offset_from_end
    assert index > len(split_first_line(bytes(data))[0])
    data[index] = data[index] ^ 0x20
    out.write_bytes(bytes(data))
    with pytest.raises(ValidationError):
        verify_adapter(source, out, Checklist("t"))


def test_an_extra_column_in_the_adapter_is_detected(source, tmp_path):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    data = out.read_bytes()
    header, rest = split_first_line(data)
    out.write_bytes(header[:-1] + b",verdict\r" + rest)
    with pytest.raises(ValidationError):
        verify_adapter(source, out, Checklist("t"))


def test_a_reordered_adapter_is_detected(source, tmp_path):
    """Order is part of the identity: a valid CSV is not a valid adapter."""
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    header, rest = split_first_line(out.read_bytes())
    body = [b for b in rest[1:].split(b"\r\n") if b]
    reordered = body[1:2] + body[0:1] + body[2:]
    out.write_bytes(header + b"\n" + b"\r\n".join(reordered) + b"\r\n")
    with pytest.raises(ValidationError):
        verify_adapter(source, out, Checklist("t"))


def test_a_renamed_digest_column_is_detected(source, tmp_path):
    """What the naive replace would have produced, refused."""
    out = tmp_path / "adapter.csv"
    header, rest = split_first_line(source.read_bytes())
    out.write_bytes(
        header.replace(SOURCE_COLUMN.encode(), ADAPTER_COLUMN.encode()) + rest)
    with pytest.raises(ValidationError):
        verify_adapter(source, out, Checklist("t"))


def test_row_order_and_every_value_are_invariant(source, tmp_path):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    source_fields, source_rows = read_rows(source)
    adapter_fields, adapter_rows = read_rows(out)
    assert [r["row_id"] for r in source_rows] == [
        r["row_id"] for r in adapter_rows]
    for before, after in zip(source_rows, adapter_rows):
        assert {(ADAPTER_COLUMN if k == SOURCE_COLUMN else k): v
                for k, v in before.items()} == after
    assert set(adapter_fields) - set(source_fields) == {ADAPTER_COLUMN}


def test_a_blank_row_id_is_refused(tmp_path, monkeypatch):
    data = make_csv(blank_id_at=7)
    monkeypatch.setattr(sir, "REQUIRED_SOURCE_SHA256", sha256_bytes(data))
    src = tmp_path / "input_manifest.csv"
    src.write_bytes(data)
    out = tmp_path / "adapter.csv"
    prepare_adapter(src, out, Checklist("t"))
    with pytest.raises(ValidationError, match="no_blank_row_id"):
        verify_adapter(src, out, Checklist("t"))


def test_duplicate_prompts_are_refused(tmp_path, monkeypatch):
    data = make_csv(unique_prompts=False)
    monkeypatch.setattr(sir, "REQUIRED_SOURCE_SHA256", sha256_bytes(data))
    src = tmp_path / "input_manifest.csv"
    src.write_bytes(data)
    out = tmp_path / "adapter.csv"
    prepare_adapter(src, out, Checklist("t"))
    with pytest.raises(ValidationError, match="unique_prompts|unique_row_ids"):
        verify_adapter(src, out, Checklist("t"))


# --------------------------------------------------------- no prompt leaks
def test_no_check_detail_carries_a_prompt(source, tmp_path):
    out = tmp_path / "adapter.csv"
    checks = Checklist("t")
    prepare_adapter(source, out, checks)
    verify_adapter(source, out, checks)
    blob = json.dumps(checks.as_dict())
    assert SENTINEL not in blob
    assert checks.all_passed


def test_the_release_record_carries_no_prompt(source, tmp_path):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    payload = build_release(source, out, Checklist("t"))
    blob = json.dumps(payload)
    assert SENTINEL not in blob
    assert payload["contains_prompt_response_gold_or_score"] is False


def test_the_command_line_prints_no_prompt(source, tmp_path):
    out = tmp_path / "adapter.csv"
    proc = subprocess.run(
        [sys.executable, str(PREPARE), "--input", str(source),
         "--output", str(out), "--dry-run"],
        capture_output=True, text=True)
    # The synthetic input fails the real digest gate, which is itself correct;
    # what matters is that nothing it printed carries a prompt.
    assert SENTINEL not in proc.stdout and SENTINEL not in proc.stderr


def test_no_public_function_offers_a_way_around_the_digest_gate():
    """The gate is a module constant, not an argument anything can pass."""
    import inspect
    for fn in (sir.assert_source_is_authoritative, sir.prepare_adapter,
               sir.verify_adapter, sir.build_release, sir.verify_release):
        names = set(inspect.signature(fn).parameters)
        assert not {n for n in names if "sha" in n or "digest" in n
                    or "required" in n}, f"{fn.__name__} takes {names}"
    for script in (PREPARE, FREEZE):
        assert "REQUIRED_SOURCE_SHA256 =" not in script.read_text(
            encoding="utf-8")


def test_the_new_files_name_no_held_out_material():
    """The repository's standing guard, restated for these files."""
    forbidden = ("rq3_sealed", "sealed_final_manifest", "sealed_final",
                 "rq3_prompt_splits")
    for path in (PREPARE, FREEZE,
                 REPO / "src/routegate/sealedinputrelease.py"):
        text = path.read_text(encoding="utf-8").lower()
        for needle in forbidden:
            assert needle not in text, f"{path.name} names '{needle}'"


# ------------------------------------------------------------ release record
def test_the_release_record_round_trips(source, tmp_path, monkeypatch):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    payload = build_release(source, out, Checklist("t"))
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    checks = Checklist("t")
    verify_release(path, checks)
    assert checks.all_passed


def test_the_release_record_pins_what_it_must(source, tmp_path):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    payload = build_release(source, out, Checklist("t"))
    assert payload["applies_to"] == "routegate_final_evaluation_v5"
    assert payload["applies_to_manifest_sha256"]
    assert payload["applies_to_manifest_record_sha256"]
    assert payload["source"]["filename"] == source.name
    assert payload["source"]["directory_stored_in_this_repository"] is False
    assert payload["rejected_predecessor"]["rows"] == 379
    assert payload["adapter"]["relative_path"].endswith(out.name)
    assert payload["adapter"]["committed_to_git"] is False
    assert payload["header_schema"]["renamed_from"] == SOURCE_COLUMN
    assert payload["header_schema"]["renamed_to"] == ADAPTER_COLUMN
    assert payload["preparation_script_sha256"]
    assert payload["bytes_after_header_identical"] is True
    for flag in ("changes_scientific_protocol", "changes_prompts",
                 "changes_row_order", "changes_generation", "changes_scoring"):
        assert payload[flag] is False


def test_the_release_record_declares_the_post_freeze_inspection(source,
                                                                tmp_path):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    inspection = build_release(source, out,
                               Checklist("t"))["post_freeze_inspection"]
    assert inspection["occurred"] is True
    assert inspection["prompt_text_printed"] is False
    assert inspection["prompt_text_semantically_reviewed"] is False
    assert inspection["outcomes_responses_scores_or_correctness_read"] is False
    assert inspection["changed_any_method_condition_or_parameter"] is False


def test_a_tampered_release_record_fails_its_own_digest(source, tmp_path):
    out = tmp_path / "adapter.csv"
    prepare_adapter(source, out, Checklist("t"))
    payload = build_release(source, out, Checklist("t"))
    payload["counts"]["rows"] = 379
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValidationError, match="record_digest"):
        verify_release(path, Checklist("t"))


def test_the_release_name_collides_with_no_committed_manifest():
    existing = {p.name for p in (REPO / "manifests").glob("*.json")}
    assert RELEASE.name == f"{sir.NAME}.json"
    assert RELEASE.name not in (existing - {RELEASE.name})


# ------------------------------------------------------- the v5 manifest
def test_the_v5_manifest_still_verifies_unchanged():
    checks = Checklist("t")
    verify_protocol(PREDECLARATION, checks)
    assert checks.all_passed


def test_this_release_changes_no_v5_generation_source():
    """The runner and its protocol are untouched by everything here."""
    assert "sealedinputrelease" not in (
        REPO / "src/routegate/finaleval.py").read_text(encoding="utf-8")
    assert "sealedinputrelease" not in (
        REPO / "experiments/final_evaluation/run_final_evaluation.py"
    ).read_text(encoding="utf-8")


@pytest.mark.skipif(not RELEASE.is_file(),
                    reason="the release record has not been frozen yet")
def test_the_committed_release_record_verifies():
    checks = Checklist("t")
    verify_release(RELEASE, checks)
    assert checks.all_passed


@pytest.mark.skipif(not RELEASE.is_file(),
                    reason="the release record has not been frozen yet")
def test_the_committed_release_record_carries_no_prompt_or_outcome():
    blob = RELEASE.read_text(encoding="utf-8")
    committed = json.loads(blob)
    assert committed["source"]["sha256"] == REQUIRED_SOURCE_SHA256
    assert committed["rejected_predecessor"]["sha256"] == (
        REJECTED_SOURCE_SHA256)
    assert committed["counts"]["rows"] == EXPECTED_ROWS
    assert committed["counts"]["unique_prompts"] == EXPECTED_ROWS
    # Outcome vocabulary may appear only inside the record's own negative
    # declarations. It must never name a column or a stored value.
    outcome = ("response", "completion", "score", "correct", "verdict",
               "gold", "answer", "label_value")
    schema = committed["header_schema"]
    for column in list(schema["original"]) + list(schema["adapter"]):
        for needle in outcome:
            assert needle not in column.lower(), (
                f"the schema names an outcome column '{column}'")
    assert committed["contains_prompt_response_gold_or_score"] is False
    assert committed["post_freeze_inspection"][
        "outcomes_responses_scores_or_correctness_read"] is False
