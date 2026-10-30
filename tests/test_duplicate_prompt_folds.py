"""The duplicate-prompt/fold diagnostic, on synthetic rows only.

The corpus currently has no duplicate prompt text at all, so the branch that
matters most -- a group spanning two folds, which would mean every cross-fitted
artifact was fitted on one copy and scored on the other -- is the one the real
data never exercises. It is exercised here.

Also under test: that the diagnostic is read-only. It reports and it never
alters a fold assignment, because what to do about a crossing duplicate has
consequences for every artifact already built on that division, and that is not
a decision a diagnostic gets to take.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.validation import ValidationError                    # noqa: E402

TOOL = REPO_ROOT / "tools" / "diagnostics" / "duplicate_prompt_folds.py"


def _tool():
    spec = importlib.util.spec_from_file_location("_duplicate_prompt_folds",
                                                  TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_duplicate_prompt_folds"] = module
    spec.loader.exec_module(module)
    return module


D = _tool()


def _row(row_id, text, label="E", digest=None):
    return {"row_id": row_id, "prompt_text": text, "final_ebh": label,
            "prompt_sha256": digest or (row_id * 8)[:64].ljust(64, "0")}


# ========================================================== the normalisation
def test_the_normalisation_is_the_declared_one():
    assert D.normalise("  Hello   World  ") == "hello world"
    assert D.normalise("HELLO\tWORLD") == "hello world"
    assert D.normalise("hello\nworld") == "hello world"
    assert D.normalise("hello world") == "hello world"


def test_normalisation_folds_compatibility_forms():
    """NFKC, so a full-width or ligature form is not a different prompt."""
    assert D.normalise("ﬁle") == D.normalise("file")
    assert D.normalise("Ｈｅｌｌｏ") == "hello"


def test_distinct_prompts_stay_distinct():
    assert D.normalise("answer this") != D.normalise("refuse this")


# ================================================================= the groups
def test_a_corpus_with_no_duplicates_reports_none():
    rows = [_row("a", "one prompt"), _row("b", "another prompt")]
    assert D.group_duplicates(rows, {"a": 0, "b": 1}) == []


def test_a_duplicate_inside_one_fold_is_reported_and_is_not_a_leak():
    rows = [_row("a", "same text"), _row("b", "same text")]
    groups = D.group_duplicates(rows, {"a": 3, "b": 3})
    assert len(groups) == 1
    assert groups[0]["rows"] == 2
    assert groups[0]["folds"] == [3]
    assert groups[0]["crosses_folds"] is False


def test_a_duplicate_across_two_folds_is_flagged_as_crossing():
    rows = [_row("a", "same text"), _row("b", "same text")]
    groups = D.group_duplicates(rows, {"a": 0, "b": 4})
    assert len(groups) == 1
    assert groups[0]["folds"] == [0, 4]
    assert groups[0]["crosses_folds"] is True
    assert groups[0]["row_ids"] == ["a", "b"]


def test_a_normalised_only_collision_is_reported_as_such():
    """Same words, different spacing: one group, two prompt digests."""
    rows = [_row("a", "Same Text", digest="1" * 64),
            _row("b", "same   text", digest="2" * 64)]
    groups = D.group_duplicates(rows, {"a": 0, "b": 1})
    assert groups[0]["crosses_folds"] is True
    assert groups[0]["byte_identical"] is False
    assert groups[0]["distinct_prompt_digests"] == 2


def test_a_byte_identical_duplicate_says_so():
    rows = [_row("a", "same text", digest="1" * 64),
            _row("b", "same text", digest="1" * 64)]
    groups = D.group_duplicates(rows, {"a": 0, "b": 1})
    assert groups[0]["byte_identical"] is True


def test_a_duplicate_spanning_two_classes_keeps_both_labels():
    rows = [_row("a", "same text", label="E"),
            _row("b", "same text", label="H")]
    groups = D.group_duplicates(rows, {"a": 0, "b": 1})
    assert groups[0]["labels"] == ["E", "H"]


def test_a_row_with_no_fold_is_refused_rather_than_assumed_safe():
    rows = [_row("a", "text"), _row("b", "text")]
    with pytest.raises(ValidationError, match="no fold"):
        D.group_duplicates(rows, {"a": 0})


# ================================================================ read-only
def test_the_tool_writes_no_fold_assignment():
    """It reports. It never re-splits, and it never edits the bank."""
    source = TOOL.read_text(encoding="utf-8")
    for banned in ("write_direction_bank", "fold_assignment.csv\", \"w",
                   "StratifiedKFold", "assign_withheld_folds",
                   "build_ebh_fold_assignment"):
        assert banned not in source, banned


def test_the_tool_loads_no_model_and_trains_nothing():
    source = TOOL.read_text(encoding="utf-8")
    for banned in ("torch", "sklearn", "load_model_and_tokenizer", "generate("):
        assert banned not in source, banned


def test_the_only_write_the_tool_offers_is_its_own_report():
    """``--json`` writes the report. Nothing else in the tool writes."""
    source = TOOL.read_text(encoding="utf-8")
    writes = [line for line in source.splitlines()
              if ".write_text(" in line or "open(" in line]
    assert len(writes) == 1, writes
    assert "args.json.write_text" in writes[0]


def test_the_tool_names_nothing_outside_development_data():
    from routegate.validation import looks_sealed
    source = TOOL.read_text(encoding="utf-8")
    for line in source.splitlines():
        assert looks_sealed(line) is None, line


def test_the_report_declares_that_it_altered_nothing():
    """A diagnostic that could have re-split says, in its own output, that it
    did not."""
    source = TOOL.read_text(encoding="utf-8")
    assert '"folds_were_not_altered": True' in source
    assert '"read_only": True' in source
