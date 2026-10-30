"""The one-pass budget check, on a synthetic model only.

Nothing here loads the frozen 30B model, reads the corpus, generates a token or
touches held-out material. The "model" is a handful of small linear blocks with
the attribute layout ``resolve_modules`` expects, so the properties under test
are properties of the hook ordering and the gates rather than of one checkpoint.

What is under test is the part of this preflight that is easy to get silently
wrong: whether the budget for a block is snapshotted *before* that block's own
edit is written, whether an upstream edit really does reach the block-33
reference through the model rather than through an arithmetic shortcut, and
whether every fail-closed gate actually closes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from routegate.onepassbudget import (                               # noqa: E402
    ACTUATOR_BLOCKS,
    ALPHA_R,
    FLOAT32_REDUCTION_SCALE,
    MATHEMATICALLY_IDENTICAL_BLOCKS,
    MEASURED_BLOCK,
    SAME_FORWARD_INVARIANT_BLOCKS,
    TOLERANCE,
    BudgetEquivalenceError,
    OnePassBudgetRecorder,
    RowBudgets,
    block_input_state,
    block_output_state,
    budget_norm,
    budget_reference_states,
    decide,
    summarise,
    verify_read_before_write,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    SealedSetError,
    ValidationError,
)

RUNNER = (REPO_ROOT / "experiments" / "single_direction_controller"
          / "budget_equivalence_check.py")
HIDDEN = 16
N_BLOCKS = 36


def _runner():
    spec = importlib.util.spec_from_file_location("_budget_check", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_budget_check"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- a tiny model
class _Block(torch.nn.Module):
    """One decoder-block stand-in: a fixed nonlinear map, output first in a tuple.

    Nonlinear on purpose. A linear block would let an upstream edit reach block
    33's reference by pure superposition, and the whole reason block 33 needs a
    real forward pass is that a real block does not behave that way.
    """

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN ** 0.5
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden):                                # noqa: D102
        return (hidden + torch.tanh(hidden @ self.weight),)


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, HIDDEN)
        self.layers = torch.nn.ModuleList(
            [_Block(1000 + i) for i in range(N_BLOCKS)])


class _Model(torch.nn.Module):
    """Enough of a causal LM for ``resolve_modules`` and the hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()

    def forward(self, input_ids, **kwargs):                   # noqa: D102
        hidden = self.model.embed_tokens(input_ids)
        for block in self.model.layers:
            hidden = block(hidden)[0]
        return hidden


@pytest.fixture
def model():
    torch.manual_seed(20260826)
    net = _Model().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


@pytest.fixture
def directions():
    generator = torch.Generator().manual_seed(4242)
    out = {}
    for block in ACTUATOR_BLOCKS:
        vector = torch.randn(HIDDEN, generator=generator)
        out[block] = (vector / torch.linalg.vector_norm(vector)).numpy()
    return out


@pytest.fixture
def tokens():
    return torch.tensor([[1, 2, 3, 4, 5]])


def _unsteered_states(model, tokens):
    """Every block output, unsteered: the two-pass reference."""
    captured = {}
    handles = []
    for index, block in enumerate(model.model.layers):
        def hook(module, args, output, index=index):
            captured[index + 1] = output[0][0, -1].detach().clone()
        handles.append(block.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    return captured


# ================================================== 1. the layer arithmetic
def test_historical_block_maps_to_the_package_state():
    assert [block_output_state(b) for b in (31, 32, 33)] == [32, 33, 34]
    assert [block_input_state(b) for b in (31, 32, 33)] == [31, 32, 33]
    assert budget_reference_states((31, 32, 33)) == (31, 32, 33)
    assert ACTUATOR_BLOCKS == (31, 32, 33)
    assert SAME_FORWARD_INVARIANT_BLOCKS == (31, 32)
    assert MATHEMATICALLY_IDENTICAL_BLOCKS == (31, 32)
    assert MEASURED_BLOCK == 33


def test_a_block_with_no_predecessor_has_no_block_input_budget():
    with pytest.raises(BudgetEquivalenceError, match="no preceding"):
        block_input_state(0)


# =============================================== 2. exactness at 31 and 32
def test_blocks_31_and_32_budgets_are_exactly_the_unsteered_ones(
        model, directions, tokens):
    """The snapshot is taken before the write. Verified, not assumed."""
    reference = _unsteered_states(model, tokens)
    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS,
                                     alpha=ALPHA_R, g_r=1.0)
    try:
        recorder.arm()
        with torch.inference_mode():
            model(tokens)
        recorder.disarm()
    finally:
        recorder.close()

    for block in MATHEMATICALLY_IDENTICAL_BLOCKS:
        expected = budget_norm(reference[block_input_state(block)])
        # Same forward, same device, same dtype: here it IS bit-identical.
        # Across two GPU forwards it is not, which is what v1 got wrong.
        assert recorder.one_pass[block] == expected, block


def test_block_33_reference_differs_because_the_edit_propagated(
        model, directions, tokens):
    """The whole reason this preflight needs a real forward pass."""
    reference = _unsteered_states(model, tokens)
    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS,
                                     alpha=ALPHA_R, g_r=1.0)
    try:
        recorder.arm()
        with torch.inference_mode():
            model(tokens)
        recorder.disarm()
    finally:
        recorder.close()

    unsteered = budget_norm(reference[block_input_state(MEASURED_BLOCK)])
    assert recorder.one_pass[MEASURED_BLOCK] != unsteered


def test_block_33_uses_the_actual_downstream_activation_not_the_old_estimate(
        model, directions, tokens):
    """The superseded shortcut added the deltas onto the cached residual and
    took a norm. That is a different number, and the check must not be it."""
    reference = _unsteered_states(model, tokens)
    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS,
                                     alpha=ALPHA_R, g_r=1.0)
    try:
        recorder.arm()
        with torch.inference_mode():
            model(tokens)
        recorder.disarm()
    finally:
        recorder.close()

    h33 = reference[block_input_state(MEASURED_BLOCK)].double()
    approximation = h33.clone()
    for block in (31, 32):
        r = budget_norm(reference[block_input_state(block)])
        approximation = approximation + ALPHA_R * r * torch.as_tensor(
            directions[block], dtype=torch.float64)
    residual_only = float(torch.linalg.vector_norm(approximation))
    measured = recorder.one_pass[MEASURED_BLOCK]

    # The measured value comes through block 32's nonlinearity; the old
    # estimate never passed through it. They must not coincide.
    assert abs(measured - residual_only) > 1e-9


def test_the_edit_actually_fired_at_every_actuator_block(
        model, directions, tokens):
    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS,
                                     alpha=ALPHA_R, g_r=1.0)
    try:
        recorder.arm()
        with torch.inference_mode():
            model(tokens)
        recorder.disarm()
    finally:
        recorder.close()
    assert sorted(recorder.applied) == list(ACTUATOR_BLOCKS)
    for block in ACTUATOR_BLOCKS:
        expected = ALPHA_R * recorder.one_pass[block]
        assert recorder.applied[block] == pytest.approx(expected, rel=1e-5)


def test_a_disarmed_recorder_changes_nothing(model, directions, tokens):
    """The reference pass must see the model exactly as it is."""
    with torch.inference_mode():
        plain = model(tokens).clone()
    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS)
    try:
        with torch.inference_mode():
            while_installed = model(tokens).clone()
        recorder.arm()
        with torch.inference_mode():
            while_armed = model(tokens).clone()
        recorder.disarm()
    finally:
        recorder.close()
    assert torch.equal(plain, while_installed)
    assert not torch.equal(plain, while_armed)


# ============================================ 3. prefill only / no decoding
def test_a_forward_beyond_the_prefill_is_counted_and_never_edited(
        model, directions, tokens):
    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS)
    try:
        recorder.arm()
        with torch.inference_mode():
            model(tokens)                       # the prefill
            model(tokens)                       # a second forward: a "decode"
        recorder.disarm()
    finally:
        recorder.close()
    assert recorder.decode_forwards > 0
    assert sorted(recorder.applied) == list(ACTUATOR_BLOCKS)


def test_arming_resets_the_previous_rows_state(model, directions, tokens):
    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS)
    try:
        recorder.arm()
        with torch.inference_mode():
            model(tokens)
        first = dict(recorder.one_pass)
        recorder.arm()
        assert recorder.one_pass == {} and recorder.decode_forwards == 0
        with torch.inference_mode():
            model(tokens)
        recorder.disarm()
        assert recorder.one_pass == first
    finally:
        recorder.close()


def test_the_runner_never_asks_for_a_generated_token():
    source = RUNNER.read_text(encoding="utf-8")
    for banned in ("model.generate", "generate_response", "max_new_tokens=",
                   "do_sample=True"):
        assert banned not in source, banned
    assert "use_cache=False" in source


# ======================================================== 4. the summaries
def _rows(values, block=MEASURED_BLOCK, reference=10.0):
    out = []
    for index, value in enumerate(values):
        out.append(RowBudgets(row_id=f"row:{index}",
                              two_pass={block: reference},
                              one_pass={block: reference * (1.0 + value)}))
    return out


def test_an_exactly_equal_set_passes():
    rows = _rows([0.0, 0.0, 0.0])
    summary = summarise(rows, MEASURED_BLOCK)
    assert summary["max"] == 0.0 and summary["within_tolerance"] is True
    assert summary["n_exceeding_tolerance"] == 0
    assert decide({MEASURED_BLOCK: summary})["decision"] == "PASS"


def test_a_nonzero_difference_inside_the_tolerance_still_passes():
    rows = _rows([0.0, 0.005, 0.009999])
    summary = summarise(rows, MEASURED_BLOCK)
    assert 0.0 < summary["max"] <= TOLERANCE
    assert summary["within_tolerance"] is True
    assert decide({MEASURED_BLOCK: summary})["decision"] == "PASS"


def test_a_difference_over_the_tolerance_fails_and_names_the_rows():
    rows = _rows([0.0, 0.02, 0.5])
    summary = summarise(rows, MEASURED_BLOCK)
    assert summary["within_tolerance"] is False
    assert summary["n_exceeding_tolerance"] == 2
    assert summary["row_ids_exceeding_tolerance"] == ["row:1", "row:2"]
    verdict = decide({MEASURED_BLOCK: summary})
    assert verdict["decision"] == "BLOCKED"
    assert verdict["blocks_exceeding_tolerance"] == [MEASURED_BLOCK]
    assert "not redefined" in verdict["statement"]


def test_the_summary_reports_every_requested_statistic():
    summary = summarise(_rows([0.0, 0.001, 0.002, 0.003, 0.02]),
                        MEASURED_BLOCK)
    for field in ("max", "mean", "median", "p95", "p99",
                  "n_exceeding_tolerance", "row_ids_exceeding_tolerance"):
        assert field in summary, field


def test_the_same_forward_invariant_holds_when_the_read_precedes_the_write():
    rows = _rows([0.0, 0.0], block=31)
    for row in rows:
        row.read_before_write = {32: 3.3, 33: 3.3}
    evidence = verify_read_before_write(rows, (31,))
    assert evidence[31]["successor"] == 32
    assert evidence[31]["min_separation"] > 0.0


def test_a_collapsed_separation_is_a_read_after_write_and_is_refused():
    rows = _rows([0.0, 0.0], block=31)
    rows[0].read_before_write = {32: 3.3}
    rows[1].read_before_write = {32: 0.0}          # the read moved
    with pytest.raises(ValidationError, match="read_before_the_activation"):
        verify_read_before_write(rows, (31,))


def test_the_invariant_cannot_be_waived_by_recording_no_evidence():
    rows = _rows([0.0, 0.0], block=31)
    rows[0].read_before_write = {32: 3.3}          # rows[1] records nothing
    with pytest.raises(ValidationError, match="evidence_exists"):
        verify_read_before_write(rows, (31,))


def test_a_zero_reference_budget_is_refused_rather_than_divided_by():
    row = RowBudgets(row_id="r", two_pass={31: 0.0}, one_pass={31: 0.0})
    with pytest.raises(BudgetEquivalenceError, match="undefined"):
        row.relative(31)


def test_summarising_nothing_is_refused():
    with pytest.raises(BudgetEquivalenceError, match="not a result"):
        summarise([], MEASURED_BLOCK)


# ======================================================= 5. the input gates
def test_the_expected_digest_is_read_from_the_committed_files():
    module = _runner()
    digest = module.default_bank_sha256()
    config = json.loads(
        (REPO_ROOT / "configs" / "dual_direction_controller.json")
        .read_text(encoding="utf-8"))
    assert digest == config["causal_screen"]["direction_bank"][
        "artifact_sha256"]
    assert len(digest) == 64


def test_a_wrong_direction_digest_is_refused(tmp_path):
    from routegate.dualsteer import load_validated_bank
    bank = Path("/home/ucabh21/routegate-data/frozen/validated_direction_bank")
    if not bank.is_dir():
        pytest.skip("the validated direction bank is not reachable here")
    with pytest.raises(ValidationError, match="refused, not adopted"):
        load_validated_bank(bank, "0" * 64, Checklist("t"))


def test_the_orthogonalised_population_bank_is_refused_as_a_substitute():
    from routegate.dualsteer import DualSteerError, load_validated_bank
    module = _runner()
    population = (REPO_ROOT / "outputs" / "causal_repair"
                  / "orthogonalized_population_scan" / "direction_bank")
    if not population.is_dir():
        pytest.skip("the population bank is not present here")
    with pytest.raises(DualSteerError):
        load_validated_bank(population, module.default_bank_sha256(),
                            Checklist("t"))


def test_a_sealed_output_directory_is_refused(tmp_path):
    module = _runner()
    with pytest.raises(SealedSetError):
        module.check_output_dir(tmp_path / "sealed_budget_run")


def test_a_sealed_row_id_is_refused():
    from routegate.validation import looks_sealed
    assert looks_sealed("sealed_eval:0001") is not None
    assert looks_sealed("false_reject_test:00005:23e9d7db72f9") is None


def test_a_non_empty_output_directory_is_refused(tmp_path):
    module = _runner()
    out = tmp_path / "run"
    out.mkdir()
    assert module.check_output_dir(out) == out          # empty is fine
    (out / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="already holds"):
        module.check_output_dir(out)


def test_a_dry_run_does_not_occupy_the_output_directory(tmp_path):
    """A dry run that wrote here would make the real run it clears the way
    for impossible."""
    module = _runner()
    out = tmp_path / "run"
    module.check_output_dir(out)
    assert not out.exists()


def test_a_missing_or_reordered_row_is_refused(tmp_path):
    module = _runner()
    fold = tmp_path / "bank"
    fold.mkdir()
    ids = [f"row:{i}" for i in range(4)]
    (fold / "fold_assignment.csv").write_text(
        "array_row_index,row_id,final_ebh,fold\n"
        + "".join(f"{i},{r},E,0\n" for i, r in enumerate(ids)),
        encoding="utf-8")

    complete = [{"row_id": r} for r in ids]
    module.assert_row_order(complete, fold, Checklist("t"))

    with pytest.raises(ValidationError, match="missing_or_added"):
        module.assert_row_order(complete[:-1], fold, Checklist("t"))

    reordered = [complete[1], complete[0]] + complete[2:]
    with pytest.raises(ValidationError, match="reordered"):
        module.assert_row_order(reordered, fold, Checklist("t"))


def test_the_frozen_tolerance_cannot_be_relaxed_on_the_command_line():
    module = _runner()
    args = module.build_parser().parse_args([
        "--direction-bank", "/nonexistent", "--output-dir", "/tmp/nonexistent",
        "--tolerance", "0.5"])
    with pytest.raises(ValidationError, match="frozen tolerance"):
        module.run(args)


def test_the_token_position_cannot_be_moved():
    module = _runner()
    args = module.build_parser().parse_args([
        "--direction-bank", "/nonexistent", "--output-dir", "/tmp/nonexistent",
        "--token-position", "last generated token"])
    with pytest.raises(ValidationError, match="final prompt token"):
        module.run(args)


def test_the_runner_exposes_the_required_options():
    module = _runner()
    options = [option for action in module.build_parser()._actions
               for option in action.option_strings]
    for required in ("--direction-bank", "--output-dir", "--dry-run",
                     "--tolerance", "--g-r", "--blocks", "--fold-dir"):
        assert required in options, required
    from routegate.validation import looks_sealed
    for option in options:
        assert looks_sealed(option) is None, option


def test_the_recorder_refuses_a_gate_outside_its_range(model, directions):
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(BudgetEquivalenceError, match="gate"):
            OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS, g_r=bad)


def test_the_recorder_refuses_a_missing_direction(model, directions):
    partial = {31: directions[31]}
    with pytest.raises(BudgetEquivalenceError, match="no direction"):
        OnePassBudgetRecorder(model, partial, ACTUATOR_BLOCKS)


# ==================================================== 6. deterministic output
def test_the_per_row_record_is_deterministic_and_hashable(model, directions,
                                                          tokens):
    import hashlib

    def once():
        recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS,
                                         alpha=ALPHA_R, g_r=1.0)
        try:
            recorder.arm()
            with torch.inference_mode():
                model(tokens)
            recorder.disarm()
            reference = _unsteered_states(model, tokens)
            return RowBudgets(
                row_id="row:0",
                two_pass={b: budget_norm(reference[block_input_state(b)])
                          for b in ACTUATOR_BLOCKS},
                one_pass=dict(recorder.one_pass),
                one_pass_runtime_input=dict(recorder.runtime_input),
                prefill_tokens=recorder.prefill_tokens).as_dict()
        finally:
            recorder.close()

    first, second = once(), once()
    assert first == second
    digest = lambda d: hashlib.sha256(
        json.dumps(d, sort_keys=True).encode()).hexdigest()
    assert digest(first) == digest(second)
    for block in ACTUATOR_BLOCKS:
        assert f"rel_diff_block_{block}" in first


# ================================================ 7. the written artifacts
class _FakeBank:
    manifest_sha256 = "b" * 64


class _FakeGeneration:
    def fingerprint(self):
        return "c" * 64


def _report_args(tmp_path, tolerance=TOLERANCE):
    import argparse
    return argparse.Namespace(
        direction_bank=tmp_path, tolerance=tolerance, g_r=1.0,
        token_position="final prompt token")


def _measured(values):
    """One RowBudgets per value, exact at 31/32 and `value` off at 33."""
    rows = []
    for index, value in enumerate(values):
        rows.append(RowBudgets(
            row_id=f"row:{index}",
            two_pass={31: 10.0, 32: 20.0, 33: 30.0},
            one_pass={31: 10.0, 32: 20.0, 33: 30.0 * (1.0 + value)},
            one_pass_runtime_input={33: 30.0 * (1.0 + value * 2)},
            read_before_write={32: 1.5, 33: 3.0},
            prefill_tokens=7))
    return rows


def _run_report(tmp_path, values, rows_expected=None):
    module = _runner()
    if rows_expected is not None:
        module.EXPECTED_ROWS = rows_expected
    out = tmp_path / "run"
    out.mkdir()
    code = module._report(
        _measured(values), ACTUATOR_BLOCKS, _report_args(tmp_path),
        Checklist("t"), out, {"dataset_sha256": "a" * 64},
        _FakeBank(), _FakeGeneration(), [1e-7])
    return module, out, code


def test_a_passing_report_writes_every_required_artifact(tmp_path):
    module, out, code = _run_report(tmp_path, [0.0, 0.001], rows_expected=2)
    assert code == 0
    for name in (module.OUTPUT_ROWS, module.OUTPUT_INPUT_HASHES,
                 module.OUTPUT_MANIFEST, module.OUTPUT_SUMMARY,
                 module.OUTPUT_HASHES):
        assert (out / name).is_file(), name

    summary = json.loads((out / module.OUTPUT_SUMMARY).read_text())
    assert summary["decision"]["decision"] == "PASS"
    assert summary["generated_tokens"] == 0
    invariant = summary["same_forward_read_before_write"]
    assert invariant["holds"] is True and invariant["error"] is None
    assert invariant["blocks"] == [31, 32]

    manifest = json.loads((out / module.OUTPUT_MANIFEST).read_text())
    assert manifest["generates_tokens"] is False
    assert manifest["trains_a_controller"] is False
    assert manifest["judges_responses"] is False
    assert manifest["alpha_S"] == 0.0
    assert manifest["alpha_R"] == ALPHA_R and manifest["g_R"] == 1.0
    assert manifest["states_edited"] == [32, 33, 34]
    assert manifest["environment"]["generation_performed"] is False


def test_the_recorded_output_hashes_are_the_files_own_digests(tmp_path):
    import hashlib
    module, out, _ = _run_report(tmp_path, [0.0, 0.002], rows_expected=2)
    recorded = json.loads((out / module.OUTPUT_HASHES).read_text())
    for name, digest in recorded.items():
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert actual == digest, name


def test_a_failing_block_33_exits_non_zero_and_names_the_rows(tmp_path):
    module, out, code = _run_report(tmp_path, [0.0, 0.2], rows_expected=2)
    assert code == 1
    summary = json.loads((out / module.OUTPUT_SUMMARY).read_text())
    assert summary["decision"]["decision"] == "BLOCKED"
    assert summary["decision"]["blocks_exceeding_tolerance"] == [33]
    assert summary["per_block"]["33"]["row_ids_exceeding_tolerance"] == ["row:1"]
    # blocks 31 and 32 still verified exact even on a failing run
    assert summary["per_block"]["31"]["max"] == 0.0
    assert summary["per_block"]["32"]["max"] == 0.0


def test_a_short_run_is_refused_but_its_evidence_is_still_written(tmp_path):
    """479 of 480 rows is not a smaller result; it is not a result.

    It is refused with a non-zero exit -- but the artifacts are written first.
    v1 raised at this point and discarded 960 completed forward passes, which
    is how a run that had already answered the question left no answer.
    """
    module, out, code = _run_report(tmp_path, [0.0, 0.0])   # EXPECTED_ROWS 480
    assert code == 2
    for name in (module.OUTPUT_ROWS, module.OUTPUT_MANIFEST,
                 module.OUTPUT_SUMMARY, module.OUTPUT_HASHES):
        assert (out / name).is_file(), name
    manifest = json.loads((out / module.OUTPUT_MANIFEST).read_text())
    failed = [c["check"] for c in manifest["checklist"]["checks"]
              if c["status"] != "PASS"]
    assert "every_row_measured" in failed


def test_a_refused_run_records_why_it_was_refused(tmp_path):
    """The checklist in the written manifest carries the failure, so the
    evidence and the reason survive together."""
    module, out, code = _run_report(tmp_path, [0.0, 0.0])
    assert code == 2
    manifest = json.loads((out / module.OUTPUT_MANIFEST).read_text())
    assert manifest["checklist"]["all_passed"] is False
    summary = json.loads((out / module.OUTPUT_SUMMARY).read_text())
    assert summary["rows"] == 2                 # the evidence it did collect


# ======================= 8. the four cases the v1 failure conflated
#
# v1 demanded that blocks 31 and 32 be bit-identical across two separate
# forward passes and refused a complete 480-prompt run over 3.527095e-07.
# These four tests hold apart what that single gate ran together.


def test_case_1_same_forward_hook_order_identity_is_exact(model, directions,
                                                          tokens):
    """Within one pass, on one device, the budget IS the unsteered value --
    bit for bit, and the separation evidence proves the ordering."""
    reference = _unsteered_states(model, tokens)
    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS,
                                     alpha=ALPHA_R, g_r=1.0)
    try:
        recorder.arm()
        with torch.inference_mode():
            model(tokens)
        recorder.disarm()
    finally:
        recorder.close()

    for block in MATHEMATICALLY_IDENTICAL_BLOCKS:
        assert recorder.one_pass[block] == budget_norm(
            reference[block_input_state(block)])
    # and the separation is the applied delta, not zero
    for block in SAME_FORWARD_INVARIANT_BLOCKS:
        successor = block + 1
        assert recorder.read_before_write[successor] > 0.0
        assert recorder.read_before_write[successor] == pytest.approx(
            recorder.applied[block], rel=1e-5)


def test_case_2_harmless_cross_forward_numerical_variation_passes():
    """Two float32 reductions of the same vector in different orders. This is
    what v1 refused; it must now pass, on the predeclared tolerance."""
    observed = 3.527095e-07                     # the real v1 block-31 maximum
    assert observed < FLOAT32_REDUCTION_SCALE   # rounding, not a difference
    rows = _rows([0.0, observed], block=31)
    for row in rows:
        row.read_before_write = {32: 3.3}

    summary = summarise(rows, 31)
    assert summary["max"] == pytest.approx(observed)
    assert summary["within_tolerance"] is True
    assert summary["n_exceeding_tolerance"] == 0
    assert decide({31: summary})["decision"] == "PASS"
    # and the same-forward invariant is untouched by it
    assert verify_read_before_write(rows, (31,))[31]["min_separation"] > 0.0


def test_case_3_a_real_pre_read_mutation_is_caught(model, directions, tokens):
    """A recorder that snapshots AFTER its own edit. The separation collapses
    to zero and the invariant refuses it -- on the same synthetic model, in
    one forward pass, with no tolerance involved."""

    class _ReadAfterWrite(OnePassBudgetRecorder):
        def _on_block_output(self, block, output):
            out = super()._on_block_output(block, output)
            successor = block + 1
            if successor in self.blocks and block in self.applied:
                # re-snapshot from the EDITED tensor: the bug being guarded
                tensor = out[0] if isinstance(out, tuple) else out
                position = tensor.shape[1] - 1
                edited = tensor[0, position].detach().clone()
                self.reference[successor] = edited
                self.one_pass[successor] = budget_norm(edited)
                self.read_before_write[successor] = 0.0
            return out

    recorder = _ReadAfterWrite(model, directions, ACTUATOR_BLOCKS,
                               alpha=ALPHA_R, g_r=1.0)
    try:
        recorder.arm()
        with torch.inference_mode():
            model(tokens)
        recorder.disarm()
    finally:
        recorder.close()

    row = RowBudgets(row_id="row:0", two_pass={31: 1.0, 32: 1.0},
                     one_pass={31: 1.0, 32: 1.0},
                     read_before_write=dict(recorder.read_before_write))
    assert row.read_before_write[32] == 0.0
    with pytest.raises(ValidationError, match="read_before_the_activation"):
        verify_read_before_write([row], (31,))


def test_case_3b_the_recorder_refuses_a_zero_separation_in_the_pass_itself(
        model, directions, tokens):
    """The invariant also fires inside the hook, not only at report time."""
    from routegate import onepassbudget

    recorder = OnePassBudgetRecorder(model, directions, ACTUATOR_BLOCKS,
                                     alpha=ALPHA_R, g_r=1.0)
    original = onepassbudget.vector_norm
    try:
        recorder.arm()
        # force the separation to read as zero, as a read-after-write would
        onepassbudget.vector_norm = lambda v: 0.0 if v.dim() == 1 and bool(
            (v == 0).all()) else original(v)
        # simpler and more faithful: make the edit a no-op so post == snapshot
        recorder.directions = {b: np.zeros_like(np.asarray(directions[b]))
                               for b in ACTUATOR_BLOCKS}
        with pytest.raises(ValidationError):
            with torch.inference_mode():
                model(tokens)
    finally:
        onepassbudget.vector_norm = original
        recorder.close()


def test_case_4_a_block_33_tolerance_failure_still_blocks():
    """The genuine empirical gate is untouched by any of the above."""
    rows = _rows([0.0, 0.05], block=MEASURED_BLOCK)
    for row in rows:
        row.read_before_write = {32: 3.3, 33: 3.3}
    summary = summarise(rows, MEASURED_BLOCK)
    assert summary["within_tolerance"] is False
    verdict = decide({MEASURED_BLOCK: summary})
    assert verdict["decision"] == "BLOCKED"
    # the invariant holding does not rescue a real tolerance failure
    assert verify_read_before_write(rows, (32,))[32]["min_separation"] > 0.0


def test_the_report_carries_the_runtime_input_variant_beside_the_gate(tmp_path):
    module, out, _ = _run_report(tmp_path, [0.0, 0.001], rows_expected=2)
    summary = json.loads((out / module.OUTPUT_SUMMARY).read_text())
    variant = summary["block_33_runtime_input_variant"]
    assert variant["measured"] is True and variant["variant"] == "runtime_input"
    # strictly worse than the declared convention, and reported as such
    assert variant["max"] >= summary["per_block"]["33"]["max"]
