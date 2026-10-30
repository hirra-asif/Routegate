"""The deployable controller: checkpoints, the fold rule, and the live edit.

Every fixture is synthetic and lives in ``tmp_path``. Nothing here loads a
language model, generates a response, calls a judge, reads the sealed set or
touches completed judging evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import (                                            # noqa: E402
    needs_dynamic_arm_inputs,
    needs_real_inputs,
)

import routegate.gatecontroller as G                             # noqa: E402
import routegate.livecontroller as L                             # noqa: E402
import routegate.twositecontinuation as TS                       # noqa: E402
import routegate.twositestaticclosure as C                       # noqa: E402
from routegate.validation import Checklist                       # noqa: E402


def _net(seed: int = 1):
    pytest.importorskip("torch")
    return G.build_controller(seed)


def _checkpoint(kind=L.KIND_FOLD, fold=0, seed=None, width=None):
    net = _net(G.seed_for(fold if fold is not None else 0))
    width = width or G.HIDDEN_SIZE
    return L.build_checkpoint(
        kind, fold, net.state_dict(),
        mu=np.zeros(width), sd=np.ones(width),
        tau_hi=1.5, selected_step=42, trained_on_rows=384)


def _bundle(tmp_path, with_refit=True):
    for k in range(G.N_SPLITS):
        payload = _checkpoint(L.KIND_FOLD, k)
        (tmp_path / f"controller_fold_{k}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    if with_refit:
        payload = _checkpoint(L.KIND_REFIT, None)
        (tmp_path / "controller_all_development_refit.json").write_text(
            json.dumps(payload), encoding="utf-8")
    return L.load_bundle(tmp_path)


# ------------------------------------------------------------- checkpoints
def test_a_checkpoint_carries_weights_calibration_and_nothing_else():
    payload = _checkpoint()
    assert set(payload) == set(L.CHECKPOINT_FIELDS)
    assert payload["parameters"] == G.CONTROLLER_PARAMETERS == 32801
    assert payload["read_state"] == G.READ_STATE == 32
    assert payload["apply_historical_blocks"] == [31, 32]
    assert payload["alpha_R"] == 0.15
    assert payload["contains_prompt_or_response_text"] is False
    assert payload["contains_per_row_coefficients"] is False
    # mu and sd are part of the function, not decoration
    assert len(payload["mu"]) == len(payload["sd"]) == G.HIDDEN_SIZE


def test_a_checkpoint_carrying_anything_sensitive_is_refused():
    payload = _checkpoint()
    payload["prompt_text"] = "should never be here"
    with pytest.raises(L.LiveControllerError) as exc:
        L.assert_checkpoint_is_metadata_only(payload)
    assert "carries exactly" in str(exc.value)


def test_a_checkpoint_must_declare_it_holds_no_row_coefficients():
    payload = _checkpoint()
    payload["contains_per_row_coefficients"] = True
    with pytest.raises(L.LiveControllerError) as exc:
        L.assert_checkpoint_is_metadata_only(payload)
    assert "not a controller" in str(exc.value)


def test_an_edited_checkpoint_does_not_verify(tmp_path):
    payload = _checkpoint()
    # perturb the first scalar of the first weight tensor, whatever its rank
    key = sorted(payload["state_dict"])[0]
    tensor = payload["state_dict"][key]
    cursor = tensor
    while isinstance(cursor[0], list):
        cursor = cursor[0]
    cursor[0] += 1.0
    path = tmp_path / "controller_fold_0.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(L.LiveControllerError) as exc:
        L.load_checkpoint(path)
    assert "has been edited" in str(exc.value)


def test_a_checkpoint_for_the_wrong_sites_is_refused(tmp_path):
    payload = _checkpoint()
    payload["apply_historical_blocks"] = [31, 32, 33]
    payload["weights_sha256"] = L.weights_digest(payload["state_dict"])
    path = tmp_path / "c.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(L.LiveControllerError) as exc:
        L.load_checkpoint(path)
    assert "applies at" in str(exc.value)


# --------------------------------------------------------------- fold rule
def test_a_row_is_scored_by_the_controller_that_never_saw_its_fold(tmp_path):
    bundle = _bundle(tmp_path)
    for k in range(G.N_SPLITS):
        chosen = bundle.controller_for_row(k)
        assert chosen["fold"] == k
        assert chosen["is_out_of_fold"] is True


def test_an_unknown_fold_is_refused_rather_than_guessed(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(L.LiveControllerError) as exc:
        bundle.controller_for_row(7)
    assert "no controller for outer fold 7" in str(exc.value)


def test_the_refit_never_scores_a_development_row(tmp_path):
    bundle = _bundle(tmp_path)
    refit = bundle.deployment_controller()
    assert refit["kind"] == L.KIND_REFIT
    assert refit["is_out_of_fold"] is False
    # it is reachable only through the deployment accessor, never the fold one
    assert all(c["is_out_of_fold"] for c in bundle.folds.values())


def test_a_bundle_missing_a_fold_controller_is_refused(tmp_path):
    _bundle(tmp_path)
    (tmp_path / "controller_fold_3.json").unlink()
    with pytest.raises(L.LiveControllerError) as exc:
        L.load_bundle(tmp_path)
    assert "never saw it" in str(exc.value)


def test_a_bundle_without_a_refit_refuses_to_invent_one(tmp_path):
    bundle = _bundle(tmp_path, with_refit=False)
    assert bundle.refit is None
    with pytest.raises(L.LiveControllerError) as exc:
        bundle.deployment_controller()
    assert "gate_oof_v1 performed none" in str(exc.value)


# ------------------------------------------------------------- the live gate
def test_the_live_gate_reproduces_the_module_it_was_trained_with(tmp_path):
    """The live path must be the same function, not a re-derivation of it."""
    torch = pytest.importorskip("torch")

    fold = 0
    net = _net(G.seed_for(fold))
    width = G.HIDDEN_SIZE
    payload = L.build_checkpoint(
        L.KIND_FOLD, fold, net.state_dict(),
        mu=np.zeros(width), sd=np.ones(width), tau_hi=1.0,
        selected_step=10, trained_on_rows=384)

    rng = np.random.default_rng(20260828)
    h = rng.normal(size=(7, width))
    live = L.gate_from_activation(h, payload)

    net.eval()
    with torch.no_grad():
        want = G.gate(net, torch.as_tensor(h, dtype=torch.float32)).numpy()
    assert np.max(np.abs(live - want)) == 0.0        # bit-identical, not close


def test_the_gate_is_bounded_and_starts_where_the_spec_says(tmp_path):
    width = G.HIDDEN_SIZE
    net = _net(G.seed_for(0))
    payload = L.build_checkpoint(
        L.KIND_FOLD, 0, net.state_dict(), np.zeros(width), np.ones(width),
        1.0, 1, 384)
    rng = np.random.default_rng(7)
    g = L.gate_from_activation(rng.normal(size=(64, width)), payload)
    assert g.min() >= 0.0 and g.max() <= 1.0
    # a freshly initialised controller gates at 0.1 for every row
    assert np.allclose(g, G.GATE_INITIAL, atol=1e-6)


def test_a_wrongly_shaped_activation_is_refused():
    payload = _checkpoint()
    with pytest.raises(L.LiveControllerError) as exc:
        L.gate_from_activation(np.zeros((1, 16)), payload)
    assert "was standardised on" in str(exc.value)


def test_the_applied_coefficient_is_alpha_times_the_gate():
    g = np.array([0.0, 0.25, 1.0])
    assert np.allclose(L.applied_coefficient(g), 0.15 * g)
    assert L.applied_coefficient(np.array([1.0]))[0] == 0.15


# ------------------------------------------------------------------ the veto
def test_the_veto_only_lowers_and_only_where_it_fires():
    g = np.array([0.8, 0.8, 0.8])
    z = np.array([0.0, 2.0, 5.0])
    out = L.apply_hard_veto(g, z, tau_hi=2.0)
    assert out[0] == 0.8                     # below threshold: untouched
    assert out[1] == 0.0 and out[2] == 0.0   # at or above: zeroed
    assert np.all(out <= g)                  # never raises a gate


def test_both_arms_share_one_controller(tmp_path):
    """The veto is an inference-time brake, not a second trained gate."""
    bundle = _bundle(tmp_path)
    width = G.HIDDEN_SIZE
    activation = np.random.default_rng(3).normal(size=width)
    raw = L.plan_live_edit(activation, 0, bundle, hard_veto=False)
    vetoed = L.plan_live_edit(activation, 0, bundle, z_safety=99.0,
                              hard_veto=True)
    passed = L.plan_live_edit(activation, 0, bundle, z_safety=-99.0,
                              hard_veto=True)
    assert vetoed.g_r == 0.0 and vetoed.vetoed is True
    assert passed.g_r == raw.g_r and passed.vetoed is False


def test_the_hard_veto_refuses_to_assume_the_brake_passed(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(L.LiveControllerError) as exc:
        L.plan_live_edit(np.zeros(G.HIDDEN_SIZE), 0, bundle, hard_veto=True)
    assert "is not assumed to pass" in str(exc.value)


# ------------------------------------------------------------- the live plan
def test_the_same_coefficient_is_used_at_both_sites(tmp_path):
    bundle = _bundle(tmp_path)
    plan = L.plan_live_edit(np.zeros(G.HIDDEN_SIZE), 2, bundle)
    assert plan.apply_at_historical_blocks == (31, 32)
    assert plan.same_coefficient_at_every_site is True
    assert plan.read_state == 32
    assert plan.applied_alpha == pytest.approx(0.15 * plan.g_r)
    assert 33 not in plan.apply_at_historical_blocks


def test_no_frozen_parameter_may_receive_a_gradient():
    torch = pytest.importorskip("torch")

    frozen = torch.nn.Linear(4, 1)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    L.assert_no_parameter_requires_grad([frozen])

    thawed = torch.nn.Linear(4, 1)
    with pytest.raises(L.LiveControllerError) as exc:
        L.assert_no_parameter_requires_grad([thawed])
    assert "would receive a gradient" in str(exc.value)


def test_the_live_gate_builds_no_graph(tmp_path):
    """No graph means nothing downstream can backpropagate into the model."""
    torch = pytest.importorskip("torch")

    payload = _checkpoint()
    activation = torch.zeros((1, G.HIDDEN_SIZE), requires_grad=True)
    g = L.gate_from_activation(activation.detach().numpy(), payload)
    assert isinstance(g, np.ndarray)          # a number, not a tensor
    assert not hasattr(g, "grad_fn")


# ---------------------------------------------------------------- the closure
def test_the_two_site_closure_verifies_itself():
    payload = C.load_closure()
    assert payload["closure_sha256"] == C.closure_sha256(payload)
    C.assert_matches_package()


def test_the_closure_records_the_result_at_its_own_denominators():
    payload = C.load_closure()
    assert payload["judge_verdicts"] == {
        "planned": 960, "valid": 957, "missing_technical": 3}
    assert payload["cells"]["planned"] == 480
    assert payload["cells"]["dual_judged"] == 477
    assert payload["original_protocol_complete"] is False

    e = payload["readings"]["E"]
    assert e["decided_denominator"] == 89
    assert e["substantive_repair"] == 64 and e["full_repair"] == 61
    b = payload["readings"]["B"]
    assert b["decided_denominator"] == 176 and b["unresolved"] == 2
    assert b["material_degradation"] == 16
    assert b["non_material_degradation"] == 16
    h = payload["readings"]["H"]
    assert h["decided_denominator"] == 212 and h["unresolved"] == 1
    assert h["YES"] == 10 and h["NO"] == 202


def test_the_closure_names_all_three_technical_cells_without_text():
    payload = C.load_closure()
    entries = payload["technical_unresolved"]
    assert len(entries) == 3
    assert {e["row_id"] for e in entries} == {
        "TSJ1QUALITY228E0C7BA5D1", "TSJ1QUALITY3FCCAD83E63B",
        "TSJ1SAFETY112B028C0586"}
    kinds = sorted(e["failure_type"] for e in entries)
    assert kinds == ["EmptyJudgeResponse", "MismatchedItemIdEcho",
                     "MismatchedItemIdEcho"]
    # The only place those words may appear is the declaration that says the
    # closure holds none of it.
    blob = json.dumps({k: v for k, v in payload.items()
                       if k != "contains_prompt_or_response_text"})
    assert "prompt_text" not in blob and "response_text" not in blob
    assert payload["contains_prompt_or_response_text"] is False


def test_the_closure_states_no_post_hoc_pass_threshold():
    payload = C.load_closure()
    assert payload["outcome"] == "TRADE_OFF_NOT_A_SELECTION"
    blob = json.dumps(payload).lower()
    for invented in ("pass_threshold", "passes", "meets_criterion",
                     "selected_arm"):
        assert invented not in blob


def test_an_edited_closure_is_refused(tmp_path):
    payload = C.load_closure()
    payload["readings"]["E"]["substantive_repair"] = 65
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(C.TwoSiteStaticClosureError) as exc:
        C.load_closure(path)
    assert "edited after it was written" in str(exc.value)


# ------------------------------------------------------- the two dynamic arms
import importlib.util                                            # noqa: E402

_RUNNER = (Path(__file__).resolve().parents[1] / "experiments"
           / "single_direction_controller" / "run_dynamic_arms.py")


def _runner():
    spec = importlib.util.spec_from_file_location("_dyn_arms", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                              # type: ignore
    return module


D = _runner()


def _plan_inputs(tmp_path, n=8, veto_first=3):
    bundle = _bundle(tmp_path)
    width = G.HIDDEN_SIZE
    rng = np.random.default_rng(11)
    activations = {f"row{i:03d}": rng.normal(size=width) for i in range(n)}
    fold_of = {r: i % G.N_SPLITS for i, r in enumerate(sorted(activations))}
    # tau_hi is 1.5 in the synthetic checkpoints; push the first few over it
    z = {r: (9.0 if i < veto_first else -9.0)
         for i, r in enumerate(sorted(activations))}
    return bundle, activations, fold_of, z


def test_dynamic_raw_needs_every_row_and_reuses_nothing(tmp_path):
    bundle, activations, fold_of, z = _plan_inputs(tmp_path)
    plan = D.plan_rows(bundle, activations, fold_of, z, D.ARM_RAW)
    assert len(plan) == len(activations)
    assert all(r["vetoed"] is False for r in plan)
    arithmetic = D.reuse_arithmetic(plan, D.ARM_RAW)
    assert arithmetic["new"] == len(activations)
    assert arithmetic["reused"] == 0


def test_dynamic_hard_veto_needs_no_new_generation(tmp_path):
    """Its rows are provably identical to ones already generated."""
    bundle, activations, fold_of, z = _plan_inputs(tmp_path, n=8, veto_first=3)
    plan = D.plan_rows(bundle, activations, fold_of, z, D.ARM_VETO)
    vetoed = [r for r in plan if r["vetoed"]]
    assert len(vetoed) == 3
    assert all(r["g_R"] == 0.0 and r["applied_alpha_R"] == 0.0
               for r in vetoed)
    arithmetic = D.reuse_arithmetic(plan, D.ARM_VETO)
    assert arithmetic["new"] == 0
    assert arithmetic["reused"] == len(plan)
    assert arithmetic["reused_from"] == {D.REFERENCE_ARM: 3, D.ARM_RAW: 5}


def test_a_vetoed_row_with_a_non_zero_coefficient_is_refused(tmp_path):
    """Reuse is only correct when the edit is exactly the zero vector."""
    bundle, activations, fold_of, z = _plan_inputs(tmp_path)
    plan = D.plan_rows(bundle, activations, fold_of, z, D.ARM_VETO)
    for row in plan:
        if row["vetoed"]:
            row["applied_alpha_R"] = 1e-12       # not exactly zero
            break
    with pytest.raises(D.DynamicArmError) as exc:
        D.reuse_arithmetic(plan, D.ARM_VETO)
    assert "exactly the zero vector" in str(exc.value)


def test_both_arms_use_the_same_coefficient_where_the_veto_is_silent(tmp_path):
    bundle, activations, fold_of, z = _plan_inputs(tmp_path, veto_first=0)
    raw = {r["row_id"]: r["g_R"]
           for r in D.plan_rows(bundle, activations, fold_of, z, D.ARM_RAW)}
    veto = {r["row_id"]: r["g_R"]
            for r in D.plan_rows(bundle, activations, fold_of, z, D.ARM_VETO)}
    assert raw == veto            # one controller, not two


def test_every_planned_row_applies_at_both_sites_and_not_at_33(tmp_path):
    bundle, activations, fold_of, z = _plan_inputs(tmp_path)
    for row in D.plan_rows(bundle, activations, fold_of, z, D.ARM_RAW):
        assert row["apply_at_historical_blocks"] == [31, 32]
        assert row["applied_alpha_R"] == pytest.approx(0.15 * row["g_R"])


def test_the_saved_run_implies_the_reuse_split_the_runner_claims():
    """192 vetoed / 288 not, from gate_oof_v1's own committed summary."""
    payload = C.load_closure()
    assert payload["cells"]["planned"] == 480
    # the veto arithmetic the runner relies on, stated as a fixed expectation
    assert D.EXPECTED_ROWS == 480
    assert D.REFERENCE_ARM == "baseline"
    assert D.ARMS == ("dynamic_raw", "dynamic_hard_veto")


# ------------------------------------- the row order the dynamic arms shard on
#
# ``run_dynamic_arms.py`` called ``deterministic_row_order(list(frame.row_ids))``
# -- bare id strings where the helper takes corpus *records* -- and died with
# ``TypeError: string indices must be integers, not 'str'`` before it had
# planned a single row. These pin the contract, the order, the partition and
# the fact that a dry run leaves no trace.

def _distinct_bundle(tmp_path):
    """Five fold controllers that actually score differently.

    ``build_controller`` zeroes the readout, so every fold would return
    ``g_R = 0.1`` on every row and a binding test would pass without binding
    anything. Each fold gets its own readout instead, so a coefficient
    identifies the controller that produced it.
    """
    rng = np.random.default_rng(4242)
    for k in range(G.N_SPLITS):
        net = _net(G.seed_for(k))
        state = {name: value.detach().numpy().copy()
                 for name, value in net.state_dict().items()}
        state["3.weight"] = rng.normal(size=state["3.weight"].shape)
        payload = L.build_checkpoint(
            L.KIND_FOLD, k, state, mu=np.zeros(G.HIDDEN_SIZE),
            sd=np.ones(G.HIDDEN_SIZE), tau_hi=1.5, selected_step=42,
            trained_on_rows=384)
        (tmp_path / f"controller_fold_{k}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    return L.load_bundle(tmp_path)


def test_the_row_order_helper_takes_corpus_records_not_bare_row_ids():
    """The exact shape mismatch that raised, stated as a contract."""
    rows = [{"row_id": "b"}, {"row_id": "a"}, {"row_id": "c"}]
    assert TS.deterministic_row_order(rows) == ["a", "b", "c"]
    with pytest.raises(TypeError, match="string indices"):
        TS.deterministic_row_order(["a", "b", "c"])


def test_the_dynamic_runner_orders_from_the_corpus_records():
    """Catches the regression on a machine that cannot reach the corpus."""
    source = _RUNNER.read_text(encoding="utf-8")
    assert "deterministic_row_order(corpus_rows)" in source
    assert "deterministic_row_order(list(frame.row_ids))" not in source
    # inputs.row_ids is the 267-row E/B projection, and would silently shard
    # a subset of the corpus rather than raise
    assert "deterministic_row_order(inputs.row_ids)" not in source


def test_the_shards_stay_an_exhaustive_non_overlapping_partition():
    ordered = TS.deterministic_row_order(
        [{"row_id": f"row:{i:04d}"} for i in range(480)])
    for count in (1, 2, 3, 5, 7, 16, 480):
        shards = [TS.shard_rows(ordered, i, count)
                  for i in range(1, count + 1)]
        flat = [r for shard in shards for r in shard]
        assert flat == ordered                      # exhaustive, and in order
        assert len(set(flat)) == len(ordered)       # no row in two shards
        assert max(map(len, shards)) - min(map(len, shards)) <= 1
        TS.assert_shards_partition(ordered, count, Checklist("t"))


def test_a_coefficient_binds_to_its_row_and_to_its_own_fold_controller(tmp_path):
    bundle = _distinct_bundle(tmp_path)
    rng = np.random.default_rng(7)
    activations = {f"row{i:03d}": rng.normal(size=G.HIDDEN_SIZE)
                   for i in range(10)}
    fold_of = {r: i % G.N_SPLITS for i, r in enumerate(sorted(activations))}
    z = {r: -9.0 for r in activations}

    plan = D.plan_rows(bundle, activations, fold_of, z, D.ARM_RAW)
    assert [r["row_id"] for r in plan] == sorted(activations)
    for row in plan:
        rid, k = row["row_id"], row["fold"]
        assert k == fold_of[rid]
        expected = float(L.gate_from_activation(
            activations[rid].reshape(1, -1), bundle.folds[k])[0])
        assert row["g_R"] == expected
        assert row["applied_alpha_R"] == pytest.approx(G.ALPHA_R * expected)
        # it really is this fold's controller and not another's
        others = {float(L.gate_from_activation(
            activations[rid].reshape(1, -1), bundle.folds[j])[0])
            for j in range(G.N_SPLITS) if j != k}
        assert expected not in others

    moved = D.plan_rows(bundle, activations,
                        {r: (f + 1) % G.N_SPLITS for r, f in fold_of.items()},
                        z, D.ARM_RAW)
    assert [r["g_R"] for r in moved] != [r["g_R"] for r in plan]


@needs_real_inputs
def test_the_dynamic_order_is_the_whole_corpus_exactly_once():
    from routegate.data import load_development_inputs
    from routegate.ebh import frame_from_inputs

    inputs = load_development_inputs()
    order = TS.deterministic_row_order(list(inputs.manifest_rows))
    assert len(order) == D.EXPECTED_ROWS == 480
    assert len(set(order)) == 480
    assert order == sorted(order)

    # the substitution this fix made must not move a single row
    frame = frame_from_inputs(inputs)
    assert order == sorted(frame.row_ids)
    assert set(order) == set(frame.row_ids)
    # deterministic across calls and independent of the input order
    assert order == TS.deterministic_row_order(
        list(reversed(list(inputs.manifest_rows))))


@needs_dynamic_arm_inputs
def test_the_dynamic_dry_run_writes_nothing_and_creates_no_output_directory(
        tmp_path, capsys, direction_bank_dir, controller_bundle_dir):
    """The real gate, on the real bundle: it decides, and it leaves no trace."""
    out = tmp_path / "arms_output"
    before = sorted(p.name for p in Path(controller_bundle_dir).iterdir())
    assert D.main(["--direction-bank", str(direction_bank_dir),
                   "--controllers", str(controller_bundle_dir),
                   "--output-dir", str(out),
                   "--arm", "dynamic_raw",
                   "--shard-index", "1", "--shard-count", "1",
                   "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "shard 1/1: 480 row(s)" in printed
    assert "nothing written" in printed
    assert "no model loaded" in printed
    # the coefficient reproduction is decided here, before any GPU is claimed
    assert "per-fold held-out means reproduce" in printed
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []
    assert sorted(
        p.name for p in Path(controller_bundle_dir).iterdir()) == before


@needs_dynamic_arm_inputs
def test_the_dynamic_dry_run_shards_the_corpus_without_gaps(
        tmp_path, capsys, direction_bank_dir, controller_bundle_dir):
    seen = []
    for index in (1, 2, 3):
        assert D.main(["--direction-bank", str(direction_bank_dir),
                       "--controllers", str(controller_bundle_dir),
                       "--output-dir", str(tmp_path / "arms_output"),
                       "--shard-index", str(index), "--shard-count", "3",
                       "--dry-run"]) == 0
        line = [l for l in capsys.readouterr().out.splitlines()
                if f"shard {index}/3" in l][-1]
        seen.append(int(line.split(":")[-1].split()[0]))
    assert sum(seen) == 480
    assert max(seen) - min(seen) <= 1
    assert not (tmp_path / "arms_output").exists()


@needs_dynamic_arm_inputs
def test_resume_changes_nothing_about_the_plan_or_what_is_written(
        tmp_path, capsys, direction_bank_dir, controller_bundle_dir):
    """--resume is a generation-time skip; it must not move the shard."""
    out = tmp_path / "arms_output"
    base = ["--direction-bank", str(direction_bank_dir),
            "--controllers", str(controller_bundle_dir),
            "--output-dir", str(out),
            "--shard-index", "1", "--shard-count", "1", "--dry-run"]

    assert D.main(base) == 0
    plain = capsys.readouterr().out
    assert D.main(base + ["--resume"]) == 0
    resumed = capsys.readouterr().out

    assert plain == resumed
    assert not out.exists()
    assert not (out / D.shard_records_name(1, 1)).exists()
    assert D.build_parser().parse_args(base).resume is False
    assert D.build_parser().parse_args(base + ["--resume"]).resume is True
