"""The single-direction coefficient controller: architecture, losses, guards.

Nothing here loads a language model, generates a token, calls a judge or reads
held-out material. Tests needing the frozen probe run and activation capture
skip when those are unreachable; the rest run on synthetic tensors.

What is under test is what a training stage gets wrong invisibly: a fold that
leaks into its own controller, a gradient that reaches something frozen, a
retired direction that comes back, a gate that exceeds its ceiling, and a
safety guard that can raise the thing it exists to lower.
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

from routegate import gatecontroller as G                            # noqa: E402
from routegate import frozenprobe as F                               # noqa: E402
from routegate.twosite import CANDIDATE_BLOCKS                       # noqa: E402
from routegate.validation import (                                   # noqa: E402
    Checklist,
    SealedSetError,
    ValidationError,
)

TRAINER = (REPO_ROOT / "experiments" / "single_direction_controller"
           / "train_gate_controller.py")
PROBE_DIR = REPO_ROOT / F.PROBE_RUN_DIRECTORY

needs_probe_run = pytest.mark.skipif(
    not (PROBE_DIR / F.PROBE_PREDICTIONS).is_file(),
    reason="the frozen probe run is not reachable on this machine")


def _load_trainer():
    spec = importlib.util.spec_from_file_location("_gate_trainer", TRAINER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_gate_trainer"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic():
    torch.manual_seed(7)
    n, dim = 40, G.HIDDEN_SIZE
    h = torch.randn(n, dim)
    x = (h - h.mean(0)) / h.std(0).clamp_min(1e-8)
    budget = torch.rand(n) * 5 + 20
    direction = torch.randn(dim)
    direction = direction / torch.linalg.vector_norm(direction)
    for t in (h, x, budget, direction):
        t.requires_grad_(False)
    return {"h": h, "x": x, "budget": budget, "direction": direction, "n": n}


def _probe(dim=G.HIDDEN_SIZE, seed=3, scale=1.0):
    rng = np.random.default_rng(seed)
    return F.FrozenProbe(head="repair", fold=0, state=32,
                         weight=rng.normal(size=dim) * scale,
                         mean=rng.normal(size=dim),
                         scale=np.abs(rng.normal(size=dim)) + 0.5,
                         intercept=0.25, n_train=100)


# ================================================== 1. the architecture
def test_the_controller_has_exactly_the_declared_parameter_count():
    net = G.build_controller(seed=1)
    assert G.count_parameters(net) == G.CONTROLLER_PARAMETERS == 32801
    assert G.CONTROLLER_HIDDEN == 16 and G.CONTROLLER_DEPTH == 1


def test_the_gate_starts_at_the_declared_initial_value(synthetic):
    net = G.build_controller(seed=1).eval()
    with torch.no_grad():
        g = G.gate(net, synthetic["x"])
    assert torch.allclose(g, torch.full_like(g, G.GATE_INITIAL), atol=1e-6)


def test_the_controller_reads_state_32_which_is_block_31():
    assert G.READ_STATE == 32 and G.READ_BLOCK == 31
    spec = G.build_training_specification()
    assert spec["controller"]["read_state"] == 32
    assert spec["controller"]["read_historical_block"] == 31
    assert spec["controller"]["token_position"] == "final prompt token"


# ================================================ 2. coefficient bounds
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_gate_is_always_in_the_unit_interval(synthetic, seed):
    net = G.build_controller(seed=seed).eval()
    with torch.no_grad():
        g = G.gate(net, synthetic["x"] * 50)      # deliberately extreme input
    assert bool((g >= G.GATE_MINIMUM).all() and (g <= G.GATE_MAXIMUM).all())


#: float32 cannot represent 0.15 exactly, so 0.15 * 1.0 lands one unit in the
#: last place above it. The ceiling is exact in real arithmetic and holds to a
#: single ulp in the arithmetic actually used; asserting a bare <= would be
#: asserting something false about float32.
FLOAT32_ULP_AT_ALPHA = float(np.spacing(np.float32(G.MAX_APPLIED_ALPHA_R)))


def test_the_applied_strength_never_exceeds_the_frozen_ceiling():
    g = torch.tensor([0.0, 0.5, 1.0])
    alpha = G.applied_alpha(g)
    assert torch.allclose(alpha, torch.tensor([0.0, 0.075, 0.15]))
    assert float(alpha.max()) <= G.MAX_APPLIED_ALPHA_R + FLOAT32_ULP_AT_ALPHA
    assert G.MAX_APPLIED_ALPHA_R == 0.15
    # and in float64, which is what the saved coefficients are written in,
    # it is exact
    assert G.ALPHA_R * np.float64(1.0) <= 0.15


def test_the_ceiling_is_arithmetic_not_a_clamp():
    """sigmoid saturates at 1 and 0.15 x 1 = 0.15; no branch enforces it."""
    huge = torch.tensor([1e6, -1e6])
    g = torch.sigmoid(huge)
    assert float(g.max()) <= 1.0
    assert float(G.applied_alpha(g).max()) <= 0.15 + FLOAT32_ULP_AT_ALPHA
    import ast
    source = (REPO_ROOT / "src" / "routegate" / "gatecontroller.py").read_text(
        encoding="utf-8")
    body = [n for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.FunctionDef) and n.name == "applied_alpha"]
    # no clamp, no min, no branch: the bound is the multiplication itself
    code = ast.unparse(body[0])
    for banned in ("clamp", "min(", "if "):
        assert banned not in code, banned


# =========================================== 3. no H-E actuator, ever
def test_no_safety_direction_is_applied_anywhere():
    spec = G.build_training_specification()
    assert spec["intervention"]["alpha_S"] == 0.0 == G.ALPHA_S
    assert spec["intervention"]["h_minus_e_is_never_applied"] is True
    assert spec["controller"]["output_count"] == 1


def test_the_module_never_reads_the_safety_direction():
    import ast
    source = (REPO_ROOT / "src" / "routegate" / "gatecontroller.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)):
                node.body = body[1:]
    code = ast.unparse(tree)
    assert "safety_direction" not in code
    assert "SAFETY_ARRAY" not in code


def test_the_trainer_only_takes_the_repair_direction_from_the_bank():
    source = TRAINER.read_text(encoding="utf-8")
    assert "d_repair, _ = bank.directions_for_state(READ_STATE)" in source
    assert "safety_direction" not in source


def test_the_intervention_edits_only_the_two_validated_sites():
    spec = G.build_training_specification()["intervention"]
    assert spec["historical_blocks"] == list(CANDIDATE_BLOCKS) == [31, 32]
    assert spec["package_states_edited"] == [32, 33]
    assert spec["budget_reference_states"] == [31, 32]
    assert 33 not in spec["historical_blocks"]
    assert 34 not in spec["package_states_edited"]


# ============================================= 4. the intervention itself
def test_the_steered_activation_is_the_budgeted_edit(synthetic):
    g = torch.full((synthetic["n"],), 0.5)
    steered = G.steer(synthetic["h"], g, synthetic["budget"],
                      synthetic["direction"])
    expected = synthetic["h"] + (
        0.15 * 0.5 * synthetic["budget"]).unsqueeze(-1) * synthetic["direction"]
    assert torch.allclose(steered, expected, atol=1e-6)


def test_a_zero_gate_is_no_edit_at_all(synthetic):
    g = torch.zeros(synthetic["n"])
    steered = G.steer(synthetic["h"], g, synthetic["budget"],
                      synthetic["direction"])
    assert torch.equal(steered, synthetic["h"])


def test_the_explicit_path_equals_the_affine_oracle(synthetic):
    """The oracle is exact because the probe is affine -- and it is used only
    here, never in training, so the trainer visibly applies d_R."""
    probe = F.TorchProbe(_probe())
    g = torch.rand(synthetic["n"])
    explicit = probe.logit(G.steer(synthetic["h"], g, synthetic["budget"],
                                   synthetic["direction"]))
    kappa = _probe().kappa(synthetic["direction"].numpy().astype(np.float64))
    oracle = G.affine_oracle(probe.logit(synthetic["h"]), g,
                             synthetic["budget"], kappa)
    assert torch.allclose(explicit, oracle, atol=1e-3)


def test_the_trainer_builds_the_steered_tensor_rather_than_the_shortcut():
    source = TRAINER.read_text(encoding="utf-8")
    assert "steer(" in source
    assert "affine_oracle" not in source


# ================================= 5. controller-only gradients
def test_gradients_reach_the_controller_and_nothing_else(synthetic):
    net = G.build_controller(seed=1)
    probe = F.TorchProbe(_probe())
    assert probe.parameters_require_grad is False

    g = G.gate(net, synthetic["x"])
    steered = G.steer(synthetic["h"], g, synthetic["budget"],
                      synthetic["direction"])
    loss = G.repair_loss(probe.logit(steered),
                         torch.ones(synthetic["n"], dtype=torch.bool))
    loss.backward()

    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in net.parameters())
    for frozen in (synthetic["h"], synthetic["x"], synthetic["budget"],
                   synthetic["direction"], probe.weight, probe.mean,
                   probe.scale, probe.intercept):
        assert frozen.grad is None
        assert frozen.requires_grad is False


def test_detaching_the_intervention_kills_the_gradient(synthetic):
    """The negative control.

    Detaching the gate disconnects the graph entirely, so backward raises
    rather than merely producing zero gradients -- a stronger guarantee than
    zero grads, and proof that the ONLY route from the loss to theta is through
    the steered tensor. If the loss reached theta by any other path this would
    not raise.
    """
    net = G.build_controller(seed=1)
    probe = F.TorchProbe(_probe())
    g = G.gate(net, synthetic["x"])
    steered = G.steer(synthetic["h"], g.detach(), synthetic["budget"],
                      synthetic["direction"])
    loss = G.repair_loss(probe.logit(steered),
                         torch.ones(synthetic["n"], dtype=torch.bool))
    assert loss.grad_fn is None and loss.requires_grad is False
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()
    assert all(p.grad is None for p in net.parameters())


def test_the_optimiser_holds_only_controller_parameters(synthetic):
    net = G.build_controller(seed=1)
    optimiser = G.controller_optimiser(net)
    probe = F.TorchProbe(_probe())
    outcome = G.assert_optimiser_is_controller_only(
        optimiser, net,
        [synthetic["h"], synthetic["direction"], probe.weight], Checklist("t"))
    assert outcome["parameters"] == 32801


def test_a_foreign_parameter_in_the_optimiser_is_refused(synthetic):
    net = G.build_controller(seed=1)
    intruder = torch.nn.Parameter(torch.zeros(3))
    optimiser = torch.optim.AdamW(
        [{"params": list(net.parameters())}, {"params": [intruder]}], lr=1e-3)
    with pytest.raises(ValidationError, match="only_controller_parameters"):
        G.assert_optimiser_is_controller_only(optimiser, net, [], Checklist("t"))


def test_a_frozen_tensor_left_trainable_is_refused(synthetic):
    net = G.build_controller(seed=1)
    optimiser = G.controller_optimiser(net)
    thawed = synthetic["h"].clone().requires_grad_(True)
    with pytest.raises(ValidationError, match="frozen_tensors_require_no_grad"):
        G.assert_optimiser_is_controller_only(optimiser, net, [thawed],
                                              Checklist("t"))


# ================================================ 6. the losses
def test_the_repair_loss_falls_as_the_repair_logit_falls():
    mask = torch.ones(4, dtype=torch.bool)
    assert float(G.repair_loss(torch.full((4,), -3.0), mask)) < \
        float(G.repair_loss(torch.full((4,), 3.0), mask))


def test_the_safety_loss_is_one_sided():
    mask = torch.ones(3, dtype=torch.bool)
    before = torch.tensor([1.0, 1.0, 1.0])
    assert float(G.safety_loss(before, before + 2.0, mask)) == 0.0   # rise free
    assert float(G.safety_loss(before, before - 2.0, mask)) == 2.0   # fall paid


def test_the_size_penalty_is_quadratic_in_the_gate():
    assert float(G.size_loss(torch.tensor([1.0, 1.0]))) == 1.0
    assert float(G.size_loss(torch.tensor([0.5, 0.5]))) == 0.25
    assert float(G.size_loss(torch.zeros(4))) == 0.0
    big = float(G.size_loss(torch.tensor([1.0])))
    small = float(G.size_loss(torch.tensor([0.5])))
    assert big / small == 4.0                       # superlinear, not L1


def test_the_size_penalty_pushes_the_gate_down(synthetic):
    net = G.build_controller(seed=1)
    optimiser = G.controller_optimiser(net)
    for _ in range(50):
        optimiser.zero_grad(set_to_none=True)
        G.size_loss(G.gate(net, synthetic["x"])).backward()
        optimiser.step()
    net.eval()
    with torch.no_grad():
        assert float(G.gate(net, synthetic["x"]).mean()) < G.GATE_INITIAL


def test_the_total_loss_uses_the_predeclared_weights():
    assert G.LAMBDA_REPAIR == 1.0
    assert G.LAMBDA_SAFETY == 4.0
    assert G.LAMBDA_SIZE == 0.25
    assert G.LOSS_WEIGHTS_ARE_PREDECLARED is True
    total = G.total_loss(torch.tensor(1.0), torch.tensor(2.0),
                         torch.tensor(4.0))
    assert float(total) == 1.0 + 4.0 * 2.0 + 0.25 * 4.0


def test_every_training_h_row_enters_the_safety_term():
    """No H row may be excluded, whatever its score."""
    z = torch.tensor([-99.0, 0.0, 99.0])
    is_h = torch.tensor([True, True, True])
    assert bool(G.protected_mask(z, is_h, tau_hi=50.0).all())


def test_safe_rows_scoring_h_like_are_protected_too():
    z = torch.tensor([-99.0, 60.0])
    is_h = torch.tensor([False, False])
    mask = G.protected_mask(z, is_h, tau_hi=50.0)
    assert bool(mask[1]) and not bool(mask[0])


def test_hyperparameter_search_is_forbidden():
    assert G.HYPERPARAMETER_SEARCH_PERMITTED is False
    spec = G.build_training_specification()
    assert spec["optimiser"]["hyperparameter_search_permitted"] is False


# ================================= 7. the veto and the clamp
def test_the_veto_threshold_fully_vetoes_every_training_h_row():
    z = [1.0, 2.0, 5.0, 0.5]
    tau = G.veto_threshold(z)
    assert tau == 0.5
    assert all(v >= tau for v in z)                 # every H row is caught


def test_an_empty_training_h_set_is_refused():
    with pytest.raises(G.GateControllerError, match="not a threshold"):
        G.veto_threshold([])


def test_the_veto_can_only_lower_a_gate():
    g = torch.tensor([0.9, 0.9, 0.9])
    z = torch.tensor([-1.0, 0.0, 1.0])
    out = G.apply_safety_veto(g, z, tau_hi=0.0)
    assert bool((out <= g).all())
    assert float(out[0]) == pytest.approx(0.9)      # below threshold, untouched
    assert float(out[1]) == 0.0 and float(out[2]) == 0.0


@pytest.mark.parametrize("tau", [-5.0, 0.0, 5.0])
def test_the_veto_never_raises_a_gate_for_any_threshold(tau):
    torch.manual_seed(11)
    g = torch.rand(200)
    z = torch.randn(200) * 5
    assert bool((G.apply_safety_veto(g, z, tau) <= g).all())


@pytest.mark.parametrize("kappa", [-2.0, -0.5, -0.01])
def test_the_clamp_never_raises_a_gate(kappa):
    torch.manual_seed(12)
    g = torch.rand(200)
    z = torch.randn(200) * 3
    budget = torch.rand(200) * 5 + 20
    out = G.apply_counterfactual_clamp(g, z, budget, kappa, tau_hi=0.0)
    assert bool((out <= g + 1e-9).all())
    assert bool((out >= 0.0).all())


def test_a_non_negative_kappa_needs_no_clamp():
    """If steering cannot lower z_H there is nothing to clamp against."""
    g = torch.rand(50)
    out = G.apply_counterfactual_clamp(g, torch.randn(50), torch.ones(50),
                                       kappa_safety=0.3, tau_hi=0.0)
    assert torch.equal(out, g)


def test_the_clamp_keeps_the_steered_logit_clear_of_the_threshold():
    """The counterfactual is exact for an affine probe, so this is checkable."""
    z = torch.tensor([2.0, 3.0, 10.0])
    budget = torch.tensor([20.0, 20.0, 20.0])
    g = torch.ones(3)
    kappa, tau = -1.0, 0.0
    clamped = G.apply_counterfactual_clamp(g, z, budget, kappa, tau)
    after = z + G.ALPHA_R * clamped * budget * kappa
    assert bool((after >= tau - 1e-5).all())


def test_neither_guard_is_trained():
    spec = G.build_training_specification()["brake"]
    assert spec["neither_is_trained"] is True
    assert spec["neither_can_raise_a_gate"] is True


# ================================= 8. fold isolation
def test_the_inner_fold_is_never_the_held_out_fold():
    for k in range(G.N_SPLITS):
        assert G.inner_fold_for(k) != k
        assert 0 <= G.inner_fold_for(k) < G.N_SPLITS
    assert {G.inner_fold_for(k) for k in range(G.N_SPLITS)} == set(range(5))


def test_an_outer_fold_outside_the_division_is_refused():
    for bad in (-1, G.N_SPLITS):
        with pytest.raises(G.GateControllerError, match="outside"):
            G.inner_fold_for(bad)


def test_each_fold_gets_its_own_seed():
    seeds = {G.seed_for(k) for k in range(G.N_SPLITS)}
    assert len(seeds) == G.N_SPLITS
    assert G.seed_for(0) == G.SEED_BASE


def test_the_trainer_trains_on_the_complement_of_the_held_out_fold():
    source = TRAINER.read_text(encoding="utf-8")
    assert "held = fold_of == k" in source
    assert "train = ~held" in source
    assert "inner_train = train & ~inner_mask_full" in source
    # the held-out fold is only ever predicted, never trained or validated on
    assert "g_held = gate(net, x[held_t])" in source


def test_the_trainer_declares_no_final_refit():
    spec = G.build_training_specification()
    assert spec["oof_protocol"]["final_refit_performed"] is False
    assert "final_refit_performed" in TRAINER.read_text(encoding="utf-8")


# ================================= 9. frozen probe reproduction
@needs_probe_run
def test_the_frozen_probes_reproduce_to_the_committed_tolerance():
    from routegate.ebh import load_ebh_frame
    from routegate.probeheads import load_bank_fold_assignment
    frame = load_ebh_frame()
    assignment, _ = load_bank_fold_assignment(
        REPO_ROOT / "outputs/causal_repair/orthogonalized_population_scan"
                    "/direction_bank")
    saved = F.read_saved_predictions(PROBE_DIR / F.PROBE_PREDICTIONS, 32)
    checks = Checklist("t")
    probes = F.reconstruct_and_verify(frame, assignment, saved, 32,
                                      F.REPRODUCTION_TOLERANCE, checks)
    assert checks.all_passed
    assert sorted(probes) == ["repair", "safety"]
    assert len(probes["repair"]) == 5


def test_a_probe_that_does_not_reproduce_is_refused():
    """The gate is fail-closed: near enough is not the committed probe."""
    checks = Checklist("t")
    with pytest.raises(ValidationError, match="reproduces"):
        checks.require("repair_fold_0_reproduces", False,
                       "max |reproduced - saved| = 1.0e-06")


def test_the_tolerance_is_the_committed_one():
    assert F.REPRODUCTION_TOLERANCE == 1e-12


def test_kappa_r_must_be_negative_and_kappa_h_is_only_reported():
    dim = 8
    direction = np.ones(dim) / np.sqrt(dim)

    def probe(head, weight):
        return F.FrozenProbe(head=head, fold=0, state=32, weight=weight,
                             mean=np.zeros(dim), scale=np.ones(dim),
                             intercept=0.0, n_train=10)

    good = {"repair": {0: probe("repair", -np.ones(dim))},
            "safety": {0: probe("safety", np.ones(dim))}}   # kappa_H > 0
    checks = Checklist("t")
    out = F.assert_repair_direction_reduces_the_repair_logit(
        good, direction, checks)
    assert out["repair"][0] < 0 and out["safety"][0] > 0
    assert checks.all_passed          # a positive kappa_H is NOT a failure

    bad = {"repair": {0: probe("repair", np.ones(dim))},
           "safety": {0: probe("safety", np.ones(dim))}}
    with pytest.raises(ValidationError, match="kappa_R_is_negative"):
        F.assert_repair_direction_reduces_the_repair_logit(bad, direction,
                                                           Checklist("t"))


def test_the_probe_is_differentiable_in_the_activation_only():
    probe = F.TorchProbe(_probe())
    h = torch.randn(4, G.HIDDEN_SIZE, requires_grad=True)
    probe.logit(h).sum().backward()
    assert h.grad is not None and float(h.grad.abs().sum()) > 0
    assert probe.weight.grad is None and probe.intercept.grad is None


# ================================= 10. no model, no generation, no judge
def test_the_trainer_loads_no_model_and_generates_nothing():
    import ast
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)):
                node.body = body[1:]
    code = ast.unparse(tree)
    for banned in ("load_model_and_tokenizer", "generate_response",
                   "AutoModelForCausalLM", "capture_prompt_states",
                   "PatchController", "adjudicat", "rubric"):
        assert banned not in code, banned


def test_the_run_declares_zero_forward_passes():
    spec = G.build_training_specification()
    assert spec["forward_passes"] == 0
    assert spec["generates_responses"] is False
    assert spec["calls_a_judge"] is False
    assert spec["reads_the_sealed_set"] is False


def test_cached_activations_are_exact_for_the_state_32_loss():
    """Nothing upstream of state 32 is edited, so the cached tensor stays
    correct even though the deployed arm also edits block 32."""
    spec = G.build_training_specification()["intervention"]
    assert spec["budget_reference_states"][0] == 31
    assert spec["package_states_edited"][0] == 32


# ================================= 11. sealed refusal, provisional status
def test_a_sealed_output_directory_is_refused(tmp_path):
    module = _load_trainer()
    with pytest.raises(SealedSetError):
        module.claim_output_dir(tmp_path / "sealed_gate_run", create=False)


def test_a_non_empty_output_directory_is_refused(tmp_path):
    module = _load_trainer()
    out = tmp_path / "run"
    out.mkdir()
    assert module.claim_output_dir(out, create=False) == out
    (out / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="already holds"):
        module.claim_output_dir(out, create=False)


def test_checking_the_output_directory_does_not_create_it(tmp_path):
    module = _load_trainer()
    out = tmp_path / "run"
    module.claim_output_dir(out, create=False)
    assert not out.exists()


def test_the_runner_exposes_no_option_reaching_held_out_material():
    from routegate.validation import looks_sealed
    module = _load_trainer()
    options = [o for a in module.build_parser()._actions
               for o in a.option_strings]
    assert options
    for option in options:
        assert looks_sealed(option) is None, option


def test_the_status_is_provisional_and_says_why():
    assert G.PROVISIONAL_STATUS == "PROVISIONAL_PENDING_ACTUATOR_VALIDATION"
    spec = G.build_training_specification()
    assert spec["status"] == G.PROVISIONAL_STATUS
    assert "have not completed behavioural adjudication" in spec["status_reason"]
    assert "no operating point" in spec["status_reason"]


def test_the_provisional_status_reaches_the_trainer_and_its_outputs():
    source = TRAINER.read_text(encoding="utf-8")
    assert "PROVISIONAL_STATUS" in source
    assert '"status": PROVISIONAL_STATUS' in source


def test_the_specification_resolves_what_v1_left_open():
    spec = G.build_training_specification()
    v1 = json.loads((REPO_ROOT / "configs"
                     / "single_direction_controller.json").read_text("utf-8"))
    open_in_v1 = {k for k in v1["unresolved"] if k != "note"}
    assert open_in_v1 <= set(spec["resolves"]["fields"])
    assert spec["resolves"]["specification"] == \
        "configs/single_direction_controller.json"
    # v1 itself is not edited
    assert v1["status"] == "SPECIFIED_NOT_IMPLEMENTED"


def test_the_specification_holds_no_prompt_or_response_text():
    spec = G.build_training_specification()
    assert spec["contains_prompt_or_response_text"] is False
    spec.pop("contains_prompt_or_response_text")
    text = json.dumps(spec)
    for banned in ("prompt_text", "response_text"):
        assert banned not in text, banned


# ============ 12. stratified diagnostics that are not features
def test_the_stratification_fields_are_the_two_frozen_labels():
    assert G.STRATIFICATION_FIELDS == ("topic_standardized", "task_frame")
    assert G.STRATIFICATION_SOURCE == {"topic_standardized": "topic",
                                       "task_frame": "task_frame"}


def test_the_forbidden_features_are_never_used():
    assert G.FORBIDDEN_FEATURES == ("source_family", "benchmark_designation",
                                    "prompt_category")
    spec = G.build_training_specification()["stratification"]
    assert spec["forbidden_features"] == list(G.FORBIDDEN_FEATURES)
    for banned in G.FORBIDDEN_FEATURES:
        assert banned not in spec["fields"]
    source = TRAINER.read_text(encoding="utf-8")
    # source_family is never read off the frame by the trainer
    assert "frame.source_family" not in source
    assert "benchmark_designation" not in source
    assert "prompt_category" not in source


def test_the_controller_input_is_the_activation_alone(synthetic):
    outcome = G.assert_covariates_are_not_features(synthetic["x"],
                                                   checks=Checklist("t"))
    assert outcome["input_width"] == G.HIDDEN_SIZE == 2048


def test_an_input_widened_by_a_covariate_is_refused(synthetic):
    """The only way a covariate reaches the network is by being concatenated,
    so the width is the thing to check."""
    widened = torch.cat([synthetic["x"], torch.ones(synthetic["n"], 3)], dim=-1)
    with pytest.raises(ValidationError,
                       match="controller_input_is_the_activation_alone"):
        G.assert_covariates_are_not_features(widened, checks=Checklist("t"))


def test_the_specification_declares_the_fields_diagnostic_only():
    spec = G.build_training_specification()["stratification"]
    assert spec["is_a_controller_input"] is False
    assert spec["is_a_loss_input"] is False
    assert spec["is_a_threshold_input"] is False
    assert spec["is_a_hyperparameter_selection_input"] is False
    assert "diagnostic only" in spec["standing"]


def test_no_loss_threshold_or_selection_function_takes_a_covariate():
    """Signature-level proof: none of them can receive a stratum at all."""
    import inspect
    for fn in (G.repair_loss, G.safety_loss, G.size_loss, G.total_loss,
               G.protected_mask, G.veto_threshold, G.apply_safety_veto,
               G.apply_counterfactual_clamp, G.gate, G.steer,
               G.inner_fold_for, G.seed_for):
        names = set(inspect.signature(fn).parameters)
        for banned in ("topic", "topic_standardized", "task_frame",
                       "source_family", "stratum", "covariate"):
            assert banned not in names, (fn.__name__, banned)


def test_the_stratified_report_covers_every_row_once():
    values = [0.0, 0.4, 0.9, 0.5]
    strata = ["a", "a", "b", "b"]
    out = G.stratified_distribution(values, strata, (0.5,))
    assert sorted(out) == ["a", "b"]
    assert out["a"]["n"] == 2 and out["b"]["n"] == 2
    assert sum(v["n"] for v in out.values()) == len(values)
    assert out["a"]["mean_g_R"] == pytest.approx(0.2)
    assert out["b"]["intervention_rate"]["0.50"] == 1.0
    assert out["a"]["mean_applied_alpha_R"] == pytest.approx(0.15 * 0.2)


def test_the_per_row_output_retains_what_a_later_router_needs():
    source = TRAINER.read_text(encoding="utf-8")
    for field in ('"topic_standardized": topic[i]', '"task_frame": task_frame[i]',
                  '"arm_id": "repair_015_blocks31_32"',
                  '"historical_blocks": "31,32"', '"g_R": float(coefficients[i])',
                  '"fold": int(fold_of[i])'):
        assert field in source, field
