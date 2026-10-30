"""State numbering, hook sites, token position, and the pinned configuration.

The claims under test are the ones that would be invisible if they were wrong:
an off-by-one between a state and a decoder block patches the wrong place and
still produces fluent text, and a configuration that quietly re-enables sampling
or thinking produces differences that look like the intervention working.

No model is loaded and no GPU is used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from routegate.intervention import (
    BLOCK_OUTPUT_SITE,
    BUDGET_REFERENCES,
    EMBEDDING_SITE,
    INTERVENTION_TYPES,
    STATE_EMBEDDING,
    InterventionError,
    budget_reference_state,
    capture_states_for,
    final_prompt_token_index,
    historical_layer_to_state,
    n_states_for,
    state_to_site,
)
from routegate.runtime import (
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_HIDDEN_STATES,
    EXPECTED_NUM_LAYERS,
    FROZEN_MODEL,
    FROZEN_REVISION,
    TOKEN_POSITION,
    GenerationConfig,
    GenerationConfigError,
    build_chat_strict,
    check_model_geometry,
    finish_flags,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "causal_repair.json"
N_LAYERS = EXPECTED_NUM_LAYERS


# --------------------------------------------------------- 1. state numbering
def test_there_are_forty_nine_states_for_forty_eight_blocks():
    assert n_states_for(N_LAYERS) == EXPECTED_HIDDEN_STATES == 49


def test_state_zero_is_the_embedding_output_not_a_block():
    site = state_to_site(STATE_EMBEDDING, N_LAYERS)
    assert site.kind == EMBEDDING_SITE
    assert site.block_index is None
    assert site.state == 0


def test_states_one_to_forty_eight_are_block_outputs():
    for state in range(1, N_LAYERS + 1):
        site = state_to_site(state, N_LAYERS)
        assert site.kind == BLOCK_OUTPUT_SITE
        assert site.block_index == state - 1
        assert site.tensor == "output"


def test_a_state_outside_the_model_is_refused():
    for bad in (-1, 49, 100):
        with pytest.raises(InterventionError):
            state_to_site(bad, N_LAYERS)


def test_a_state_must_be_an_integer():
    for bad in (1.0, "3", True, None):
        with pytest.raises(InterventionError):
            state_to_site(bad, N_LAYERS)


# ------------------------------------------- 2. states map to hook locations
def test_every_state_maps_to_exactly_one_module():
    sites = [state_to_site(s, N_LAYERS) for s in range(n_states_for(N_LAYERS))]
    keys = [(s.kind, s.block_index) for s in sites]
    assert len(set(keys)) == len(keys)
    assert sum(1 for s in sites if s.kind == EMBEDDING_SITE) == 1
    assert sum(1 for s in sites if s.kind == BLOCK_OUTPUT_SITE) == N_LAYERS


def test_the_historical_layer_band_is_states_thirty_two_to_thirty_four():
    """RQ1's 'layers 31,32,33' are zero-based decoder blocks, not states."""
    assert [historical_layer_to_state(i) for i in (31, 32, 33)] == [32, 33, 34]
    for block in (31, 32, 33):
        assert state_to_site(historical_layer_to_state(block),
                             N_LAYERS).block_index == block


def test_the_config_records_both_numberings():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    recovered = config["recovered_from_history"]
    assert recovered["historical_layers"] == [31, 32, 33]
    assert recovered["historical_layers_as_states"] == [32, 33, 34]


# ------------------------------------------------- the recorded strengths
# Corrected 2026-08-21. The config previously recorded `rq1_grid_best: 0.2`,
# attributing the population-scan pilot value to the RQ1 grid's selection.
#
# What the completed summaries support, exactly — the wording was narrowed a
# second time on the same day because the first correction overstated it:
#
#   * at layers 31-33, 0.15 outranked 0.2 in ALL THREE completed summaries.
#     That is the band-level preference.
#   * 0.15's OVERALL ranks were 2 of 18, 2 of 8 and 1 of 2, so it was first
#     overall ONLY in the final 512 confirmation, and second in the two
#     earlier runs behind a window-6 configuration at the same band.
#   * 0.2 placed 11 of 18, 3 of 8 and 2 of 2, with one harmful break in each
#     completed run.
#   * every count is a provisional keyword readout, not a confirmed optimum.
#
# These tests keep the corrected attribution from drifting back, and keep the
# overstatement from creeping back in. See docs/decision_log.md, 2026-08-21.
def _strengths():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return config["recovered_from_history"]["historical_strengths"]


def test_the_supported_rq1_selected_strength_is_zero_point_one_five():
    strengths = _strengths()
    assert strengths["rq1_grid_selected"] == 0.15
    assert "rq1_grid_best" not in strengths, (
        "the superseded attribution must live under 'superseded_value', not as "
        "a current claim")
    assert strengths["rq1_grid_selected_evidence"]
    assert "PROVISIONAL" in strengths["rq1_grid_selected_caveat"]


def test_the_rq1_evidence_is_a_band_preference_not_a_clean_sweep():
    """0.15 outranked 0.2 everywhere; it was first *overall* in one run only."""
    strengths = _strengths()
    evidence = strengths["rq1_grid_selected_evidence"]

    # the claim that holds across all three completed summaries
    assert "outranks L31-33_w1_s0.2 in ALL THREE completed summaries" in evidence
    assert "band-level preference" in evidence

    # the claim that does not, stated as a limit rather than left to inference
    assert "first overall ONLY in that final 512 confirmation" in evidence
    assert strengths["rq1_grid_selected_basis"]
    assert "band-level preference" in strengths["rq1_grid_selected_basis"]

    # the key carries the negation; the value names the claim that must not be
    # made and then says what actually holds
    disclaimer = strengths["rq1_grid_selected_is_not"]
    assert "ranked first overall in every completed run" in disclaimer
    assert "It did not" in disclaimer
    assert "first overall only in pathway_confirmation_512" in disclaimer
    assert "outranked 0.2 at this band" in disclaimer


def test_the_recorded_rank_numbers_are_preserved_exactly():
    """The ranks are the evidence; the wording is downstream of them."""
    evidence = _strengths()["rq1_grid_selected_evidence"]
    for rank in ("2 of 18", "2 of 8", "1 of 2",          # 0.15
                 "11 of 18", "3 of 8", "2 of 2"):        # 0.2
        assert rank in evidence, rank
    assert "1 harmful break in every completed run" in evidence
    for run in ("pathway_supervisor_layer_extension_20260707_234542",
                "pathway_confirmation_256_20260708_023720",
                "pathway_confirmation_512_20260708_172847"):
        assert run in evidence, run


def test_no_recorded_text_claims_zero_point_one_five_ranked_first_everywhere():
    """The overstatement this correction removed must not return."""
    text = CONFIG.read_text(encoding="utf-8")
    for forbidden in ("L31-33_w1_s0.15 is first",
                      "ranks 0.15 first",
                      "0.15 is first"):
        assert forbidden not in text, forbidden


def test_the_frozen_rq2_actuator_strength_is_zero_point_one_five():
    strengths = _strengths()
    assert strengths["rq2_frozen_actuator"] == 0.15
    assert "rq2_final_method_freeze_20260721_150015" in (
        strengths["rq2_frozen_actuator_evidence"])


def test_zero_point_two_is_named_only_as_the_population_scan_pilot_value():
    strengths = _strengths()
    assert strengths["population_scan_pilot_value"] == 0.2
    status = strengths["population_scan_pilot_value_status"]
    assert "NOT a measured optimum" in status
    assert "NOT what the RQ1 grid selected" in status
    assert "NOT the frozen RQ2 actuator" in status


def test_the_superseded_value_is_preserved_with_its_correction():
    superseded = _strengths()["superseded_value"]
    assert superseded["rq1_grid_best"] == 0.2
    assert superseded["why_it_was_wrong"]
    assert "decision_log" in superseded["evidence_and_reasoning"]


def test_no_strength_in_the_config_is_called_a_measured_optimum():
    """Every mention of the phrase must be a denial, not a claim."""
    text = CONFIG.read_text(encoding="utf-8")
    for match in re.finditer(r"[^\"]*measured optimum[^\"]*", text):
        assert re.search(r"(NOT|not) a measured optimum", match.group(0)), \
            match.group(0)


def test_the_scan_strength_is_unchanged_and_described_as_a_pilot_value():
    """The protocol was never wrong; only its description was."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    intervention = config["intervention"]
    assert intervention["strength"] == 0.2
    rationale = intervention["strength_rationale"]
    assert "population-scan pilot value" in rationale
    assert "not a measured optimum" in rationale
    assert "0.15" in rationale          # the actuator's, named as different


# ------------------------------------------------ budget reference selection
def test_the_budget_reference_is_the_preceding_state_by_default():
    for state in (1, 24, 48):
        reference, note = budget_reference_state(state, "block_input")
        assert reference == state - 1
        assert "block's input" in note


def test_state_zero_references_itself_and_says_so():
    reference, note = budget_reference_state(0, "block_input")
    assert reference == 0
    assert "deviation" in note


def test_the_block_output_reference_is_the_patched_tensor():
    for state in (0, 7, 48):
        reference, note = budget_reference_state(state, "block_output")
        assert reference == state
        assert "own norm" in note


def test_an_unknown_budget_reference_is_refused():
    with pytest.raises(InterventionError):
        budget_reference_state(3, "whatever")
    assert set(BUDGET_REFERENCES) == {"block_input", "block_output"}


def test_a_run_captures_the_patched_states_and_their_references():
    assert capture_states_for([32, 33, 34]) == [31, 32, 33, 34]
    assert capture_states_for([0]) == [0]
    assert capture_states_for([1]) == [0, 1]
    assert capture_states_for([5], "block_output") == [5]


# --------------------------------------------- 3. final-prompt-token indexing
def test_the_patched_row_is_the_last_prompt_token():
    assert final_prompt_token_index(1) == 0
    assert final_prompt_token_index(17) == 16
    assert final_prompt_token_index(768) == 767


def test_an_empty_prompt_is_refused():
    for bad in (0, -1):
        with pytest.raises(InterventionError):
            final_prompt_token_index(bad)


def test_the_token_position_constant_is_minus_one():
    assert TOKEN_POSITION == -1
    assert GenerationConfig().token_position == -1


# ------------------------------------------------- 4. configuration validation
def test_the_default_configuration_is_the_frozen_one():
    config = GenerationConfig()
    config.validate()
    assert config.model == FROZEN_MODEL == "Qwen/Qwen3-30B-A3B"
    assert config.revision == FROZEN_REVISION
    assert config.tokenizer_revision == FROZEN_REVISION
    assert config.dtype == "bfloat16"
    assert config.attn_implementation == "eager"
    assert config.batch_size == 1
    assert config.use_cache is True


def test_the_model_revision_is_pinned_to_the_recorded_commit():
    assert FROZEN_REVISION == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    with pytest.raises(GenerationConfigError, match="revision"):
        GenerationConfig(revision="main").validate()
    with pytest.raises(GenerationConfigError, match="revision"):
        GenerationConfig(tokenizer_revision="main").validate()


def test_a_different_model_is_refused():
    with pytest.raises(GenerationConfigError, match="pinned"):
        GenerationConfig(model="Qwen/Qwen3-8B").validate()


def test_thinking_must_be_disabled():
    with pytest.raises(GenerationConfigError, match="thinking"):
        GenerationConfig(enable_thinking=True).validate()


def test_thinking_must_be_fail_closed():
    with pytest.raises(GenerationConfigError, match="fail-closed"):
        GenerationConfig(enable_thinking_fail_closed=False).validate()


def test_sampling_is_refused():
    with pytest.raises(GenerationConfigError, match="greedy"):
        GenerationConfig(do_sample=True).validate()
    for kwargs in ({"temperature": 0.7}, {"top_p": 0.9}, {"top_k": 50}):
        with pytest.raises(GenerationConfigError, match="unset"):
            GenerationConfig(**kwargs).validate()


def test_a_batch_larger_than_one_is_refused():
    with pytest.raises(GenerationConfigError, match="batch"):
        GenerationConfig(batch_size=4).validate()


def test_a_system_prompt_is_refused():
    with pytest.raises(GenerationConfigError, match="system prompt"):
        GenerationConfig(system_prompt="be helpful").validate()


def test_a_non_eager_attention_implementation_is_refused():
    with pytest.raises(GenerationConfigError, match="eager"):
        GenerationConfig(attn_implementation="sdpa").validate()


def test_a_non_bfloat16_dtype_is_refused():
    with pytest.raises(GenerationConfigError, match="bfloat16"):
        GenerationConfig(dtype="float16").validate()


def test_the_fingerprint_changes_with_any_setting():
    base = GenerationConfig()
    assert base.fingerprint() == GenerationConfig().fingerprint()
    assert base.fingerprint() != GenerationConfig(max_new_tokens=512).fingerprint()


def test_every_intervention_type_is_declared():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert set(config["intervention"]["available_types"]) == set(INTERVENTION_TYPES)


# --------------------------------- thinking-disabled prompt rendering, mocked
class _Tokenizer:
    """The smallest thing that can be asked to render a chat prompt."""

    def __init__(self, accepts_enable_thinking: bool = True):
        self.accepts = accepts_enable_thinking
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        if "enable_thinking" in kwargs and not self.accepts:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.calls.append({"messages": messages, "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt,
                           **kwargs})
        return "<|im_start|>user\n" + messages[-1]["content"] + "<|im_end|>\n"


def test_the_chat_template_is_called_with_thinking_disabled():
    tokenizer = _Tokenizer()
    build_chat_strict(tokenizer, "hello")
    call = tokenizer.calls[-1]
    assert call["enable_thinking"] is False
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False
    assert call["messages"] == [{"role": "user", "content": "hello"}]


def test_a_tokenizer_that_cannot_disable_thinking_aborts_the_run():
    tokenizer = _Tokenizer(accepts_enable_thinking=False)
    with pytest.raises(GenerationConfigError, match="enable_thinking"):
        build_chat_strict(tokenizer, "hello")
    assert tokenizer.calls == []      # nothing was rendered


def test_no_system_message_is_ever_rendered():
    tokenizer = _Tokenizer()
    build_chat_strict(tokenizer, "hello")
    roles = [m["role"] for m in tokenizer.calls[-1]["messages"]]
    assert roles == ["user"]


# ----------------------------------------------------------- model geometry
class _Config:
    def __init__(self, num_hidden_layers, hidden_size):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size


class _Model:
    def __init__(self, num_hidden_layers=EXPECTED_NUM_LAYERS,
                 hidden_size=EXPECTED_HIDDEN_SIZE):
        self.config = _Config(num_hidden_layers, hidden_size)


def test_the_expected_model_geometry_is_accepted():
    assert check_model_geometry(_Model())["n_states"] == EXPECTED_HIDDEN_STATES


def test_a_model_of_the_wrong_shape_is_refused():
    with pytest.raises(GenerationConfigError, match="blocks of width"):
        check_model_geometry(_Model(num_hidden_layers=32))
    with pytest.raises(GenerationConfigError, match="blocks of width"):
        check_model_geometry(_Model(hidden_size=4096))


# ------------------------------------------------------------- finish flags
def test_finish_flags_are_decided_explicitly():
    assert finish_flags([1, 2, 7], 10, 7) == (True, False)
    assert finish_flags([1, 2, 3], 3, 7) == (False, True)
    assert finish_flags([], 10, 7) == (False, False)
    assert finish_flags([1, 7], 2, 7) == (True, True)
