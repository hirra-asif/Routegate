"""The pinned model, the pinned prompt rendering, and deterministic generation.

Everything that could make two runs disagree is nailed down here and nowhere
else: which weights, which tokeniser, which chat template, which dtype, which
attention implementation, whether thinking is on, and how tokens are chosen.

The values are not invented. They are the ones the completed development
generation and residual extraction used, which is what makes a patched response
comparable to the recorded original response at all:

    model / tokenizer   Qwen/Qwen3-30B-A3B @ ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
    dtype / attention   bfloat16, eager
    chat template       apply_chat_template(add_generation_prompt=True,
                        enable_thinking=False), FAIL-CLOSED
    system prompt       none, ever; exactly one user message
    decoding            greedy (do_sample=False), use_cache=True
    batch size          1

Two of those deserve their reason stated.

**Thinking is fail-closed.** The historical pathway probe caught the ``TypeError``
from a tokeniser that does not accept ``enable_thinking`` and retried without
it, which silently produces thinking-enabled text. The later frozen generation
code refuses instead, and so does this: a run that cannot prove thinking is off
does not run.

**Batch size is pinned, not defaulted.** The intervention position is "the final
prompt token". With one prompt per forward there is no padding, so the last row
*is* that token. Add padding and the last row becomes whatever the padding put
there, and every edit lands in the wrong place while everything still appears to
work.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .validation import ValidationError, sha256_file, sha256_text

FROZEN_MODEL = "Qwen/Qwen3-30B-A3B"
FROZEN_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
FROZEN_TOKENIZER_CONFIG_SHA256 = (
    "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101")
FROZEN_CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8")

# The development responses this stage compares against were generated at 768.
# RQ1's causal-repair runs used 512; the difference is recorded rather than
# reconciled, because 768 is what the "original response" in the corpus is.
FROZEN_MAX_NEW_TOKENS = 768

EXPECTED_HIDDEN_STATES = 49          # 48 transformer blocks plus the embedding
EXPECTED_HIDDEN_SIZE = 2048
EXPECTED_NUM_LAYERS = 48

TOKEN_POSITION = -1
TOKEN_POSITION_MEANING = ("the final token of the rendered chat prompt, "
                          "immediately before the first generated answer token")


class GenerationConfigError(ValidationError):
    """The generation configuration is not the frozen one."""


@dataclass(frozen=True)
class GenerationConfig:
    """Every knob that could change what the model produces."""

    model: str = FROZEN_MODEL
    revision: str = FROZEN_REVISION
    tokenizer_revision: str = FROZEN_REVISION
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    device_map: str = "auto"
    trust_remote_code: bool = True

    enable_thinking: bool = False
    enable_thinking_fail_closed: bool = True
    add_generation_prompt: bool = True
    system_prompt: Optional[str] = None
    messages_per_prompt: str = "single user message"

    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = FROZEN_MAX_NEW_TOKENS
    use_cache: bool = True
    batch_size: int = 1

    token_position: int = TOKEN_POSITION
    verify_snapshot_hashes: bool = True

    def validate(self) -> None:
        """Refuse anything that is not the frozen configuration."""
        if self.model != FROZEN_MODEL:
            raise GenerationConfigError(
                f"model {self.model!r} is not the pinned {FROZEN_MODEL!r}")
        if self.revision != FROZEN_REVISION or self.tokenizer_revision != FROZEN_REVISION:
            raise GenerationConfigError(
                f"revision must be pinned to {FROZEN_REVISION} for both model "
                f"and tokenizer; got {self.revision!r} / "
                f"{self.tokenizer_revision!r}. refs/main is never consulted.")
        if self.dtype != "bfloat16":
            raise GenerationConfigError(
                f"dtype {self.dtype!r} is not the bfloat16 the corpus was "
                f"produced with")
        if self.attn_implementation != "eager":
            raise GenerationConfigError(
                f"attn_implementation {self.attn_implementation!r} is not eager; "
                f"a fused kernel can reorder the arithmetic a hook reads")
        if self.enable_thinking:
            raise GenerationConfigError(
                "thinking must be disabled at this stage")
        if not self.enable_thinking_fail_closed:
            raise GenerationConfigError(
                "enable_thinking must be fail-closed: a tokeniser that rejects "
                "the flag must abort the run, not silently generate with "
                "thinking on")
        if self.system_prompt is not None:
            raise GenerationConfigError(
                "no system prompt is used at any point in this study")
        if self.do_sample:
            raise GenerationConfigError(
                "decoding must be greedy; sampling makes a patched response "
                "differ from its baseline for reasons that are not the patch")
        for name in ("temperature", "top_p", "top_k"):
            if getattr(self, name) is not None:
                raise GenerationConfigError(
                    f"{name} must be unset under greedy decoding, got "
                    f"{getattr(self, name)!r}")
        if self.batch_size != 1:
            raise GenerationConfigError(
                f"batch size must be 1 so that token position -1 is the final "
                f"prompt token and never a pad token; got {self.batch_size}")
        if self.token_position != TOKEN_POSITION:
            raise GenerationConfigError(
                f"token position must be {TOKEN_POSITION} ({TOKEN_POSITION_MEANING}); "
                f"got {self.token_position}")
        if self.max_new_tokens < 1:
            raise GenerationConfigError(
                f"max_new_tokens must be positive, got {self.max_new_tokens}")

    def as_dict(self) -> Dict[str, Any]:
        record = asdict(self)
        record["token_position_meaning"] = TOKEN_POSITION_MEANING
        return record

    def fingerprint(self) -> str:
        """A hash of the settings, for pinning a resumed run to its origin."""
        return sha256_text(json.dumps(self.as_dict(), sort_keys=True,
                                      separators=(",", ":")))


# ------------------------------------------------------------------- prompting
def build_chat_strict(tokenizer: Any, prompt: str,
                      config: Optional[GenerationConfig] = None) -> str:
    """Render one user message, thinking disabled, no fallback.

    A ``TypeError`` from ``apply_chat_template`` means this tokeniser does not
    know the flag, which means the run cannot demonstrate that thinking is off.
    That is a refusal, not a warning.
    """
    config = config or GenerationConfig()
    config.validate()
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=config.add_generation_prompt,
            enable_thinking=config.enable_thinking,
        )
    except TypeError as exc:
        raise GenerationConfigError(
            "tokenizer.apply_chat_template rejected enable_thinking="
            f"{config.enable_thinking}; refusing to generate without the frozen "
            "flag") from exc


def finish_flags(token_ids: Sequence[int], max_new_tokens: int,
                 eos_token_id: Optional[int]) -> Tuple[bool, bool]:
    """(eos reached, hit the cap), each decided explicitly.

    Neither is read off a ``finish_reason`` string: a truncated answer and a
    finished one are different outcomes and the record must not guess.
    """
    eos_reached = bool(token_ids) and eos_token_id is not None \
        and token_ids[-1] == eos_token_id
    hit_max = len(token_ids) >= max_new_tokens
    return eos_reached, hit_max


# ----------------------------------------------------------------- model load
def resolve_snapshot(config: Optional[GenerationConfig] = None) -> Tuple[str, Path]:
    """Resolve the local snapshot for the pinned revision, offline.

    ``local_files_only=True`` and an explicit revision: this never reaches the
    network and never resolves refs/main. The tokeniser config and the chat
    template are hashed too, because the revision pins the weights while the
    template is what decides where the final prompt token actually is.
    """
    config = config or GenerationConfig()
    config.validate()
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(config.model, revision=config.revision,
                                  local_files_only=True))
    resolved = path.name
    if resolved != config.revision:
        raise GenerationConfigError(
            f"resolved snapshot commit {resolved} != pinned revision "
            f"{config.revision}")
    if not config.verify_snapshot_hashes:
        return resolved, path

    tokenizer_config = path / "tokenizer_config.json"
    digest = sha256_file(tokenizer_config)
    if digest != FROZEN_TOKENIZER_CONFIG_SHA256:
        raise GenerationConfigError(
            f"tokenizer_config.json SHA256 {digest} != frozen "
            f"{FROZEN_TOKENIZER_CONFIG_SHA256}")
    template = json.loads(tokenizer_config.read_text(encoding="utf-8")).get(
        "chat_template")
    if not isinstance(template, str):
        raise GenerationConfigError(
            "the snapshot's tokenizer_config.json carries no chat_template")
    template_digest = sha256_text(template)
    if template_digest != FROZEN_CHAT_TEMPLATE_SHA256:
        raise GenerationConfigError(
            f"chat_template SHA256 {template_digest} != frozen "
            f"{FROZEN_CHAT_TEMPLATE_SHA256}")
    return resolved, path


def load_model_and_tokenizer(config: Optional[GenerationConfig] = None):
    """Load the pinned model and tokeniser. Never falls back to refs/main."""
    config = config or GenerationConfig()
    config.validate()
    resolve_snapshot(config)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.tokenizer_revision,
        trust_remote_code=config.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=dtype, device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        low_cpu_mem_usage=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    check_model_geometry(model)
    return model, tokenizer


def check_model_geometry(model: Any) -> Dict[str, int]:
    """Refuse a model whose shape is not the one every state index assumes."""
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    if not isinstance(n_layers, int) or not isinstance(hidden, int):
        raise GenerationConfigError(
            f"the model config does not declare num_hidden_layers and "
            f"hidden_size as integers (got {n_layers!r}, {hidden!r})")
    if n_layers != EXPECTED_NUM_LAYERS or hidden != EXPECTED_HIDDEN_SIZE:
        raise GenerationConfigError(
            f"the loaded model has {n_layers} blocks of width {hidden}, but this "
            f"stage expects {EXPECTED_NUM_LAYERS} of width {EXPECTED_HIDDEN_SIZE} "
            f"({EXPECTED_HIDDEN_STATES} states: the embedding output plus one "
            f"per block). State indices would not mean what they say.")
    return {"num_hidden_layers": n_layers, "hidden_size": hidden,
            "n_states": n_layers + 1}


def model_input_device(model: Any) -> Any:
    """Where to put the input ids for a possibly sharded model."""
    return model.model.embed_tokens.weight.device


# ------------------------------------------------------------------- running
#: How a prompt reaches the tokeniser.
#:
#: ``chat`` renders one user turn through the frozen chat template. It is what
#: every development stage used and is the only mode the sealed evaluation
#: uses.
#:
#: ``raw_completion`` passes the prompt through byte-for-byte. It exists for
#: the tinyBenchmarks capability items, whose ``input_formatted`` value is a
#: few-shot COMPLETION prompt ending at ``Answer:``. Wrapping such a prompt as
#: a chat turn asks an instruction-tuned model to reply in prose instead of
#: continuing the pattern, which is a different task from the one the
#: benchmark defines.
PROMPT_MODE_CHAT = "chat"
PROMPT_MODE_RAW = "raw_completion"
PROMPT_MODES: Tuple[str, ...] = (PROMPT_MODE_CHAT, PROMPT_MODE_RAW)

# ------------------------------------------------------- generation bounds
#: Why a generation stopped. Recorded so a truncated completion can never be
#: mistaken for a complete one, or the reverse.
STOP_REASON_EOS = "eos"
STOP_REASON_STOP_STRING = "stop_string"
STOP_REASON_MAX_NEW_TOKENS = "max_new_tokens"
STOP_REASONS: Tuple[str, ...] = (
    STOP_REASON_EOS, STOP_REASON_STOP_STRING, STOP_REASON_MAX_NEW_TOKENS)


def first_stop(text: str, stop_strings: Optional[Sequence[str]]
               ) -> Tuple[str, Optional[str]]:
    """Cut ``text`` at the EARLIEST complete stop sequence.

    Returns ``(kept text, the stop that fired or None)``. Operating on decoded
    text rather than token ids is what makes a multi-token stop sequence work:
    a sequence split across two tokens is still one substring once decoded.
    """
    if not stop_strings:
        return text, None
    best_at = len(text)
    fired: Optional[str] = None
    for stop in stop_strings:
        if not stop:
            continue
        at = text.find(stop)
        if at != -1 and at < best_at:
            best_at, fired = at, stop
    if fired is None:
        return text, None
    return text[:best_at], fired


def new_text_stop_criteria(tokenizer: Any, prompt_length: int,
                           stop_strings: Sequence[str]) -> Any:
    """Stop when the NEWLY GENERATED text contains a stop sequence.

    Scoped to ``input_ids[:, prompt_length:]`` on purpose. A few-shot prompt
    contains ``Question:`` many times; a criterion that looked at the whole
    sequence would fire before a single token was produced.
    """
    from transformers import StoppingCriteria

    class _NewTextStop(StoppingCriteria):
        def __init__(self) -> None:
            self.tokenizer = tokenizer
            self.prompt_length = int(prompt_length)
            self.stop_strings = [s for s in stop_strings if s]

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[0, self.prompt_length:]
            if generated.numel() == 0:
                return False
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return any(stop in text for stop in self.stop_strings)

    return _NewTextStop()


def render_prompt(tokenizer: Any, prompt: str,
                  config: Optional[GenerationConfig] = None,
                  prompt_mode: str = PROMPT_MODE_CHAT) -> str:
    """The exact text handed to the tokeniser, by declared mode."""
    if prompt_mode not in PROMPT_MODES:
        raise GenerationConfigError(
            f"unknown prompt mode {prompt_mode!r}; this runtime renders "
            f"{list(PROMPT_MODES)}")
    if prompt_mode == PROMPT_MODE_RAW:
        return prompt
    return build_chat_strict(tokenizer, prompt, config)


def tokenize_prompt(tokenizer: Any, prompt: str, model: Any,
                    config: Optional[GenerationConfig] = None,
                    prompt_mode: str = PROMPT_MODE_CHAT) -> Dict[str, Any]:
    """The rendered, tokenised prompt on the model's input device, batch of one."""
    config = config or GenerationConfig()
    text = render_prompt(tokenizer, prompt, config, prompt_mode)
    batch = tokenizer(text, return_tensors="pt")
    if int(batch["input_ids"].shape[0]) != config.batch_size:
        raise GenerationConfigError(
            f"expected a batch of {config.batch_size}, got "
            f"{int(batch['input_ids'].shape[0])}")
    device = model_input_device(model)
    return {key: value.to(device) for key, value in batch.items()}


def capture_prompt_states(model: Any, tokenizer: Any, prompt: str,
                          controller: Any, capture_name: str, example_id: str,
                          config: Optional[GenerationConfig] = None,
                          prompt_mode: str = PROMPT_MODE_CHAT) -> int:
    """One forward pass over the prompt, capturing the requested states.

    No text is generated: the vectors this stage needs all live in the prefill
    forward, and generating a donor's answer to obtain them would cost a full
    decode for something never read. Returns the prompt length in tokens, which
    is how the donor's own final-token position is located -- the donor and the
    target are tokenised independently and need not be the same length.
    """
    import torch

    config = config or GenerationConfig()
    config.validate()
    batch = tokenize_prompt(tokenizer, prompt, model, config, prompt_mode)
    controller.reset_run(mode="capture", example_id=example_id,
                         capture_name=capture_name)
    with torch.inference_mode():
        model(**batch, use_cache=False)
    return int(batch["input_ids"].shape[1])


def generate_response(model: Any, tokenizer: Any, prompt: str,
                      config: Optional[GenerationConfig] = None,
                      controller: Any = None,
                      mode: str = "off",
                      plans: Optional[Dict[int, Any]] = None,
                      example_id: Optional[str] = None,
                      prompt_mode: str = PROMPT_MODE_CHAT,
                      stop_strings: Optional[Sequence[str]] = None
                      ) -> Dict[str, Any]:
    """Greedy generation, optionally with an intervention installed.

    The controller, when given, is reset for this generation and verified
    afterwards: a patched run whose plan never fired raises rather than being
    recorded as an intervention that did nothing.
    """
    import torch

    config = config or GenerationConfig()
    config.validate()
    if controller is not None:
        controller.reset_run(mode=mode, example_id=example_id, plans=plans)

    batch = tokenize_prompt(tokenizer, prompt, model, config, prompt_mode)
    prompt_length = int(batch["input_ids"].shape[1])
    criteria = None
    if stop_strings:
        from transformers import StoppingCriteriaList
        criteria = StoppingCriteriaList([
            new_text_stop_criteria(tokenizer, prompt_length, stop_strings)])
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **batch,
            do_sample=config.do_sample,
            max_new_tokens=config.max_new_tokens,
            use_cache=config.use_cache,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **({"stopping_criteria": criteria} if criteria is not None else {}),
        )
    elapsed = time.time() - started

    generated = output[0, prompt_length:]
    token_ids = [int(t) for t in generated.tolist()]
    eos_reached, hit_max = finish_flags(token_ids, config.max_new_tokens,
                                        tokenizer.eos_token_id)
    if controller is not None and mode == "patch":
        controller.verify_applied()

    # The stop sequence and everything after it are removed BEFORE the
    # completion is stored, so nothing downstream can parse past the boundary.
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    kept, fired = first_stop(raw_text, stop_strings)
    if fired is not None:
        stop_reason = STOP_REASON_STOP_STRING
        hit_max = False
    elif eos_reached:
        stop_reason = STOP_REASON_EOS
    elif hit_max:
        stop_reason = STOP_REASON_MAX_NEW_TOKENS
    else:
        stop_reason = STOP_REASON_EOS

    return {
        "response_text": kept.strip(),
        "stop_strings": list(stop_strings or []),
        "stop_fired": fired is not None,
        "stop_reason": stop_reason,
        "prompt_token_count": prompt_length,
        "completion_token_count": len(token_ids),
        "eos_reached": bool(eos_reached),
        "hit_max_new_tokens": bool(hit_max),
        "elapsed_seconds": round(elapsed, 3),
        "prompt_mode": prompt_mode,
        "hooks_fired": int(sum(int(v) for v in getattr(
            getattr(controller, "run", None), "hits", {}).values())
            if controller is not None else 0),
        "applied_by_state": dict(getattr(getattr(controller, "run", None),
                                         "applied", {}) or {}),
    }
