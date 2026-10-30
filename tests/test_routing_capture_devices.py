"""RouterCapture: cross-device aggregation and host-side buffering.

Qwen3-30B-A3B is loaded with ``device_map="auto"``, so its MoE blocks are spread
over several GPUs and each block's router output arrives on that block's own
CUDA device. Nothing may be stacked or concatenated across blocks while its
members still live on an accelerator.

These tests run entirely on the CPU. ``OffHostTensor`` is an ordinary CPU tensor
that merely *reports* another device and turns into a plain tensor when it is
moved to ``"cpu"``, which is enough to exercise the placement logic without a
GPU. No Qwen, no weights, no CUDA.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

SCRIPT = (Path(__file__).resolve().parents[1]
          / "experiments/routegate_routing/extract_routing_v1.py")

N_BLOCKS = 4
N_EXPERTS = 6
SEQ_LEN = 5


def _module():
    """Load the extraction script by path: it lives outside the package."""
    spec = importlib.util.spec_from_file_location("extract_routing_v1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # safe: no heavy imports at module scope
    return mod


# The script resolves its inputs from configs/paths.local.json when imported.
if not (Path(__file__).resolve().parents[1] / "configs" / "paths.local.json").is_file():
    pytest.skip("configs/paths.local.json is absent", allow_module_level=True)

M = _module()


class OffHostTensor(torch.Tensor):
    """A CPU tensor that claims to live on another device.

    ``.detach()``, slicing and dtype casts keep the claim, exactly as they
    would on a real accelerator; ``.to("cpu")`` is the only thing that gives it
    up, and it returns a plain tensor, which is how a genuine host transfer
    behaves. The claim lives on the class rather than the instance because a
    Tensor subclass propagates its type through operations but not its Python
    attributes.
    """

    FAKE_DEVICE = "cuda:0"

    @property
    def device(self):
        return torch.device(self.FAKE_DEVICE)

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target is not None and str(target) == "cpu":
            return self.clone().as_subclass(torch.Tensor)
        return super().to(*args, **kwargs)


_OFF_HOST_CLASSES: dict = {}


def off_host(t: torch.Tensor, device: str = "cuda:0") -> OffHostTensor:
    cls = _OFF_HOST_CLASSES.get(device)
    if cls is None:
        cls = type(f"OffHost_{device.replace(':', '_')}", (OffHostTensor,),
                   {"FAKE_DEVICE": device})
        _OFF_HOST_CLASSES[device] = cls
    return t.as_subclass(cls)


class ToyRouter(nn.Module):
    """Stands in for an MoE block's router Linear.

    ``block_device`` is the device the block would sit on under
    ``device_map="auto"``; every block reports a different one, so a run that
    aggregates before transferring would be stacking across devices.
    """

    def __init__(self, block_index: int, rank: int = 3):
        super().__init__()
        self.block_index = block_index
        self.rank = rank
        torch.manual_seed(1000 + block_index)
        self.logits = torch.randn(1, SEQ_LEN, N_EXPERTS, dtype=torch.bfloat16)

    def forward(self, x):
        out = self.logits if self.rank == 3 else self.logits[0]
        return off_host(out.clone(), f"cuda:{self.block_index % 4}")


def _moe(rank: int = 3):
    return [dict(block_index=i, _module=ToyRouter(i, rank)) for i in range(N_BLOCKS)]


def _fire(cap, moe):
    for m in moe:
        m["_module"](torch.zeros(1))
    return cap


# ------------------------------------------------------------------ the hook
def test_the_toy_blocks_really_span_several_devices():
    """Guards the guard: if the fixture degenerated to one device the tests
    below would pass on code that cannot survive device_map="auto"."""
    moe = _moe()
    devices = {str(m["_module"](torch.zeros(1)).device) for m in moe}
    assert devices == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}, devices


def test_hook_leaves_nothing_on_the_accelerator():
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        assert set(cap.buf) == {m["block_index"] for m in moe}
        for bi, t in cap.buf.items():
            assert t.device.type == "cpu", f"block {bi} still on {t.device}"


def test_hook_records_the_full_shape_for_validation():
    moe = _moe(rank=3)
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        assert all(s == (1, SEQ_LEN, N_EXPERTS) for s in cap.shapes.values())
    moe2 = _moe(rank=2)
    with M.RouterCapture(moe2) as cap:
        _fire(cap, moe2)
        assert all(s == (SEQ_LEN, N_EXPERTS) for s in cap.shapes.values())


def test_clear_drops_buffers_and_shapes():
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        cap.clear()
        assert cap.buf == {} and cap.shapes == {}


def test_hooks_are_removed_on_exit():
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        pass
    _fire(cap, moe)
    assert cap.buf == {}


# ----------------------------------------------------------- the aggregation
@pytest.mark.parametrize("rank", [3, 2])
def test_aggregation_is_bit_identical_to_direct_indexing(rank):
    np = pytest.importorskip("numpy")
    moe = _moe(rank)
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        got = cap.final_token_logits(SEQ_LEN)

    # the reference: index the final prompt token straight out of the source
    # tensor, as the pre-fix code did once every block was on one device
    want = torch.stack([m["_module"].logits[0, -1, :].to(torch.float32)
                        for m in moe], dim=0).numpy()
    assert got.shape == (N_BLOCKS, N_EXPERTS)
    assert got.dtype == np.float32
    assert (got == want).all(), "the captured value is no longer the raw final-token logit"


def test_nothing_is_stacked_while_off_host(monkeypatch):
    """The regression: every element handed to torch.stack must be on the CPU."""
    seen = []
    real_stack = torch.stack
    monkeypatch.setattr(torch, "stack",
                        lambda ts, **kw: (seen.append([str(t.device) for t in ts]),
                                          real_stack(ts, **kw))[1])
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        cap.final_token_logits(SEQ_LEN)
    assert seen and all(d == "cpu" for devs in seen for d in devs), seen


def test_aggregation_transfers_anything_still_off_host(monkeypatch):
    """The stated principle: each per-block tensor is detached and moved to the
    CPU individually BEFORE the stack, so a value that reached the buffer still
    claiming a device is transferred rather than stacked across devices."""
    seen = []
    real_stack = torch.stack
    monkeypatch.setattr(torch, "stack",
                        lambda ts, **kw: (seen.append([str(t.device) for t in ts]),
                                          real_stack(ts, **kw))[1])
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        reference = cap.final_token_logits(SEQ_LEN)
        # as if the hook had not moved this one off its device
        cap.buf[0] = off_host(cap.buf[0].clone(), "cuda:3")
        got = cap.final_token_logits(SEQ_LEN)
    assert all(d == "cpu" for devs in seen for d in devs), seen
    assert (got == reference).all()


class StuckTensor(OffHostTensor):
    """A tensor whose transfer to the host never takes: it keeps claiming a
    device. Models a capture path that fails to reach host memory."""

    def to(self, *args, **kwargs):
        return self


def test_aggregation_refuses_to_stack_a_tensor_that_never_reached_the_host():
    """Belt and braces: the stack is refused with a clear message rather than
    raising a raw cross-device RuntimeError out of torch."""
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        cap.buf[0] = cap.buf[0].clone().as_subclass(StuckTensor)
        with pytest.raises(M.ExtractionError, match="on the CPU before aggregation"):
            cap.final_token_logits(SEQ_LEN)


# ------------------------------------------------- validation is not weakened
def test_missing_block_fails():
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        del cap.buf[2]
        with pytest.raises(M.ExtractionError, match="produced no tensor"):
            cap.final_token_logits(SEQ_LEN)


def test_wrong_seq_len_fails():
    moe = _moe(rank=3)
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        with pytest.raises(M.ExtractionError, match="router output seq"):
            cap.final_token_logits(SEQ_LEN + 1)
    moe2 = _moe(rank=2)
    with M.RouterCapture(moe2) as cap:
        _fire(cap, moe2)
        with pytest.raises(M.ExtractionError,
                           match="cannot identify the final prompt token"):
            cap.final_token_logits(SEQ_LEN + 1)


def test_unexpected_rank_fails():
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        cap.shapes[1] = (1, 1, SEQ_LEN, N_EXPERTS)
        with pytest.raises(M.ExtractionError, match="unexpected router output rank"):
            cap.final_token_logits(SEQ_LEN)


def test_batch_greater_than_one_fails():
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        cap.shapes[0] = (2, SEQ_LEN, N_EXPERTS)
        with pytest.raises(M.ExtractionError, match="router output batch"):
            cap.final_token_logits(SEQ_LEN)


# ------------------------------------------------------ derived features only
def test_derive_is_unchanged_by_placement():
    """derive() consumes the host array, so it is placement-independent."""
    np = pytest.importorskip("numpy")
    moe = _moe()
    with M.RouterCapture(moe) as cap:
        _fire(cap, moe)
        lg = cap.final_token_logits(SEQ_LEN)
    p, order, ent, margin = M.derive(lg)
    assert p.shape == (N_BLOCKS, N_EXPERTS)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert (margin >= 0).all() and (ent > 0).all()
