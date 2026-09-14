# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hook-path reload for offline FP8 per-block quantization (doc section 8).

Simulates the three backend layout families without GPU kernels:
identity (Triton/CUTLASS dense), FlashInfer CUTLASS MoE (w31 swap +
scale clamp), and DeepGemm (scale staging + layout transform at finish).
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.model_loader.reload.fp8_block import (
    DeepGemmScaleHook,
    DeepGemmUe8m0Hook,
    Fp8BlockClampHook,
    Fp8BlockIdentityHook,
    Fp8MoeW31SwapHook,
)
from vllm.model_executor.model_loader.reload.hooks import (
    HookState,
    ReloadContext,
    ReloadRejected,
    RuntimeSlot,
)
from vllm.model_executor.model_loader.reload.inplace import (
    finalize_reload,
    initialize_reload,
    install_hook_reload_observers,
    reload_used_hooks,
    supports_hook_reload,
)

E, N_MOE, K_MOE, BLOCK = 4, 8, 16, 4  # block 4x4 keeps tensors tiny


def moe_loader(param, loaded_weight, weight_name, shard_id, expert_id):
    # mimics vLLM's fused MoE loader for w13 (shard_id w1/w3) and w2
    if shard_id in ("w1", "w3"):
        offset = 0 if shard_id == "w1" else N_MOE
        param.data[expert_id, offset : offset + N_MOE].copy_(loaded_weight)
    else:
        param.data[expert_id].copy_(loaded_weight)


def moe_scale_loader(param, loaded_weight, weight_name, shard_id, expert_id):
    # scale tensors: w13 scale (E, 2N/B, K/B), w2 scale (E, K/B, N/B)
    if shard_id in ("w1", "w3"):
        half = N_MOE // BLOCK
        offset = 0 if shard_id == "w1" else half
        param.data[expert_id, offset : offset + half].copy_(loaded_weight)
    else:
        param.data[expert_id].copy_(loaded_weight)


def _bind(loader, param, w, sid, e):
    return inspect.signature(loader).bind(param, w, "weight_name", sid, e)


def test_w31_swap_hook_routes_shards_to_swapped_halves():
    # runtime slot is w31-swapped: rows [0,N) hold up (w3), [N,2N) gate (w1)
    param = torch.nn.Parameter(torch.zeros(E, 2 * N_MOE, K_MOE))
    slot = RuntimeSlot("layer.w13_weight", param)
    hook = Fp8MoeW31SwapHook(slot, moe_loader)
    for e in range(E):
        hook.tracker.add_slot(("w1", e), (N_MOE, K_MOE))
        hook.tracker.add_slot(("w3", e), (N_MOE, K_MOE))
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()

    hook.load_from_loader(
        _bind(moe_loader, param, torch.full((N_MOE, K_MOE), 1.0), "w1", 0)
    )
    assert not hook.complete
    # w1 content must land in the swapped (upper) half of the runtime slot
    assert torch.all(slot.tensor[0, :N_MOE] == 0)
    assert torch.all(slot.tensor[0, N_MOE:] == 1.0)
    # the tracker records the checkpoint-side shard id
    assert hook.tracker.has_arrived(("w1", 0))

    hook.load_from_loader(
        _bind(moe_loader, param, torch.full((N_MOE, K_MOE), 2.0), "w3", 0)
    )
    assert torch.all(slot.tensor[0, :N_MOE] == 2.0)
    assert hook.tracker.has_arrived(("w3", 0))


def test_w31_swap_rejects_unknown_shard_before_write():
    param = torch.nn.Parameter(torch.zeros(E, 2 * N_MOE, K_MOE))
    slot = RuntimeSlot("layer.w13_weight", param)
    hook = Fp8MoeW31SwapHook(slot, moe_loader)
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    before = param.detach().clone()
    with pytest.raises(ReloadRejected):
        hook.load_from_loader(
            _bind(moe_loader, param, torch.ones(N_MOE, K_MOE), "wX", 0)
        )
    torch.testing.assert_close(slot.tensor, before)


def test_clamp_hook_defers_clamp_to_finish_load():
    param = torch.nn.Parameter(torch.full((E, 2, 2), 1e-12))
    slot = RuntimeSlot("layer.w2_weight_scale_inv", param)
    hook = Fp8BlockClampHook(slot, moe_scale_loader, clamp_min=1e-10)
    for e in range(E):
        hook.tracker.add_slot((None, e), (2, 2))
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    for e in range(E):
        payload = torch.full((2, 2), 1e-12)
        hook.load_from_loader(_bind(moe_scale_loader, param, payload, None, e))
        if e < E - 1:
            # clamp must not run before the last shard commits
            assert torch.all(slot.tensor == 1e-12)
    assert hook.complete
    assert torch.all(slot.tensor == 1e-10)


def test_deepgemm_scale_hook_stages_and_transforms_at_finish():
    runtime_scale = torch.zeros(3, 3)  # pretend runtime layout differs
    slot = RuntimeSlot("layer.weight_scale_inv", runtime_scale)

    def fake_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    def fake_transform(ws: torch.Tensor) -> torch.Tensor:
        assert ws.shape == (2, 2)
        # stand-in for transform_sf_into_required_layout (pads to a (3,3)
        # atom-aligned layout)
        return torch.nn.functional.pad(ws, (0, 1, 0, 1)) + 1

    hook = DeepGemmScaleHook(
        slot,
        fake_loader,
        staging_shape=(2, 2),
        block_shape=(BLOCK, BLOCK),
        mn=8,
        k=8,
        transform=fake_transform,
    )
    hook.tracker.add_slot("full", (2, 2))
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    src = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 1
    hook.load_from_loader(inspect.signature(fake_loader).bind(slot.tensor, src))
    assert hook.complete
    assert hook._staging is None  # staging freed after finish_load
    torch.testing.assert_close(slot.tensor, fake_transform(src))
    # runtime storage untouched until finish_load
    assert slot.data_ptr == runtime_scale.data_ptr()


def test_deepgemm_scale_hook_shape_mismatch_is_hard_error():
    param = torch.nn.Parameter(torch.zeros(2, 2))
    slot = RuntimeSlot("s", param)
    hook = DeepGemmScaleHook(
        slot,
        lambda param, loaded_weight: param.data.copy_(loaded_weight),
        staging_shape=(2, 2),
        block_shape=(BLOCK, BLOCK),
        mn=8,
        k=8,
        transform=lambda ws: torch.zeros(3, 3),  # wrong runtime shape
    )
    hook.tracker.add_slot("full", (2, 2))
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    with pytest.raises(RuntimeError, match="DeepGemm scale transform"):
        hook.load_from_loader(
            inspect.signature(hook._original_loader).bind(
                param, torch.ones(2, 2)
            )
        )


def test_deepgemm_ue8m0_pair_commits_weight_and_scale_together(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(1, 1)
    weight_slot = RuntimeSlot("weight", weight)
    scale_slot = RuntimeSlot("scale", scale)

    def loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    def fake_process(wq, ws, **kwargs):
        assert kwargs["use_e8m0"] is True
        return torch.full_like(wq, 2), ws + 2

    monkeypatch.setattr(
        fp8_utils, "deepgemm_post_process_fp8_weight_block", fake_process
    )
    hook = DeepGemmUe8m0Hook(
        weight_slot=weight_slot,
        weight_loader=loader,
        weight_param=weight,
        weight_name="weight",
        scale_slot=scale_slot,
        scale_loader=loader,
        scale_param=scale,
        scale_name="scale",
        block_shape=(4, 4),
        mn=4,
        k=4,
    )
    hook.add_slot("weight", "full", (4, 4))
    hook.add_slot("scale", "full", (1, 1))
    ctx = ReloadContext()
    ctx.register(hook, "scale")
    ctx.start_reload()
    assert list(ctx.hooks) == ["weight"]
    assert ctx.hook("scale") is hook

    hook.load_from_loader(
        inspect.signature(loader).bind(weight, torch.ones_like(weight)),
        "weight",
    )
    assert torch.all(weight == 0)
    assert not hook.complete

    hook.load_from_loader(
        inspect.signature(loader).bind(scale, torch.ones_like(scale)),
        "scale",
    )
    assert torch.all(weight.float() == 2)
    assert torch.all(scale == 3)
    assert hook.complete
    assert hook._weight_staging is None
    assert hook._scale_staging is None


# ---------- dispatch-level integration ----------


class FakeQuantMethod:
    """Stands in for Fp8MoEMethod: identity hooks for all params."""

    def __init__(self, supported=True):
        self._supported = supported
        self.made_hooks = []

    def supports_hook_reload(self):
        return self._supported

    def make_reload_hook(
        self, layer, param_name, slot, original_loader, cold_param=None
    ):
        if not self._supported:
            return None
        hook = Fp8BlockIdentityHook(slot, original_loader)
        self.made_hooks.append(param_name)
        return hook


class FakeFp8Model(torch.nn.Module):
    def __init__(self, quant_method):
        super().__init__()
        self.moe = torch.nn.Module()
        self.moe.w13_weight = torch.nn.Parameter(torch.zeros(E, 2 * N_MOE, K_MOE))
        self.moe.w2_weight = torch.nn.Parameter(torch.zeros(E, K_MOE, N_MOE))
        self.moe.w13_weight_scale_inv = torch.nn.Parameter(
            torch.ones(E, 2 * N_MOE // BLOCK, K_MOE // BLOCK)
        )
        self.moe.w2_weight_scale_inv = torch.nn.Parameter(
            torch.ones(E, K_MOE // BLOCK, N_MOE // BLOCK)
        )
        self.moe.w13_weight.weight_loader = moe_loader
        self.moe.w2_weight.weight_loader = moe_loader
        self.moe.w13_weight_scale_inv.weight_loader = moe_scale_loader
        self.moe.w2_weight_scale_inv.weight_loader = moe_scale_loader
        self.moe.quant_method = quant_method
        self.norm = torch.nn.Parameter(torch.zeros(4))
        self.norm.weight_loader = lambda param, w: param.data.copy_(w)

    def load_weights(self, weights):
        for name, w in weights:
            if name == "norm.weight":
                self.norm.weight_loader(self.norm, w)
            elif "w13_scale" in name:
                e = int(name.split(".")[1])
                sid = "w1" if name.endswith("w1") else "w3"
                self.moe.w13_weight_scale_inv.weight_loader(
                    self.moe.w13_weight_scale_inv, w, name, sid, e
                )
            elif "w2_scale" in name:
                e = int(name.split(".")[1])
                self.moe.w2_weight_scale_inv.weight_loader(
                    self.moe.w2_weight_scale_inv, w, name, None, e
                )
            elif ".w1." in name or ".w3." in name:
                e = int(name.split(".")[1])
                sid = "w1" if ".w1." in name else "w3"
                self.moe.w13_weight.weight_loader(
                    self.moe.w13_weight, w, name, sid, e
                )
            elif ".w2." in name:
                e = int(name.split(".")[1])
                self.moe.w2_weight.weight_loader(
                    self.moe.w2_weight, w, name, None, e
                )
            else:
                raise ValueError(name)
        return set()


def make_fp8_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    ckpt = {"norm.weight": rnd(4)}
    for e in range(E):
        ckpt[f"experts.{e}.w1.weight"] = rnd(N_MOE, K_MOE)
        ckpt[f"experts.{e}.w3.weight"] = rnd(N_MOE, K_MOE)
        ckpt[f"experts.{e}.w2.weight"] = rnd(K_MOE, N_MOE)
        ckpt[f"experts.{e}.w13_scale.w1"] = rnd(
            N_MOE // BLOCK, K_MOE // BLOCK
        ).abs()
        ckpt[f"experts.{e}.w13_scale.w3"] = rnd(
            N_MOE // BLOCK, K_MOE // BLOCK
        ).abs()
        ckpt[f"experts.{e}.w2_scale"] = rnd(K_MOE // BLOCK, N_MOE // BLOCK).abs()
    return ckpt


FP8_MODEL_CONFIG = SimpleNamespace(quantization="fp8")
NONQUANT_NONE = SimpleNamespace(quantization=None)
FP8_QUANT_CONFIG = SimpleNamespace(
    weight_block_size=[BLOCK, BLOCK], is_checkpoint_fp8_serialized=True
)


def test_fp8_block_model_uses_hook_path():
    qm = FakeQuantMethod()
    model = FakeFp8Model(qm)
    install_hook_reload_observers(model)
    assert supports_hook_reload(FP8_MODEL_CONFIG, False, FP8_QUANT_CONFIG)

    model.load_weights(list(make_fp8_checkpoint(1).items()))
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}

    initialize_reload(model, FP8_MODEL_CONFIG, quant_config=FP8_QUANT_CONFIG)
    model.load_weights(list(make_fp8_checkpoint(2).items()))
    finalize_reload(model, FP8_MODEL_CONFIG)

    assert reload_used_hooks(model)
    # every quantized param was built by the quant method's factory
    assert set(qm.made_hooks) == {
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    }
    reference = FakeFp8Model(FakeQuantMethod())
    reference.load_weights(list(make_fp8_checkpoint(2).items()))
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)
    for name, p in model.named_parameters():
        assert p.data_ptr() == ptrs[name]


def test_fp8_unsupported_backend_raises_not_falls_back(monkeypatch):
    # hook reload never falls back to layerwise; unsupported configs raise
    from vllm.model_executor.model_loader.reload.hooks import (
        HookReloadUnsupportedError,
    )

    calls = []
    import vllm.model_executor.model_loader.reload.inplace as inplace
    import vllm.model_executor.model_loader.reload.layerwise as layerwise

    monkeypatch.setattr(
        layerwise,
        "initialize_layerwise_reload",
        lambda model: calls.append("layerwise"),
    )
    qm = FakeQuantMethod(supported=False)
    model = FakeFp8Model(qm)
    install_hook_reload_observers(model)
    model.load_weights(list(make_fp8_checkpoint(1).items()))
    with pytest.raises(HookReloadUnsupportedError):
        inplace.initialize_reload(
            model, FP8_MODEL_CONFIG, quant_config=FP8_QUANT_CONFIG
        )
    assert calls == []  # layerwise was not invoked
    assert not reload_used_hooks(model)


def test_quantized_gate_raises_not_falls_back():
    from vllm.model_executor.model_loader.reload.hooks import (
        HookReloadUnsupportedError,
    )

    model = FakeFp8Model(FakeQuantMethod())
    install_hook_reload_observers(model)
    model.load_weights(list(make_fp8_checkpoint(1).items()))
    # per-tensor fp8 (no block size): unsupported on the hook path
    with pytest.raises(HookReloadUnsupportedError):
        initialize_reload(
            model,
            FP8_MODEL_CONFIG,
            quant_config=SimpleNamespace(
                weight_block_size=None, is_checkpoint_fp8_serialized=True
            ),
        )


def test_supports_hook_reload_gates():
    assert supports_hook_reload(SimpleNamespace(quantization=None), False)
    assert supports_hook_reload(FP8_MODEL_CONFIG, False, FP8_QUANT_CONFIG)
    # per-tensor fp8 (no block size) stays on the layerwise path
    assert not supports_hook_reload(
        FP8_MODEL_CONFIG,
        False,
        SimpleNamespace(weight_block_size=None, is_checkpoint_fp8_serialized=True),
    )
    # online quantization (not fp8-serialized) is out of scope
    assert not supports_hook_reload(
        FP8_MODEL_CONFIG,
        False,
        SimpleNamespace(
            weight_block_size=[128, 128], is_checkpoint_fp8_serialized=False
        ),
    )
    assert not supports_hook_reload(FP8_MODEL_CONFIG, True, FP8_QUANT_CONFIG)


def test_mixed_model_unquantized_sublayers_use_generic_hook():
    # FP8 checkpoints still carry unquantized sublayers (embedding, lm_head,
    # MoE gate) whose quant_method is Unquantized*; they must not force a
    # whole-model fallback.
    class UnquantizedLinearMethod:
        pass

    class Model(FakeFp8Model):
        def __init__(self, quant_method):
            super().__init__(quant_method)
            self.gate = torch.nn.Module()
            self.gate.weight = torch.nn.Parameter(torch.zeros(4, 4))
            self.gate.weight.weight_loader = (
                lambda param, w: param.data.copy_(w)
            )
            self.gate.quant_method = UnquantizedLinearMethod()

        def load_weights(self, weights):
            rest = []
            for name, w in weights:
                if name == "gate.weight":
                    self.gate.weight.weight_loader(self.gate.weight, w)
                else:
                    rest.append((name, w))
            return super().load_weights(rest)

    qm = FakeQuantMethod()
    model = Model(qm)
    install_hook_reload_observers(model)
    ckpt = make_fp8_checkpoint(1)
    ckpt["gate.weight"] = torch.ones(4, 4)
    model.load_weights(list(ckpt.items()))

    initialize_reload(model, FP8_MODEL_CONFIG, quant_config=FP8_QUANT_CONFIG)
    ckpt2 = make_fp8_checkpoint(2)
    ckpt2["gate.weight"] = torch.zeros(4, 4)
    model.load_weights(list(ckpt2.items()))
    finalize_reload(model, FP8_MODEL_CONFIG)

    assert reload_used_hooks(model)
    assert torch.all(model.gate.weight == 0)


def test_reload_restores_param_class_dropped_by_pwal():
    # PWAL's replace_parameter re-registers a plain Parameter, dropping the
    # custom subclass that the weight loader depends on. The hook path must
    # restore the cold-load class before delegating.
    class CustomParam(torch.nn.Parameter):
        def custom_write(self, w):
            self.data.mul_(0).add_(w)

    def class_dependent_loader(param, loaded_weight):
        param.custom_write(loaded_weight)  # missing on a plain Parameter

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = CustomParam(torch.zeros(4, 4))
            self.weight.weight_loader = class_dependent_loader

        def load_weights(self, weights):
            for name, w in weights:
                self.weight.weight_loader(self.weight, w)
            return set()

    model = Model()
    install_hook_reload_observers(model)
    model.load_weights([("weight", torch.ones(4, 4))])

    # simulate PWAL: plain Parameter replacement keeps only weight_loader
    plain = torch.nn.Parameter(model.weight.detach().clone())
    plain.weight_loader = model.weight.weight_loader
    model.weight = plain
    assert not hasattr(model.weight, "custom_write")

    initialize_reload(model, NONQUANT_NONE)
    model.load_weights([("weight", torch.full((4, 4), 3.0))])
    finalize_reload(model, NONQUANT_NONE)

    assert isinstance(model.weight, CustomParam)
    assert torch.all(model.weight == 3.0)


# ---------- Marlin dense block hooks ----------


def _cold_param(shape, dtype=torch.float32):
    p = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
    p.loader_attr = "marker"  # loader-visible attribute to carry over
    return p


def test_marlin_weight_hook_stages_full_shape_and_tracks_shards(monkeypatch):
    from vllm.model_executor.model_loader.reload.marlin import (
        MarlinFp8WeightHook,
    )

    # the marlin repack needs CUDA ops; here we only verify the staging
    # mechanics and that finish_load fires exactly once at completion
    finished = []
    monkeypatch.setattr(
        MarlinFp8WeightHook, "finish_load", lambda self: finished.append(1)
    )

    # merged QKV-like param: block path is size_k_first=False, so the
    # checkpoint layout staging is (size_n, size_k)
    runtime = torch.nn.Parameter(
        torch.zeros(24, 8, dtype=torch.int32), requires_grad=False
    )
    slot = RuntimeSlot("layer.qkv.weight", runtime)
    cold = _cold_param((24, 8), torch.float8_e4m3fn)
    hook = MarlinFp8WeightHook(
        slot, None, cold, size_k=8, size_n=24, size_k_first=False,
        group_size=4,
    )
    for sid in ("q", "k", "v"):
        hook.tracker.add_slot((sid, None), (8, 8))

    def merged_loader(param, loaded_weight, loaded_shard_id):
        rows = {"q": (0, 8), "k": (8, 8), "v": (16, 8)}[loaded_shard_id]
        param.data[rows[0] : rows[0] + rows[1]].copy_(loaded_weight)

    hook._original_loader = merged_loader
    bind3 = lambda p, w, sid: inspect.signature(merged_loader).bind(  # noqa: E731
        p, w, loaded_shard_id=sid
    )
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()

    for i, sid in enumerate(("q", "k", "v")):
        hook.load_from_loader(
            bind3(
                slot.tensor,
                torch.full((8, 8), float(i + 1), dtype=torch.float8_e4m3fn),
                sid,
            )
        )
        if i == 0:
            # pre_reload fired at the first shard and allocated the
            # full checkpoint-layout staging buffer
            assert hook._staging is not None
            assert tuple(hook._staging.shape) == (24, 8)
            # cold-load loader attributes carried onto the staging buffer
            assert hook._staging.loader_attr == "marker"
    assert hook.complete
    # runtime untouched until finish_load (marlin repack needs CUDA)
    assert torch.all(slot.tensor == 0)
    staging = hook._staging
    for i in range(3):
        assert torch.all(
            staging[8 * i : 8 * i + 8].float() == float(i + 1)
        )
    assert finished == [1]


def test_marlin_hook_rejects_shard_before_pre_reload():
    from vllm.model_executor.model_loader.reload.marlin import (
        MarlinFp8ScaleHook,
    )

    slot = RuntimeSlot("layer.weight_scale_inv", torch.zeros(2, 2))
    cold = _cold_param((2, 2))
    hook = MarlinFp8ScaleHook(
        slot,
        lambda param, loaded_weight: param.data.copy_(loaded_weight),
        cold,
        size_k=8,
        size_n=8,
        size_k_first=False,
        block_size=(4, 4),
        input_dtype=None,
    )
    hook.tracker.add_slot("full", (2, 2))
    # bypass pre_reload to prove the guard
    hook.state = HookState.RECEIVING
    with pytest.raises(ReloadRejected, match="pre_reload"):
        hook.load_from_loader(
            inspect.signature(hook._original_loader).bind(
                slot.tensor, torch.ones(2, 2)
            )
        )
