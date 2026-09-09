# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hook-path reload through the vLLM weight_loader seam.

Uses a fake model whose ``load_weights`` routes checkpoint names to fused
parameters exactly like vLLM models do (merged QKV / gate_up with
``shard_id``, fused MoE with ``expert_id``, padded vocab embedding), then
drives the real adapter: observers record the plan at cold load, and
``initialize_reload`` / ``finalize_reload`` run the hook reload.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.model_loader.reload.hooks import (
    ReloadIncomplete,
    ReloadRejected,
)
from vllm.model_executor.model_loader.reload.inplace import (
    finalize_reload,
    initialize_reload,
    install_hook_reload_observers,
    reload_used_hooks,
    supports_hook_reload,
)

torch.manual_seed(0)

H, I, V_REAL, V_PAD, E, N_MOE, K_MOE = 8, 16, 11, 16, 4, 6, 8


def dense_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def make_merged_loader(shard_rows: dict[str, tuple[int, int]]):
    # mirrors vLLM's QKVParallelLinear/MergedColumnParallelLinear signature,
    # which names the shard argument `loaded_shard_id`, not `shard_id`
    def merged_loader(param, loaded_weight, loaded_shard_id):
        start, rows = shard_rows[loaded_shard_id]
        param.data[start : start + rows].copy_(loaded_weight)

    return merged_loader


def vocab_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    # padded vocab: only the real rows are loaded
    param.data[: loaded_weight.shape[0]].copy_(loaded_weight)


def moe_w13_loader(param, loaded_weight, weight_name, shard_id, expert_id):
    half = param.shape[1] // 2
    offset = 0 if shard_id == "w1" else half
    param.data[expert_id, offset : offset + half].copy_(loaded_weight)


def moe_w2_loader(param, loaded_weight, weight_name, shard_id, expert_id):
    param.data[expert_id].copy_(loaded_weight)


class FakeVllmModel(torch.nn.Module):
    """Mimics a vLLM model's parameter layout and load_weights routing."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.zeros(V_PAD, H))
        self.embed.weight_loader = vocab_loader
        self.qkv = torch.nn.Parameter(torch.zeros(3 * H, H))
        self.qkv.weight_loader = make_merged_loader(
            {"q": (0, H), "k": (H, H), "v": (2 * H, H)}
        )
        self.gate_up = torch.nn.Parameter(torch.zeros(2 * I, H))
        self.gate_up.weight_loader = make_merged_loader(
            {"gate": (0, I), "up": (I, I)}
        )
        self.norm = torch.nn.Parameter(torch.zeros(H))
        self.norm.weight_loader = dense_loader
        self.w13 = torch.nn.Parameter(torch.zeros(E, 2 * N_MOE, K_MOE))
        self.w13.weight_loader = moe_w13_loader
        self.w2 = torch.nn.Parameter(torch.zeros(E, K_MOE, N_MOE))
        self.w2.weight_loader = moe_w2_loader
        # buffers can be load targets too (e.g. attention scales)
        self.register_buffer("k_scale", torch.ones(1))
        self.k_scale.weight_loader = dense_loader

    def load_weights(self, weights):
        loaded = set()
        for name, w in weights:
            if name == "embed.weight":
                self.embed.weight_loader(self.embed, w)
                loaded.add("embed")
            elif name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
                shard_id = name.split(".")[-2][0]  # q / k / v
                self.qkv.weight_loader(self.qkv, w, loaded_shard_id=shard_id)
                loaded.add("qkv")
            elif name.endswith(("gate_proj.weight", "up_proj.weight")):
                shard_id = "gate" if "gate" in name else "up"
                self.gate_up.weight_loader(
                    self.gate_up, w, loaded_shard_id=shard_id
                )
                loaded.add("gate_up")
            elif name == "norm.weight":
                self.norm.weight_loader(self.norm, w)
                loaded.add("norm")
            elif name == "k_scale":
                self.k_scale.weight_loader(self.k_scale, w)
                loaded.add("k_scale")
            elif ".w1." in name or ".w3." in name:
                expert_id = int(name.split(".")[1])
                shard_id = "w1" if ".w1." in name else "w3"
                self.w13.weight_loader(self.w13, w, name, shard_id, expert_id)
                loaded.add("w13")
            elif ".w2." in name:
                expert_id = int(name.split(".")[1])
                self.w2.weight_loader(self.w2, w, name, None, expert_id)
                loaded.add("w2")
            else:
                raise ValueError(f"unexpected weight {name}")
        return loaded


def make_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    ckpt = {
        "embed.weight": rnd(V_REAL, H),
        "q_proj.weight": rnd(H, H),
        "k_proj.weight": rnd(H, H),
        "v_proj.weight": rnd(H, H),
        "gate_proj.weight": rnd(I, H),
        "up_proj.weight": rnd(I, H),
        "norm.weight": rnd(H),
        "k_scale": rnd(1),
    }
    for e in range(E):
        ckpt[f"experts.{e}.w1.weight"] = rnd(N_MOE, K_MOE)
        ckpt[f"experts.{e}.w3.weight"] = rnd(N_MOE, K_MOE)
        ckpt[f"experts.{e}.w2.weight"] = rnd(K_MOE, N_MOE)
    return ckpt


NONQUANT = SimpleNamespace(quantization=None)


def _reference_model(ckpt) -> FakeVllmModel:
    model = FakeVllmModel()
    model.load_weights(list(ckpt.items()))
    return model


def _assert_equal(a: FakeVllmModel, b: FakeVllmModel) -> None:
    params = lambda m: list(m.named_parameters()) + list(m.named_buffers())
    for (n1, p1), (n2, p2) in zip(params(a), params(b)):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)


def test_hook_reload_through_load_weights():
    model = FakeVllmModel()
    install_hook_reload_observers(model)
    ckpt_a = make_checkpoint(1)
    model.load_weights(list(ckpt_a.items()))  # cold load: observers record

    ckpt_b = make_checkpoint(2)
    reference = _reference_model(ckpt_b)
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}

    initialize_reload(model, NONQUANT)
    model.load_weights(list(ckpt_b.items()))
    finalize_reload(model, NONQUANT)

    _assert_equal(model, reference)
    for name, p in model.named_parameters():
        assert p.data_ptr() == ptrs[name], f"{name} storage moved"


def test_hook_reload_roundtrip_reuses_hooks():
    model = FakeVllmModel()
    install_hook_reload_observers(model)
    model.load_weights(list(make_checkpoint(1).items()))
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}

    for seed in (2, 1, 2):
        ckpt = make_checkpoint(seed)
        reference = _reference_model(ckpt)
        initialize_reload(model, NONQUANT)
        model.load_weights(list(ckpt.items()))
        finalize_reload(model, NONQUANT)
        _assert_equal(model, reference)
    for name, p in model.named_parameters():
        assert p.data_ptr() == ptrs[name]


def test_hook_reload_missing_shard_raises_incomplete():
    model = FakeVllmModel()
    install_hook_reload_observers(model)
    model.load_weights(list(make_checkpoint(1).items()))

    ckpt_b = make_checkpoint(2)
    missing = ckpt_b.pop("experts.2.w3.weight")
    initialize_reload(model, NONQUANT)
    model.load_weights(list(ckpt_b.items()))
    with pytest.raises(ReloadIncomplete, match="w13"):
        finalize_reload(model, NONQUANT)
    del missing


def test_hook_reload_rejects_unknown_shard_before_write():
    model = FakeVllmModel()
    install_hook_reload_observers(model)
    model.load_weights(list(make_checkpoint(1).items()))
    before = model.qkv.detach().clone()

    initialize_reload(model, NONQUANT)
    with pytest.raises(ReloadRejected):
        model.qkv.weight_loader(model.qkv, torch.zeros(H, H), "bad_shard")
    torch.testing.assert_close(model.qkv, before)


def test_hook_reload_rejects_shape_mismatch_before_write():
    model = FakeVllmModel()
    install_hook_reload_observers(model)
    model.load_weights(list(make_checkpoint(1).items()))
    before = model.norm.detach().clone()

    initialize_reload(model, NONQUANT)
    with pytest.raises(ReloadRejected):
        model.norm.weight_loader(model.norm, torch.zeros(H + 1))
    torch.testing.assert_close(model.norm, before)


def test_quantized_model_falls_back_to_layerwise():
    assert not supports_hook_reload(SimpleNamespace(quantization="fp8"), False)
    assert supports_hook_reload(NONQUANT, False)
    assert not supports_hook_reload(NONQUANT, True)  # lora: param identity moves


def test_reload_used_hooks_flag():
    model = FakeVllmModel()
    install_hook_reload_observers(model)
    assert not reload_used_hooks(model)  # no reload round yet
    model.load_weights(list(make_checkpoint(1).items()))
    initialize_reload(model, NONQUANT)
    model.load_weights(list(make_checkpoint(2).items()))
    finalize_reload(model, NONQUANT)
    # hook path handled the round: callers must skip legacy name-set warnings
    assert reload_used_hooks(model)


def test_hook_reload_tracks_buffers():
    model = FakeVllmModel()
    install_hook_reload_observers(model)
    model.load_weights(list(make_checkpoint(1).items()))
    ptr_before = model.k_scale.data_ptr()

    initialize_reload(model, NONQUANT)
    model.load_weights(list(make_checkpoint(2).items()))
    finalize_reload(model, NONQUANT)

    reference = _reference_model(make_checkpoint(2))
    torch.testing.assert_close(model.k_scale, reference.k_scale, rtol=0, atol=0)
    assert model.k_scale.data_ptr() == ptr_before
    # the buffer was observed and is hook-tracked, not silently bypassed
    assert reload_used_hooks(model)


def test_hook_reload_missing_buffer_shard_raises_incomplete():
    model = FakeVllmModel()
    install_hook_reload_observers(model)
    model.load_weights(list(make_checkpoint(1).items()))

    ckpt_b = make_checkpoint(2)
    ckpt_b.pop("k_scale")
    initialize_reload(model, NONQUANT)
    model.load_weights(list(ckpt_b.items()))
    with pytest.raises(ReloadIncomplete, match="k_scale"):
        finalize_reload(model, NONQUANT)


def test_merged_qkv_shards_tracked_not_degraded():
    # Regression: linear loaders name the shard argument `loaded_shard_id`;
    # missing it collapsed q/k/v into one key and degraded the parameter
    from vllm.model_executor.model_loader.reload.inplace import _PLANS

    model = FakeVllmModel()
    install_hook_reload_observers(model)
    model.load_weights(list(make_checkpoint(1).items()))

    plan = _PLANS[model]
    qkv = plan.records["qkv"]
    assert qkv.expected is not None, "merged QKV must stay tracked"
    assert set(qkv.expected) == {("q", None), ("k", None), ("v", None)}
    gate_up = plan.records["gate_up"]
    assert gate_up.expected is not None
    assert set(gate_up.expected) == {("gate", None), ("up", None)}

    initialize_reload(model, NONQUANT)
    assert qkv.hook is not None
    assert gate_up.hook is not None
    model.load_weights(list(make_checkpoint(2).items()))
    finalize_reload(model, NONQUANT)
    _assert_equal(model, _reference_model(make_checkpoint(2)))
