# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end reload flow: hooks mounted on a model's real weights.

Builds a toy model with the fused structures vLLM produces at cold load
(merged QKV, gate_up MLP, padded + tied embedding, fused MoE w13/w2),
cold-loads checkpoint A, binds reload hooks, then reloads checkpoint B
through the ReloadContext and verifies:
- every weight matches a fresh cold load of B (bit-identical),
- no runtime storage moved (data_ptr stable),
- forward outputs match the cold-loaded-B model exactly,
- repeated reloads (B -> A -> B) stay pointer-stable and correct.
"""

import pytest
import torch

from vllm.model_executor.model_loader.reload.hooks import (
    ReloadIncomplete,
    ReloadRejected,
    WeightShard,
)
from vllm.model_executor.model_loader.reload.registry import (
    WeightPlan,
    build_reload_context,
    expand_shards,
    reload_from_state_dict,
)

torch.manual_seed(0)

H, I, V_REAL, V_PAD, E, N_MOE, K_MOE = 8, 16, 11, 16, 4, 6, 8


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Parameter(torch.zeros(3 * H, H))
        self.o_proj = torch.nn.Parameter(torch.zeros(H, H))
        self.gate_up = torch.nn.Parameter(torch.zeros(2 * I, H))
        self.down = torch.nn.Parameter(torch.zeros(H, I))
        self.norm = torch.nn.Parameter(torch.zeros(H))
        self.w13 = torch.nn.Parameter(torch.zeros(E, 2 * N_MOE, K_MOE))
        self.w2 = torch.nn.Parameter(torch.zeros(E, K_MOE, N_MOE))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.zeros(V_PAD, H))
        self.block = ToyBlock()
        self.lm_head = self.embed  # tied weights share storage

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.embedding(ids, self.embed[:V_REAL])
        b = self.block
        x = x + torch.nn.functional.layer_norm(x, (H,), b.norm, None)
        q, k, v = (x @ b.qkv.T).split(H, dim=-1)
        attn = torch.softmax(q @ k.transpose(-1, -2) / H**0.5, dim=-1) @ v
        x = x + attn @ b.o_proj
        gate, up = (x @ b.gate_up.T).split(I, dim=-1)
        x = x + (torch.nn.functional.silu(gate) * up) @ b.down.T
        # top-1 routed MoE over fused w13/w2
        scores = x[..., : E].softmax(-1)
        e = scores.argmax(-1)
        h = x[..., :K_MOE]
        w1, w3 = b.w13[e][:, :N_MOE], b.w13[e][:, N_MOE:]
        moe = (torch.nn.functional.silu(torch.einsum("bi,bni->bn", h, w1))
               * torch.einsum("bi,bni->bn", h, w3))
        moe = torch.einsum("bn,bkn->bk", moe, b.w2[e])
        x = x + torch.nn.functional.pad(moe, (0, H - K_MOE))
        return x @ self.lm_head[:V_REAL].T


def make_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """Checkpoint layout: unfused per-shard tensors, as on disk."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    ckpt = {
        "embed.weight": rnd(V_REAL, H),
        "q_proj.weight": rnd(H, H),
        "k_proj.weight": rnd(H, H),
        "v_proj.weight": rnd(H, H),
        "o_proj.weight": rnd(H, H),
        "gate_proj.weight": rnd(I, H),
        "up_proj.weight": rnd(I, H),
        "down.weight": rnd(H, I),
        "norm.weight": rnd(H),
    }
    for e in range(E):
        ckpt[f"experts.{e}.w1.weight"] = rnd(N_MOE, K_MOE)
        ckpt[f"experts.{e}.w3.weight"] = rnd(N_MOE, K_MOE)
        ckpt[f"experts.{e}.w2.weight"] = rnd(K_MOE, N_MOE)
    return ckpt


PLANS = [
    WeightPlan(
        "embed", source="embed.weight", aliases=("lm_head",),
        vocab_rows=V_REAL,
    ),
    WeightPlan(
        "block.qkv",
        shard_sources={
            "q": "q_proj.weight",
            "k": "k_proj.weight",
            "v": "v_proj.weight",
        },
    ),
    WeightPlan("block.o_proj", source="o_proj.weight"),
    WeightPlan(
        "block.gate_up",
        shard_sources={"gate": "gate_proj.weight", "up": "up_proj.weight"},
    ),
    WeightPlan("block.down", source="down.weight"),
    WeightPlan("block.norm", source="norm.weight"),
    WeightPlan(
        "block.w13",
        expert_sources={
            "w1": "experts.{e}.w1.weight",
            "w3": "experts.{e}.w3.weight",
        },
    ),
    WeightPlan("block.w2", expert_sources={"w2": "experts.{e}.w2.weight"}),
]


def cold_load(model: ToyModel, ckpt: dict[str, torch.Tensor]) -> None:
    """Install a checkpoint the way the cold-load weight loaders do."""
    with torch.no_grad():
        model.embed[:V_REAL].copy_(ckpt["embed.weight"])
        model.block.qkv.copy_(
            torch.cat([ckpt["q_proj.weight"], ckpt["k_proj.weight"],
                       ckpt["v_proj.weight"]])
        )
        model.block.o_proj.copy_(ckpt["o_proj.weight"])
        model.block.gate_up.copy_(
            torch.cat([ckpt["gate_proj.weight"], ckpt["up_proj.weight"]])
        )
        model.block.down.copy_(ckpt["down.weight"])
        model.block.norm.copy_(ckpt["norm.weight"])
        for e in range(E):
            model.block.w13[e].copy_(
                torch.cat([ckpt[f"experts.{e}.w1.weight"],
                           ckpt[f"experts.{e}.w3.weight"]])
            )
            model.block.w2[e].copy_(ckpt[f"experts.{e}.w2.weight"])


def _assert_model_matches(model: ToyModel, other: ToyModel) -> None:
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), other.named_parameters()
    ):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)


def test_reload_matches_cold_load_bit_exact():
    model = ToyModel()
    cold_load(model, make_checkpoint(1))
    ids = torch.randint(0, V_REAL, (5,))

    reference = ToyModel()
    ckpt_b = make_checkpoint(2)
    cold_load(reference, ckpt_b)
    expected_logits = reference(ids)

    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}
    ctx = build_reload_context(model, PLANS, checkpoint=make_checkpoint(1))

    reload_from_state_dict(ctx, PLANS, ckpt_b)

    _assert_model_matches(model, reference)
    torch.testing.assert_close(model(ids), expected_logits, rtol=0, atol=0)
    for name, p in model.named_parameters():
        assert p.data_ptr() == ptrs[name], f"{name} storage moved"
    # tied weights still share storage
    assert model.lm_head.data_ptr() == model.embed.data_ptr()


def test_consecutive_reloads_roundtrip():
    model = ToyModel()
    cold_load(model, make_checkpoint(1))
    ctx = build_reload_context(model, PLANS, checkpoint=make_checkpoint(1))
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}

    for seed in (2, 1, 2):
        ckpt = make_checkpoint(seed)
        reference = ToyModel()
        cold_load(reference, ckpt)
        reload_from_state_dict(ctx, PLANS, ckpt)
        _assert_model_matches(model, reference)
    for name, p in model.named_parameters():
        assert p.data_ptr() == ptrs[name]


def test_reload_rejects_corrupt_shard_before_write():
    model = ToyModel()
    ckpt_a = make_checkpoint(1)
    cold_load(model, ckpt_a)
    before = {n: p.clone() for n, p in model.named_parameters()}
    ctx = build_reload_context(model, PLANS, checkpoint=ckpt_a)

    ckpt_b = make_checkpoint(2)
    ckpt_b["k_proj.weight"] = torch.zeros(H + 1, H)  # corrupt shape
    ctx.start_reload()
    try:
        for plan in PLANS:
            num_experts = E if plan.expert_sources else None
            for shard in expand_shards(plan, ckpt_b, num_experts):
                ctx.deliver(shard)
        raise AssertionError("corrupt shard was not rejected")
    except ReloadRejected:
        pass
    # embed committed before the corrupt qkv shard arrived; everything after
    # is untouched. Runtime state is only guaranteed for un-arrived weights.
    torch.testing.assert_close(model.block.gate_up, before["block.gate_up"])
    torch.testing.assert_close(model.block.w13, before["block.w13"])


def test_reload_missing_expert_shard_raises_incomplete():
    model = ToyModel()
    ckpt_a = make_checkpoint(1)
    cold_load(model, ckpt_a)
    ctx = build_reload_context(model, PLANS, checkpoint=ckpt_a)

    ckpt_b = make_checkpoint(2)
    del ckpt_b["experts.3.w3.weight"]
    ctx.start_reload()
    with pytest.raises((KeyError, ReloadIncomplete)):
        for plan in PLANS:
            num_experts = E if plan.expert_sources else None
            for shard in expand_shards(plan, ckpt_b, num_experts):
                ctx.deliver(shard)
        ctx.finish()
