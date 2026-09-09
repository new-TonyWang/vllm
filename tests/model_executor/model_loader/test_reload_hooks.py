# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the non-quantized reload hooks (design doc section 7)."""

import pytest
import torch

from vllm.model_executor.model_loader.reload.hooks import (
    HookState,
    ReloadContext,
    ReloadIncomplete,
    ReloadRejected,
    RuntimeSlot,
    WeightShard,
)
from vllm.model_executor.model_loader.reload.nonquant_hooks import (
    DenseWeightHook,
    MergedRowsWeightHook,
    MoeW13WeightHook,
    MoeW2WeightHook,
    VocabParallelWeightHook,
)


def _slot(name: str, *shape: int, dtype: torch.dtype = torch.bfloat16):
    return RuntimeSlot(name, torch.zeros(*shape, dtype=dtype))


def test_dense_hook_in_place_roundtrip():
    # case 1: plain dense weight, single shard, pointer stability
    slot = _slot("mlp.o_proj.weight", 4, 8)
    ptr = slot.data_ptr
    hook = DenseWeightHook(slot)
    ctx = ReloadContext()
    ctx.register(hook)

    ctx.start_reload()
    new = torch.randn(4, 8, dtype=torch.bfloat16)
    ctx.deliver(WeightShard("mlp.o_proj.weight", new))
    assert hook.complete
    ctx.finish()

    torch.testing.assert_close(slot.tensor, new)
    assert slot.tensor.data_ptr() == ptr
    assert hook.state is HookState.IDLE


def test_dense_hook_cast_is_not_conversion():
    # case 8: fp32 checkpoint -> bf16 runtime via implicit copy_ cast
    slot = _slot("norm.weight", 8, dtype=torch.bfloat16)
    hook = DenseWeightHook(slot)
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    src = torch.randn(8, dtype=torch.float32)
    ctx.deliver(WeightShard("norm.weight", src))
    ctx.finish()
    torch.testing.assert_close(slot.tensor, src.to(torch.bfloat16))


def test_dense_hook_rejects_bad_shape_before_write():
    slot = _slot("w", 4, 8)
    before = slot.tensor.clone()
    hook = DenseWeightHook(slot)
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    with pytest.raises(ReloadRejected):
        ctx.deliver(WeightShard("w", torch.zeros(5, 8, dtype=torch.bfloat16)))
    # pre-write rejection: runtime storage untouched
    torch.testing.assert_close(slot.tensor, before)


def test_merged_rows_hook_requires_all_shards():
    # case 2: merged QKV, q/k/v shard_ids write distinct row ranges
    slot = _slot("attn.qkv_proj.weight", 12, 4)
    hook = MergedRowsWeightHook(
        slot, {"q": (0, 4), "k": (4, 4), "v": (8, 4)}
    )
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()

    q, k, v = (torch.full((4, 4), float(i), dtype=torch.bfloat16) for i in (1, 2, 3))
    ctx.deliver(WeightShard("attn.qkv_proj.weight", q, shard_id="q"))
    assert not hook.complete
    ctx.deliver(WeightShard("attn.qkv_proj.weight", k, shard_id="k"))
    ctx.deliver(WeightShard("attn.qkv_proj.weight", v, shard_id="v"))
    assert hook.complete
    ctx.finish()

    torch.testing.assert_close(slot.tensor, torch.cat([q, k, v]))


def test_merged_rows_hook_rejects_duplicate_shard():
    slot = _slot("w", 8, 4)
    before = slot.tensor.clone()
    hook = MergedRowsWeightHook(slot, {"gate": (0, 4), "up": (4, 4)})
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    ctx.deliver(
        WeightShard("w", torch.ones(4, 4, dtype=torch.bfloat16), shard_id="gate")
    )
    with pytest.raises(ReloadRejected, match="twice"):
        ctx.deliver(
            WeightShard(
                "w", torch.zeros(4, 4, dtype=torch.bfloat16), shard_id="gate"
            )
        )
    # the rejected duplicate did not overwrite the arrived gate rows
    torch.testing.assert_close(slot.tensor[:4], before[:4] + 1)


def test_vocab_parallel_hook_writes_real_rows_only():
    # case 3: padded vocab, padding tail keeps cold-load values
    slot = _slot("embed_tokens.weight", 8, 4)
    slot.tensor.fill_(7)
    hook = VocabParallelWeightHook(slot, vocab_rows=5)
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    new = torch.randn(5, 4, dtype=torch.bfloat16)
    ctx.deliver(WeightShard("embed_tokens.weight", new))
    assert hook.complete
    ctx.finish()

    torch.testing.assert_close(slot.tensor[:5], new)
    assert torch.all(slot.tensor[5:] == 7)


def test_tied_weights_alias_dedupes_second_delivery():
    # case 4: embedding/lm_head share one slot via an alias
    slot = _slot("embed_tokens.weight", 6, 4)
    hook = VocabParallelWeightHook(slot, vocab_rows=6)
    ctx = ReloadContext()
    ctx.register(hook, "lm_head.weight")
    ctx.start_reload()
    new = torch.randn(6, 4, dtype=torch.bfloat16)
    ctx.deliver(WeightShard("embed_tokens.weight", new))
    assert hook.complete
    # alias arrival of the same storage is deduplicated, not an error
    ctx.deliver(WeightShard("lm_head.weight", torch.zeros(6, 4, dtype=torch.bfloat16)))
    ctx.finish()
    torch.testing.assert_close(slot.tensor, new)


def test_moe_w2_hook_tracks_per_expert_arrival():
    # case 5: fused w2 (E, K, N), one slot per expert
    slot = _slot("experts.w2_weight", 4, 3, 2)
    hook = MoeW2WeightHook(slot)
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    for e in (2, 0, 3):
        payload = torch.full((3, 2), float(e), dtype=torch.bfloat16)
        ctx.deliver(WeightShard("experts.w2_weight", payload, expert_id=e))
        assert not hook.complete
    ctx.deliver(
        WeightShard(
            "experts.w2_weight",
            torch.full((3, 2), 1.0, dtype=torch.bfloat16),
            expert_id=1,
        )
    )
    assert hook.complete
    ctx.finish()
    for e in range(4):
        assert torch.all(slot.tensor[e] == e)


def test_moe_w2_hook_rejects_unknown_expert_before_write():
    slot = _slot("experts.w2_weight", 2, 3, 2)
    before = slot.tensor.clone()
    hook = MoeW2WeightHook(slot)
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    with pytest.raises(ReloadRejected, match="expert_id"):
        ctx.deliver(
            WeightShard(
                "experts.w2_weight",
                torch.ones(3, 2, dtype=torch.bfloat16),
                expert_id=2,
            )
        )
    torch.testing.assert_close(slot.tensor, before)


def test_moe_w13_hook_tracks_expert_and_half():
    # case 6: fused w13 (E, 2N, K), w1 -> lower half rows, w3 -> upper half
    e_total, n, k = 2, 3, 2
    slot = _slot("experts.w13_weight", e_total, 2 * n, k)
    hook = MoeW13WeightHook(slot)
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()

    for e in range(e_total):
        w1 = torch.full((n, k), float(e) + 0.1, dtype=torch.bfloat16)
        ctx.deliver(
            WeightShard("experts.w13_weight", w1, shard_id="w1", expert_id=e)
        )
        assert not hook.complete  # w3 half still missing
    for e in range(e_total):
        w3 = torch.full((n, k), float(e) + 0.2, dtype=torch.bfloat16)
        ctx.deliver(
            WeightShard("experts.w13_weight", w3, shard_id="w3", expert_id=e)
        )
    assert hook.complete
    ctx.finish()

    for e in range(e_total):
        assert torch.all(slot.tensor[e, :n] == e + 0.1)
        assert torch.all(slot.tensor[e, n:] == e + 0.2)


def test_finish_with_missing_shards_raises_incomplete():
    slot = _slot("w", 8, 4)
    hook = MergedRowsWeightHook(slot, {"gate": (0, 4), "up": (4, 4)})
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    ctx.deliver(
        WeightShard("w", torch.ones(4, 4, dtype=torch.bfloat16), shard_id="gate")
    )
    with pytest.raises(ReloadIncomplete, match="up"):
        ctx.finish()
    # context is closed after the failure; engine must restart / cold load
    with pytest.raises(ReloadRejected):
        ctx.deliver(
            WeightShard(
                "w", torch.ones(4, 4, dtype=torch.bfloat16), shard_id="up"
            )
        )


def test_empty_finish_is_noop():
    slot = _slot("w", 4, 4)
    before = slot.tensor.clone()
    hook = DenseWeightHook(slot)
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    ctx.finish()
    torch.testing.assert_close(slot.tensor, before)
    # repeated finish without start is rejected, not silently accepted
    with pytest.raises(ReloadRejected):
        ctx.finish()


def test_deliver_before_start_rejected():
    slot = _slot("w", 4, 4)
    ctx = ReloadContext()
    ctx.register(DenseWeightHook(slot))
    with pytest.raises(ReloadRejected):
        ctx.deliver(WeightShard("w", torch.ones(4, 4, dtype=torch.bfloat16)))


def test_consecutive_reloads_keep_pointer_stable():
    slot = _slot("w", 4, 4)
    ptr = slot.data_ptr
    hook = DenseWeightHook(slot)
    ctx = ReloadContext()
    ctx.register(hook)
    for i in range(3):
        ctx.start_reload()
        new = torch.full((4, 4), float(i), dtype=torch.bfloat16)
        ctx.deliver(WeightShard("w", new))
        ctx.finish()
        torch.testing.assert_close(slot.tensor, new)
        assert slot.tensor.data_ptr() == ptr


def test_slot_write_tile_region():
    # 2D tile write-back, e.g. a block-scale grid chunk
    slot = _slot("scale_inv", 8, 8)
    tile = torch.full((3, 2), 0.5, dtype=torch.bfloat16)
    slot.write(tile, (slice(2, 5), slice(3, 5)))
    assert torch.all(slot.tensor[2:5, 3:5] == 0.5)
    assert torch.all(slot.tensor[:2] == 0)
    assert torch.all(slot.tensor[5:] == 0)


def test_slot_write_int_plane():
    # int index pins one plane of a fused MoE slot
    slot = _slot("experts.w2_weight", 4, 3, 2)
    payload = torch.full((3, 2), 7.0, dtype=torch.bfloat16)
    slot.write(payload, (2,))
    assert torch.all(slot.tensor[2] == 7)
    assert torch.all(slot.tensor[1] == 0)


def test_slot_write_rejects_out_of_bounds_before_write():
    slot = _slot("w", 4, 8)
    before = slot.tensor.clone()
    payload = torch.ones(2, 8, dtype=torch.bfloat16)
    with pytest.raises(ReloadRejected):
        slot.write(payload, (slice(3, 5),))  # 3+2 > 4 rows
    with pytest.raises(ReloadRejected):
        slot.write(payload, (4,))  # plane index out of bounds
    with pytest.raises(ReloadRejected):
        slot.write(payload, (slice(0, 4, 2),))  # strided writes unsupported
    with pytest.raises(ReloadRejected):
        slot.write(payload, (0, 0, 0))  # more index dims than the slot
    torch.testing.assert_close(slot.tensor, before)


def test_slot_write_region_shape_must_match_exactly():
    slot = _slot("w", 4, 8)
    before = slot.tensor.clone()
    with pytest.raises(ReloadRejected):
        slot.write(torch.ones(2, 8, dtype=torch.bfloat16), (slice(0, 3),))
    torch.testing.assert_close(slot.tensor, before)


def test_slot_write_casts_dtype_implicitly():
    slot = _slot("w", 4, 8, dtype=torch.bfloat16)
    src = torch.randn(2, 8, dtype=torch.float32)
    slot.write(src, (slice(1, 3),))
    torch.testing.assert_close(slot.tensor[1:3], src.to(torch.bfloat16))


def test_post_load_runs_per_shard_finish_load_once():
    from vllm.model_executor.model_loader.reload.hooks import WeightReloadHook

    calls = {"pre": 0, "post_load": 0, "finish": 0}

    class CountingHook(MergedRowsWeightHook):
        def pre_reload(self):
            calls["pre"] += 1

        def post_load(self):
            calls["post_load"] += 1

        def finish_load(self):
            calls["finish"] += 1

    slot = _slot("w", 8, 4)
    hook = CountingHook(slot, {"gate": (0, 4), "up": (4, 4)})
    ctx = ReloadContext()
    ctx.register(hook)
    ctx.start_reload()
    for shard_id in ("gate", "up"):
        ctx.deliver(
            WeightShard(
                "w", torch.ones(4, 4, dtype=torch.bfloat16), shard_id=shard_id
            )
        )
    ctx.finish()
    assert calls == {"pre": 1, "post_load": 2, "finish": 1}
