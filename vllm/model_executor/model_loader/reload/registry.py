# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assembly of reload hooks onto model weights and the reload driver.

DECLARE: :class:`WeightPlan` records, per runtime weight, its kind and how
checkpoint tensors map onto it (fusion shard sources, expert templates,
vocab padding, tied aliases). ``build_reload_context`` binds one
:class:`~vllm.model_executor.model_loader.reload.hooks.RuntimeSlot` and the
matching non-quantized hook to every planned weight and registers them in a
:class:`~vllm.model_executor.model_loader.reload.hooks.ReloadContext`.

SOURCE -> VALIDATE -> COMMIT -> FINISH: ``reload_from_state_dict`` expands
each plan into :class:`WeightShard` deliveries and finishes the round. All
validation runs before each write; a partial round raises
``ReloadIncomplete`` from ``finish`` and leaves engine state undefined.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .hooks import ReloadContext, RuntimeSlot, WeightReloadHook, WeightShard
from .nonquant_hooks import (
    DenseWeightHook,
    MergedRowsWeightHook,
    MoeW13WeightHook,
    MoeW2WeightHook,
    VocabParallelWeightHook,
)

__all__ = [
    "WeightPlan",
    "build_reload_context",
    "expand_shards",
    "reload_from_state_dict",
]


@dataclass(frozen=True)
class WeightPlan:
    """Declarative reload plan for one runtime weight.

    Args:
        runtime_name: qualified parameter/buffer name on the model.
        source: checkpoint tensor name for dense/vocab weights; defaults
            to ``runtime_name``.
        aliases: extra checkpoint names bound to the same slot (tied
            weights); duplicate delivery after commit is deduplicated.
        vocab_rows: real vocab rows for vocab-parallel weights; the padding
            tail is never written. None for non-vocab weights.
        shard_sources: merged-row weights: shard_id -> checkpoint tensor
            name, in fusion order (e.g. ``{"q": ..., "k": ..., "v": ...}``
            or ``{"gate": ..., "up": ...}``).
        expert_sources: fused MoE weights: ``{"w2": template}`` or
            ``{"w1": template, "w3": template}`` where the template contains
            ``{e}`` for the expert id, e.g. ``"experts.{e}.w2.weight"``.
    """

    runtime_name: str
    source: str | None = None
    aliases: tuple[str, ...] = ()
    vocab_rows: int | None = None
    shard_sources: Mapping[str, str] | None = None
    expert_sources: Mapping[str, str] | None = None

    def kind(self) -> str:
        if self.expert_sources is not None:
            keys = set(self.expert_sources)
            if keys == {"w2"}:
                return "moe_w2"
            if keys == {"w1", "w3"}:
                return "moe_w13"
            raise ValueError(
                f"expert_sources must be {{'w2'}} or {{'w1','w3'}}, got {keys}"
            )
        if self.shard_sources is not None:
            return "merged"
        if self.vocab_rows is not None:
            return "vocab"
        return "dense"


def _resolve_runtime_tensor(model: torch.nn.Module, name: str) -> torch.Tensor:
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    try:
        return tensors[name]
    except KeyError:
        raise ValueError(f"Plan targets unknown model tensor {name!r}") from None


def _make_hook(plan: WeightPlan, slot: RuntimeSlot) -> WeightReloadHook:
    kind = plan.kind()
    if kind == "dense":
        return DenseWeightHook(slot)
    if kind == "vocab":
        assert plan.vocab_rows is not None
        return VocabParallelWeightHook(slot, plan.vocab_rows)
    if kind == "merged":
        # merged hooks need checkpoint shapes for row ranges; built by
        # build_reload_context directly
        raise ValueError("merged plans require build_reload_context")
    if kind == "moe_w2":
        return MoeW2WeightHook(slot)
    if kind == "moe_w13":
        return MoeW13WeightHook(slot)
    raise ValueError(f"Unknown plan kind {kind!r}")


def build_reload_context(
    model: torch.nn.Module,
    plans: list[WeightPlan],
    checkpoint: Mapping[str, torch.Tensor] | None = None,
) -> ReloadContext:
    """DECLARE: bind hooks to the model's runtime weights.

    Args:
        model: the loaded model whose current parameter storages become the
            stable runtime slots.
        plans: one plan per reloadable weight.
        checkpoint: required when any plan is row-merged, to derive each
            shard's row count from the cold-load checkpoint shapes.
    """
    ctx = ReloadContext()
    for plan in plans:
        tensor = _resolve_runtime_tensor(model, plan.runtime_name)
        slot = RuntimeSlot(plan.runtime_name, tensor)
        if plan.kind() == "merged":
            if checkpoint is None:
                raise ValueError(
                    f"Merged plan {plan.runtime_name!r} needs checkpoint shapes"
                )
            assert plan.shard_sources is not None
            row_start = 0
            shard_rows: dict[str, tuple[int, int]] = {}
            for shard_id, ckpt_name in plan.shard_sources.items():
                rows = checkpoint[ckpt_name].shape[0]
                shard_rows[shard_id] = (row_start, rows)
                row_start += rows
            hook: WeightReloadHook = MergedRowsWeightHook(slot, shard_rows)
        else:
            hook = _make_hook(plan, slot)
        ctx.register(hook, *plan.aliases)
    return ctx


def expand_shards(
    plan: WeightPlan,
    checkpoint: Mapping[str, torch.Tensor],
    num_experts: int | None = None,
) -> list[WeightShard]:
    """SOURCE: expand one plan into shard deliveries from a checkpoint."""
    name = plan.runtime_name
    kind = plan.kind()
    if kind in ("dense", "vocab"):
        source = plan.source
        if source is None or source not in checkpoint:
            if source is None and name in checkpoint:
                source = name
            else:
                source = _alias_in(plan, checkpoint)
        return [WeightShard(name, checkpoint[source])]
    if kind == "merged":
        assert plan.shard_sources is not None
        return [
            WeightShard(name, checkpoint[ckpt_name], shard_id=shard_id)
            for shard_id, ckpt_name in plan.shard_sources.items()
        ]
    if num_experts is None:
        raise ValueError(f"MoE plan {name!r} requires num_experts")
    assert plan.expert_sources is not None
    shards = []
    for expert_id in range(num_experts):
        for half, template in plan.expert_sources.items():
            shard_id = None if half == "w2" else half
            shards.append(
                WeightShard(
                    name,
                    checkpoint[template.format(e=expert_id)],
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
            )
    return shards


def _alias_in(plan: WeightPlan, checkpoint: Mapping[str, torch.Tensor]) -> str:
    for alias in plan.aliases:
        if alias in checkpoint:
            return alias
    raise KeyError(
        f"Neither {plan.runtime_name!r} nor its aliases {plan.aliases} are "
        "present in the checkpoint"
    )


def reload_from_state_dict(
    ctx: ReloadContext,
    plans: list[WeightPlan],
    checkpoint: Mapping[str, torch.Tensor],
) -> None:
    """Run one full reload round: SOURCE -> VALIDATE -> COMMIT -> FINISH.

    Shards are delivered per plan in declaration order (no layerwise
    streaming); every write is validated before it lands, and ``finish``
    enforces that the declared set arrived exactly once.
    """
    ctx.start_reload()
    try:
        for plan in plans:
            num_experts = None
            if plan.kind() in ("moe_w2", "moe_w13"):
                num_experts = ctx.hook(plan.runtime_name).slot.shape[0]
            for shard in expand_shards(plan, checkpoint, num_experts):
                ctx.deliver(shard)
        ctx.finish()
    except BaseException:
        # If the round broke mid-delivery, finish() state was not reached;
        # the context stays active for the caller to hard-fail the engine.
        raise
