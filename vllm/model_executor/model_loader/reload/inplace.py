# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hook-based in-place reload for non-quantized models.

Adapts the hook model (docs/design/weight-update/weight-reload-abstraction.md
sections 6-7) onto vLLM's existing ``weight_loader`` seam, so all three
weight-update flows (``reload_weights`` from disk, NCCL broadcast, CUDA IPC)
share one entry pair: :func:`initialize_reload` / :func:`finalize_reload`.

How it works:

- **Observe (cold load).** :func:`install_hook_reload_observers` wraps every
  parameter's weight loader at model construction time. The cold load flows
  through the wrappers, which record which shard keys (``shard_id`` /
  ``expert_id`` from the loader arguments) and shapes each parameter
  received. This observed plan is the DECLARE stage: no per-architecture
  fusion tables are needed, because ``model.load_weights`` has already
  resolved checkpoint names to fused parameters.
- **Reload (hook path).** For non-quantized models (``quantization is
  None``; bf16/fp16/fp32), ``initialize_reload`` builds one
  :class:`LoaderWeightHook` per tracked parameter from the observed plan and
  arms a :class:`ReloadContext`. Loader calls then run pre-write validation
  (unknown slot, duplicate shard, shape mismatch -> :class:`ReloadRejected`,
  runtime untouched) and delegate the actual write to the original loader,
  which already copies in place into the runtime storage. No meta-device
  detour, no staging buffer, pointer-stable.
- **Fallback.** Quantized models (or models without an observed plan) keep
  the layerwise path unchanged; the dispatcher picks per model.

Failure semantics follow the design: pre-write errors raise before any
write; a partial round raises :class:`ReloadIncomplete` from
``finalize_reload`` and the engine must stop serving.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary, ref

import torch

from .hooks import (
    ArrivalTracker,
    ReloadContext,
    ReloadRejected,
    RuntimeSlot,
    WeightReloadHook,
    WeightShard,
)
from .meta import SKIP_LOAD_TENSORS

if TYPE_CHECKING:
    from vllm.config import ModelConfig

__all__ = [
    "LoaderWeightHook",
    "install_hook_reload_observers",
    "initialize_reload",
    "finalize_reload",
    "reload_used_hooks",
    "supports_hook_reload",
]

_FULL = "full"


def _shard_key(bound_args: inspect.BoundArguments) -> Hashable:
    """Shard identity from loader arguments: (shard_id, expert_id).

    Loader argument names differ across layer types: fused MoE loaders use
    ``shard_id``/``expert_id`` while the linear layers
    (``QKVParallelLinear``, ``MergedColumnParallelLinear``) use
    ``loaded_shard_id``. Both name the same concept; check both.
    """
    arguments = bound_args.arguments
    shard_id = arguments.get("shard_id")
    if shard_id is None:
        shard_id = arguments.get("loaded_shard_id")
    expert_id = arguments.get("expert_id")
    if shard_id is None and expert_id is None:
        return _FULL
    return (
        str(shard_id) if shard_id is not None else None,
        expert_id,
    )


class LoaderWeightHook(WeightReloadHook):
    """Per-parameter hook driven through the vLLM weight_loader seam.

    The original loader performs the actual write (it already does
    ``param.data.narrow(...).copy_(...)`` in place); this hook adds the
    arrival validation and completeness tracking around it.
    """

    def __init__(
        self,
        slot: RuntimeSlot,
        original_loader: Callable,
    ) -> None:
        super().__init__(slot)
        self._original_loader = original_loader

    def load_from_loader(self, bound_args: inspect.BoundArguments) -> Any:
        """Entry point for the wrapped weight loader."""
        key = _shard_key(bound_args)
        if not self._begin_shard(str(key)):
            return None  # deduplicated tied-weight arrival
        loaded = bound_args.arguments.get("loaded_weight")
        if not isinstance(loaded, torch.Tensor):
            raise ReloadRejected(
                f"Loader for {self.name!r} received a non-tensor payload"
            )
        self.tracker.validate(key, loaded.shape)
        result = self._original_loader(*bound_args.args, **bound_args.kwargs)
        self.tracker.mark_arrived(key)
        self.post_load()
        self._commit_if_complete()
        return result

    def _write_shard(self, shard: WeightShard) -> None:
        raise NotImplementedError(
            "LoaderWeightHook is driven through load_from_loader"
        )


@dataclass
class _ParamRecord:
    """Observed cold-load plan and live hook for one parameter."""

    name: str
    param: torch.Tensor
    original_loader: Callable
    loader_signature: inspect.Signature
    # shard key -> observed shape; None means the param is not tracked
    # (duplicate writes at cold load, or the slot is not a stable view)
    expected: dict[Hashable, tuple[int, ...]] | None = field(default_factory=dict)
    # counts of writes per key at cold load; >1 disqualifies tracking
    observed_counts: dict[Hashable, int] = field(default_factory=dict)
    hook: LoaderWeightHook | None = None


@dataclass
class _ModelHookPlan:
    records: dict[str, _ParamRecord] = field(default_factory=dict)
    frozen: bool = False
    ctx: ReloadContext | None = None
    active: bool = False


_PLANS: WeakKeyDictionary[torch.nn.Module, _ModelHookPlan] = WeakKeyDictionary()


def install_hook_reload_observers(model: torch.nn.Module) -> None:
    """Wrap every parameter's weight loader with a cold-load observer.

    Idempotent; call once at model construction, before the cold load. The
    observer records the arrival plan and delegates unchanged.
    """
    if model in _PLANS:
        return
    plan = _ModelHookPlan()
    _PLANS[model] = plan
    model_ref = ref(model)
    # Buffers are load targets too (e.g. attention k_scale/v_scale with their
    # own weight_loader); skip buffers that are never loaded from checkpoints,
    # matching the layerwise path's SKIP_LOAD_TENSORS.
    tensors = list(model.named_parameters())
    tensors += [
        (name, buffer)
        for name, buffer in model.named_buffers()
        if name.rsplit(".", 1)[-1] not in SKIP_LOAD_TENSORS
    ]
    for name, param in tensors:
        original_loader = getattr(param, "weight_loader", None)
        if original_loader is None:
            from vllm.model_executor.model_loader.weight_utils import (
                default_weight_loader,
            )

            original_loader = default_weight_loader
        try:
            signature = inspect.signature(original_loader)
        except (TypeError, ValueError):
            continue  # uninspectable loader: leave untracked
        record = _ParamRecord(
            name=name,
            param=param,
            original_loader=original_loader,
            loader_signature=signature,
        )
        plan.records[name] = record
        param.weight_loader = _make_observer_wrapper(model_ref, record)


def _make_observer_wrapper(
    model_ref: ref[torch.nn.Module],
    record: _ParamRecord,
) -> Callable:
    original_loader = record.original_loader

    @wraps(original_loader)
    def hook_aware_weight_loader(*args, **kwargs):
        bound = record.loader_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        plan = _PLANS.get(model_ref()) if model_ref() is not None else None
        if plan is None:
            return original_loader(*args, **kwargs)
        if plan.active and record.hook is not None:
            # reload round: route through the hook (validates + delegates)
            return record.hook.load_from_loader(bound)
        if not plan.frozen:
            _observe_write(record, bound)
        return original_loader(*args, **kwargs)

    return hook_aware_weight_loader


def _observe_write(record: _ParamRecord, bound: inspect.BoundArguments) -> None:
    loaded = bound.arguments.get("loaded_weight")
    if not isinstance(loaded, torch.Tensor) or record.expected is None:
        return
    key = _shard_key(bound)
    record.observed_counts[key] = record.observed_counts.get(key, 0) + 1
    if record.observed_counts[key] > 1:
        # legitimate duplicate loads (shared weight written twice) cannot be
        # told apart from corrupt duplicates; stop tracking this parameter
        record.expected = None
        return
    if record.expected is not None:
        record.expected[key] = tuple(loaded.shape)


def reload_used_hooks(model: torch.nn.Module) -> bool:
    """Whether this model's reloads go through the hook path.

    True once a hook ReloadContext exists for the model. On the hook path,
    per-shard completeness is already enforced by ``ReloadContext.finish``
    (``ReloadIncomplete``), so callers must skip legacy name-set completeness
    warnings: model ``load_weights`` return values use checkpoint shard
    names, which do not match the fused runtime parameter namespace.
    """
    plan = _PLANS.get(model)
    return plan is not None and plan.ctx is not None


def supports_hook_reload(model_config: ModelConfig, lora_enabled: bool) -> bool:
    """Non-quantized models (bf16/fp16/fp32) reload through hooks."""
    return model_config.quantization is None and not lora_enabled


def initialize_reload(
    model: torch.nn.Module,
    model_config: ModelConfig,
    *,
    lora_enabled: bool = False,
) -> None:
    """Enter a reload round; hook path for non-quantized models.

    Falls back to the layerwise path when hooks do not apply. Pair with
    :func:`finalize_reload`.
    """
    plan = _PLANS.get(model)
    use_hooks = (
        plan is not None
        and any(r.expected for r in plan.records.values())
        and supports_hook_reload(model_config, lora_enabled)
    )
    if plan is not None:
        plan.frozen = True
    if not use_hooks:
        from .layerwise import initialize_layerwise_reload

        initialize_layerwise_reload(model)
        return

    assert plan is not None
    if plan.ctx is None:
        ctx = ReloadContext()
        for record in plan.records.values():
            if not record.expected:
                continue
            try:
                slot = RuntimeSlot(record.name, record.param)
            except (RuntimeError, ValueError):
                record.expected = None  # no stable row view; stop tracking
                continue
            hook = LoaderWeightHook(slot, record.original_loader)
            for key, shape in record.expected.items():
                hook.tracker.add_slot(key, shape)
            record.hook = hook
            ctx.register(hook)
        plan.ctx = ctx
    plan.ctx.start_reload()
    plan.active = True


def finalize_reload(
    model: torch.nn.Module,
    model_config: ModelConfig,
    updated_parameter_names: frozenset[str] | None = None,
) -> None:
    """Complete a reload round started by :func:`initialize_reload`."""
    plan = _PLANS.get(model)
    if plan is None or not plan.active:
        from .layerwise import finalize_layerwise_reload

        finalize_layerwise_reload(model, model_config)
        return

    assert plan.ctx is not None
    try:
        # raises ReloadIncomplete on a partial round; engine must stop
        plan.ctx.finish()
    finally:
        plan.active = False

    from .selective import refresh_derived_state

    refresh_derived_state(model, updated_parameter_names)
