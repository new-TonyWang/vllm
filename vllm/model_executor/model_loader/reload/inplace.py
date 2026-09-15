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
- **No fallback.** Unsupported configurations (other quantization schemes,
  unsupported FP8 backends, missing cold-load plans, LoRA) raise
  :class:`HookReloadUnsupportedError` instead of falling back to the
  layerwise path; see
  docs/design/weight-update/reload-hook-unsupported.md.

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
    HookReloadUnsupportedError,
    ReloadContext,
    ReloadRejected,
    RuntimeSlot,
    WeightReloadHook,
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

    def load_from_loader(
        self,
        bound_args: inspect.BoundArguments,
        tensor_name: str | None = None,
    ) -> Any:
        """Entry point for the wrapped weight loader."""
        key = self._arrival_key(bound_args)
        if not self._begin_shard(str(key)):
            return None  # deduplicated tied-weight arrival
        loaded = bound_args.arguments.get("loaded_weight")
        if not isinstance(loaded, torch.Tensor):
            raise ReloadRejected(
                f"Loader for {self.name!r} received a non-tensor payload"
            )
        self.tracker.validate(key, loaded.shape)
        bound_args = self._pre_delegate(bound_args)
        result = self._original_loader(*bound_args.args, **bound_args.kwargs)
        self.tracker.mark_arrived(key)
        self.post_load()
        self._commit_if_complete()
        return result

    def _arrival_key(self, bound_args: inspect.BoundArguments) -> Hashable:
        """Tracking key for one loader call; overridable by mapping hooks."""
        return _shard_key(bound_args)

    def _pre_delegate(
        self, bound_args: inspect.BoundArguments
    ) -> inspect.BoundArguments:
        """Adjust bound args after validation, before the original loader.

        Mapping hooks override this to redirect the write (e.g. swap the
        shard's target half, or substitute a staging buffer).
        """
        return bound_args


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
            return record.hook.load_from_loader(bound, record.name)
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


def supports_hook_reload(
    model_config: ModelConfig,
    lora_enabled: bool,
    quant_config: Any = None,
) -> bool:
    """Non-quantized and offline FP8 per-block models reload through hooks."""
    if lora_enabled:
        return False
    if model_config.quantization is None:
        return True
    # Offline FP8 per-block: checkpoint is already FP8 + block scales.
    # Per-backend support is decided per layer at hook construction; an
    # unsupported backend falls the whole model back to the layerwise path.
    return (
        model_config.quantization == "fp8"
        and quant_config is not None
        and getattr(quant_config, "weight_block_size", None) is not None
        and getattr(quant_config, "is_checkpoint_fp8_serialized", False)
    )


def initialize_reload(
    model: torch.nn.Module,
    model_config: ModelConfig,
    *,
    lora_enabled: bool = False,
    quant_config: Any = None,
) -> None:
    """Enter a reload round on the hook path. Pair with :func:`finalize_reload`.

    Never falls back to the layerwise path: unsupported configurations
    raise :class:`HookReloadUnsupportedError` (see
    docs/design/weight-update/reload-hook-unsupported.md).
    """
    plan = _PLANS.get(model)
    if plan is not None:
        plan.frozen = True
    if plan is None or not any(r.expected for r in plan.records.values()):
        raise HookReloadUnsupportedError(
            "No observed cold-load plan for this model; hook reload "
            "requires a prior cold load (dummy-init models are unsupported)"
        )
    if not supports_hook_reload(model_config, lora_enabled, quant_config):
        raise HookReloadUnsupportedError(
            f"Hook reload is not supported for "
            f"quantization={model_config.quantization!r} "
            f"(lora_enabled={lora_enabled}); see "
            "docs/design/weight-update/reload-hook-unsupported.md"
        )

    if plan.ctx is None:
        ctx = _build_hook_context(model, model_config, plan)
        if ctx is None:
            raise HookReloadUnsupportedError(
                "A quantized layer does not support hook reload "
                f"(quantization={model_config.quantization!r}); see "
                "docs/design/weight-update/reload-hook-unsupported.md"
            )
        plan.ctx = ctx
    plan.ctx.start_reload()
    plan.active = True


def _restore_param_identity(cold: torch.Tensor, live: torch.Tensor) -> None:
    """Restore the cold-load parameter identity on a PWAL-replaced param.

    ``replace_parameter`` re-registers a plain ``torch.nn.Parameter``,
    dropping the ``BasevLLMParameter`` subclass and loader attributes that
    weight loaders rely on (e.g. ``load_qkv_weight``, ``tp_rank``,
    ``output_dim``). Restore the cold-load class and any dropped attributes
    on the live object; attributes PWAL set on the replacement win.
    """
    if type(live) is not type(cold):
        live.__class__ = type(cold)
    # drop a stale plain attribute so the BasevLLMParameter
    # `weight_loader` property (restored with the class) is not shadowed
    live.__dict__.pop("weight_loader", None)
    for attr, value in cold.__dict__.items():
        live.__dict__.setdefault(attr, value)


def _build_hook_context(
    model: torch.nn.Module,
    model_config: ModelConfig,
    plan: _ModelHookPlan,
) -> ReloadContext | None:
    """Build the per-parameter hooks. Returns None if any quantized layer
    does not support the hook path (caller raises HookReloadUnsupportedError).
    """
    quantized = model_config.quantization is not None

    # Parameters may have been replaced by process_weights_after_loading
    # (replace_parameter carries our wrapper over to the new object);
    # re-resolve every record against the live model.
    live_tensors = dict(model.named_parameters())
    live_tensors.update(dict(model.named_buffers()))
    for record in plan.records.values():
        live = live_tensors.get(record.name)
        if live is None:
            record.expected = None  # parameter gone; stop tracking it
            continue
        if live is not record.param:
            _restore_param_identity(record.param, live)
            record.param = live

    # Quant hook factories: qualified param name -> (quant_method, module, local)
    makers: dict[str, tuple[Any, torch.nn.Module, str]] = {}
    if quantized:
        modules = dict(model.named_modules())
        for record in plan.records.values():
            if not record.expected:
                # never loaded from the checkpoint (e.g. attention kv-scale
                # buffers): nothing to track, so the owning module needs no
                # hook support
                continue
            module_name, _, local_name = record.name.rpartition(".")
            module = modules.get(module_name)
            quant_method = getattr(module, "quant_method", None) if module else None
            if quant_method is None:
                continue
            # Unquantized sublayers of a mixed-quantization model (e.g. the
            # embedding / lm_head / MoE gate of an FP8 checkpoint) hold plain
            # bf16 params; the generic hook covers them.
            if type(quant_method).__name__ in (
                "UnquantizedLinearMethod",
                "UnquantizedEmbeddingMethod",
            ):
                continue
            maker = getattr(quant_method, "make_reload_hook", None)
            capability = getattr(quant_method, "supports_hook_reload", None)
            if not callable(maker) or not (callable(capability) and capability()):
                return None
            makers[record.name] = (quant_method, module, local_name)

    ctx = ReloadContext()
    slots: dict[str, RuntimeSlot] = {}
    for record in plan.records.values():
        if not record.expected:
            continue
        try:
            slot = RuntimeSlot(record.name, record.param)
        except (RuntimeError, ValueError):
            record.expected = None  # no stable row view; stop tracking
            continue
        slots[record.name] = slot

    grouped_hooks: dict[str, WeightReloadHook] = {}
    hook_groups: dict[tuple[str, int], dict[str, Any]] = {}
    for record in plan.records.values():
        if record.name not in slots or record.name not in makers:
            continue
        quant_method, module, local_name = makers[record.name]
        try:
            supports_group = "hook_group" in inspect.signature(
                quant_method.make_reload_hook
            ).parameters
        except (TypeError, ValueError):
            supports_group = False
        if not supports_group:
            continue
        module_key = (record.name.rsplit(".", 1)[0], id(quant_method))
        group = hook_groups.setdefault(
            module_key,
            {"method": quant_method, "module": module, "names": {}},
        )
        group["names"][local_name] = record.name

    for group in hook_groups.values():
        names = group["names"]
        group_slots = {
            local: slots[name]
            for local, name in names.items()
            if name in slots
        }
        group_loaders = {
            local: plan.records[name].original_loader
            for local, name in names.items()
            if name in slots
        }
        group_params = {
            local: plan.records[name].param
            for local, name in names.items()
            if name in slots
        }
        if set(group_slots) != set(names):
            continue
        made = group["method"].make_reload_hook(
            group["module"],
            None,
            None,
            None,
            hook_group={
                "slots": group_slots,
                "loaders": group_loaders,
                "cold_params": group_params,
                "names": names,
            },
        )
        if made is not None:
            for local, hook in made.items():
                grouped_hooks[names[local]] = hook

    registered: set[int] = set()
    for record in plan.records.values():
        if record.name not in slots:
            continue
        slot = slots[record.name]
        if record.name in makers:
            quant_method, module, local_name = makers[record.name]
            hook = grouped_hooks.get(record.name)
            if hook is None:
                hook = quant_method.make_reload_hook(
                    module,
                    local_name,
                    slot,
                    record.original_loader,
                    cold_param=record.param,
                )
            if hook is None:
                return None
        else:
            hook = LoaderWeightHook(slot, record.original_loader)
        for key, shape in record.expected.items():
            if hook is grouped_hooks.get(record.name):
                role = (
                    "weight"
                    if record.name == getattr(hook, "_weight_name", None)
                    else "scale"
                )
                hook.add_slot(role, key, shape)
            else:
                hook.tracker.add_slot(key, shape)
        record.hook = hook
        if id(hook) not in registered:
            aliases = []
            scale_name = getattr(hook, "_scale_name", None)
            if scale_name and scale_name != hook.name:
                aliases.append(scale_name)
            ctx.register(hook, *aliases)
            registered.add(id(hook))
    return ctx


def finalize_reload(
    model: torch.nn.Module,
    model_config: ModelConfig,
    updated_parameter_names: frozenset[str] | None = None,
) -> None:
    """Complete a reload round started by :func:`initialize_reload`."""
    plan = _PLANS.get(model)
    if plan is None or not plan.active:
        raise HookReloadUnsupportedError(
            "finalize_reload without a hook reload round in progress"
        )

    assert plan.ctx is not None
    try:
        # raises ReloadIncomplete on a partial round; engine must stop
        plan.ctx.finish()
    finally:
        plan.active = False

    from .selective import refresh_derived_state

    refresh_derived_state(model, updated_parameter_names)
