# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core reload hook abstractions: slots, arrival tracking, reload context.

Implements the hook model of docs/design/weight-update/
weight-reload-abstraction.md section 6: every reloadable weight is a
:class:`RuntimeSlot` with a :class:`WeightReloadHook` binding pre_reload /
load_shard / finish_load. Quantized backends build CONVERT-buffer hooks
on top of the same primitives; non-quantized hooks live in
``nonquant_hooks``.

Failure semantics:
- :class:`ReloadRejected`: raised by pre-write validation. Runtime storage
  has not been touched; the old weights are fully intact.
- :class:`ReloadIncomplete`: raised by ``ReloadContext.finish`` when the
  arrival set is partial. In-place writes have already happened and no old
  weights are kept, so engine state is undefined; the engine must stop
  serving (restart or cold load).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import Enum, auto

import torch

__all__ = [
    "ReloadRejected",
    "ReloadIncomplete",
    "HookReloadUnsupportedError",
    "WeightShard",
    "ArrivalTracker",
    "RuntimeSlot",
    "HookState",
    "WeightReloadHook",
    "ReloadContext",
]


class ReloadRejected(RuntimeError):
    """Pre-write validation failure; runtime storage was not modified."""


class ReloadIncomplete(RuntimeError):
    """Partial arrival detected at finish; engine state is undefined."""


class HookReloadUnsupportedError(RuntimeError):
    """This model/configuration has no hook reload support yet.

    Raised instead of falling back to the layerwise path. See
    docs/design/weight-update/reload-hook-unsupported.md for the list of
    currently unsupported scenarios.
    """


@dataclass(frozen=True)
class WeightShard:
    """One incoming piece of a weight delivered by the transfer engine.

    Args:
        name: checkpoint name of the target weight.
        tensor: shard payload. May have a different dtype than the runtime
            storage; ``copy_`` casts implicitly (not a conversion).
        shard_id: logical shard within a fused weight, e.g. ``"q"``/``"k"``/
            ``"v"`` for merged QKV or ``"w1"``/``"w3"`` for fused MoE w13.
        expert_id: expert index within a fused MoE weight.
    """

    name: str
    tensor: torch.Tensor
    shard_id: str | None = None
    expert_id: int | None = None


@dataclass
class _SlotRecord:
    expected_shape: tuple[int, ...]
    arrived: bool = False


class ArrivalTracker:
    """Table of shard slots that must each be filled exactly once.

    The table shape encodes the weight structure: a single slot for dense
    weights, one slot per shard_id for fused dense weights, E slots for
    fused MoE w2, E x 2 slots for fused MoE w13. Completion is uniform:
    every slot filled.
    """

    def __init__(self) -> None:
        self._slots: dict[Hashable, _SlotRecord] = {}

    def add_slot(self, key: Hashable, expected_shape: Iterable[int]) -> None:
        key = self._normalize(key)
        if key in self._slots:
            raise ValueError(f"Duplicate arrival slot: {key!r}")
        self._slots[key] = _SlotRecord(tuple(expected_shape))

    def expected_shape(self, key: Hashable) -> tuple[int, ...]:
        return self._record(key).expected_shape

    def validate(self, key: Hashable, shape: torch.Size) -> None:
        """Pre-write check: known slot, expected shape, not yet arrived."""
        record = self._record(key)
        if record.arrived:
            raise ReloadRejected(f"Shard arrived twice for slot {key!r}")
        if tuple(shape) != record.expected_shape:
            raise ReloadRejected(
                f"Shard shape {tuple(shape)} does not match expected "
                f"{record.expected_shape} for slot {key!r}"
            )

    def mark_arrived(self, key: Hashable) -> None:
        self._record(key).arrived = True

    def has_arrived(self, key: Hashable) -> bool:
        return self._record(key).arrived

    def num_arrived(self) -> int:
        return sum(record.arrived for record in self._slots.values())

    @property
    def complete(self) -> bool:
        return all(record.arrived for record in self._slots.values())

    def missing(self) -> list[Hashable]:
        return [key for key, record in self._slots.items() if not record.arrived]

    def reset(self) -> None:
        """Clear per-round arrival flags, keeping the slot plan."""
        for record in self._slots.values():
            record.arrived = False

    def _record(self, key: Hashable) -> _SlotRecord:
        key = self._normalize(key)
        try:
            return self._slots[key]
        except KeyError:
            raise ReloadRejected(f"Shard targets unknown slot {key!r}") from None

    @staticmethod
    def _normalize(key: Hashable) -> Hashable:
        # tuples sort/hash consistently for MoE (expert_id, half) keys
        return tuple(key) if isinstance(key, list) else key


class RuntimeSlot:
    """A reloadable runtime tensor with a stable pointer.

    The storage exists since cold load; reloads only ever ``copy_`` into it,
    so ``data_ptr``, the Parameter object, and CUDA-graph captures stay
    byte-identical across reloads.

    All writes go through :meth:`write`, which mirrors torch indexing: an
    ``int`` pins one plane along its dimension (e.g. an expert id in a fused
    MoE slot), a ``slice`` selects a range (row block for merged weights,
    1D/2D tile block for block-wise scales), and omitted trailing dims are
    taken whole. This single primitive covers dense full writes, fused-row
    writes, and tile-level scale write-back uniformly.
    """

    def __init__(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.is_meta:
            raise ValueError(f"Runtime slot {name!r} must be materialized")
        self.name = name
        self.tensor = tensor
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self._data_ptr = tensor.data_ptr()

    @property
    def data_ptr(self) -> int:
        return self._data_ptr

    def check_pointer_stability(self) -> None:
        if self.tensor.data_ptr() != self._data_ptr:
            raise RuntimeError(
                f"Runtime slot {self.name!r} storage moved; reload requires "
                "stable pointers for CUDA graph compatibility"
            )

    def write(
        self,
        src: torch.Tensor,
        index: tuple[int | slice, ...] = (),
    ) -> None:
        """In-place write of ``src`` into the selected region.

        ``index`` follows torch basic indexing per leading dimension: an int
        collapses that dim (selecting one plane, e.g. expert ``e``), a slice
        selects a half-open range (e.g. ``slice(2*N*e, 2*N*e+N)`` for a w13
        gate half, or a ``(slice(r0, r1), slice(c0, c1))`` tile in a
        block-scale grid). Unspecified trailing dims are taken whole, so
        ``write(src)`` with no index is a full-tensor write.

        All validation (rank, bounds, exact region shape) runs before any
        write, so rejection leaves the runtime untouched. ``copy_``
        performs an implicit dtype cast when source and runtime dtypes
        differ.
        """
        if len(index) > len(self.shape):
            raise ReloadRejected(
                f"Write index {index} has more dims than slot "
                f"{self.name!r} with shape {self.shape}"
            )
        normalized: list[int | slice] = []
        for dim, idx in enumerate(index):
            size = self.shape[dim]
            if isinstance(idx, slice):
                if idx.step not in (None, 1):
                    raise ReloadRejected(
                        f"Strided writes are not supported (slot "
                        f"{self.name!r}, dim {dim})"
                    )
                start = 0 if idx.start is None else idx.start
                stop = size if idx.stop is None else idx.stop
                if start < 0 or stop < start or stop > size:
                    raise ReloadRejected(
                        f"Write range [{start}, {stop}) exceeds dim {dim} "
                        f"of slot {self.name!r} with size {size}"
                    )
                normalized.append(slice(start, stop))
            else:
                plane = int(idx)
                if plane < 0 or plane >= size:
                    raise ReloadRejected(
                        f"Plane index {plane} out of bounds for dim {dim} "
                        f"of slot {self.name!r} with size {size}"
                    )
                normalized.append(plane)
        region = self.tensor[tuple(normalized)]
        if tuple(src.shape) != tuple(region.shape):
            raise ReloadRejected(
                f"Shard shape {tuple(src.shape)} does not match region "
                f"{tuple(region.shape)} of slot {self.name!r}"
            )
        self.check_pointer_stability()
        with torch.no_grad():
            region.copy_(src)


class HookState(Enum):
    IDLE = auto()
    ARMED = auto()
    RECEIVING = auto()
    COMMITTED = auto()


class WeightReloadHook(ABC):
    """Per-weight reload hook bound to one :class:`RuntimeSlot`.

    Lifecycle: ``arm`` (start_reload) -> first shard triggers ``pre_reload``
    -> each shard written via ``load_shard`` then ``post_load`` -> completion
    triggers ``finish_load`` exactly once.

    - ``post_load`` runs after every individual shard write. It is a no-op
      by default: per-shard work (offset-mapped ``copy_``, arrival
      registration) already lives in ``_write_shard``. Quantized backends
      may use it for incremental per-shard work, but the design defers all
      fixups to completion.
    - ``finish_load`` runs once when the arrival table is full: convert +
      ``copy_`` into runtime storage, scale clamp/repack, refresh derived
      slots, free the CONVERT buffer. Non-quantized hooks write in place,
      so ``finish_load`` is usually a no-op.
    """

    def __init__(self, slot: RuntimeSlot) -> None:
        self.slot = slot
        self.tracker = ArrivalTracker()
        self.state = HookState.IDLE
        # Set by ReloadContext when aliases are registered: duplicate delivery
        # of an already-committed slot is deduplicated (tied weights).
        self.allow_duplicate_after_commit = False

    @property
    def name(self) -> str:
        return self.slot.name

    @property
    def complete(self) -> bool:
        return self.state is HookState.COMMITTED

    @property
    def received_any(self) -> bool:
        return self.state in (HookState.RECEIVING, HookState.COMMITTED)

    def arm(self) -> None:
        """Enter the reload state before any shard arrives.

        Resets per-round arrival records so hooks are reusable across
        consecutive reloads.
        """
        if self.state is not HookState.IDLE:
            raise ReloadRejected(f"Hook {self.name!r} re-armed mid-reload")
        self.tracker.reset()
        self.state = HookState.ARMED

    def load_shard(self, shard: WeightShard) -> None:
        """Route one incoming shard; auto-commits when complete."""
        if self._begin_shard(shard.name):
            self._write_shard(shard)
            self.post_load()
            self._commit_if_complete()

    def _begin_shard(self, shard_label: str) -> bool:
        """Advance the state machine for an incoming shard.

        Returns True when the shard should be written; False when it is a
        deduplicated post-commit arrival (tied weights).
        """
        if self.state is HookState.IDLE:
            raise ReloadRejected(
                f"Shard for {self.name!r} arrived before start_reload"
            )
        if self.state is HookState.COMMITTED:
            if self.allow_duplicate_after_commit:
                return False
            raise ReloadRejected(
                f"Shard {shard_label!r} arrived after {self.name!r} committed"
            )
        if self.state is HookState.ARMED:
            self.pre_reload()
            self.state = HookState.RECEIVING
        return True

    def _commit_if_complete(self) -> None:
        if self.tracker.complete:
            self.finish_load()
            self.slot.check_pointer_stability()
            self.state = HookState.COMMITTED

    def missing(self) -> list[Hashable]:
        return self.tracker.missing()

    def pre_reload(self) -> None:
        """First-shard hook: metadata checks, optional CONVERT buffer alloc."""

    def post_load(self) -> None:
        """Per-shard hook: runs after each individual shard write."""

    def finish_load(self) -> None:
        """Completion hook: convert + copy_, refresh derived slots, free."""

    @abstractmethod
    def _write_shard(self, shard: WeightShard) -> None:
        """Validate and route one shard into the slot (in place or buffer)."""


class ReloadContext:
    """Lifecycle owner for one reload round across all registered hooks.

    start_reload arms every hook; deliver routes shards by checkpoint name
    (resolving tied-weight aliases); finish enforces the completeness
    invariant: either nothing arrived (no-op) or every armed hook committed,
    otherwise raise :class:`ReloadIncomplete`.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, WeightReloadHook] = {}
        self._aliases: dict[str, str] = {}
        self._active = False

    def register(self, hook: WeightReloadHook, *aliases: str) -> None:
        if hook.name in self._hooks:
            raise ValueError(f"Duplicate hook registration: {hook.name!r}")
        self._hooks[hook.name] = hook
        if aliases:
            hook.allow_duplicate_after_commit = True
            for alias in aliases:
                self.register_alias(alias, hook.name)

    def register_alias(self, alias: str, canonical: str) -> None:
        """Map a second checkpoint name onto an existing hook (tied weights)."""
        if canonical not in self._hooks:
            raise ValueError(f"Alias {alias!r} targets unknown hook {canonical!r}")
        if alias in self._hooks or alias in self._aliases:
            raise ValueError(f"Duplicate alias registration: {alias!r}")
        self._aliases[alias] = canonical
        self._hooks[canonical].allow_duplicate_after_commit = True

    def hook(self, name: str) -> WeightReloadHook:
        canonical = self._aliases.get(name, name)
        try:
            return self._hooks[canonical]
        except KeyError:
            raise ReloadRejected(f"No reload hook registered for {name!r}") from None

    @property
    def hooks(self) -> dict[str, WeightReloadHook]:
        return dict(self._hooks)

    def start_reload(self) -> None:
        if self._active:
            raise ReloadRejected("start_reload called while a reload is active")
        for hook in self._hooks.values():
            hook.arm()
        self._active = True

    def deliver(self, shard: WeightShard) -> None:
        if not self._active:
            raise ReloadRejected(
                f"Shard {shard.name!r} delivered outside an active reload"
            )
        self.hook(shard.name).load_shard(shard)

    def finish(self) -> None:
        """Complete the reload. No-op if no shard arrived at all."""
        if not self._active:
            raise ReloadRejected("finish called without start_reload")
        hooks = list(self._hooks.values())
        arrived = [hook for hook in hooks if hook.received_any]
        try:
            if not arrived:
                return  # empty reload is a side-effect-free no-op
            incomplete = {
                hook.name: hook.missing() for hook in hooks if not hook.complete
            }
            if incomplete:
                raise ReloadIncomplete(
                    "Reload finished with missing shards; runtime storage is "
                    f"partially overwritten and engine state is undefined: "
                    f"{incomplete}"
                )
        finally:
            for hook in hooks:
                hook.state = HookState.IDLE
            self._active = False
