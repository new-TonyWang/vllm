# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-quantized reload hooks (design doc section 7).

Runtime storage has the same layout as the checkpoint (dtype casts are
handled implicitly by ``copy_`` and are not conversions), so every hook
writes in place with zero staging. The cases differ only in
arrival-tracking granularity and row-offset computation:

- :class:`DenseWeightHook`: plain dense weight, single arrival slot.
- :class:`MergedRowsWeightHook`: row-fused dense weight (merged QKV,
  gate_up), one slot per logical shard_id.
- :class:`VocabParallelWeightHook`: vocab-parallel weight with padding;
  only the real vocab rows are written.
- Tied weights: no dedicated hook; register the same hook with alias names
  via ``ReloadContext.register(hook, "alias.name")`` and duplicate delivery
  of the shared slot is deduplicated.
- :class:`MoeW2WeightHook`: fused expert w2 (E x K x N), one slot per expert.
- :class:`MoeW13WeightHook`: fused expert w13 (E x 2N x K), one slot per
  (expert, w1/w3 half).
"""

from __future__ import annotations

from collections.abc import Mapping

from .hooks import (
    ReloadRejected,
    RuntimeSlot,
    WeightReloadHook,
    WeightShard,
)

__all__ = [
    "DenseWeightHook",
    "MergedRowsWeightHook",
    "VocabParallelWeightHook",
    "MoeW2WeightHook",
    "MoeW13WeightHook",
]

_FULL = "full"


class DenseWeightHook(WeightReloadHook):
    """Plain dense weight: one shard fills the whole slot (cases 1, 8)."""

    def __init__(self, slot: RuntimeSlot) -> None:
        super().__init__(slot)
        self.tracker.add_slot(_FULL, slot.shape)

    def _write_shard(self, shard: WeightShard) -> None:
        self.tracker.validate(_FULL, shard.tensor.shape)
        self.slot.write(shard.tensor)
        self.tracker.mark_arrived(_FULL)


class MergedRowsWeightHook(WeightReloadHook):
    """Row-fused dense weight: q/k/v or gate/up shards into row ranges.

    Args:
        slot: fused runtime slot.
        shard_rows: shard_id -> (row_start, num_rows) in registration order,
            matching the fusion order recorded by the cold-load weight
            loader (q then k then v; gate then up). Ranges must be
            contiguous and cover the whole row dimension.
    """

    def __init__(
        self,
        slot: RuntimeSlot,
        shard_rows: Mapping[str, tuple[int, int]],
    ) -> None:
        super().__init__(slot)
        self._shard_rows = dict(shard_rows)
        covered = 0
        for shard_id, (row_start, num_rows) in self._shard_rows.items():
            if row_start != covered or num_rows <= 0:
                raise ValueError(
                    f"Shard {shard_id!r} row range must be contiguous and in "
                    "registration order"
                )
            covered += num_rows
            self.tracker.add_slot(shard_id, (num_rows, slot.shape[-1]))
        if covered != slot.shape[0]:
            raise ValueError(
                f"Shard rows cover {covered} of {slot.shape[0]} rows in "
                f"slot {slot.name!r}"
            )

    def _write_shard(self, shard: WeightShard) -> None:
        if shard.shard_id is None:
            raise ReloadRejected(
                f"Fused weight {self.name!r} requires shard_id on every shard"
            )
        self.tracker.validate(shard.shard_id, shard.tensor.shape)
        row_start, num_rows = self._shard_rows[shard.shard_id]
        self.slot.write(shard.tensor, (slice(row_start, row_start + num_rows),))
        self.tracker.mark_arrived(shard.shard_id)


class VocabParallelWeightHook(WeightReloadHook):
    """Vocab-parallel weight with padding: embeddings and lm_head (case 3).

    Only the first ``vocab_rows`` rows are written; the padding tail keeps
    its cold-load values.
    """

    def __init__(self, slot: RuntimeSlot, vocab_rows: int) -> None:
        super().__init__(slot)
        if not 0 < vocab_rows <= slot.shape[0]:
            raise ValueError(
                f"vocab_rows {vocab_rows} outside slot {slot.name!r} with "
                f"{slot.shape[0]} rows"
            )
        self._vocab_rows = vocab_rows
        self.tracker.add_slot(_FULL, (vocab_rows, slot.shape[-1]))

    def _write_shard(self, shard: WeightShard) -> None:
        self.tracker.validate(_FULL, shard.tensor.shape)
        self.slot.write(shard.tensor, (slice(0, self._vocab_rows),))
        self.tracker.mark_arrived(_FULL)


class MoeW2WeightHook(WeightReloadHook):
    """Fused MoE w2 (E x K x N): one arrival slot per expert (case 5).

    Each shard carries an ``expert_id`` and a (K, N) payload written to
    plane ``e`` of the (E, K, N) slot.
    """

    def __init__(self, slot: RuntimeSlot) -> None:
        super().__init__(slot)
        if len(slot.shape) != 3:
            raise ValueError(
                f"MoeW2 slot {slot.name!r} must be 3D (E, K, N), got {slot.shape}"
            )
        self._num_experts, self._expert_rows = slot.shape[0], slot.shape[1]
        for expert_id in range(self._num_experts):
            self.tracker.add_slot(expert_id, slot.shape[1:])

    def _write_shard(self, shard: WeightShard) -> None:
        expert_id = self._checked_expert_id(shard)
        self.tracker.validate(expert_id, shard.tensor.shape)
        self.slot.write(shard.tensor, (expert_id,))
        self.tracker.mark_arrived(expert_id)

    def _checked_expert_id(self, shard: WeightShard) -> int:
        expert_id = shard.expert_id
        if expert_id is None or not 0 <= expert_id < self._num_experts:
            raise ReloadRejected(
                f"Shard {shard.name!r} has invalid expert_id {expert_id}; "
                f"expected [0, {self._num_experts})"
            )
        return expert_id


class MoeW13WeightHook(MoeW2WeightHook):
    """Fused MoE w13 (E x 2N x K): one slot per (expert, half) (case 6).

    w1 (gate) shards write rows [2N*e, 2N*e+N); w3 (up) shards write rows
    [2N*e+N, 2N*e+2N). Completion requires all E x 2 slots filled.
    """

    _HALVES = ("w1", "w3")

    def __init__(self, slot: RuntimeSlot) -> None:
        WeightReloadHook.__init__(self, slot)
        if len(slot.shape) != 3 or slot.shape[1] % 2 != 0:
            raise ValueError(
                f"MoeW13 slot {slot.name!r} must be 3D (E, 2N, K), got "
                f"{slot.shape}"
            )
        self._num_experts = slot.shape[0]
        self._expert_rows = slot.shape[1]
        self._half_rows = slot.shape[1] // 2
        for expert_id in range(self._num_experts):
            for half in self._HALVES:
                self.tracker.add_slot(
                    (expert_id, half), (self._half_rows, slot.shape[2])
                )

    def _write_shard(self, shard: WeightShard) -> None:
        expert_id = self._checked_expert_id(shard)
        if shard.shard_id not in self._HALVES:
            raise ReloadRejected(
                f"Shard {shard.name!r} has invalid half {shard.shard_id!r}; "
                f"expected one of {self._HALVES}"
            )
        key = (expert_id, shard.shard_id)
        self.tracker.validate(key, shard.tensor.shape)
        half_start = self._half_rows if shard.shard_id == "w3" else 0
        self.slot.write(
            shard.tensor,
            (expert_id, slice(half_start, half_start + self._half_rows)),
        )
        self.tracker.mark_arrived(key)
