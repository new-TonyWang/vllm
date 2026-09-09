# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline FP8 per-block reload hooks (design doc section 8).

The source checkpoint is already FP8 + block scales, so dtypes match the
runtime and the only differences are backend layouts:

- identity backends (Triton / CUTLASS dense / Torch reference): shards are
  written in place by the original loader, tracking only;
- DeepGemm (dense + MoE, also FlashInfer-DeepGEMM dense): the scale factor
  layout is produced by ``transform_sf_into_required_layout`` (an atom
  swizzle, not a view), so scale shards are staged in checkpoint layout and
  transformed once at ``finish_load`` (doc case #9-adjacent: the scale
  tensors are tiny, the weights stay in place);
- FlashInfer CUTLASS block-wise MoE: cold load swaps w13 gate/up halves
  (w13 -> w31) and clamps block scales. The swap is a view-level mapping,
  applied at write time by flipping the loader's shard_id (w1 <-> w3); the
  clamp is deferred to ``finish_load`` and applied to the whole scale
  tensor in place.

UE8M0 requant (SM100) rewrites weights and is therefore unsupported here:
such models fall back to the layerwise path.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch

from .hooks import ReloadRejected, RuntimeSlot
from .inplace import LoaderWeightHook

__all__ = [
    "Fp8BlockIdentityHook",
    "Fp8BlockClampHook",
    "Fp8MoeW31SwapHook",
    "DeepGemmScaleHook",
]


class Fp8BlockIdentityHook(LoaderWeightHook):
    """Identity-layout FP8 block tensor: track arrivals, write in place."""


class Fp8BlockClampHook(LoaderWeightHook):
    """Identity write plus an in-place scale clamp at finish_load.

    Matches the FlashInfer CUTLASS block-wise MoE cold-load behavior of
    clamping near-zero block scales (dead experts) to avoid NaNs.
    """

    def __init__(
        self,
        slot: RuntimeSlot,
        original_loader: Callable,
        *,
        clamp_min: float,
    ) -> None:
        super().__init__(slot, original_loader)
        self._clamp_min = clamp_min

    def finish_load(self) -> None:
        with torch.no_grad():
            self.slot.tensor.clamp_(min=self._clamp_min)


class Fp8MoeW31SwapHook(LoaderWeightHook):
    """Write-time mapping for the FlashInfer CUTLASS w13 -> w31 swap.

    Cold load stores w13 in [up; gate] (w31) order; checkpoint shards arrive
    as w1 (gate) / w3 (up). Flipping the loader's shard_id routes each shard
    to the opposite half of the runtime slot, which is exactly the cold-load
    swap expressed as a write-time mapping.
    """

    _SWAP = {"w1": "w3", "w3": "w1"}

    def __init__(
        self,
        slot: RuntimeSlot,
        original_loader: Callable,
        *,
        clamp_min: float | None = None,
    ) -> None:
        super().__init__(slot, original_loader)
        self._clamp_min = clamp_min

    def _pre_delegate(
        self, bound_args: inspect.BoundArguments
    ) -> inspect.BoundArguments:
        shard_id = bound_args.arguments.get("shard_id")
        if shard_id is not None:
            try:
                bound_args.arguments["shard_id"] = self._SWAP[str(shard_id)]
            except KeyError:
                raise ReloadRejected(
                    f"w13 hook {self.name!r} got unexpected shard_id "
                    f"{shard_id!r}"
                ) from None
        return bound_args

    def finish_load(self) -> None:
        if self._clamp_min is not None:
            with torch.no_grad():
                self.slot.tensor.clamp_(min=self._clamp_min)


class DeepGemmScaleHook(LoaderWeightHook):
    """Staged reload for DeepGemm block scale factors.

    Scale shards are written into a checkpoint-layout staging buffer by the
    original loader; ``finish_load`` runs the DeepGemm scale layout
    transform once and copies the result into the runtime storage, then
    frees the buffer. The FP8 weights themselves are layout-identical and
    use :class:`Fp8BlockIdentityHook`.
    """

    def __init__(
        self,
        slot: RuntimeSlot,
        original_loader: Callable,
        *,
        staging_shape: tuple[int, ...],
        block_shape: tuple[int, int],
        mn: int,
        k: int,
        num_groups: int = 1,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        staging_template: torch.Tensor | None = None,
    ) -> None:
        super().__init__(slot, original_loader)
        self._staging_shape = staging_shape
        self._staging_template = staging_template
        self._block_shape = block_shape
        self._mn = mn
        self._k = k
        self._num_groups = num_groups
        self._staging: torch.Tensor | None = None
        self._transform = transform

    def pre_reload(self) -> None:
        staging = torch.nn.Parameter(
            torch.empty(
                self._staging_shape,
                dtype=torch.float32,
                device=self.slot.tensor.device,
            ),
            requires_grad=False,
        )
        # the loader routes through param attributes (load_qkv_weight,
        # tp_rank, output_dim, ...); give the staging buffer the cold-load
        # parameter's class and attributes so those paths work unchanged
        template = self._staging_template
        if template is not None:
            # MoE scale params are plain Parameters whose loader attributes
            # (quant_method, tp_rank, ...) live in __dict__; BasevLLMParameter
            # subclasses additionally need their class restored
            if type(template) is not torch.nn.Parameter:
                staging.__class__ = type(template)
            for attr, value in template.__dict__.items():
                staging.__dict__.setdefault(attr, value)
        self._staging = staging

    def _pre_delegate(
        self, bound_args: inspect.BoundArguments
    ) -> inspect.BoundArguments:
        if self._staging is None:
            raise ReloadRejected(
                f"DeepGemm scale hook {self.name!r} received a shard before "
                "pre_reload allocated its staging buffer"
            )
        bound_args.arguments["param"] = self._staging
        return bound_args

    def finish_load(self) -> None:
        assert self._staging is not None
        if self._transform is not None:
            transformed = self._transform(self._staging)
        else:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                deepgemm_post_process_weight_scale_block,
            )

            # the DeepGemm layout transform requires a 3D grouped tensor;
            # mirror the cold-load helper's unsqueeze/squeeze for 2D dense
            # scales
            staging = self._staging
            squeeze = staging.ndim == 2
            if squeeze:
                staging = staging.unsqueeze(0)
            transformed = deepgemm_post_process_weight_scale_block(
                ws=staging,
                mn=self._mn,
                k=self._k,
                quant_block_shape=self._block_shape,
                num_groups=self._num_groups,
            )
            if squeeze:
                transformed = transformed.squeeze(0)
        if tuple(transformed.shape) != self.slot.shape:
            raise RuntimeError(
                f"DeepGemm scale transform for {self.name!r} produced "
                f"{tuple(transformed.shape)}, runtime slot is {self.slot.shape}"
            )
        self.slot.write(transformed)
        self._staging = None
