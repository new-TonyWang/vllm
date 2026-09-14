# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staged hook reload for offline FP8 Marlin linear weights."""

from __future__ import annotations

import torch

from .hooks import ReloadRejected
from .inplace import LoaderWeightHook


def _copy_loader_attrs(template: torch.Tensor, staging: torch.Tensor) -> None:
    if type(template) is not torch.nn.Parameter:
        staging.__class__ = type(template)
    for attr, value in template.__dict__.items():
        staging.__dict__.setdefault(attr, value)


class _MarlinStagedHook(LoaderWeightHook):
    def __init__(
        self,
        slot,
        original_loader,
        cold_param,
        staging_shape,
        staging_dtype: torch.dtype,
    ) -> None:
        super().__init__(slot, original_loader)
        self._cold_param = cold_param
        self._staging_shape = staging_shape
        # checkpoint-side dtype: the cold_param is the post-PWAL runtime
        # tensor (int32-packed weight / bf16 scales for Marlin), so its dtype
        # is not the staging dtype
        self._staging_dtype = staging_dtype
        self._staging: torch.Tensor | None = None

    def pre_reload(self) -> None:
        # the conversion covers the whole tensor, so stage the full
        # checkpoint-layout parameter; merged shards (q/k/v, gate/up) write
        # into it at their loader offsets
        self._staging = torch.nn.Parameter(
            torch.empty(
                self._staging_shape,
                dtype=self._staging_dtype,
                device=self._cold_param.device,
            ),
            requires_grad=False,
        )
        _copy_loader_attrs(self._cold_param, self._staging)

    def _pre_delegate(self, bound_args):
        if self._staging is None:
            raise ReloadRejected(
                f"Marlin hook {self.name!r} received a shard before "
                "pre_reload allocated its staging buffer"
            )
        bound_args.arguments["param"] = self._staging
        return bound_args



class MarlinFp8WeightHook(_MarlinStagedHook):
    def __init__(
        self,
        slot,
        original_loader,
        cold_param,
        *,
        size_k: int,
        size_n: int,
        size_k_first: bool,
        group_size: int,
    ) -> None:
        self._staging_shape = (
            size_k,
            size_n,
        ) if size_k_first else (size_n, size_k)
        super().__init__(
            slot,
            original_loader,
            cold_param,
            self._staging_shape,
            torch.float8_e4m3fn,  # checkpoint weights are fp8
        )
        self._size_k = size_k
        self._size_n = size_n
        self._size_k_first = size_k_first
        self._group_size = group_size

    def finish_load(self) -> None:
        assert self._staging is not None
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_pad_qweight,
            marlin_padded_nk,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            pack_fp8_to_int32,
        )

        padded_n, padded_k = marlin_padded_nk(
            self._size_n, self._size_k, self._group_size
        )
        qweight = pack_fp8_to_int32(self._staging, self._size_k_first)
        if not self._size_k_first:
            qweight = qweight.T.contiguous()
        qweight = marlin_pad_qweight(
            qweight, self._size_n, self._size_k, padded_n, padded_k
        )
        from vllm import _custom_ops as ops

        converted = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            perm=torch.empty(0, dtype=torch.int, device=qweight.device),
            size_k=padded_k,
            size_n=padded_n,
            num_bits=8,
        )
        self.slot.write(converted)
        self._staging = None


class MarlinFp8ScaleHook(_MarlinStagedHook):
    def __init__(
        self,
        slot,
        original_loader,
        cold_param,
        *,
        size_k: int,
        size_n: int,
        size_k_first: bool,
        block_size: tuple[int, int],
        input_dtype: torch.dtype | None,
    ) -> None:
        self._staging_shape = (
            (size_k + block_size[1] - 1) // block_size[1],
            (size_n + block_size[0] - 1) // block_size[0],
        )
        if not size_k_first:
            self._staging_shape = self._staging_shape[::-1]
        super().__init__(
            slot,
            original_loader,
            cold_param,
            self._staging_shape,
            torch.float32,  # checkpoint block scales are fp32
        )
        self._size_k = size_k
        self._size_n = size_n
        self._size_k_first = size_k_first
        self._block_size = block_size
        self._input_dtype = input_dtype
        # Marlin's cold-load path converts checkpoint fp32 scales to the
        # layer's original dtype before fusing the exponent bias.
        self._runtime_scale_dtype = cold_param.dtype

    def finish_load(self) -> None:
        assert self._staging is not None
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_pad_scales,
            marlin_padded_nk,
            marlin_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            fp8_fused_exponent_bias_into_scales,
        )

        block_n, block_k = self._block_size
        padded_n, padded_k = marlin_padded_nk(
            self._size_n, self._size_k, block_k
        )
        scales = self._staging
        if not self._size_k_first:
            scales = scales.T.contiguous()
        scales = scales.repeat_interleave(block_n, 1)[:, : self._size_n]
        scales = marlin_pad_scales(
            scales, self._size_n, self._size_k, padded_n, padded_k, block_k
        )
        converted = marlin_permute_scales(
            s=scales, size_k=padded_k, size_n=padded_n, group_size=block_k
        )
        if self._input_dtype != torch.float8_e4m3fn:
            converted = converted.to(self._runtime_scale_dtype)
            converted = fp8_fused_exponent_bias_into_scales(converted)
        self.slot.write(converted)
        self._staging = None
