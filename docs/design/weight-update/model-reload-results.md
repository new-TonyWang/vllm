# Model Reload Validation

This record separates the vLLM day-0-kit NCCL protocol from the earlier direct
`collective_rpc("reload_weights")` smoke tests. A model is marked **NCCL
day0 PASS** only when the day0-kit publisher completes
`start_weight_update -> NCCL update buckets -> finish_weight_update` against a
server started with `--weight-transfer-config '{"backend":"nccl"}'`.

## Runtime

- Host: `qz_dev`, H200 workload submitted with `ai4qz run h200_ncu`
- Python: `/inspire/hdd/global_user/wangtongyu-25057/miniconda3/envs/vllm/bin/python`
- Checkout: `/inspire/hdd/global_user/wangtongyu-25057/vllm-git-test`
- day-0 kit: `/inspire/hdd/global_user/wangtongyu-25057/vllm-rl-day0-kit`
- Code commits: selective-reload base `066ec5f59e52a8f6afd65dc2b9f6ecfcb820b051`;
  DeepSeek V4 partial-bucket fix `cc1f202883`
- day0-kit dtype fix: `c446b55`
- NCCL settings: `backend=nccl`, `NCCL_SOCKET_IFNAME=lo`

## Results

| Model/checkpoint | Protocol | Result | Evidence |
|---|---|---|---|
| `Qwen2.5-7B-Instruct-2layer` | day0-kit NCCL | PASS | `/inspire/hdd/global_user/wangtongyu-25057/day0-result.json`; `send_weights_completed=true`, `weight_epoch=1`, 27 tensors, 3,112,227,840 bytes; H200 `rc=0` |
| `Qwen3.8-27B-FP8-2layer` | direct vLLM RPC reload | PASS (non-day0) | H200 `rc=0`; reload loaded 3 shards in 2.75s and second generation completed |
| `Qwen3-30B-A3B-FP8` | direct vLLM RPC reload | PASS (non-day0) | H200 `rc=0`; persistent result `qwen30-reload.json` has identical pre/post output `" _"` |
| `Qwen3-30B-A3B-FP8` | day0-kit NCCL | PASS | `day0-qwen30-rerun6.json`; `send_weights_completed=true`, epoch/update version 1, 37,491 tensors, 32,444,792,832 bytes, 52 buckets; H200 `rc=0` |
| `DeepSeek-V3-FP8-2layer` | direct vLLM RPC reload | PASS (non-day0) | H200 `rc=0`; reload loaded the checkpoint and second generation completed |
| `DeepSeek-V3-FP8-2layer` | day0-kit NCCL | PASS | `day0-deepseek-v3-rerun.json`; `send_weights_completed=true`, epoch/update version 1, 1,581 tensors, 15,802,320,320 bytes, 5 buckets; H200 `rc=0` |
| `DeepSeek-V4-Flash-FP8-2layer` | day0-kit NCCL | PASS | `day0-deepseek-v4-rerun5.json`; `send_weights_completed=true`, epoch/update version 1, 4,711 tensors, 12,844,479,536 bytes, 7 buckets; H200 `rc=0` |
| `Qwen3.8-27B-FP8-2layer` | day0-kit NCCL | PASS | `day0-qwen38-rerun4.json`; `send_weights_completed=true`, epoch/update version 1, 398 tensors, 7,251,989,472 bytes, 7 buckets; H200 `rc=0` |

## Day-0 rerun requirements

All current matrix entries have completed the day0 NCCL protocol. A new model is
not promoted to **NCCL day0 PASS** until the result records start/update/finish,
reports `send_weights_completed=true`, and the completed H200 workload returns
`rc=0`.

## Fixed-environment build evidence (2026-09-03)

- CMake was configured in the fixed `vllm` environment with `TORCH_CUDA_ARCH_LIST=9.0a`.
- DeepGEMM, CUTLASS, FMHA, FlashMLA, FlashKDA, and QUTLASS sources were fetched
  into the checkout build directory; no packages were installed into `kernel_dev`.
- `_moe_C_stable_libtorch.abi3.so` (24 targets) and `_C_stable_libtorch.abi3.so`
  (45 targets) compiled successfully and were copied into the fixed environment.
- Container-native `_C_stable_libtorch.abi3.so` and
  `_moe_C_stable_libtorch.abi3.so` builds completed under the H200 container's
  CUDA 13.0/glibc 2.35 runtime and import together in the fixed environment.
- The source checkout is selected through `PYTHONPATH`, keeping Python wrappers
  aligned with the rebuilt extensions. FlashMLA extension links are available
  from the checkout and import successfully.
- Qwen3.8, Qwen3-30B-A3B, and DeepSeek V3 now complete the day0 NCCL protocol;
  the earlier ABI, FlashAttention, and mixed host/container build blockers are
  resolved.

## Background-job behavior

`ai4qz` branch `fix/detached-run` adds `run --detach`. It submits the workload
under `nohup`, immediately returns job ID/PID/log/return-code paths, and keeps
the process alive after the temporary Jupyter terminal is deleted. An H200
control job survived terminal cleanup, wrote its delayed artifact, and exposed
the expected return code. `status=submitted` confirms launch only; model PASS
still requires the job return-code file and the day0 result JSON.
