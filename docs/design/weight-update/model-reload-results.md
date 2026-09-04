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
| --- | --- | --- | --- |
| `Qwen2.5-7B-Instruct-2layer` | day0-kit NCCL | PASS | `/inspire/hdd/global_user/wangtongyu-25057/day0-result.json`; `send_weights_completed=true`, `weight_epoch=1`, 27 tensors, 3,112,227,840 bytes; H200 `rc=0` |
| `Qwen2.5-7B-Instruct` | day0-kit NCCL | PASS | `day0-qwen25-full-day0-rerun.json`; `send_weights_completed=true`, epoch/update version 1, 339 tensors, 15,231,233,024 bytes, 30 buckets; H200 `rc=0` |
| `Qwen3-8B-2layer` | day0-kit NCCL | PASS | `day0-qwen3-8b-day0.json`; `send_weights_completed=true`, epoch/update version 1, 25 tensors, 3,261,113,344 bytes, 4 buckets; H200 `rc=0` |
| `Qwen3.8-27B-FP8-2layer` | direct vLLM RPC reload | PASS (non-day0) | H200 `rc=0`; reload loaded 3 shards in 2.75s and second generation completed |
| `Qwen3-30B-A3B-FP8` | direct vLLM RPC reload | PASS (non-day0) | H200 `rc=0`; persistent result `qwen30-reload.json` has identical pre/post output `" _"` |
| `Qwen3-30B-A3B-FP8` | day0-kit NCCL | PASS | `day0-qwen30-rerun6.json`; `send_weights_completed=true`, epoch/update version 1, 37,491 tensors, 32,444,792,832 bytes, 52 buckets; H200 `rc=0` |
| `Qwen3-30B-A3B` | day0-kit NCCL | PASS | `day0-qwen30-bf16-full-day0.json`; `send_weights_completed=true`, epoch/update version 1, 18,867 tensors, 61,064,245,248 bytes, 54 buckets; H200 `rc=0` |
| `Step-3.7-Flash` | day0-kit NCCL, TP=4 | PASS | `day0-step37-full-day0-rerun.json`; inference/rendezvous world size 4/5, `send_weights_completed=true`, epoch/update version 1, 1,471 tensors, 402,730,656,512 bytes, 273 buckets; H200 detached job `rc=0` |
| `Kimi-K2-Instruct-0905-2layer` | day0-kit NCCL | PASS | `day0-kimi-k2-2layer-day0-rerun.json`; `send_weights_completed=true`, epoch/update version 1, 2,351 tensors, 22,261,573,056 bytes, 5 buckets; H200 detached job `rc=0` |
| `Llama-3.2-1B-Instruct` | day0-kit NCCL | PASS | `day0-llama32-1b-day0-rerun.json`; `send_weights_completed=true`, epoch/update version 1, 146 tensors, 2,471,628,800 bytes, 5 buckets, post-finish health 200; H200 detached job `rc=0` |
| `Llama-3.2-1B-Instruct` online block FP8 | day0-kit NCCL | PASS | `day0-llama32-online-block-fp8-day0.json`; `Fp8PerBlockOnlineLinearMethod` with DeepGEMM, `send_weights_completed=true`, epoch/update version 1, 146 tensors, 2,471,628,800 bytes, 5 buckets, post-finish health 200; H200 detached job `rc=0` |
| `Mixtral-8x7B-Instruct-v0.1-2layer` | day0-kit NCCL | PASS | `day0-mixtral-2layer-day0-rerun.json`; `send_weights_completed=true`, epoch/update version 1, 65 tensors, 6,329,376,768 bytes, 4 buckets, post-finish health 200; H200 detached job `rc=0` |
| `Mixtral-8x7B-Instruct-v0.1-2layer` online per-tensor FP8 | day0-kit NCCL | PASS | `day0-mixtral-online-tensor-fp8-day0.json`; `Fp8PerTensorOnlineLinearMethod` with CUTLASS and TRITON FP8 MoE, `send_weights_completed=true`, epoch/update version 1, 65 tensors, 6,329,376,768 bytes, 4 buckets, post-finish health 200; H200 detached job `rc=0` |
| `Qwen2.5-7B-Instruct-GPTQ-Int4` | day0-kit NCCL, Marlin | PASS | `day0-qwen25-gptq-marlin-day0.json`; `MarlinLinearKernel`, `send_weights_completed=true`, epoch/update version 1, 927 tensors, 5,575,277,568 bytes, 9 buckets, post-finish health 200; H200 detached job `rc=0` |
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
- The complete Qwen2.5 directory had a reduced-checkpoint index copied into it.
  The original ModelScope index was restored before validation; the prior index
  remains as `model.safetensors.index.json.reduced-backup-20260903`.

## Unified evidence audit

`/inspire/hdd/global_user/wangtongyu-25057/verify_day0_reload.py` validates every
PASS row's JSON and matching server log. It checks the NCCL backend and trainer
transport, start/update/finish ordering, bucket indices and aggregate
tensor/byte counts, epoch/update version, and `send_weights_completed=true`.
The fifteen-case audit completed on H200 with `status=ok rc=0`. It also
checks inference and rendezvous world sizes: 4/5 for the TP=4 Step server and
1/2 for each single-rank server.

The reduced Kimi checkpoint keeps global tensors and layers 0–1 from the full
checkpoint index. Its three shard files are hard links to the original files.
The copied custom tokenizer imports `bytes_to_unicode` from its current
Transformers module; no package downgrade or alternate runtime was used. The
day0 runner now exits early if the vLLM server dies during its health loop.
The Llama checkpoint was downloaded from ModelScope without modification; its
2,471,645,608-byte safetensors file matches repository SHA-256
`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`.
The runner also checks server health after the publisher returns, and the
Llama verifier requires that health response to occur after finish.

The reduced Mixtral checkpoint contains layers 0–1 and every expert in those
layers. Merely reducing its index was insufficient because vLLM iterates every
tensor in a selected physical shard; the final checkpoint therefore rewrites
three compact safetensors files containing only the 65 indexed tensors. Each
source shard was verified against its ModelScope SHA-256 before compaction.
The failed index-only attempt exited with H200 job `rc=1`; the compact rerun
completed with `rc=0` and post-finish health 200.

The complete Qwen2.5 GPTQ checkpoint contains two repository-verified
safetensors shards and uses 4-bit symmetric weights with group size 128 and
`desc_act=false`. The first auto-backend control run selected Machete and was
not counted as Marlin evidence. The accepted run set
`DAY0_LINEAR_BACKEND=marlin`; its server log records
`Using MarlinLinearKernel for AutoGPTQLinearMethod`, followed by nine successful
update buckets, finish, and post-finish health 200.

The online FP8 cases reuse BF16 checkpoints rather than serialized FP8 weights.
The Llama run selected `DeepGemmFp8BlockScaledMMKernel` for
`Fp8PerBlockOnlineLinearMethod`, covering 128x128 online block quantization.
The Mixtral run selected `CutlassFP8ScaledMMLinearKernel` for
`Fp8PerTensorOnlineLinearMethod` and the TRITON FP8 MoE backend, covering fused
w1/w3 expert quantization. Both reloads completed through the current layerwise
restore, requantize, and stable-runtime-storage copy path. These same-checkpoint
protocol checks do not replace the later A/B value and CUDA graph replay tests.

## Background-job behavior

`ai4qz` branch `fix/detached-run` adds `run --detach`. It submits the workload
under `nohup`, immediately returns job ID/PID/log/return-code paths, and keeps
the process alive after the temporary Jupyter terminal is deleted. An H200
control job survived terminal cleanup, wrote its delayed artifact, and exposed
the expected return code. `status=submitted` confirms launch only; model PASS
still requires the job return-code file and the day0 result JSON.
Job paths are inside the target notebook container, so they must be queried via
`ai4qz run <target>` rather than read directly on the `qz_dev` submission host.
