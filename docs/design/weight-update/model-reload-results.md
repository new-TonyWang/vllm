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
  DeepSeek V4 partial-bucket fix `cc1f202883`; packed wire negotiation
  `38cf4a5`; packed stream ordering `c95957f3b8`; old-NCCL compatibility and
  disabled-communicator fail-closed `59bdf7cf92`; post-update cache invalidation
  `730eab3782`
- day0-kit dtype fix: `c446b55`; configurable prefix-cache oracle `152c2c0`
- NCCL settings: `backend=nccl`, `NCCL_SOCKET_IFNAME=lo`

## Results

| Model/checkpoint | Protocol | Result | Evidence |
| --- | --- | --- | --- |
| `Qwen2.5-7B-Instruct-2layer` | day0-kit NCCL | LEGACY CONTROL ONLY | `/inspire/hdd/global_user/wangtongyu-25057/day0-result.json`; transaction completed before `59bdf7cf92`, so data transfer is unproven |
| `Qwen2.5-7B-Instruct` | day0-kit NCCL | **PASS** | `day0-qwen25-full-real-nccl-rerun.json`; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 339 tensors, 15,231,233,024 bytes, 30 buckets; verifier PASS; H200 job `720901bac91d`, `rc=0` |
| `Qwen3-8B-2layer` | day0-kit NCCL | **PASS** | `day0-qwen3-8b-real-nccl-rerun.json`; publisher log proves NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 25 tensors, 3,261,113,344 bytes, 4 buckets; H200 job `ae8ebda6d236`, `rc=0` |
| `Qwen3.8-27B-FP8-2layer` | direct vLLM RPC reload | PASS (non-day0) | H200 `rc=0`; reload loaded 3 shards in 2.75s and second generation completed |
| `Qwen3-30B-A3B-FP8` | direct vLLM RPC reload | PASS (non-day0) | H200 `rc=0`; persistent result `qwen30-reload.json` has identical pre/post output `" _"` |
| `Qwen3-30B-A3B-FP8` | day0-kit NCCL | **PASS** | `day0-qwen30-fp8-real-nccl-rerun.json`; DeepGEMM linear and TRITON FP8 MoE selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 37,491 tensors, 32,444,792,832 bytes, 52 buckets; verifier PASS; H200 job `e17734c6a0f2`, `rc=0` |
| `Qwen3-30B-A3B` | day0-kit NCCL | **PASS** | `day0-qwen30-bf16-real-nccl-rerun.json`; TRITON unquantized MoE selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 18,867 tensors, 61,064,245,248 bytes, 54 buckets; verifier PASS; H200 job `eff7a3ceca39`, `rc=0` |
| `Step-3.7-Flash` | day0-kit NCCL, TP=4 | **PASS** | `day0-step37-real-nccl-rerun2.json`; TRITON unquantized MoE selected; NCCL 2.28.9 rank 0 / GPU 4 / nranks 5 Init COMPLETE; 1,471 tensors, 402,730,656,512 bytes, 273 buckets; verifier PASS; H200 job `f9dd24a4d2bd`, `rc=0` |
| `Kimi-K2-Instruct-0905-2layer` | day0-kit NCCL | **PASS** | `day0-kimi-k2-2layer-real-nccl-rerun.json`; DeepGEMM FP8, FLASH_ATTN MLA, and TRITON FP8 MoE selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 2,351 tensors, 22,261,573,056 bytes, 5 buckets; verifier PASS; H200 job `68e4ff7b10ef`, `rc=0` |
| `Llama-3.2-1B-Instruct` | day0-kit NCCL | LEGACY CONTROL ONLY | `day0-llama32-1b-day0-rerun.json`; superseded by the verified A/B row below |
| `Llama-3.2-1B-Instruct` online block FP8 | day0-kit NCCL | **PASS** | `day0-llama32-online-block-fp8-real-nccl-rerun.json`; DeepGEMM online block-FP8 kernel selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 146 tensors, 2,471,628,800 bytes, 5 buckets; verifier PASS; H200 job `79491f51e73d`, `rc=0` |
| `Mixtral-8x7B-Instruct-v0.1-2layer` | day0-kit NCCL | **PASS** | `day0-mixtral-2layer-real-nccl-rerun.json`; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 65 tensors, 6,329,376,768 bytes, 4 buckets; verifier PASS; H200 job `731feb61e985`, `rc=0` |
| `Mixtral-8x7B-Instruct-v0.1-2layer` online per-tensor FP8 | day0-kit NCCL | **PASS** | `day0-mixtral-online-tensor-fp8-real-nccl-rerun.json`; CUTLASS online linear and TRITON FP8 MoE selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 65 tensors, 6,329,376,768 bytes, 4 buckets; verifier PASS; H200 job `19d780454524`, `rc=0` |
| `Qwen2.5-7B-Instruct-GPTQ-Int4` | day0-kit NCCL, Marlin | **PASS** | `day0-qwen25-gptq-marlin-real-nccl-rerun.json`; Marlin kernel selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 927 tensors, 5,575,277,568 bytes, 9 buckets; verifier PASS; H200 job `49320ed2015d`, `rc=0` |
| `DeepSeek-V3-FP8-2layer` | direct vLLM RPC reload | PASS (non-day0) | H200 `rc=0`; reload loaded the checkpoint and second generation completed |
| `DeepSeek-V3-FP8-2layer` | day0-kit NCCL | **PASS** | `day0-deepseek-v3-real-nccl-rerun.json`; DeepGEMM FP8, FLASH_ATTN MLA, and TRITON FP8 MoE selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 1,581 tensors, 15,802,320,320 bytes, 5 buckets; verifier PASS; H200 job `26882e7afa56`, `rc=0` |
| `DeepSeek-V4-Flash-FP8-2layer` | day0-kit NCCL | **PASS** | `day0-deepseek-v4-real-nccl-rerun2.json`; DeepGEMM FP8, fp8_ds_mla KV cache, and MARLIN MXFP4 MoE selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 4,711 tensors, 12,844,479,536 bytes, 7 buckets; verifier PASS; H200 job `162b2ff0ae52`, `rc=0` |
| `Qwen3.8-27B-FP8-2layer` | day0-kit NCCL | **PASS** | `day0-qwen38-real-nccl-rerun.json`; DeepGEMM FP8 kernel selected; NCCL 2.28.9 rank 0 / GPU 1 / nranks 2 Init COMPLETE; 398 tensors, 7,251,989,472 bytes, 7 buckets; verifier PASS; H200 job `891707d27c82`, `rc=0` |
| `Llama-3.2-1B-Instruct` → zero-`model.norm.weight` A/B | day0-kit NCCL, packed | **PASS** | `day0-llama32-bf16-nccl-enabled-ab-ab-compare.json`; all five repeat/difference/cold-warm checks pass. `day0-llama32-bf16-nccl-enabled-ab-update.json`: 146 tensors, 2,471,628,800 bytes, epoch/version 1; NCCL 2.28.9; H200 job `93794cf12aff`, `rc=0` |
| `Llama-3.2-1B-Instruct` → zero-layer-0-`v_proj` A/B with prefix cache | day0-kit NCCL, packed | **PASS** | `day0-v12-prefix-cache-fixed-ab-compare.json`; no manual reset endpoint; cold-A differs from warm-B and warm-B exactly equals cold-B for token IDs and completion logprobs. 146 tensors, 2,471,628,800 bytes, 5 buckets; H200 job `ad8abb41a011`, `rc=0` |

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
The original fifteen-case audit checked HTTP lifecycle and accounting, but it
predated `59bdf7cf92`. Thirteen of its logs have no NCCL initialization marker;
the remaining Qwen3.8 and TP=4 Step NCCL markers
cannot distinguish weight transfer from runtime process-group traffic. Those
rows are therefore historical control-plane evidence pending V11 reruns.

The accepted V10 run fixes the data-plane gap. A two-GPU packed NCCL integration
test passes with the fixed environment's NCCL 2.28.9. The model stage probe then
shows the source and receiver embedding range match after bucket 0, while the B
norm is zero before finish and tied embedding/lm_head storage is restored after
finish. The full fixed-token comparison reports all checks true: each oracle is
repeatable, cold-A differs from warm-B, and warm-B exactly equals independent
cold-B for token IDs, completion logprobs, and prompt logprobs.

The day0-kit verifier at commit `67a92f0` (diagnostic follow-up `915a3c8`)
requires both control-plane lifecycle/accounting and a completed publisher-side
NCCL rank-0 communicator with the expected transfer world size. It optionally
requires an A/B comparison result. In the fixed environment its tests pass and
it accepts the new Qwen3-8B and Llama A/B results; a synthetic legacy result
with only `send_weights_completed=true` is rejected.

The V12 prefix-cache oracle uses a 36-token prompt without prompt logprobs,
because prompt-logprob requests currently bypass prefix-cache queries. On the
pre-fix code, zeroing layer 0 `v_proj` left warm-B on A's cached output while an
independent cold-B produced a different token sequence; metrics recorded 96
cumulative hit tokens after the two warm-B requests. Commit `730eab3782`
invalidates local and connector prefix state, renderer/worker multimodal state,
and encoder state before publishing the new weight version. The same run then
produced warm-B == cold-B, with 64 cumulative hit tokens: the first warm-B
request rebuilt B's cache and the repeat hit it. Four synchronous/asynchronous
ordering and invalidation-failure tests pass in the fixed environment.

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
