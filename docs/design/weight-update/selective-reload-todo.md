# Selective Reload TODO

目标：把当前 PWAL 中适合 weight update 的逻辑拆到
`refresh_derived_state()`，使其满足：

- 原地：只使用已有 storage 的 `copy_()`，或已记录原始 shape 的
  `resize_()` + `copy_()`；不能替换 graph-visible tensor。
- 幂等：连续调用任意次数结果相同；禁止依赖
  `_already_called_process_weights_after_loading`，禁止在已融合结果上重复
  `mul_`/`div_`。
- 可增量：只在依赖参数实际更新时计算；不需要派生状态的模块直接返回。

审计基线：当前 `67033bc5c1`，共 195 个
`process_weights_after_loading` 定义（约 145 个包含实际逻辑）。

当前实现状态（`selective-reload-first-five`）：批次 0 的契约、调度骨架和基础测试已落地；
批次 1 已覆盖 HpcRopeNorm、attention sinks，以及标准 per-tensor KV scale，
compressed-tensors per-head 路径仍显式 fallback；批次 2 已覆盖
TrtLlmNvFp4ExpertsBase、Cutlass、FlashInfer CuteDSL（含 batched）和 FlashInfer NVFP4，
含 batched CuteDSL/B12x；无安全后处理状态的 backend 明确 fallback；批次 3 已覆盖 BF16 MLA、DCP 和 AITER
MXFP4/FP8 和 Kimi K3，AMX 明确 fallback；批次 4 已完成 mHC 广播状态刷新，含
Mega-MoE 的 DeepSeek V4 显式 fallback。以下勾选项表示已验证的子路径，
未勾选项仍是前五批次的待办，不将 H200 NCCL 传输成功误认为 selective
reload 全量完成。

## 验证批次 V0：day0 NCCL 模型矩阵和运行基础设施

- [x] 固定唯一 vLLM Python、源码 checkout、day0-kit 和模型目录。
- [x] 在 H200 容器内构建并导入主扩展、MoE 扩展和 FlashMLA 扩展。
- [x] Qwen2.5-7B、Qwen3.8-27B-FP8、Qwen3-30B-A3B-FP8、
  DeepSeek-V3-FP8 完成 start/update/finish，且
  `send_weights_completed=true`。
- [x] 为 `ai4qz` 增加 `run --detach`，验证 terminal 删除后任务、日志和退出码
  仍可用。
- [x] day0 publisher 支持 DeepSeek V4 checkpoint 的 `F8_E8M0` scale tensor。
- [x] layerwise transaction 中延迟 DeepSeek V4 模型级 post-load，避免 partial
  bucket 对 meta tensor 做派生计算。
- [x] DeepSeek V4 完整通过 NCCL reload，并保存结果 JSON 与 H200 `rc=0`。

本批只验证传输、loader 生命周期和当前已 opt-in 的派生状态；不替代下列 backend
审计与最终 CUDA graph replay 验收。

## 验证批次 V1：低成本 dense checkpoint 扩展

- [x] `Qwen3-8B-2layer` 完成 day0 NCCL reload，并保存事务日志与结果 JSON。
- [x] 完整 `Qwen2.5-7B-Instruct` 完成 day0 NCCL reload，并保存事务日志与结果
  JSON。
- [x] 使用统一 verifier 核对 backend、start/update/finish、bucket 汇总、epoch、
  update version、`send_weights_completed` 和 H200 `rc=0`。

本批不包含 Qwen3-30B BF16、Step-3.7-Flash、Kimi-K2、GR00T 或 MOSS；这些模型
需要先分别确定显存并行方案或专用 multimodal runner，再作为后续独立批次落地。

## 验证批次 V2：完整 BF16 MoE checkpoint

- [x] 完整 `Qwen3-30B-A3B` 使用独立 server/publisher GPU 完成 day0 NCCL
  reload。
- [x] 将 server GPU memory utilization 提高到足以容纳 57G checkpoint，同时
  保持 publisher 固定在 GPU 1，禁止 NCCL duplicate GPU。
- [x] 将结果加入统一 verifier，核对完整事务、bucket 汇总和 H200 `rc=0`。

Step-3.7-Flash（376G）和 Kimi-K2（959G）需要不同的多卡 server 方案；当前 8 卡
节点还必须为 publisher 保留一张独立 GPU。Step 可用 TP=4 继续验证，Kimi 即使
TP=7 也没有安全显存余量。它们不与本批混跑。

## 验证批次 V3：Step-3.7 多卡 server

- [x] runner 支持配置 TP size、publisher device 和 server health 等待时间。
- [x] `Step-3.7-Flash` 使用 TP=4（GPU 0–3）启动，publisher 独占 GPU 4。
- [x] 完成 day0 NCCL reload，并核对 4 个 inference ranks 的传输事务、结果 JSON
  与 H200 `rc=0`。

Kimi-K2 继续留在后续资源批次：959G 权重用 TP=7 后每卡原始权重约 137G，尚未计入
runtime、KV cache 和临时 buffer，无法在 143,771 MiB H200 上安全启动。

## 验证批次 V4：Kimi-K2 FP8 reduced checkpoint

- [x] 从完整 `Kimi-K2-Instruct-0905` 构造两层 checkpoint；只保留全局参数和
  layer 0–1 的 index 项，并用硬链接复用对应的 3 个 shard。
- [x] 使用 TP=1 在 GPU 0 启动 `DeepseekV3ForCausalLM` server，publisher 独占
  GPU 1。
- [x] 完成 day0 NCCL reload，将结果加入统一 verifier，并核对事务、bucket 汇总
  与 H200 `rc=0`。

两层 checkpoint 含 2,351 个 tensor、22,261,573,056 bytes，用于覆盖 Kimi-K2
的 FP8 block-scale loader。它不替代完整 959G checkpoint 验证；后者仍需额外节点或
支持独立 publisher 的资源布局。

## 验证批次 V5：Llama BF16 dense checkpoint

- [x] 从 ModelScope 下载完整 `Llama-3.2-1B-Instruct` BF16 checkpoint，并核对
  config、safetensors header 和实际 tensor 字节数。
- [x] 使用 TP=1 在 GPU 0 启动 `LlamaForCausalLM` server，publisher 独占
  GPU 1。
- [x] 完成 day0 NCCL reload，将结果加入统一 verifier，并核对事务、bucket 汇总
  与 H200 `rc=0`。

本批补齐最终验收矩阵中的 Llama BF16 架构，保持 checkpoint 完整，不做减层或
量化格式替换；finish 后额外 health check 必须返回 200。

## 验证批次 V6：Mixtral BF16 reduced checkpoint

- [x] 从 ModelScope `Mixtral-8x7B-Instruct-v0.1` 构造两层 BF16 checkpoint；
  下载全局参数与 layer 0–1 所需的 3 个 shard，并按 safetensors header 重建
  index metadata。
- [x] 使用 TP=1 在 GPU 0 启动 `MixtralForCausalLM` server，publisher 独占
  GPU 1。
- [x] 完成 day0 NCCL reload，将结果加入统一 verifier，并核对事务、bucket 汇总、
  post-finish health 与 H200 `rc=0`。

本批覆盖最终验收矩阵的 Mixtral BF16 MoE loader。两层 checkpoint 保留每层全部
8 个 expert；它验证架构和 loader 生命周期，不替代完整 93,405,585,408-byte
checkpoint 的容量测试。

## 验证批次 V7：GPTQ Marlin INT4 checkpoint

- [x] 从 ModelScope 下载完整 `Qwen2.5-7B-Instruct-GPTQ-Int4` checkpoint，按
  repository SHA-256 核对 2 个 safetensors shard。
- [x] 确认 vLLM 选择 `AutoGPTQLinearMethod`/Marlin kernel；使用 TP=1 在
  GPU 0 启动 server，publisher 独占 GPU 1。
- [x] 完成 day0 NCCL reload，将结果加入统一 verifier，并核对事务、bucket 汇总、
  post-finish health 与 H200 `rc=0`。

该 checkpoint 是 4-bit、group size 128、symmetric、`desc_act=false`，满足当前
AutoGPTQ Marlin 支持条件，用于补齐最终验收矩阵的 Marlin/INT4 路径。

## 验证批次 V8：online block FP8 dense checkpoint

- [x] 使用完整 `Llama-3.2-1B-Instruct` BF16 checkpoint 和
  `--quantization fp8_per_block`，确认选择在线 128x128 block FP8 linear 路径。
- [x] 使用 TP=1 在 GPU 0 启动 server，publisher 独占 GPU 1；完成 day0 NCCL
  reload，并核对事务、bucket 汇总、post-finish health 与 H200 `rc=0`。
- [x] 将结果加入统一 verifier，并要求 server log 中存在
  `Fp8PerBlockOnlineLinearMethod` 对应 kernel 选择证据。

本批验证 BF16 checkpoint 在 reload 时恢复 checkpoint layout、重新执行在线 block
量化，并将结果复制回已有 runtime storage；不以 serialized block-FP8 checkpoint
替代在线量化路径。

## 验证批次 V9：online fused per-tensor FP8 MoE checkpoint

- [x] 使用 `Mixtral-8x7B-Instruct-v0.1-2layer` BF16 checkpoint 和
  `--quantization fp8_per_tensor`，确认 dense linear 与 fused MoE 均进入在线
  per-tensor FP8 路径。
- [x] 使用 TP=1 在 GPU 0 启动 server，publisher 独占 GPU 1；完成 day0 NCCL
  reload，并核对事务、bucket 汇总、post-finish health 与 H200 `rc=0`。
- [x] 将结果加入统一 verifier，并要求 server log 中存在在线 per-tensor linear 和
  MoE backend 的选择证据。

本批专门覆盖 fused w1/w3 依赖同时到齐后重新量化的 MoE 路径；两层 checkpoint
保留每层全部 8 个 expert，可验证融合参数的加载与完成 accounting。

## 验证批次 V10：Llama BF16 A/B 输出对照

- [x] 从完整 `Llama-3.2-1B-Instruct` 构造结构相同的 B checkpoint，仅将
  `model.norm.weight` 置零；保留 A checkpoint 不变并记录两个 manifest hash。
- [x] 在同一 server 上记录 cold-A oracle，使用 day0-kit 将 B checkpoint reload
  后记录 warm-B oracle，要求 A/B 结果不同且重复请求各自稳定。
- [x] 独立 cold-start B server 并记录 cold-B oracle，要求 warm-B 与 cold-B 的
  token IDs、logprobs 和 prompt logprobs 一致。
- [x] 核对 B 更新的 start/update/finish、post-finish health、epoch/version、完整
  tensor/byte accounting、detached H200 `rc=0`，并加入统一 verifier。

完成证据（2026-09-04）：固定环境使用 NCCL 2.28.9，但 vLLM 把 2.29.7 才有的
`ncclCommSuspend`/`ncclCommResume` 当作 CUDA 必需符号；TCPStore 初始化捕获该
异常后把 communicator 静默标记为 disabled，造成控制面全 200、数据面实际未发送。
commit `59bdf7cf92` 将 suspend/resume 改为真正的可选符号，并让 weight transfer
对 disabled communicator fail-closed。两张 H200 的真实 packed NCCL 回归测试
通过，阶段探针中 bucket 0 receiver embedding 与 source 完全一致，finish 后
embedding/lm_head 共享原 `data_ptr` 且 norm 全零。完整对照结果
`day0-llama32-bf16-nccl-enabled-ab-ab-compare.json` 为 PASS：cold-A、warm-B、
cold-B 各自重复稳定，A != warm-B，warm-B == cold-B。A/B 的 metadata manifest
hash 均为 `f022632b...`（该 hash 不包含 tensor 内容），实际 safetensors SHA-256
分别为 `1ff795ff...` 和 `0a93a136...`。

本批用明确的权重值变化证明 reload 后推理实际读取了 B 权重；它补强同 checkpoint
协议测试，但仍不替代后续 CUDA graph replay 与不同量化格式的 A/B 对照。

## 验证批次 V11：真实 NCCL 数据面矩阵重验

- [x] 修复旧 NCCL 缺少 suspend/resume 时 communicator 静默 disabled，并为
  weight transfer 增加 fail-closed 检查。
- [x] 使用两张独立 H200 完成真实 packed NCCL tensor 内容回归，并用 Llama A/B
  cold/warm/cold 对照证明模型实际读取更新权重。
- [x] 将 V0–V9 的十五个旧结果降级为历史控制面证据；统一 verifier 必须要求
  fail-closed 修复后的提交、NCCL 初始化证据，并优先使用 A/B 值变化证明数据面。
- [x] 重验 `Qwen3-8B-2layer`：publisher rank 0 / GPU 1 的两 rank NCCL
  communicator 完成初始化，25 tensors / 3,261,113,344 bytes / 4 buckets
  全部发送，post-finish health 与 H200 `rc=0`。
- [x] 重验其余低成本 dense/BF16：Qwen2.5-7B 完成 339 tensors /
  15,231,233,024 bytes / 30 buckets；Mixtral-2layer 完成 65 tensors /
  6,329,376,768 bytes / 4 buckets；两者统一 verifier 和 H200 `rc=0` 均通过。
- [x] 重验格式变换：
    - [x] GPTQ Marlin：Marlin kernel 选择证据、927 tensors /
    5,575,277,568 bytes / 9 buckets、统一 verifier 和 H200 `rc=0` 均通过。
    - [x] online block FP8：DeepGEMM kernel 选择证据、146 tensors /
    2,471,628,800 bytes / 5 buckets、统一 verifier 和 H200 `rc=0` 均通过。
    - [x] online per-tensor FP8 MoE：CUTLASS linear 与 TRITON FP8 MoE 选择证据、
    65 tensors / 6,329,376,768 bytes / 4 buckets、统一 verifier 和 H200
    `rc=0` 均通过。
    - [x] serialized FP8 MoE class evidence：Qwen3-30B-FP8 的运行时 probe 明确
    枚举到 `model.layers.*.mlp.experts` 与 `routed_experts` 的
    `Fp8MoEMethod`（`vllm.model_executor.layers.quantization.fp8`，TRITON，
    block quant），并与 day0 reload 结果关联；不能只用 backend 日志替代类证据。
- [x] 重验大型/多卡模型：
    - [x] Qwen3.8 FP8 两层：DeepGEMM kernel 选择证据、398 tensors /
    7,251,989,472 bytes / 7 buckets、统一 verifier 和 H200 `rc=0` 均通过。
    - [x] Qwen3-30B FP8：DeepGEMM linear 与 TRITON FP8 MoE 选择证据、
    37,491 tensors / 32,444,792,832 bytes / 52 buckets、统一 verifier 和 H200
    `rc=0` 均通过。
    - [x] Qwen3-30B BF16：TRITON unquantized MoE 选择证据、18,867 tensors /
    61,064,245,248 bytes / 54 buckets、统一 verifier 和 H200 `rc=0` 均通过。
    - [x] Step-3.7：TP=4 server、GPU 4 publisher、rank 0 / nranks 5 NCCL，
    1,471 tensors / 402,730,656,512 bytes / 273 buckets、统一 verifier 和 H200
    `rc=0` 均通过。
    - [x] Kimi-K2 reduced：DeepGEMM FP8、FLASH_ATTN MLA 和 TRITON FP8 MoE
    选择证据、2,351 tensors / 22,261,573,056 bytes / 5 buckets、统一 verifier
    和 H200 `rc=0` 均通过。
    - [x] DeepSeek V3 reduced：DeepGEMM FP8、FLASH_ATTN MLA 和 TRITON FP8
    MoE 选择证据、1,581 tensors / 15,802,320,320 bytes / 5 buckets、统一
    verifier 和 H200 `rc=0` 均通过。
    - [x] DeepSeek V4 reduced：DeepGEMM FP8、fp8_ds_mla KV cache 和 MARLIN
    MXFP4 MoE 选择证据、4,711 tensors / 12,844,479,536 bytes / 7 buckets、
    统一 verifier 和 H200 `rc=0` 均通过。

旧结果的 HTTP 事务、bucket accounting 和 H200 `rc=0` 仍可用于控制面回归，但在
`59bdf7cf92` 之前没有证明 weight bytes 经 NCCL 到达 receiver；V11 完成前不得
恢复为 NCCL data-plane PASS。

统一 verifier 已写入 day0-kit commit `67a92f0`（诊断改进 `915a3c8`），会同时
检查 result accounting、HTTP 生命周期、post-finish health、publisher client 的
NCCL 版本和 `rank 0 / nranks N / Init COMPLETE`，并可选要求 A/B comparison PASS。
它已在固定环境对 Qwen3-8B 与 Llama A/B 两份新结果返回 PASS，旧控制面结果因
缺少 NCCL 初始化证据会 fail-closed。

V11 完成证据（2026-09-04）：所有重验项均由 publisher 在独立 GPU 上完成真实
NCCL communicator 初始化，并通过统一 verifier。单卡项 transfer world size 为 2；
Step-3.7 使用 TP=4 server 和 GPU 4 publisher，transfer world size 为 5。所有成功
任务均有 `[h200_ncu] status=ok rc=0`。V11 目标中的旧控制面结果已由新结果取代；
表中仅保留 Qwen2.5 reduced 和 Llama same-checkpoint 两项历史记录，分别由完整
Qwen2.5 重验和 Llama A/B 重验覆盖。

## 验证批次 V12：缓存一致性与事务失败语义

- [x] 使用 Llama A/B 先以 A 权重填充 prefix cache，确认重复请求产生 cache hit；
  不调用 `/reset_prefix_cache` reload B，并将 warm-B 与独立 cold-B 的 token IDs 和
  completion logprobs 精确对照。当前带 prompt logprobs 的请求不会进入 prefix-cache
  查询，因此不能用它验证陈旧 KV。
- [x] 成功 `finish_weight_update` 后统一失效 prefix cache、multimodal cache 和
  encoder cache；缓存清理完成前不得发布新的 `weight_version`。
- [ ] 明确 cache generation 与 weight generation 的提交顺序，禁止新权重复用旧
  权重生成的 KV block；cache invalidation 失败时 transaction 必须 fail-closed。
- [x] 为 START 后已经写入 live storage 的异常定义语义：要么恢复完整旧 generation，
  要么停止 serving，禁止在混合 generation 上继续推理。
- [ ] 拒绝缺失 tensor、重复 tensor name、未声明的 sparse update、bucket 中断和任一
  receiver rank 失败；所有 rank 必须对 generation、tensor accounting 和最终状态达成
  一致：
    - [x] dense NCCL/IPC 拒绝单 bucket 内及跨 bucket 的重复 tensor name。
    - [x] transaction active 或 start/update/finish 任一阶段失败后禁止 serving；只有
      finish 成功或完整 disk reload 恢复后才重新开放。
    - [x] START manifest 声明 generation id、完整 tensor 集合和 sparse policy；
      legacy 空 START 继续兼容。
    - [x] UPDATE 在接收 NCCL payload 前拒绝未声明及重复 tensor name；FINISH 拒绝
      缺失 tensor 以及缺失或不一致的 generation id。
    - [x] 只有 manifest 显式设置 `allow_partial=true` 时才允许集合不完整。
- [x] 多 receiver rank 必须对 generation、received-name accounting 和最终提交结果
      达成一致；注入单 rank 失败并验证其余 rank 不得提交。
- [ ] 为同步 `LLM`、`AsyncLLM` 和 draft-model update 覆盖相同的提交与缓存失效
  契约，并补充失败注入单元测试。

本批先解决成功提交后的陈旧缓存这一可独立验证的问题，再以失败注入固定 transaction
边界。HTTP 200、`send_weights_completed=true` 或 NCCL bytes 到达均不能单独证明
cache coherence 或 rollback 正确。

阶段证据（2026-09-04）：只清零最终 norm 的旧 A/B 变体不改变 prefix KV，不能作为
敏感 oracle。改为清零 `model.layers.0.self_attn.v_proj.weight` 后，未修复版本在
32-token prefix hit 下让 warm-B 继续输出 A 的 `[12366, 13, 128009]`，而 cold-B
输出 `[128009]`，复现陈旧 KV。commit `730eab3782` 在同步和异步 finish 中按
prefix（含 connector）→ multimodal → encoder → version 顺序提交；四个顺序/失败
单测通过。修复后 warm-B 与 cold-B 均为 `[128009]`，五项 A/B 精确检查全部通过；
warm 阶段累计 prefix hits 从未修复的 96 降为 64，证明 reload 后第一次 B 请求未
复用 A cache、第二次 B 请求仍能正常命中新 cache。H200 job `ad8abb41a011` 返回
`[h200_ncu] status=ok rc=0`。

失败语义证据（2026-09-04）：commit `142e770e57` 为 start/update/finish 异常增加
不可自动清除的 worker failure latch，并让完整 `reload_weights` 成为显式恢复路径；
commit `50d1fc8667` 进一步在 transaction active 期间拒绝 execute/sample，覆盖
publisher 丢 bucket 后没有 receiver exception 的情况。dense NCCL/IPC 的 bucket 内
和跨 bucket 重名均在接收数据前拒绝。固定环境共 18 个 worker/metadata 测试通过，
H200 返回 `status=ok rc=0`；真实服务只调用 START 后的 completion 返回 HTTP 500，
`v12-interruption-result.json` 记录 `serving_blocked=true`。完整正常 reload 回归仍通过
146 tensors / 2,471,628,800 bytes / 5 buckets 和统一 verifier，H200 job
`eee342b8d025` 返回 `rc=0`。当时缺失 tensor 和多 rank 一致性仍需 START manifest，
未在该阶段宣称完成；下述 follow-up 已关闭单 receiver 的缺失 tensor 项。

START manifest 阶段证据（2026-09-04）：commit `ff1a001db8` 将可选 manifest 接入
HTTP、Ray、同步 `LLM`、`AsyncLLM` 和 worker，保留 legacy 无参数调用；day0-kit
commit `c0edf22` 为每轮完整 checkpoint update 生成唯一 generation，并在 START、
FINISH 和结果 JSON 中贯穿同一值。固定环境 28 个聚焦测试通过，day0 verifier 的
3 个 generation evidence 测试亦通过。真实 H200 missing-name 与 unknown-name 注入均
使操作请求和后续 completion 返回 HTTP 500，结果分别写入
`v12-manifest-missing-result.json` 和 `v12-manifest-unknown-result.json`，两次任务均为
`status=ok rc=0`。正常 Llama zero-`v_proj` A/B 回归传输 146 tensors、
2,471,628,800 bytes、5 buckets，generation
`day0-b882e9e2184243b29671d9f11f3a9315` 通过增强后的统一 verifier；cold-A 与
warm-B 不同，warm-B 与 cold-B 完全一致，H200 返回 `status=ok rc=0`。多 receiver
rank 分歧仍未验证，因此父项保持未完成。

## 验证批次 V13：storage ownership 与 backend 收口

- [ ] 以 storage/alias group 为更新所有权单位，覆盖 tied embedding/lm_head、共享
  Parameter 和跨 module alias；同一 storage 不得被重复写入或只刷新部分别名。
- [ ] 建立 graph-visible state census，纳入非 Parameter tensor、buffer、量化 workspace
  和 backend descriptor；地址稳定性验收不能只遍历 `named_parameters()`。
- [ ] per-tensor FP8 restore 保留 `ModelWeightParameter` 等 Parameter 子类、weight
  loader、shard metadata 和自定义属性；禁止以普通 `Parameter` 替换后仅恢复 shape。
- [ ] mixed selective/layerwise 模型按实际 module/backend 分区执行，禁止任一 fallback
  触发无关 module 的 model-wide post-load 初始化。
- [ ] `supports_selective_reload()` 基于运行时选中的 backend/kernel method，而非只看
  method class；backend 变化或能力不确定时显式 fallback。
- [ ] online quantization 先完成 checkpoint name 和 TP shard routing，再对本 rank 的
  shard 做 quantize/pack；补齐 fused expert、scale 和 padding 的覆盖。
- [ ] sharded RDT 在 commit 前调用并验证 `drain_pending()`，保证异步传输全部完成且
  错误已传播。
- [ ] 用 CUDA graph capture/replay 验证 reload 前后所有 graph-visible 地址和 kernel
  descriptor 有效，并用不同 A/B 权重证明 replay 读取新 generation。

本批依赖 V12 的 generation/transaction 契约。开始实现前需要对照 RFC 相关工作
`#49459`、`#51378`、`#53438`、`#49789` 和 `#52497`，避免复制已有实现或建立冲突
的更新协议。

## 批次 0：契约和调度骨架

- [x] 在 `QuantizeMethodBase` 增加默认的
  `refresh_derived_state(layer, updated_parameter_names)`，默认 no-op。
- [x] 增加能力声明（建议
  `supports_selective_reload()`），未声明的 quant/backend 必须 fail-closed。
- [x] 明确 finish 阶段的调用顺序和 transaction 边界：所有 checkpoint 权重
  完成后才允许 refresh。
- [x] 定义依赖集合语义：空集合表示完整更新，非空集合只刷新命中的派生状态。
- [x] 保留旧 layerwise 路径作为显式 fallback，不在未验证模块上隐式切换。
- [x] 建立通用测试：连续 refresh、A→B 值变化、`data_ptr()` 稳定性、缺少依赖
  时不刷新。

验收：BF16 dense 小模型完成两次 reload；输出等价、所有 graph-visible
parameter/buffer 地址不变；旧 layerwise 回归通过。

## 批次 1：低风险、纯原地派生状态

优先实现，这些逻辑已经接近目标契约。

- [x] `HpcRopeNorm`：
  `vllm/model_executor/layers/hpc/rope_norm.py:228`。
  将 fallback norm 的 float32 拷贝正式改为 refresh；首次加载预分配
  `qnorm_weight/knorm_weight`，后续只 `copy_()`。
- [x] Attention sinks：
  `vllm/v1/attention/backends/b12x.py:840`、
  `vllm/v1/attention/backends/flashinfer.py:1869`。
  首次加载建立稳定的 FP32 sink storage，后续禁止 `self.sinks =` 替换。
- [x] KV cache scales（标准 per-tensor 路径；compressed per-head 保持 fallback）：
  `vllm/model_executor/layers/quantization/kv_cache.py:74`。
  `_q/_k/_v/_prob_scale` 和 CPU 镜像全部采用原地刷新；默认 scale 的生成
  必须是可重复计算。
- [x] 清理上述路径中与 reload 冲突的重复初始化或重新注册逻辑：selective
  attention scale storage 不再重复调用 `create_weights()`。

验收：每个路径使用不同的 A/B scale 或 sink 值；比较 cold-B 与 warm-reload-B，
同时检查 pointer census。

## 批次 2：MoE 派生 scale/alpha

- [x] `CutlassExpertsFp4`、`FlashInferCuteDSL*Experts`、
  `FlashInferExperts`：把 `weight_scale_2 *= input_scale` 改成从原始 source
  scale 每次重算后 `copy_()`。
- [x] `TrtLlmNvFp4ExpertsBase`：refresh `g1/g2` scale、clamp/beta 和
  `g1_scale_c`；保留原始 scale，禁止 destructive fusion。
- [x] `FlashInferB12xExperts`：refresh MMA-layout scale、`a2_gscale` 等
  graph-visible tensor；refresh 原地更新既有 layout，不重建 experts/kernel。
- [x] `FlashInferCuteDSLBatchedExperts`：从 raw `scale_2` 快照重算 input
  scale 派生值，按依赖原地刷新且不重建 kernel。
- [x] `FusedMoEExperts` 增加默认 no-op refresh，quant method 只负责委派。
- [x] 已验证 opt-in update 路径跳过 `process_weights_after_loading()`，因此不调用
  `make_*kernel()`/`_setup_kernel()`；未 opt-in backend 继续走 fallback。

验收：A/B 使用不同 weight/input scale；warm reload 后旧 graph replay 与 cold-B
一致，派生 tensor 地址和值均正确。

## 批次 3：MLA 派生权重

- [x] `MLAAttention`：从 `kv_b_proj.weight` 刷新 BF16
  `W_UK_T/W_UV`，目标 storage 在初始化时预分配。
- [x] AITER MXFP4/FP8 路径刷新 `W_K/W_V` 及其 scales；refresh 禁止 Triton
  预编译和任何 Parameter 替换。
- [x] DCP 路径按 `kv_b_proj -> W_UK_T -> W_UK_T_dcp_qrep` 顺序刷新，
  `all_gather` 结果复制到稳定 buffer。
- [x] Kimi K3 MLA 同步实现上述 refresh，并刷新 `_q_scale_inv/_k_scale_inv`；
  仅在派生 storage 已预分配时 opt-in。
- [x] AMX MLA 明确标记 fallback：当前会释放 `kv_b_proj.weight`，在保留源
  storage 前不得进入 selective reload。

验收：覆盖 BF16、AITER FP8、AITER MXFP4、DCP；A/B 输出、派生值、地址和
kernel 编译次数均验证。

## 批次 4：DeepSeek V4 模型派生状态

- [x] `finalize_mhc_broadcast_weights()` 改为正式 refresh 接口，使用稳定的
  `hc_attn_fn_broadcast.copy_()`。
- [x] `finalize_mega_moe_weights()` / `_finalize_moe()` 分析其每个子操作，
  只迁移纯派生计算；任何 kernel/config 构造留在 cold-start 或 fallback。
- [x] NVIDIA、AMD、XPU、MTP、DSpark 变体分别确认能力声明，不能由顶层
  `process_weights_after_loading()` 统一假设可 refresh。

审计结论（2026-09）：NVIDIA Mega-MoE 暂不能迁移到 selective refresh。
`DeepseekV4MegaMoEExperts.finalize_weights()` 会调用
`transform_sf_into_required_layout()` 和 `transform_weights_for_mega_moe()`，
为 L1/L2 权重及 scale 分配新的布局；随后将 loader 侧的
`w13_weight`/`w2_weight`/scale 参数置为 `None`。共享专家融合还会替换
`shared_experts.gate_up_proj.weight.data`，并按 EP/world-size、token 上限和
共享专家数创建对称内存及 DeepGEMM runtime。FlashInfer Mega-MoE 另有
`build_fi_mega_layer()` runtime，XPU 实现同样释放源参数，MTP/DSpark 只是
转调 finalize。因此当前没有可复用的原始 storage，也没有 transform 的
稳定 `out` API；更新后无法证明地址稳定、EPLB 重排一致或不重建 runtime。
这些变体必须继续显式 fallback，直到 restore + 预分配布局以及完整 runtime
生命周期协议落地。

验收：mHC 和 Mega-MoE 使用值不同的 A/B 权重，验证派生值刷新、地址稳定和
各平台未重复构建 runtime object。

## 批次 5：线性层格式变换（需 restore + 预分配）

这一批不是简单的 derived state，必须先保存 checkpoint/runtime 两种布局的
storage 语义，再实现 refresh。

- [x] AWQ 原生 GEMM 的 refresh 不再调用 PWAL：该后端直接消费 checkpoint
  布局，无需重新注册 Parameter。固定 vLLM 环境的行为测试在冷启动后禁用
  PWAL，连续两次 reload 检查 qweight/qzeros/scales 值、对象和地址稳定。
- [ ] AWQ 分类①接通逐参数 eager copy，避免沿用整层缓冲；Marlin repack
  独立实现并验证。上述单测不等价于完整 AWQ day0 验收。
- [ ] FP8 per-tensor transpose/requant：`fp8.py`、scaled-mm FP8 实现需要先把
  checkpoint-layout 参数暂存，再由 `refresh_derived_state` 重建 runtime layout；
  当前实验性 opt-in 的 refresh 仍调用 PWAL，不满足目标契约。
  Qwen3-30B-FP8 是 block-FP8，不能证明 per-tensor 覆盖；需要重新实现并验证。
- [x] DeepGEMM、FlashInfer、AITER 的 block-scale/layout shuffle 已审计；
  依赖 runtime/layout 生命周期的路径保留 fallback。
- [x] Marlin、Machete、WNA16、MXFP4/MXFP8/NVFP4 repack 已审计；
  shape/layout 变化或 source release 的路径保留 fallback。
- [x] Humming、AllSpark、Conch、FlyDSL、RDNA3、XPU、CPU VNNI/AMX 专有布局
  已审计；未证明 storage 可复用的路径保留 fallback。
- [x] 在通用 layerwise 初始化中接入
  `restore_weights_before_loading()`；具体 backend 仅在能记录并复用原始
  shape/stride/layout 时才允许后续 opt-in。
- [ ] 落实约束：冷启动允许 replace，refresh 只能写入既有 runtime storage。
  当前实验包装器仍调用 PWAL；需拆出纯转换并接通 per-param copy/staging，
  使用冷启动后将 PWAL 替换为抛异常的测试证明 reload 不再调用它。
- [x] 已明确 shape-changing 与 data_ptr 不可稳定的 backend，继续 fallback。

审计备注：compressed-tensors FP8 路径仍显式 fallback；它包含跨 logical shard
requant、transpose 和 Parameter 重注册。原生 `Fp8LinearMethod` per-tensor 路径
只有实验性包装器，尚未通过无 PWAL 的 per-tensor reload 验收。通用
`restore_weights_before_loading()` 调用点现已接入 layerwise 初始化，但这些
后端仍没有可复用的原始 storage/布局协议，暂不能证明 storage identity。
CutlassBatchedExpertsFp8 无额外后处理状态，BatchedDeepGemmExperts 依赖
DeepGEMM runtime/layout；两者均保持显式 fallback。

验收：`restore -> load -> refresh -> refresh` 循环；TP shard 正确；每次循环
pointer 不变；不得出现 double transpose、double repack 或 stale layout。

## 批次 6：在线量化迁移到 `weight_loader`

- [ ] `online/fp8.py`：per-block/channel/tensor 路径区分单参数和跨参数场景。
- [ ] `online/mxfp4.py`、`online/mxfp8.py`、`online/nvfp4.py`、`online/int8.py`
  的量化逻辑移入永久 loader wrapper。
- [ ] 单参数 block/channel quantization 到达即量化并 `copy_()`。
- [ ] fused per-tensor FP8、NVFP4 w1+w3 使用显式 staging buffer，所有依赖
  到齐后才完成量化。
- [ ] refresh 对这些路径保持 no-op；禁止再次运行在线量化 PWAL。
- [ ] dtype 不匹配且没有 loader 能力时 fail-closed。

验收：BF16→量化格式、格式匹配更新、fused 参数乱序/分块到达均覆盖；检查
完成 accounting、TP routing、值正确和地址稳定。

## 批次 7：最终 engine 集成和收口

- [ ] NCCL/IPC/gpu model runner 统一使用 start(restore) → load → finish(refresh)。
- [ ] 增加 `VLLM_FORCE_LAYERWISE_RELOAD=1` 逃生开关。
- [ ] selective 与 layerwise per-module 混用时，验证不会重复处理同一模块。
- [ ] 建立全量 PWAL 能力清单：selective、fallback、明确不支持三态。
- [ ] 运行 #48312 全部正确性维度：storage identity、derived refresh、loader
  lifecycle、state preservation、routing/sharding、name mapping、cache coherence。
- [ ] 删除已迁移路径中仅用于冷启动二次调用的 `_already_called` workaround。

最终验收：Llama BF16、Mixtral BF16、DeepSeek block FP8、MLA、Marlin/INT4、
在线 block FP8 和 fused per-tensor FP8 的 cold/warm/cold 对照及 CUDA/HIP graph
replay 全部通过。

## 明确不迁移到 refresh

- kernel/config/workspace/routing table 创建或重建
- Triton/AITER 预编译
- 参数注册、删除、`replace_parameter`、`.data = new_tensor`
- `_release_source_parameters`
- 无法证明 storage 可复用的 backend
- 依赖上一次派生结果继续累乘/累加的 destructive 操作
