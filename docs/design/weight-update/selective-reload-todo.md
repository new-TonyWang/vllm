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

- [x] FP8 per-tensor transpose/requant：`fp8.py`、scaled-mm FP8 实现已审计，
  因跨 shard requant、transpose 和 Parameter 替换保留 fallback。
- [x] DeepGEMM、FlashInfer、AITER 的 block-scale/layout shuffle 已审计；
  依赖 runtime/layout 生命周期的路径保留 fallback。
- [x] Marlin、Machete、WNA16、MXFP4/MXFP8/NVFP4 repack 已审计；
  shape/layout 变化或 source release 的路径保留 fallback。
- [x] Humming、AllSpark、Conch、FlyDSL、RDNA3、XPU、CPU VNNI/AMX 专有布局
  已审计；未证明 storage 可复用的路径保留 fallback。
- [x] 在通用 layerwise 初始化中接入
  `restore_weights_before_loading()`；具体 backend 仅在能记录并复用原始
  shape/stride/layout 时才允许后续 opt-in。
- [x] 约束已固化：冷启动允许 replace，refresh 只能写入既有 storage；未能
  提取纯变换函数的 backend 保持 fallback。
- [x] 已明确 shape-changing 与 data_ptr 不可稳定的 backend，继续 fallback。

审计备注：当前 FP8 per-tensor 与 compressed-tensors FP8 路径仍显式 fallback；
它们包含跨 logical shard requant、transpose 和 Parameter 重注册。通用
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
