# 模型角色审计 #12 — openai/gpt-oss-20b

基本信息
- 审计基线: vLLM main @ c7ce03bcbd(**注意:K3 debug worktree 中已修复的 permute-cache / gemm1_* graph-orphan 修复不在本 checkout,以下按 main 现状审计**)
- 模型实现: `vllm/model_executor/models/gpt_oss.py`(1249 行)
- 量化路径: 官方 checkpoint 原生 mxfp4 → `GptOssMxfp4Config` / `GptOssMxfp4MoEMethod`(`vllm/model_executor/layers/quantization/mxfp4.py`)
- 架构要点: attention sinks(nn.Parameter)、每偶数层 sliding window、YaRN RoPE(float32)、MoE 32 experts / top-4、swigluoai 激活(alpha=1.702, limit=7.0)
- 官方审计草稿: 无(本文为独立审计)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `embedding` / `lm_head` / `norm` / 各层 RMSNorm / qkv_proj / o_proj / router | gpt_oss.py:309-322, 119-135, 205-212 | R1 | P3 copy-back + P4 | RESTORABLE | 无 |
| `sinks`(nn.Parameter,每层,按 TP 头切分) | gpt_oss.py:111-113;加载 gpt_oss.py:542-548(`param.data.copy_` 原位) | **R1 checkpoint 参数**(不是 buffer) | P3 copy-back;flash_attn backend 持引用(flash_attn.py:783)原位写即同步 | RESTORABLE | 无(FA3 路径);flashinfer 路径见特殊发现 4 |
| YaRN rotary `cos_sin_cache`(dtype=float32) | gpt_oss.py:91-107(get_rope);rotary_embedding/base.py:59-63 `register_buffer(persistent=False)` | R2 config 派生 buffer | P2 `_sleep_saved_buffers`(gpu_worker.py:270-275, 311-316) | RESTORABLE | 无 |
| `_ROPE_DICT` 全局 rope 实例缓存 | rotary_embedding/__init__.py:30, 83-84, 383 | R6 全局缓存 | 无(进程级,跨 model 实例存活) | 需 RECOMPUTE 或随 P2 覆盖(实例被各层引用,其 buffer 在 named_buffers 中) | 低 |
| MoE `w13_weight` / `w2_weight` / `w13/w2_weight_scale` / `w13/w2_bias` | mxfp4.py:203-275(create_weights) | R1(uint8 打包 + uint8 scale) | P3 + P4;**但 PWAL 后被替换/删除,见 R3c 行** | RESTORABLE(需先恢复 checkpoint 形态) | 中(依赖 reload 机制正确处理形态切换) |
| **`GptOssMxfp4MoEMethod._cache_permute_indices`(dict[Size, CUDA Tensor],每层一个 method 实例各持一份)** | mxfp4.py:145;消费于 oracle/mxfp4.py:756-870;memoize 实现 flashinfer core.py:170-190 | **R4 有状态转换缓存**(非 scratch:按 shape 作 key、永不失效、跨 sleep/reload 存活) | **无**。不在 named_buffers(P2 不覆盖)、不在 named_parameters(P3/P4 不重写) | 应声明 SCRATCH/RECOMPUTE(每次 PWAL 后清空) | **高 — 已知 root-cause-1 模式,main 未修复**:cache 张量在首次 PWAL(weights 池上下文)分配,sleep-L2 丢页 wake 后归零;reload 重跑 PWAL 时按相同 shape 命中 cache,用全零 permute 索引 shuffle 权重 → 权重全错(TRTLLM backend,SM100/SM120) |
| **`TrtLlmMxfp4ExpertsBase.gemm1_alpha` / `gemm1_beta` / `gemm1_clamp_limit`(CUDA float32 张量,由常量 1.702/1.0/7.0 物化)** | experts/trtllm_mxfp4_moe.py:55-80;常量注入 mxfp4.py:410-420(get_fused_moe_quant_config: gemm1_alpha=1.702, gemm1_beta=1.0, swiglu_limit=7.0) | **R5 graph 捕获 kernel 常量** | **无**。挂在 experts 对象(非 module),P2/P3/P4 均不触及;PWAL 重跑会 new 一个 experts 对象和新张量(mxfp4.py:369-378),但**已捕获的 CUDA graph 仍引用旧地址** | 应声明 PRESERVE(池外分配)或 RECOMPUTE+重捕获 | **高 — 已知 root-cause-2 模式(graph-orphan),main 未修复**:旧张量在池内被归零,graph replay 读到 alpha=beta=clamp=0 → swiglu 激活错误输出 |
| kernel 对象 `self.moe_kernel` / `moe_quant_config` | mxfp4.py:146, 367-378 | R3b/R5 权重派生 kernel 状态 | P4 重跑 `process_weights_after_loading`(mxfp4.py:380-391)无条件重建,无 guard | RECOMPUTE | 中(重建本身可行,但见上 graph-orphan;且重建以旧 layer 张量形态为输入) |
| TRITON backend 下的 `layer.w13_weight`/`w2_weight`(triton wrapped tensor,**非 Parameter**)与 `self.w13/w2_precision_config` | mxfp4.py:346-355;oracle/mxfp4.py:1136-1174(`del layer.w13_weight ... del layer.w2_weight_scale` 后以普通属性赋回) | R3a 权重派生(swizzle 后) + method-held 状态 | **无**:PWAL 后 w13/w2 从 named_parameters 消失,scale Parameter 被 del;P2/P3 均不见它们 | RESTORABLE(须由 reload 先重建 checkpoint 形态参数再重跑 PWAL) | **高(R3c 幂等锁死)**:_setup_kernel 的 shape 断言(mxfp4.py:293-315)要求 checkpoint 形态;首跑后 TRTLLM 路径 scale 已变为 float8 interleave(oracle/mxfp4.py:872-885)、TRITON 路径参数已被删除——直接重跑 PWAL 必炸或静默错 |
| Attention `_q/_k/_v/_prob_scale` buffers | attention/attention.py:127-130, 184 | R2 | P2 | RESTORABLE | 无 |
| sliding window(每偶数层) | gpt_oss.py:157 | config 派生标量,无张量 | N/A | N/A | 无 |

## 特殊发现

1. **root-cause-1 现场确认(main 未修复)**:`_cache_permute_indices` 在 mxfp4.py:145 定义为 method 实例属性;`get_quant_method`(mxfp4.py:88-89)对每个 `RoutedExperts` 层 new 一个 `GptOssMxfp4MoEMethod`,故每层一份 dict,全部随 quant method 跨 sleep/reload 存活。flashinfer 的 `get_w2_permute_indices_with_cache`(site-packages flashinfer/fused_moe/core.py:170-190)注释明言 "Memoize permute indices as recompute is **very** costly",按 `("w2", shape)` 作 key、`.to(dst.device)` 后缓存 CUDA 张量、无任何失效逻辑。K3 debug worktree 的修复不在本 checkout。
2. **root-cause-2 现场确认(main 未修复)**:`TrtLlmMxfp4ExpertsBase.__init__`(trtllm_mxfp4_moe.py:54-80)在 `torch.accelerator.current_device_index()` 上以 `torch.tensor([...] * local_num_experts)` 物化 gemm1_alpha/beta/clamp_limit,并在 monolithic/modular 两条 apply 路径中作为 kernel 入参(trtllm_mxfp4_moe.py:195-197, 308-310)——CUDA graph 捕获期间地址被固化。
3. **PWAL 非幂等(R3c)**:`process_weights_after_loading`(mxfp4.py:380-391)对已转换的 layer 再跑一次会失败:TRTLLM 路径 scale 已 `view(float8_e4m3fn)` + interleave;TRITON 路径 `del layer.w13_weight / w2_weight / w13_weight_scale / w2_weight_scale`(oracle/mxfp4.py:1162-1168)。reload 机制必须先恢复 create_weights 形态(uint8 checkpoint 打包)再重放 load + PWAL。
4. **正面对照——flashinfer attention sinks 已按目标范式修复**:flashinfer.py:1575-1585 保存 `_sinks_source`(原 Parameter 引用,注释 "Keep the source so RL weight updates can refresh the runtime tensor"),并在 backend 级 `process_weights_after_loading`(flashinfer.py:1638-1647)优先 `self.sinks.copy_(source_sinks)` 原位刷新 float32 副本,避免 graph-orphan。这正是 R5 状态应有的处理模板,可作为 mxfp4 gemm1_* 修复的院内先例。
5. mxfp4 加载路径特有:`load_weights` 按 `quant_method` 三分支(gpt_oss.py:1138-1167);`"mxfp4"→"gpt_oss_mxfp4"` 规范化存在三处(gpt_oss.py:1126-1139 注释自述)。`sinks` 加载用裸 `param.data.copy_`(gpt_oss.py:546)而非 weight_loader。
6. 非 EP 时 `w2_bias` 在 tp_rank!=0 直接 `weight.zero_()`(gpt_oss.py:534-536)——**mutate 的是加载迭代器交付的源张量**;若 reload 权重来源是共享 host 缓冲(IPC/NCCL 广播张量),会污染源数据,重构时应改为 clone-then-zero。

## 结论

gpt-oss-20b 是本批模型中唯一携带活跃已知 bug 模式的模型:**main 现状同时具备 root-cause-1(R4 `_cache_permute_indices` 有状态缓存,mxfp4.py:145)与 root-cause-2(R5 `gemm1_alpha/beta/clamp_limit` graph-orphan,trtllm_mxfp4_moe.py:55-80)两个未修复实例**,在 SM100+FlashInfer TRTLLM backend 下 sleep-L2 + reload 必然产生静默数值错误;TRITON backend 下另有 R3c 幂等锁死(参数被 del/替换为非 Parameter)。新权重生命周期必须为该模型提供:R4 缓存的 PWAL 前失效钩子、R5 常量的池外分配或原位刷新(参照 flashinfer sinks 模板)、以及 checkpoint 形态参数的可恢复快照。
