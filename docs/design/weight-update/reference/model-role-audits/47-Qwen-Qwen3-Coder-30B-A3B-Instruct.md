# 模型角色审计 #47 — Qwen/Qwen3-Coder-30B-A3B-Instruct

基本信息
- 审计 HEAD: c7ce03bcbd
- 架构: `Qwen3MoeForCausalLM` → `qwen3_moe.py`(registry.py:198)。纯 MoE Transformer(无 GDN),128 专家 top-8,无 shared expert、无 e_score_correction_bias。
- **本报告覆盖两条路径**:① bf16 主仓库(UnquantizedFusedMoEMethod);② 官方 FP8 变体 `Qwen3-Coder-30B-A3B-Instruct-FP8`(128×128 block FP8 → `Fp8Config`/`Fp8MoEMethod`,fp8.py:95/492 + `Fp8LinearMethod`,fp8.py:267)。
- MoE 后端:bf16 → Blackwell FLASHINFER_TRTLLM/CUTLASS、否则 Triton(oracle/unquantized.py:69-93);FP8 block → Blackwell FLASHINFER_TRTLLM(oracle/fp8.py:110)、Hopper TP=TRITON / EP=FLASHINFER_CUTLASS(oracle/fp8.py:112-122)、DeepGEMM 可选。

## 状态角色清单

共通(两条路径一致):

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens / lm_head / 各 RMSNorm / qkv・o_proj | qwen3_moe.py:466-477,577-585,293-334,404-407 | R1 | P3+P4 | RESTORABLE | 低 |
| router `gate`(ReplicatedLinear;传入 quant_config,FP8 ckpt 通常 ignore router → 实际 bf16) | qwen3_moe.py:172-178 | R1 | P3+P4 | RESTORABLE | 低 |
| rotary `cos_sin_cache`(non-persistent buffer,`_ROPE_DICT` R6 共享) | rotary_embedding/base.py:59-71; __init__.py:30,83,383 | R2+R6 | P2(gpu_worker.py:270-279,310-316) | RECOMPUTE/PRESERVE | 低 |
| `_expert_map` / `expert_mask` / EPLB 路由表 | routed_experts.py:235-245 | R2 | P2 + SKIP_TENSORS(reload/meta.py:25-32) | PRESERVE | 低 |
| MoERunner `_combined_gate_weight`(`if None` 守卫永不失效) | moe_runner.py:275,332-344 | R3c 锁死隐患 | 无 | RECOMPUTE | 本模型不触发(无 shared_expert_gate) |
| workspace / permute 缓存 | modular_kernel.py:1075-1112; oracle/unquantized.py:338 | R4 scratch | 集中管理 | SCRATCH | 无 |

bf16 路径:

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `w13_weight`/`w2_weight`(bf16) | unquantized_fused_moe_method.py:88-138 | R1 | P3+P4 | RESTORABLE | 低 |
| PWAL shuffle → kernel 布局 | unquantized_fused_moe_method.py:155-177 | R3a | 守卫 `is_weight_update` + `prefer_copy=True` 保 data_ptr(:175-177),layerwise copy-back 双保险 | RECOMPUTE(保地址) | 低 |
| `moe_kernel` 仅首次构建 | unquantized_fused_moe_method.py:184-199 | R3c 守卫(正确) | — | PRESERVE | 低 |
| TrtLlmBf16Experts | trtllm_bf16_moe.py:31-96 | — | 无派生 GPU 常量 | — | 无 |

FP8 变体路径:

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `w13_weight`/`w2_weight`(fp8e4m3)、`w13/w2_weight_scale_inv`(block scale) | fp8.py:576-643 | R1 | P3+P4(restore_metadata=ckpt 格式,model_loader/utils.py:64) | RESTORABLE | 低 |
| PWAL: `_setup_kernel` convert(FI 布局/DeepGEMM 重排/AITER shuffle,oracle/fp8.py:457-537)+ `replace_parameter` **无 prefer_copy** | fp8.py:685-701,720-763 | R3a | layerwise `_copy_and_restore_kernel_tensors` 回写原 storage(layerwise.py:445-461)→ graph 指针安全 | RECOMPUTE(建议补 prefer_copy 对齐 bf16 样板) | 低-中 |
| **`self.moe_kernel = make_fp8_moe_kernel(...)` 每次 PWAL 无条件重建** | fp8.py:711-718 | R3c 反面(无守卫) | 无;每次 RL reload 重建 experts + prepare_finalize | RECOMPUTE→需加守卫 | **中**(TP 单机低;EP/all2all 下通信 buffer 重建的孤儿面,需部署验证) |
| TrtLlmFp8 block 路径 kernel scale 引用 = layer 参数(`quant_config.w1_scale/w2_scale`) | trtllm_fp8_moe.py:240,245,412,417 | R5→参数引用 | P3 copy-back 保地址 | RESTORABLE | 低 |
| TrtLlmFp8 `gemm1_alpha/beta/clamp`(MXFP8+SwiGLU-OAI 专用)、per-tensor 路径 `_g1_alphas/_g2_alphas/_g1_scale_c` | trtllm_fp8_moe.py:62-89,286-292 | **R5 裸属性未参数化**(已知 rebuild-orphan 模式,未修复;对比 NVFP4 已修复:trtllm_nvfp4_moe.py:121-170) | 无 | 应参数化→RESTORABLE | **block+silu(本模型)= None/不走 → 不触发;per-tensor FP8 ckpt = HIGH** |
| 非 MoE FP8 线性:block 量化权重 + `weight_scale_inv`;kernel 于 create_weights 构建(fp8.py:387-396),PWAL 走 kernel 内 replace_parameter(kernels/linear/base.py:249-271) | fp8.py:322-444 | R1/R3a | P3+P4 | RESTORABLE | 低 |
| per-tensor FP8 线性 requant(w13 max-scale 合并;仅非 block ckpt) | fp8.py:416-437,742-758 | R3a(PWAL 内一次性) | reload 先恢复 ckpt 值再重跑 → 幂等成立 | RECOMPUTE | 低(本模型 block 量化不走) |
| Attention k/v_scale | layerwise.py:360-383 `_reload_attention_scales`(create_weights 哨兵重建 + PWAL) | R1/R3 | P3+P4(attention 延后 finalize,layerwise.py:309-312) | RESTORABLE | 低 |

## 特殊发现

1. **同一模型文件、两种量化 = 两种生命周期成熟度**。bf16 路径是守卫+prefer_copy 的金标准(unquantized_fused_moe_method.py:168-199,注释明确为 RL weight update 设计);FP8 路径每次 reload 无条件重建 kernel、重注册参数(fp8.py:698-718)。当前不炸的原因完全依赖 layerwise copy-back 兜底(layerwise.py:423-426 注释 "preserves cudagraph refs")——即 FP8 的 graph 安全性由 reload 框架而非量化方法自身保证。任何绕过 layerwise 框架的权重通道(如 IPC 引擎直写参数后手动触发 PWAL)在 FP8 下会立即产生新参数对象/新 kernel,graph 指针失效。
2. **per-tensor FP8 的 `_g1_alphas/_g2_alphas/_g1_scale_c`**(trtllm_fp8_moe.py:286-292)是 prompt 所述已知 HIGH 模式在 FP8 侧的现存实例:experts 对象构造时派生、裸属性、随 kernel 重建被丢弃;sleep-L2 将旧张量(位于 weights pool)清零而 graph 仍指向旧地址。本模型官方 FP8 为 block 量化不触发,但同文件同类模型(如任何 per-tensor FP8 Qwen 微调 ckpt)会踩中。修复样板即隔壁 NVFP4 的参数化方案。
3. `Qwen3MoeModel.load_weights` 的 `ignore_unexpected_suffixes` 含 `.weight_scale/_weight_scale/.input_scale/...`(qwen3_moe.py:526-538):bf16 引擎吃 FP8 ckpt 不报错、静默丢 scale。RL 权重下发链路必须自行校验发送端/接收端量化格式一致。
4. FP8 MoE 的 `get_fused_moe_quant_config`(fp8.py:774-803)在每次 PWAL 重建 quant_config;block 路径其成员全为参数引用故无孤儿,与 NVFP4 的 `1.0/a13_scale` 派生张量(HIGH,见 #09)形成对照——FP8 block 是 R5 面最小的量化路径。

## 结论

- bf16 主路径:sleep-L2 + RL reload **安全**(与 #27/#37 同级),声明:参数 RESTORABLE、expert map PRESERVE、shuffle RECOMPUTE(保地址)、workspace SCRATCH。
- FP8 变体:**参数值安全但结构脆弱**——安全性寄生于 layerwise copy-back;`moe_kernel` 无条件重建应加守卫(对齐 bf16 样板),`replace_parameter` 应补 `prefer_copy=True`;per-tensor FP8 ckpt 形态下存在现役 R5 HIGH(`_g1_alphas` 族)。
- 权重生命周期重设计的最小改动集:fp8.py:698-701 加 prefer_copy;fp8.py:711 加 `if self.moe_kernel is None` 守卫(或显式声明重建后旧 experts 张量的 copy-back);trtllm_fp8_moe.py:286-292 参数化。
