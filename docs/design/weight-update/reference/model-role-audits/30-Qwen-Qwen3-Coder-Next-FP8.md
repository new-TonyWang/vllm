# 模型角色审计 #30 — Qwen/Qwen3-Coder-Next-FP8

基本信息
- 审计 HEAD: c7ce03bcbd
- 架构: Qwen3-Coder-Next 沿用 `Qwen3NextForCausalLM` → `qwen3_next.py`(registry.py:111)——**混合结构:GDN 线性注意力 + 全注意力交错**(layer_types 驱动,qwen3_next.py:624-633),MoE 带 shared expert + shared_expert_gate(qwen3_next.py:148-195)。
- 量化: FP8(Qwen 官方 FP8 为 128×128 block 量化)→ `Fp8Config`/`Fp8MoEMethod`(fp8.py:95/492,block_quant=True,scale 名 `weight_scale_inv`)+ `Fp8LinearMethod`(fp8.py:267)。
- MoE 后端: Blackwell → `FLASHINFER_TRTLLM`(oracle/fp8.py:110)→ `trtllm_fp8_moe.py`;Hopper → TP 用 TRITON、EP 用 FLASHINFER_CUTLASS(oracle/fp8.py:112-122),DeepGEMM 可选。
- MTP: `Qwen3NextMTP` 官方存在(registry.py:646,qwen3_next_mtp.py)。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens / lm_head / GemmaRMSNorm / layer_scale 参数 | qwen3_next.py:619-641,810-815,468-490 | R1 | P3+P4 | RESTORABLE | 低 |
| 全注意力 qkv_proj(含 output-gate 2x q 布局)/o_proj/q_norm/k_norm | qwen3_next.py:267-320 | R1(FP8 线性) | P3+P4 | RESTORABLE | 低 |
| **GDN**: `conv1d.weight`(ColumnParallelLinear,__init__ 即 `unsqueeze(1)`,qwen_gdn_linear_attn.py:396;自定义 `mamba_v2_sharded_weight_loader` :425-433) | qwen_gdn_linear_attn.py:390-396 | R1 | P3+P4;record_metadata 在 init 后(utils.py:64)故 restore 形状=unsqueeze 后形状,无错位 | RESTORABLE | 低 — **decode_conv1d_weight 式 R3b 派生副本在本实现中不存在**:decode/prefill 全用 `self.conv1d.weight.view(...)` 现场视图(:1245-1246,1519-1520,1591-1592) |
| GDN `A_log` / `dt_bias`(fp32,sharded_weight_loader) | qwen_gdn_linear_attn.py:439-450 | R1 | P3+P4 | RESTORABLE | 低 |
| GDN in_proj_qkvz / in_proj_ba / out_proj / RMSNormGated | qwen_gdn_linear_attn.py:403-419,459-476 | R1 | P3+P4;in_proj_ba 不支持 blockwise fp8 → maybe_disable_tp(:411-420),replicated 参数 reload 由 layerwise.py:417-421 `update_param_tp_status` 重新对齐(近期修复) | RESTORABLE | 低 |
| GDN conv/SSM 递归状态 | mamba state cache(qwen3_next.py:840-875 shape/dtype/copy calculators) | 会话状态(kv_cache 池) | P7 域(post_kv_cache_wake_up,gpu_worker.py:326-329) | SCRATCH(唤醒后必须清零/重建) | 低(属 kv 生命周期) |
| ChunkGatedDeltaRule | qwen_gdn_linear_attn.py:213-338 | — | 无持久张量(纯 kernel 分发) | — | 无 |
| router `gate`(quant_config=None,恒 bf16)/ shared_expert_gate | qwen3_next.py:140-154 | R1 | P3+P4 | RESTORABLE | 低 |
| rotary `cos_sin_cache`(fused_qk_rmsnorm_rope_gate 直接引用,qwen3_next.py:352-364) | rotary_embedding/base.py:59-71 | R2+R6 | P2 | RECOMPUTE/PRESERVE | 低 |
| MoE ckpt 参数: `w13_weight`/`w2_weight`(fp8e4m3)、`w13/w2_weight_scale_inv`(block scale) | fp8.py:576-643 | R1 | P3+P4 | RESTORABLE | 低 |
| PWAL: fnuz 归一(ROCm)、block 路径无 requant;`_setup_kernel` convert(DeepGEMM 重排/AITER shuffle/FI 布局,oracle/fp8.py:457-537)+ `replace_parameter`(**无 prefer_copy**) | fp8.py:720-763,685-701 | R3a | P3 copy-back 兜底回写原 storage(layerwise.py:445-461)| RECOMPUTE(建议补 prefer_copy) | 低-中 |
| **`self.moe_kernel = make_fp8_moe_kernel(...)` 每次 PWAL 无条件重建** | fp8.py:711-718 | R3c 反面(无守卫;对比 unquantized_fused_moe_method.py:175-199 的守卫模式) | 无 | RECOMPUTE→应加守卫 | **中-高**(见特殊发现 1) |
| TrtLlmFp8 block 路径 kernel 引用的 scale = `quant_config.w1_scale/w2_scale`(layer 参数引用) | trtllm_fp8_moe.py:240,245,412,417 | R5→参数引用 | P3 copy-back 保地址 | RESTORABLE | 低 |
| TrtLlmFp8 `gemm1_alpha/beta/clamp_limit`(仅 MXFP8+SwiGLU-OAI)与 per-tensor 路径 `_g1_alphas/_g2_alphas/_g1_scale_c` | trtllm_fp8_moe.py:62-89,286-292 | **R5 裸属性,未参数化** | 无(对比 trtllm_nvfp4_moe.py:121-170 已修复版本) | 应参数化→RESTORABLE | **本模型(block+silu)全为 None/不走此路径 → 不触发**;若换 per-tensor FP8 ckpt 即为 HIGH,记录在案 |
| `_expert_map`/`expert_mask`/EPLB 表 | routed_experts.py:235-245 | R2 | P2 + SKIP_TENSORS(reload/meta.py:25-32) | PRESERVE | 低 |
| MoERunner `_combined_gate_weight` | moe_runner.py:275,332-344 | R3c 锁死 | 无 | RECOMPUTE(reload 后置 None) | 默认 CUDA 不触发(shared_expert 非 None → shared_expert_gate 不传入 FusedMoE,qwen3_next.py:191-194);**ROCm FSE 路径下 reload 后 router 用旧权重 = 静默腐坏** |
| Fp8 线性 kernel(init_fp8_linear_kernel,create_weights 期构建;PWAL 走 kernel.process_weights_after_loading,replace_parameter 模式) | fp8.py:387-396,398-444; kernels/linear/base.py:249-271 | R3a | P3+P4 | RESTORABLE | 低 |
| Attention k_scale/v_scale 等 | reload 专有通道 `_reload_attention_scales`(layerwise.py:360-383,含 create_weights 哨兵重建) | R1/R3 | P3+P4(attention 延后处理,layerwise.py:280-282,309-312) | RESTORABLE | 低 |
| **MTP(Qwen3NextMTP)**: fc、MTP decoder 层、pre_fc_norm×2 | qwen3_next_mtp.py:75-104 | R1(独立 draft 模型) | **无**:主模型 load_weights skip `mtp.`(qwen3_next.py:884);reload_weights 只覆盖主模型(gpu_model_runner.py:5505-5530);P2 只备份主模型 buffers(gpu_worker.py:271-273) | RESTORABLE(需接入) | **高(若启用 MTP)** |

## 特殊发现

1. **Fp8MoEMethod 与 UnquantizedFusedMoEMethod 的不对称**是本模型最大结构性风险:同一 reload 流程下,unquantized 走守卫+prefer_copy(unquantized_fused_moe_method.py:175-199),FP8 却每次重建 kernel、重注册参数(fp8.py:698-718)。参数值因 copy-back 兜底不坏,但每次 reload 都会:重建 prepare_finalize(EP/DP all2all 通信 buffer 重新分配)、丢弃旧 experts 对象。block+silu 下旧对象无 GPU 常量故无孤儿;一旦 prepare_finalize 持有被 graph 捕获的通信 buffer(DeepEP/all2all 场景),就是 fp8 版 rebuild-orphan。建议对齐守卫模式。
2. **GDN 是干净的**:传闻中的 conv1d 派生 decode 权重(R3b)在当前实现不存在,所有 kernel 调用即时 `.view`。GDN 的三个非常规点——init 期 `unsqueeze(1)`、fp32 A_log/dt_bias、自定义 sharded loader——都与 reload 机制兼容(record_metadata 时机 + CopyCounter 对 Mamba `mixer.D` 类尾参数的修正,reload/meta.py:190-199)。
3. **in_proj_ba disable_tp** 与 replicated 参数在 reload 后的 tp_rank 漂移问题已由 layerwise.py:417-421(git log d96aee0951)修复,本模型是该修复的直接受益者,回归测试应包含 hybrid FP8。
4. MTP 缺口同 #09(双重:reload 不触达 + P2 不备份);rotary 经 `_ROPE_DICT` 共享获得意外保护,但 MTP `fc`/decoder 权重 sleep-L2 后为零。
5. Qwen3-Next 家族的 shared expert 融合加载 `maybe_fuse_shared_experts`(qwen3_next.py:716-724)是 loader 侧变换(R3b 输入侧),reload 复用同一 load_weights 路径,幂等。

## 结论

- 常规 TP 部署(CUDA、block-FP8、silu、无 MTP)下,今日 sleep-L2 + RL reload **参数值层面安全**,依赖链为 P2(buffers)+ P3 copy-back + P4 重跑 PWAL。
- 风险排序:① 启用 MTP → draft 模型完全脱保(高);② `moe_kernel`/prepare_finalize 无条件重建在 EP/all2all 下的通信 buffer 孤儿(中-高,需针对部署形态验证);③ ROCm FSE 下 `_combined_gate_weight` 锁死(高,但仅该配置);④ per-tensor FP8 变体的 `_g1_alphas/_g1_scale_c` 裸属性(该 ckpt 形态下 HIGH,本模型 block 量化不触发)。
- 终态声明:全部 ckpt 参数 RESTORABLE;expert map/EPLB PRESERVE;kernel 格式转换与 moe_kernel RECOMPUTE(需加守卫/保地址);GDN 递归状态 SCRATCH;MTP 需要新增 RESTORABLE 通道。
