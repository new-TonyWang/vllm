# 模型角色审计 #50 — deepseek-ai/DeepSeek-V3

基本信息
- 实现文件：与 DeepSeek-R1 完全同一实现——`vllm/model_executor/models/deepseek_v2.py`（`DeepseekV3ForCausalLM(DeepseekV2ForCausalLM)`，deepseek_v2.py:1916-1917）
- MTP draft：`vllm/model_executor/models/deepseek_mtp.py`（官方 `DeepSeekMTP`；V3 checkpoint 自带 1 层 nextn）
- 量化：官方 FP8 block-quantized（同 R1：`Fp8LinearMethod` block 路径 + `Fp8MoEMethod` block_quant；H100 DEEPGEMM/TRITON，B200 DEEPGEMM ue8m0 requant 或 FLASHINFER_TRTLLM，oracle/fp8.py:81-127）
- 注意力：MLA，`topk_method="noaux_tc"` → `e_score_correction_bias` 存在（deepseek_v2.py:314-317）；无 `index_topk` → 无 v3.2 indexer
- 审计基线：fork HEAD c7ce03bcbd

本报告为独立报告；状态清单与 #10（DeepSeek-R1）逐行一致（同一份代码、同一 config 家族），下表全文列出，R1/V3 差异在"特殊发现"注明。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| 各线性层 FP8 block 权重 + `weight_scale_inv`（fused_qkv_a_proj/q_b/kv_b/o_proj/shared_experts/lm_head 等） | deepseek_v2.py:1002-1050, 1829-1836 | R1 | P3 copy-back（layerwise.py:445-461）+ P4 PWAL 重跑 | RESTORABLE | 低 |
| MoE experts `w13_weight/w2_weight` + block scales | fp8.py:642-643, 698-701 | R1 | P3 + P4 | RESTORABLE | 低 |
| `gate.weight` | deepseek_v2.py:308-313 | R1 | P3 | RESTORABLE | 低 |
| **`gate.e_score_correction_bias`（fp32 Param，noaux_tc 必有）** | deepseek_v2.py:315-317 | R1 | SKIP_TENSORS（meta.py:31）→ reload 不 capture/不 wrap，仅当权重流含该键时由原 loader 原地写回；**Param → P2 不覆盖** | RESTORABLE（条件） | **中-高**：sync 子集不含该键 → wake 后永久全零，专家选择分布静默漂移 |
| MLA `W_UK_T`/`W_UV` | mla_attention.py:908-995 | R3a | P4 deferred attention finalize（layerwise.py:281, 343-357）+ `replace_parameter(prefer_copy=True)` 地址保持（utils.py:82-90） | RECOMPUTE | 低（reload 后重算；wake 未 reload 前为零） |
| rotary `cos_sin_cache`（deepseek_yarn，persistent=False） | rotary_embedding/base.py:63; deepseek_scaling_rope.py:248 | R2 | P2（gpu_worker.py:270-316）；reload 不重算 | PRESERVE（P2） | 主模型低 |
| attention `_q/_k/_v/_prob_scale` buffers | attention.py:124-135, 184 | R2/R1 | P2 + PWAL 重置/`_reload_attention_scales`（layerwise.py:360-383） | RESTORABLE | 低 |
| expert maps（`_expert_map/expert_mask/...`，register_buffer） | routed_experts.py:235-245 | R2 | SKIP_TENSORS + P2 | PRESERVE（P2） | 主模型低 |
| FP8 MoE kernel 格式产物（DEEPGEMM ue8m0 / TRTLLM shuffle） | oracle/fp8.py:457-531 | R3a | P4 重跑 + copy-back | RECOMPUTE | 低 |
| `moe_quant_config`/`moe_kernel`（含 TRTLLM R5 语义常量引用） | fp8.py:708-718, 774-803 | R5/R4 | P4 重建 | RECOMPUTE | 低 |
| `_use_min_latency_gemm`（dsv3_fused_a_gemm 开关，7168×2112 bf16 特判） | deepseek_v2.py:922-934 | R2（host） | init 一次 | PRESERVE | 无 |
| kv_cache 页 | — | R7 | P7 | SCRATCH | 无 |

### MTP draft（deepseek_mtp.py）

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| enorm/hnorm/eh_proj/shared_head | deepseek_mtp.py:91-117 | R1 | draft reload（PR #46725） | RESTORABLE | 低 |
| mtp_block（= DeepseekV2DecoderLayer：MLA 派生权重、FP8 MoE、bias） | deepseek_mtp.py:112-117 | 同主模型 | 同主模型 | 同上 | 同上 |
| **draft 全部 registered buffers（cos_sin_cache、attention scales、expert maps）** | gpu_worker.py:272-273 | R2 | **无**（P2 仅主模型；`_build_fused_kv_buffers` 钩子 DeepSeekMTP 不具备，gpu_worker.py:277-279） | 无人认领 | **CRITICAL**：wake 后 draft rope/scale/expert-map 全零，draft reload 也不恢复（非 checkpoint 键）→ MTP 接受率崩塌 |
| MTP 逐层完整性校验 raise | deepseek_mtp.py:509-526 | — | — | — | 逐权重流式 `load_weights([(name,w)])`（nccl_engine.py:317）会误触发 ValueError |

## 特殊发现

1. 与 R1 的差异仅在 checkpoint/config 数值层面（V3 базовый chat 模型、R1 为推理蒸馏），vLLM 状态生命周期完全同构；两模型审计结论可互相引用，修复一处即同时覆盖两者（以及 `DeepseekForCausalLM`、`GlmMoeDsaForCausalLM` 等同文件别名，deepseek_v2.py:1912-1921）。
2. `e_score_correction_bias` 是 V3 论文的 aux-loss-free 负载均衡核心状态，训练期会更新——**在 RL 场景它恰恰属于"会变的权重"**，若 transfer 白名单按"requires_grad/trainable"过滤会漏掉它（requires_grad 恒 False），SKIP_TENSORS + P2 盲区叠加后果被放大。
3. FP16 溢出缩放路径（deepseek_v2.py:1302-1313, 1335-1341）为纯 forward 数值逻辑，无持久状态，不参与生命周期。
4. `_pending_indexer_wk_fp8` 加载缓冲（deepseek_v2.py:1522-1524）挂在 model 对象上跨 load_weights 调用持久——V3 无 indexer 不触发；但注意它是 dict 裸属性，若日后 v3.2 系模型走逐权重流式 reload，其 partial 状态跨 session 残留是隐患（记录备查）。
5. 主模型 buffers 的安全性 100% 依赖 P2 的"named_buffers 无差别备份"；重设计若改为按角色声明（R2→RECOMPUTE/PRESERVE），cos_sin_cache 应配显式重建钩子而非备份。

## 结论

- **今日会腐蚀（sleep-L2 + RL reload）**：
  1. **MTP draft 全部 buffers** —— CRITICAL（同 #10；draft cos_sin_cache 清零 → 接受率崩塌）。
  2. **e_score_correction_bias** —— 条件性 HIGH（权重流不含该键时静默清零；V3/R1 均为 noaux_tc，必存在该状态）。
- **时序依赖**：W_UK_T/W_UV 与 MoE kernel 格式产物在 wake→reload 窗口内为零，禁止 wake 后未 reload 即服务。
- **安全**：R1 类 checkpoint 参数（P3/P4）、主模型 R2 buffers（P2）、kv_cache（P7）。
