# 模型角色审计 #10 — deepseek-ai/DeepSeek-R1

基本信息
- 实现文件：`vllm/model_executor/models/deepseek_v2.py`（`DeepseekV3ForCausalLM` = `DeepseekV2ForCausalLM` 别名，deepseek_v2.py:1916）
- MTP draft：`vllm/model_executor/models/deepseek_mtp.py`（`DeepSeekMTP`，官方实现）——本报告一并审计
- 量化：官方 FP8 block-quantized（`Fp8LinearMethod` block 路径 + `Fp8MoEMethod` block_quant）
- 注意力：MLA（`DeepseekV2MLAAttention` → `MultiHeadLatentAttentionWrapper` → `MLAAttention`）；R1 无 index_topk → 无 v3.2 Indexer/topk_indices_buffer（deepseek_v2.py:1359-1369 走 else 分支）
- 审计基线：fork HEAD c7ce03bcbd

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| q_a/q_b/kv_a/kv_b/o_proj/fused_qkv_a_proj 权重 + `weight_scale_inv`（FP8 block） | deepseek_v2.py:1002-1050 | R1 | P3 copy-back（layerwise.py:445-461）+ P4 PWAL 重跑 | RESTORABLE | 低（reload 全量重写；wake 后未 reload 前为零） |
| MoE experts `w13_weight/w2_weight/w13_weight_scale_inv/w2_weight_scale_inv` | fp8.py:642-643, 698-701 | R1 | P3 + P4（`replace_parameter`，layerwise 复制回原存储） | RESTORABLE | 低 |
| `gate.weight`（GateLinear，router） | deepseek_v2.py:308-313 | R1 | P3 | RESTORABLE | 低 |
| **`gate.e_score_correction_bias`（fp32 nn.Parameter）** | deepseek_v2.py:315-317 | R1（但被当 R2 对待） | **仅 P3 的"原地 in-place 写入"路径**：在 SKIP_TENSORS（reload/meta.py:31）→ 不 capture、不 wrap（layerwise.py:165），原存储始终存活；**是 Param 不是 buffer → P2 (`named_buffers`) 不覆盖**（gpu_worker.py:272-273） | RESTORABLE（条件） | **中-高**：sleep-L2 清零后，只有当 RL 权重流包含该键（`mlp.gate.e_score_correction_bias`，noaux_tc routing）时才被原 loader 原地写回；若 sync 子集不含它 → 静默保持全零，routing 偏置永久丢失 |
| MLA `W_UK_T` / `W_UV`（kv_b_proj 派生 bf16 bmm 权重） | mla_attention.py:908-995 | R3a | P4：attention 层 deferred finalize（layerwise.py:281, 310-312, 343-357）重跑 PWAL；`replace_parameter(prefer_copy=True)`（mla_attention.py:993-995 → model_executor/utils.py:82-90）**同形状原地 copy_，地址保持** —— 已知安全范式 | RECOMPUTE | 低（reload 后重算）；wake 未 reload 前为零 → 输出损坏 |
| `kv_b_proj` FP8 → bf16 反量化中间（`get_and_maybe_dequant_weights`） | mla_attention.py:912-914 | R3a 瞬态 | 随 P4 重算 | RECOMPUTE | 无 |
| rotary `cos_sin_cache`（deepseek_yarn，persistent=False buffer） | rotary_embedding/base.py:63; deepseek_scaling_rope.py:248 | R2 | **P2**：`named_buffers` 无差别备份/回写（gpu_worker.py:270-316）。注意 reload 不重算它（无 loader，finalize 走 `_place_kernel_tensors` 原样放回，layerwise.py:295-297） | PRESERVE（P2） | 主模型低；**draft 见下** |
| attention `_q_scale/_k_scale/_v_scale/_prob_scale`（register_buffer） | attention.py:124-135, 184 | R2/R1 | P2 备份；reload 时 `_reload_attention_scales` + PWAL `set_default_quant_scales(register_buffer=False)` 原地 fill_（mla_attention.py:1000-1006） | RESTORABLE | 低 |
| `q_range/k_range/v_range`、`_k_scale_cpu` 等 host 张量/float | attention.py:137-150; mla_attention.py:544-546 | R2（host） | 不在 GPU pool | PRESERVE | 无 |
| expert maps：`_expert_map/expert_mask/expert_global_to_physical/...`（register_buffer） | routed_experts.py:235-245 | R2 | SKIP_TENSORS（meta.py:26-30）使 reload 不碰；**P2 备份内容**；EPLB 变更走 `update_expert_map()`（deepseek_v2.py:1780） | PRESERVE（P2） | 主模型低（完全依赖 P2）；**draft 见下** |
| FP8 MoE kernel 格式转换产物（DEEPGEMM ue8m0 requant / TRTLLM shuffle） | oracle/fp8.py:457-531; fp8.py:685-718 | R3a | P4 重跑 `_setup_kernel` → `replace_parameter` + copy-back 回原地址 | RECOMPUTE | 低 |
| `moe_quant_config` / `moe_kernel`（PWAL 构建，持 scale 引用；TRTLLM 路径含 R5 语义常量） | fp8.py:708-718, 774-803 | R5/R4 | P4 重建。注意重建后 config 引用的是临时物化张量，graph 捕获的是被 copy-back 的原地址 → 数值一致 | RECOMPUTE | 低（有内存双持轻微开销） |
| `DeepSeekV2FusedQkvAProjLinear._use_min_latency_gemm` | deepseek_v2.py:922-934 | R2（host bool） | init 一次 | PRESERVE | 无 |
| kv_cache 页（MLA spec） | mla_attention.py:531 附近 | R7 | P7 memset-0 + scale 重置 | SCRATCH | 无 |

### MTP draft（deepseek_mtp.py，官方）

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| enorm/hnorm/eh_proj/shared_head（norm+lm_head） | deepseek_mtp.py:91-117 | R1 | draft reload（PR #46725 `start_draft_weight_update` 重定向 engine → 同一 layerwise 生命周期） | RESTORABLE | 低（须执行 draft reload） |
| mtp_block = DeepseekV2DecoderLayer（含 MLA W_UK_T/W_UV、FP8 MoE、e_score_correction_bias） | deepseek_mtp.py:112-117 | 同主模型各行 | 同主模型（draft reload 重跑 PWAL） | 同上 | 同上 + e_score_correction_bias 同样的 SKIP_TENSORS 缺口 |
| **draft rotary `cos_sin_cache`、`_q/_k/_v_scale` buffers、expert maps —— 整个 draft 模型的 registered buffers** | gpu_worker.py:272-273 | R2 | **无**：P2 只遍历 `self.model_runner.model.named_buffers()`（主模型）；draft 唯一钩子是 `_build_fused_kv_buffers` 重建（gpu_worker.py:277-279, 318-324），DeepSeekMTP **没有**该方法；draft reload 也不重写 R2 buffer（非 checkpoint 键，`_copy_and_restore_kernel_tensors` 跳过未加载的 non-persistent buffer，layerwise.py:455-458） | 应为 PRESERVE，实际 **无人认领** | **CRITICAL**：sleep-L2 wake 后 draft cos_sin_cache 全零 → draft RoPE 全错 → MTP 接受率崩塌（与 `k3-dspark-l2-acceptance-collapse-root-cause-20260724.md` 根因同构）；draft expert_map 全零 → draft MoE 路由错乱 |
| MTP load_weights 尾部逐层完整性校验 | deepseek_mtp.py:509-526 | —（loader 行为） | — | — | reload 兼容性隐患：若 transfer engine 逐权重调 `load_weights([(name,w)])`（nccl_engine.py:317），每次调用都会因"MTP 层缺权重"raise。IPC 批量路径（ipc_engine.py:305）不受影响 |

## 特殊发现

1. **e_score_correction_bias 的双重豁免恰好形成盲区**：SKIP_TENSORS 让 reload 机制绕过它（本意是保留原存储），`named_buffers` 让 P2 绕过它（因为它是 Param）。两个"保护"的成员判据都不以"可恢复性"为谓词。sleep-L2 后它是否恢复完全取决于权重流是否携带该键 —— 这在 RL 部分同步（仅 trainable 子集）场景下会静默出错，且零偏置不会崩溃、只会悄悄改变专家选择分布，属最难察觉的一类腐蚀。
2. **W_UK_T/W_UV 是全库的正面范例**：`replace_parameter(prefer_copy=True)`（utils.py:63-90 显式注明为 RL/cudagraph 设计）+ attention deferred finalize，使 R3a 派生权重在 reload 中原地重算。建议重设计将其作为 R3a 的规范模板。
3. **cos_sin_cache 依赖 P2 而非重算**：reload 全程不触碰 rotary buffer（无对应 checkpoint 键，`rotary_emb.inv_freq` 被显式 continue，deepseek_v2.py:1562）。若未来 P2 被移除/收窄，主模型 rope 也会失守；deepseek_yarn 的 cache 是 config 派生（R2），理想终态是 RECOMPUTE（重建钩子）而非依赖无差别备份。
4. draft 模型的保护矩阵覆盖为空集（除 fused-kv 特例钩子）——对 DeepSeek 系 MTP 是结构性缺口，不是单点遗漏。
5. R1 无 v3.2 indexer 路径（无 `index_topk`），故 `topk_indices_buffer`、`DeepseekV32IndexerCache`、fp8 indexer wk 融合加载（deepseek_v2.py:820-864）均不参与本模型。

## 结论

- **今日会腐蚀（sleep-L2 + RL reload 流程下）**：
  1. **MTP draft 全部 registered buffers（首当其冲 cos_sin_cache）** —— CRITICAL，wake 后无任何机制恢复；表现为 MTP 接受率崩塌，主模型输出仍正确（易误判为"spec decode 参数问题"）。
  2. **e_score_correction_bias** —— 条件性 HIGH：权重流不含该键时静默清零。
- **依赖时序正确性**：W_UK_T/W_UV、FP8 MoE kernel 格式产物在 wake 之后、reload 完成之前是零 —— 流程上必须保证"wake → reload → 服务"顺序，任何 wake 后直接采样的路径都会产生垃圾输出。
- **安全**：主模型 buffers（P2）、所有 R1 checkpoint 参数（P3+P4）、kv_cache（P7）、host 常量。
