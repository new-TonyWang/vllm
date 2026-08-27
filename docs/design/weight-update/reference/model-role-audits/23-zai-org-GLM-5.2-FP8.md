# 模型角色审计 #23 — zai-org/GLM-5.2-FP8

## 基本信息

- **实现文件**: `vllm/model_executor/models/deepseek_v2.py` — HF 架构 `GlmMoeDsaForCausalLM` 直接映射到 `DeepseekV2ForCausalLM` 空子类 (deepseek_v2.py:1920-1921, registry.py:116)。`model_type=glm_moe_dsa` 复用 DeepSeek V3.2 DSA 架构 (vllm/models/deepseek_v32/__init__.py:5-9 的注释确认，但该硬件隔离入口未被 registry 引用，实际走 deepseek_v2.py)。
- **注意力**: MLA(非 GQA)。`glm_moe_dsa` 在 `is_deepseek_mla` 列表 (transformers_utils/model_arch_config_convertor.py:263-270)，`kv_lora_rank` 存在 → `DeepseekV2MLAAttention` (deepseek_v2.py:952)。rope 仅作用于 `qk_rope_head_dim` 部分（结构性 partial rope），yarn mscale 折入 python float `self.scaling` (deepseek_v2.py:1070-1073)。
- **DSA 稀疏注意力**: `index_topk` 存在 → 每层 `Indexer` (deepseek_v2.py:643) + `DeepseekV32IndexerCache`。
- **量化**: 官方 FP8 = block-128 fp8 → `Fp8MoEMethod` (fp8.py:492) + `Fp8LinearMethod` (fp8.py:267)。MoE experts 后端: Blackwell 自动优先 `FLASHINFER_TRTLLM`→`TrtLlmFp8Experts*` (oracle/fp8.py:80-110, trtllm_fp8_moe.py); Hopper TP → `TRITON`，EP → `FLASHINFER_CUTLASS` (oracle/fp8.py:112-122)。
- **MTP draft**: `glm_moe_dsa` 被 `hf_config_override` 重写为 `deepseek_mtp` / `DeepSeekMTPModel` (config/speculative.py:331-342) → `vllm/model_executor/models/deepseek_mtp.py`（不是 glm4_moe_mtp.py）。draft 内嵌完整 `DeepseekV2DecoderLayer`（MLA + Indexer + MoE 全套）。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| 各 Linear/MoE checkpoint 参数 (w13/w2 fp8 + weight_scale_inv 等) | fp8.py:576-643 | R1(+R3a kernel 重排) | P4+P3: layerwise reload 按 restore_metadata 还原原始格式→重载→PWAL 重跑→copy-back 原地址 (reload/layerwise.py:385-429, 445-461) | RESTORABLE | 低（RL reload 走 layerwise 生命周期时） |
| `gate.e_score_correction_bias` (fp32 Param) | deepseek_v2.py:314-317 | R1 | 在 SKIP_TENSORS (reload/meta.py:31)，loader 不包装、直接原地加载；FusedMoE 持有别名 (deepseek_v2.py:378, routed_experts.py:114) | RESTORABLE | 低（reload 原地写，别名不失效；但 L2 唤醒后至 reload 完成前为 0） |
| MLA `W_UV`/`W_UK_T`（由 kv_b_proj 吸收变换） | mla_attention.py:908-995 | R3a | P4: attention 层在 finalize 阶段重跑 `process_weights_after_loading` (reload/layerwise.py:343-357)；`replace_parameter(prefer_copy=True)` 保地址 (mla_attention.py:993-995) | RECOMPUTE | 低 |
| Attention `_q/_k/_v/_prob_scale` (registered buffer) | attention.py:124-135,184 | R2/R1' | P2 (gpu_worker.py:272-274) + reload `_reload_attention_scales` (layerwise.py:360-383) 或 PWAL 内 `set_default_quant_scales` 重置 (mla_attention.py:1005-1006) | RESTORABLE | 低（仅主模型） |
| rotary `cos_sin_cache` (persistent=False buffer) | rotary_embedding/base.py:59-63 | R2 (+R6: `_ROPE_DICT` 全局实例缓存 rotary_embedding/__init__.py:30) | P2 覆盖（named_buffers 含非持久 buffer） | RESTORABLE | 低（主模型；draft 靠与主模型共享同一 `_ROPE_DICT` 实例才被间接恢复） |
| Indexer `wk_weights_proj.weight`（FP8 wk 在 load_weights 中反量化并与 weights_proj 融合） | deepseek_v2.py:674-684, 820-860 | R1+P5(loader 副作用融合) | P5: reload 走 `model.load_weights` 时重放 `_try_load_fp8_indexer_wk` | RESTORABLE | 低；但若用 kernel-format 直写路径 (`is_checkpoint_format=False`, gpu_model_runner.py:5538-5542) 则 checkpoint 名对不上融合参数名，需注意 |
| `topk_indices_buffer`（model 级 plain attr，穿透到所有层/MLA/indexer） | deepseek_v2.py:1359-1368,689 | R4 (scratch，graph 按地址捕获) | CuMem VA 保留；内容每步重写 | SCRATCH | 低（唤醒清零无害；地址不变） |
| Indexer `k_cache` (DeepseekV32IndexerCache) | deepseek_v2.py:694-699 | R7（KV-cache tag 管辖） | kv_cache wake tag | SCRATCH | 低 |
| `RoutedExperts` `_expert_map`/`expert_mask`/`expert_*_to_*` (registered buffers) | routed_experts.py:235-245 | R2 (EP 派生) | P2 覆盖（主模型）；reload 时在 SKIP_TENSORS 中被完全跳过 (reload/meta.py:25-32) | PRESERVE | 低（主模型）；**draft 上 HIGH，见下** |
| `Fp8MoEMethod.moe_kernel` + `moe_quant_config`（PWAL 无条件重建） | fp8.py:708-718, 761-763 | R5 | P4 重建；quant_config 引用的 w1/w2_scale 均为 layer 参数，copy-back 后地址内容一致 | RECOMPUTE | 中：见特殊发现 1 |
| trtllm fp8 per-tensor `_g1_alphas/_g2_alphas/_g1_scale_c` | trtllm_fp8_moe.py:286-291 | R5（已知 HIGH 模式） | 无 | RECOMPUTE | **本模型不触发**：GLM-5.2 为 block-128 量化，走 `_apply_block_scale` (trtllm_fp8_moe.py:353-435)，scale 直接取 layer 参数；`gemm1_alpha/beta/clamp` 对 silu 为 None (trtllm_fp8_moe.py:63-89) |
| `expert_weights` EPLB 别名列表 | interfaces.py:916-919 | R7(内部别名) | copy-back 保留原张量对象 | PRESERVE | 低 |
| MTP draft 全部参数/buffer（enorm/hnorm/eh_proj/shared_head + 完整 decoder 层） | deepseek_mtp.py:83-118 | R1/R2/R3a | **P2 不覆盖 draft**：`sleep()` 只备份 `model_runner.model.named_buffers()` (gpu_worker.py:270-274)；draft 与主模型同在 "weights" 池加载 (gpu_worker.py:506-513) | RESTORABLE(需 draft reload) | **高**，见特殊发现 2 |
| draft 自建 `topk_indices_buffer` | deepseek_mtp.py:97-107 | R4 | 无需恢复 | SCRATCH | 低 |

## 特殊发现

1. **FP8 MoE PWAL 无条件重建 kernel（已知 HIGH 模式的 fp8 变体）**: `Fp8MoEMethod.process_weights_after_loading` 每次都调 `_setup_kernel` (fp8.py:761-763)：kernel 格式转换（trtllm block 路径做 W13→W31 交换 + BlockMajorK shuffle，flashinfer_utils.py:493-510）+ `replace_parameter`（未用 prefer_copy，fp8.py:698-701 → 每次重跑生成新 Parameter 对象/新地址）+ `make_fp8_moe_kernel` 重建。**今日的救赎**是 layerwise reload 的 copy-back（reload/layerwise.py:445-461）把处理结果拷回初始 kernel 张量并重新挂回，cudagraph 捕获的地址得以保全。但该保护要求 reload 严格走 initialize/finalize 生命周期；任何"只重跑 PWAL 不 rematerialize"的路径都会因 shuffle 非幂等（对已 shuffle 权重再 shuffle）而锁死 → R3c。
2. **MTP draft 是 sleep-L2 的最大缺口（HIGH）**: draft 不在 `_sleep_saved_buffers` 备份范围内；wake 后 draft 的 `_k_scale/_v_scale`、`cos_sin_cache`（若与主模型 rope 参数不同则为独立实例）、`RoutedExperts` expert-map buffers 全部清零。且 expert-map 系列在 SKIP_TENSORS（reload/meta.py:25-32）中，**连 draft 权重重传（`_set_draft_weight_update_target`, gpu_worker.py:1017-1035）也不会恢复它们** —— 被清零后无任何机制回填，EP>1 时 draft 路由静默指向本地 expert 0。draft 的 W_UV/W_UK_T、g1 缩放常量只有在对 draft 做完整 layerwise 重载时才会重算；若 RL 流程只更新主模型，draft 在 wake 后是全零/残破状态（spec decode 的 rejection sampling 保证最终输出正确性，但会产生 NaN 崩溃或接受率归零的隐性故障）。
3. **DSA indexer 的融合加载依赖 load_weights 路径**: `_try_load_fp8_indexer_wk` (deepseek_v2.py:820-860) 在 load_weights 中把 FP8 wk + scale 反量化后写入融合参数 `wk_weights_proj.weight`。这是 P5 型副作用，checkpoint-format 的 reload 可重放；trainer 若按 vLLM 参数名直推 kernel-format 权重则必须自行完成该融合。
4. `deepseek_v2.py:389-395` 对 e_score_correction_bias 的 dtype 原地改写仅在 ROCm AITER 分支，NVIDIA 部署不触发。
5. GLM-5.2 的 MTP 层 indexer 永不允许 skip_topk (deepseek_v2.py:1095-1103, 1160-1165)，draft indexer 状态与主模型同构，无额外持久状态。

## 结论

主模型（target）在"sleep-L2 → wake → layerwise RL reload"流程下基本健康：R1 由 reload 重写、R2 buffer 由 P2 备份恢复、R3a（MLA 吸收、fp8 shuffle）由 PWAL 重跑且 copy-back 保地址、block-fp8 的 trtllm kernel 常量都落在受保护的 layer 参数上（本模型不踩 per-tensor `_g1_alphas` 的已知 HIGH 雷区）。**今日真实会腐坏的是 MTP draft**：draft buffer 无 P2 备份、expert-map 被 SKIP_TENSORS 永久排除在恢复之外、attention scale 清零后仅在 draft 完整重载时经 PWAL 重置。启用 MTP + sleep-L2 的 RL 部署若不显式对 draft 做权重更新（且即便做了，expert-map 也不会恢复），draft 在 wake 后静默损坏。次要风险是 fp8 MoE PWAL 的非幂等 shuffle（R3c），依赖 layerwise rematerialization 兜底，禁止裸重跑 PWAL。
