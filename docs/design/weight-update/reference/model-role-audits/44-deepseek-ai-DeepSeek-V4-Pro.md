# 模型角色审计 #44 — deepseek-ai/DeepSeek-V4-Pro

基本信息
- 实现位置：与 DeepSeek-V4-Flash **同一份代码**——`vllm/models/deepseek_v4/`（fork 专属目录）。`DeepseekV4ForCausalLM`（nvidia/model.py）、`DeepSeekV4MTP`（nvidia/mtp.py）、`DSparkDeepseekV4ForCausalLM`（nvidia/dspark.py）；registry.py:94, 609, 631。
- Pro 与 Flash 的区别在 config/checkpoint 而非实现文件（见"与 Flash 的差异"）。fork 代码中对 Pro 的直接引用：`common/ops/fused_indexer_q.py:57` 的 MXFP4 clamp 常量注明取自 `DeepSeek-V4-Pro/inference/kernel.py`。
- 量化：`DeepseekV4FP8Config`（quant_config.py:29-160）——线性/attention FP8 block；experts 按 `expert_dtype`（fp4 → Mxfp4/NVFP4，fp8 → Fp8MoEMethod）。旗舰 Pro 检查点为 fp4 experts。
- MoE 后端：Pro 的目标部署（B200/SM100 + EP）是 **MegaMoE（deep_gemm_mega_moe）** 的主要用户（model.py:307-315 硬性要求 SM100；fp4 experts + sqrtsoftplus routing，model.py:543-552）。
- 稀疏注意力：C4A（compress_ratio=4，带 indexer）/ C128 / SWA 分层（attention.py:187-190, 253-272）；NSA 式 topk 索引 + compressor 状态缓存。
- 审计基线：fork HEAD c7ce03bcbd

状态清单与 #24（V4-Flash）逐行同构（同一实现），下表完整列出并按 Pro 的部署画像标注风险权重；细节论证见 #24，此处引用相同 file:line。

## 状态角色清单

### 主模型

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| **MegaMoE `_transformed_l1/l2_weights`（deep_gemm 变换产物，裸属性）** | nvidia/model.py:241-242, 317-355 | **R3c** | 幂等守卫（model.py:318-319）锁死重算；finalize 销毁 w13/w2 参数（model.py:352-355）断掉 reload 链 | 应 RECOMPUTE，实际被锁 | **CRITICAL**：Pro 旗舰部署（MegaMoE+EP+SM100）必中；reload-only=静默旧权重，sleep-L2=全零垃圾 |
| MegaMoE `w13_weight/w2_weight/w13_weight_scale/w2_weight_scale`（uint8 packed fp4 + ue8m0 scale） | nvidia/model.py:195-239 | R1 | init 时 record_metadata 捕获；但 reload 时因已被置 None，`_place_kernel_tensors` 丢弃新载权重（layerwise.py:464-474） | 应 RESTORABLE，实际断链 | **CRITICAL**（与上一条为同一故障的两面） |
| `hc_attn_fn_broadcast`（layer0 派生，裸属性） | nvidia/model.py:831, 1299-1308 | R3b | P5（load_weights 尾部重算）；非 buffer → P2 盲区 | RECOMPUTE | **HIGH**：wake 未 reload → 首层 MHC 损坏 |
| `attn_sink`（Param，-inf padding，raw copy_ 加载） | attention.py:198-201; nvidia/model.py:1238-1244 | R1 | 绕过 weight_loader → layerwise reload 不兼容 | 断链 | **HIGH**（Pro 头数更多、TP 切分下 padding 槽更多） |
| hc_attn_fn/hc_ffn_fn/…（每层 6 个 fp32 MHC Param）+ hc_head_fn/base/scale | nvidia/model.py:824-866, 1023-1041 | R1 | P3+P4 | RESTORABLE | 低 |
| `gate.tid2eid`（hash-MoE 表；Pro 的 num_hash_layers 层） | nvidia/model.py:564-578 | R1 | P3（不在 SKIP_TENSORS） | RESTORABLE | 中（sync 白名单漏掉 → hash 层全路由 expert 0） |
| `gate.e_score_correction_bias`（noaux_tc 层） | nvidia/model.py:579-583 | R1 | SKIP_TENSORS + 非 buffer（P2 盲区） | RESTORABLE（条件） | 中-高 |
| attention/compressor/indexer 各 FP8 block 线性（fused_wqa_wkv/wq_b/wo_a/wo_b/fused_wkv_wgate/weights_proj）+ `ape` | attention.py:203-239, 700-713; compressor.py:251-269 | R1 | P3+P4 | RESTORABLE | 低 |
| rotary `cos_sin_cache`（分层双 theta：compress_rope_theta vs rope_theta，persistent=False） | common/rope.py:9-36; base.py:63 | R2 | P2（主模型）；reload 不重算 | PRESERVE（P2） | 主模型低；5 处 kernel 直读（attention.py:543-599; compressor.py:369,424; flashmla.py:46） |
| `_flashinfer_fp8_*` buffers / host scales / `_einsum_recipe` | flashinfer_sparse.py:196-218; flashmla.py:40 | R2 | P2 / host | PRESERVE | 低 |
| `topk_indices_buffer`、`_mtp_hidden_buffer` | nvidia/model.py:991-995, 1048-1055 | R4（地址被 graph 捕获） | 每步全量重写 + CuMem 同 VA | SCRATCH | 低 |
| `_symm_buffer_cache`（类级，deep_gemm 对称显存） | nvidia/model.py:162, 357-384 | R6/R7 | pool 外分配，sleep 不触及 | PRESERVE | 低（正确性 OK；**Pro 的 symm buffer 按 max_num_tokens×topk×hidden 计，体量大，sleep 期间不释放**） |
| compressor `state_cache` / SWA cache / indexer k_cache（kv 页） | compressor.py:271-276; attention.py:294-300, 735-741 | R7 | P7 memset | SCRATCH | 无 |
| `_compress_scratch`（two-stage 补偿 fp32 scratch，ROCm-only） | compressor.py:241-248 | R4 | 每步重写 | SCRATCH | 无（NVIDIA 不分配） |

### MTP draft（nvidia/mtp.py）与 DSpark（nvidia/dspark.py）

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| draft R1 集（e_proj/h_proj/enorm/hnorm/shared_head/hc_head_*） | nvidia/mtp.py:84-124; dspark.py:104-111 | R1 | draft reload（PR #46725） | RESTORABLE | 低 |
| **draft MegaMoE `_transformed_*`（mtp_block.ffn / dspark 层）** | nvidia/mtp.py:478, 482-484; dspark.py:456-462 | R3c | 同主模型守卫锁死 | 被锁 | **CRITICAL** |
| **draft 全部 registered buffers（cos_sin_cache 等）** | gpu_worker.py:272-279 | R2 | **无**（P2 仅主模型；`_build_fused_kv_buffers` 钩子两种 V4 draft 均无） | 无人认领 | **CRITICAL**：wake 后 draft rope 全零 → 接受率崩塌 |
| draft `attn_sink` raw copy_ | nvidia/mtp.py:435-440 | R1 | 同主模型缺陷 | 断链 | HIGH |
| MTP load_weights 末尾逐层校验 raise | nvidia/mtp.py:461-477 | — | — | — | 逐权重流式 reload（nccl_engine.py:317）误触发 ValueError |

## 与 Flash 的差异（config 层面，影响风险权重而非清单结构）

1. **规模**：Pro 为旗舰大杯（更多层/专家/头数）→ MegaMoE + EP 几乎是必选部署形态，R3c CRITICAL 从"可能踩中"升级为"默认踩中"；`_symm_buffer_cache` 与变换产物的显存占用也更大，sleep-L2 的"省显存"收益被 pool 外常驻进一步稀释。
2. **量化**：Pro 检查点 fp4 experts（`expert_dtype="fp4"`，quant_config.py:57-84 惰性解析）+ FP8 block 线性；若启用 `moe_quant_algo=NVFP4`（quant_config.py:143-151）则走 `ModelOptNvFp4FusedMoE`——该路径的 R5 图捕获常量（gemm1_alpha/_g1_alphas 族，fused_moe/config.py:353, 501-628）随 PWAL 重建，行为同 FusedMoE 系（RESTORABLE/RECOMPUTE），无 MegaMoE 守卫问题。
3. **NSA/稀疏结构**：Pro 与 Flash 同用 compress_ratios 分层（C4A 带 indexer、C128、SWA，attention.py:187-190, 253-272）；差异是 pattern 长度与 index_topk 值——状态角色不变。C128 层无 indexer，只有 compressor —— compressor 滚动状态在 kv 页（P7 自洽）。
4. Flash-Base 类 fp8-expert 变体在 Pro 家族同样可能存在；该形态无 MegaMoE 风险（quant_config.py:153-155 落回 Fp8MoEMethod）。

## 特殊发现

1. Pro 的默认高性能路径把三个 CRITICAL 叠在同一条链上：MegaMoE（主模型）→ MegaMoE（MTP draft）→ draft buffers 无 P2。sleep-L2 + reload 后哪怕修好 MegaMoE，draft rope 仍是零；反之只修 draft 也救不了主模型。**必须视为一组联动修复**。
2. `finalize_weights` 的注释（model.py:346-351）明确说明 L2 变换权重与原 `w2_weight` 存储存在别名关系（"the L2 weight is the only tensor that aliases the original storage"）——即便解开守卫重算，`transform_weights_for_mega_moe` 为 L1 分配新张量：graph 捕获的 `_transformed_l1` 地址与重算产物地址不同 → 必须实现"重算进原地址"（copy_ 回写）才对 cudagraph 安全，直接重新赋值属 rebuild 反模式。
3. dspark.py 的 `_finalize_moe`（dspark.py:456-462）与 MTP 一样在 load_weights 尾部调 `finalize_mega_moe_weights`——三个入口（主模型/MTP/DSpark）共享同一守卫缺陷，修 `DeepseekV4MegaMoEExperts.finalize_weights` 一处即可覆盖三者。
4. `get_expert_weights`（model.py:400-427，EPLB 视图）直接暴露 `_transformed_*` 存储给 EPLB 重排——EPLB 原地写这些视图不会触发守卫问题（无需重算），但 reload 修复若重建张量会使 EPLB 持有的旧视图悬空，需同步设计。
5. per-weight 流式 reload 下 `DeepseekV4ForCausalLM.load_weights` 尾部两个 finalize（model.py:1471-1472）每次调用都执行：MegaMoE finalize 因守卫为 no-op（掩盖问题），mhc broadcast 反复重算（中间态可能在 meta 上）——批式（ipc_engine.py:305）无此问题。建议 reload 契约明确"finalize 只在会话结束时跑一次"。

## 结论

- **今日会腐蚀（sleep-L2 + RL reload，Pro 默认 MegaMoE+MTP 部署）**：
  1. **MegaMoE `_transformed_l1/l2_weights`（主模型 + MTP/DSpark draft 共三处入口）** —— CRITICAL：reload-only 静默旧权重；sleep-L2 全零。R3c 幂等守卫 + loader 参数销毁双重断链。
  2. **draft 全部 buffers（cos_sin_cache）** —— CRITICAL：P2 不覆盖 draft，MTP 接受率崩塌。
  3. **`hc_attn_fn_broadcast`** —— HIGH（wake 未 reload 即损坏）。
  4. **`attn_sink`** —— HIGH（清零 + raw copy_ 加载与 layerwise reload 不兼容）。
  5. `tid2eid`/`e_score_correction_bias` —— 条件性（权重流键集合）。
- **安全**：FP8 线性/attention/compressor R1 参数（P3/P4）、主模型 R2 buffers（P2）、kv/state cache（P7）、R4 workspace（每步重写 + CuMem 同 VA）、host 常量。
