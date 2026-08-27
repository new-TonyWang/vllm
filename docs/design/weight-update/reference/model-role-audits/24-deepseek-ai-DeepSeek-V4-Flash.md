# 模型角色审计 #24 — deepseek-ai/DeepSeek-V4-Flash

基本信息
- 实现位置（fork 专属，`vllm/models/` 而非 `vllm/model_executor/models/`）：
  - 主模型：`vllm/models/deepseek_v4/nvidia/model.py`（`DeepseekV4ForCausalLM`；注册于 registry.py:94，经 `vllm/models/deepseek_v4/__init__.py:27-32` 按平台分发）
  - MTP draft：`vllm/models/deepseek_v4/nvidia/mtp.py`（`DeepSeekV4MTP`，registry.py:631）；另有 DSpark draft `nvidia/dspark.py`（registry.py:609）
  - 注意力：`vllm/models/deepseek_v4/attention.py`（`DeepseekV4Attention` 基类）+ `nvidia/flashmla.py` / `nvidia/flashinfer_sparse.py`
  - 量化：`vllm/models/deepseek_v4/quant_config.py`（`DeepseekV4FP8Config`：线性层 FP8 block；experts 按 `expert_dtype` 分发——Flash 默认 `fp4` → `Mxfp4MoEMethod` 或 NVFP4；`fp8` 变体走 `Fp8MoEMethod`，quant_config.py:134-155）
  - MoE：`DeepseekV4MoE` 双后端——FusedMoE 或 **MegaMoE**（`--kernel-config moe_backend=deep_gemm_mega_moe`，SM100 专属，model.py:524-532, 605-645）
- 架构要点：MHC 多流 hidden（hc_mult 流 + sinkhorn 混合）、稀疏 MLA（compress_ratio 4/128 分层 + indexer）、SWA cache、attn sink、hash-MoE（前 num_hash_layers 层用 tid2eid 表路由）
- 审计基线：fork HEAD c7ce03bcbd

## 状态角色清单

### 主模型

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| **MegaMoE `_transformed_l1_weights` / `_transformed_l2_weights`（deep_gemm 变换后 kernel 权重，裸属性 tuple）** | nvidia/model.py:241-242, 317-355 | **R3c** | 名义 P5（`load_weights` 末尾 `finalize_mega_moe_weights`，model.py:1471）+ forward 兜底再调（model.py:492-494），但 **`finalize_weights` 幂等守卫 `if self._transformed_l1_weights is not None: return`（model.py:318-319）把重算锁死** | 应为 RECOMPUTE，实际被锁 | **CRITICAL（双重失效）**：(a) 仅 reload：新 expert 权重永远进不了 `_transformed_*` → 静默用旧 policy；(b) sleep-L2：变换产物分配于 weights pool → wake 后清零，守卫仍拦截重算 → 输出彻底损坏。与 kimi_k3 `nvidia/linear.py` 已知 CRITICAL 同构 |
| MegaMoE `w13_weight/w2_weight/w13_weight_scale/w2_weight_scale` finalize 后被置 None | nvidia/model.py:352-355 | R1（被销毁） | record_metadata 在 init 时捕获过它们（model_loader/utils.py:64），但 reload 时 `initialize_layerwise_reload` 捕获的 `kernel_tensors` 为空（params 已是 None，reload/utils.py:29 过滤 None）→ finalize 阶段 `_place_kernel_tensors`（layerwise.py:464-474）**把刚流入的新权重全部 delattr 丢弃** | 应为 RESTORABLE，实际断链 | **CRITICAL**（同上问题的 reload 侧机制细节；新权重被丢 + 守卫拦重算） |
| **`hc_attn_fn_broadcast`（layer0 派生和，裸属性，非 param/buffer）** | nvidia/model.py:831, 1299-1308 | R3b | P5：`load_weights` 末尾 `finalize_mhc_broadcast_weights` 重算（model.py:1472）。非 buffer → **P2 不覆盖** | RECOMPUTE | **HIGH**：wake 未 reload → 全零 → 首层 `mhc_pre_broadcast_tilelang`（model.py:883-896）输出损坏。reload 会重算（逐权重流式时中间调用可能在 meta 张量上求和产生 meta 结果，最终一次调用在 layer0 参数已回填后执行才正确——时序脆弱） |
| **`attn_sink`（Param，-inf padding，raw `copy_` 加载绕过 weight_loader）** | attention.py:198-201; 加载 nvidia/model.py:1238-1244 | R1 | 加载不经 weight_loader → layerwise reload 期间该 param 在 meta 上，`params_dict[name][:n].copy_()` 会失败或其 load 不计入 `load_numel` → finalize 按"未加载"走 `_place_kernel_tensors` 放回旧值（layerwise.py:286-297） | 应为 RESTORABLE，实际 reload 不兼容 | **HIGH**：sleep-L2 清零后 padding 槽从 -inf 变 0（0 ≠ "无 sink"）且 reload 无法正确写回 → attention 分布畸变 |
| `hc_attn_fn/hc_ffn_fn/hc_attn_base/hc_ffn_base/hc_attn_scale/hc_ffn_scale`（每层 fp32 Param） | nvidia/model.py:824-866 | R1 | P3 + P4（普通 layerwise 流程；DeepseekV4DecoderLayer 直属 params） | RESTORABLE | 低 |
| `hc_head_fn/hc_head_base/hc_head_scale`（model 级 Param） | nvidia/model.py:1023-1041 | R1 | P3 | RESTORABLE | 低 |
| `gate.tid2eid`（hash-MoE 路由表 Param，int32/int64，vocab×topk） | nvidia/model.py:564-578 | R1（非 trainable 语义表） | P3（不在 SKIP_TENSORS，正常 capture/wrap） | RESTORABLE | 中：reload 流含该键则恢复；RL sync 白名单若按 trainable 过滤会漏 → 全零表把 hash 层全部 token 路由到 expert 0 |
| `gate.e_score_correction_bias`（非 hash 层，noaux_tc） | nvidia/model.py:579-583 | R1 | SKIP_TENSORS（meta.py:31）豁免 + 非 buffer → P2 不覆盖（同 V2 系分析） | RESTORABLE（条件） | 中-高（同 #10/#50） |
| `fused_wqa_wkv/wq_b/wo_a/wo_b` FP8 block 权重 + `weight_scale_inv` | attention.py:203-239; 消费于 nvidia/ops/o_proj.py:63-70 | R1 | P3 + P4 | RESTORABLE | 低 |
| compressor `ape`（fp32 Param）、`fused_wkv_wgate`、`norm.weight` | compressor.py:251-269 | R1 | P3 | RESTORABLE | 低 |
| indexer `wq_b/weights_proj` | attention.py:700-713 | R1 | P3 | RESTORABLE | 低 |
| rotary `cos_sin_cache`（每层两种 theta：compress_rope_theta/rope_theta，persistent=False buffer） | common/rope.py:9-36; rotary_embedding/base.py:63 | R2 | P2（主模型）；reload 不重算 | PRESERVE（P2) | 主模型低；被 5 处 kernel 直读（attention.py:543, 554-599; compressor.py:369, 424; flashmla.py:46） |
| `_flashinfer_fp8_q_scale/_q_scale_inv/_kv_scale`（register_buffer, FlashInfer fp8 KV 路径） | nvidia/flashinfer_sparse.py:196-216 | R2 | P2 | PRESERVE（P2） | 低 |
| `_flashinfer_fp8_bmm1/2_scale`（host float）、`_einsum_recipe/_tma_aligned_scales` | flashinfer_sparse.py:217-218; flashmla.py:40 | R2（host） | 不在 GPU | PRESERVE | 无 |
| `topk_indices_buffer`（model 级共享，裸 GPU 张量） | nvidia/model.py:991-995 | R4 | 内容每步由 indexer 重写；地址被 graph 捕获 → 依赖 CuMem 同 VA 唤醒 | SCRATCH | 低（内容清零无害；地址由 CuMem 保证） |
| `_mtp_hidden_buffer`（pre-hc_head 残差 stash，graph 内 copy_ 目标） | nvidia/model.py:1048-1055, 1133-1135 | R4/R7（地址敏感） | 每次 target forward 全量 copy_ 重写；注释明确"stable address（cudagraph pool 外）" | SCRATCH | 低（内容 wake 后首个 target step 即重写；draft 在 target step 前读它的路径不存在） |
| `DeepseekV4MegaMoEExperts._symm_buffer_cache`（类级 dict → deep_gemm 对称显存） | nvidia/model.py:162, 357-384 | R6/R7 | 首次 forward 分配（pool 外）→ sleep 不丢弃；key 按 (group,device,shape) 稳定 | PRESERVE | 低（正确性无虞；但 sleep 期间这块对称显存不释放，蚕食 sleep 的省显存目标——记录） |
| `EplbLayerState`（引用 runner 侧 tensors） | nvidia/model.py:192, 386-398 | R7（外部引用） | EPLB 子系统自管 | PRESERVE | 低 |
| SWA cache / indexer k_cache / compressor state_cache（kv_cache 页） | attention.py:294-300, 735-741; compressor.py:271-276 | R7 | P7 memset-0（compressor 滚动状态随请求重建，语义自洽） | SCRATCH | 无 |
| `aux_stream_list`/`ln_events`（CUDA stream/event 对象） | nvidia/model.py:988; attention.py:275-279 | R7（非显存状态） | 进程生命周期 | PRESERVE | 无 |

### MTP draft（nvidia/mtp.py）

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| e_proj/h_proj（fp8）、enorm/hnorm、shared_head、每层 hc_head_fn/base/scale | nvidia/mtp.py:84-124 | R1 | draft reload（PR #46725） | RESTORABLE | 低 |
| **mtp_block.ffn 的 MegaMoE `_transformed_*` + 幂等守卫** | nvidia/mtp.py:478, 482-484 → model.py:317-355 | R3c | 同主模型——`load_weights` 末尾 `finalize_mega_moe_weights` 被守卫拦截 | 被锁 | **CRITICAL**（同主模型双重失效，draft 侧同样存在） |
| draft `attn_sink` raw copy_ 加载 | nvidia/mtp.py:435-440 | R1 | 同主模型缺陷 | reload 不兼容 | HIGH |
| **draft 全部 registered buffers（cos_sin_cache、_flashinfer_fp8_*）** | gpu_worker.py:272-279 | R2 | **无**：P2 仅主模型；`_build_fused_kv_buffers` 钩子（gpu_worker.py:318-324）V4 MTP/DSpark 均无 | 无人认领 | **CRITICAL**：wake 后 draft rope 全零 → MTP 接受率崩塌 |
| draft `topk_indices_buffer`（draft 自有，mtp.py:178-182） | nvidia/mtp.py:178-182 | R4 | 每步重写 | SCRATCH | 低 |
| MTP load_weights 末尾逐层校验 raise + finalize 副作用 | nvidia/mtp.py:461-480 | — | — | — | 逐权重流式 reload（nccl_engine.py:317）会误触发 ValueError；`finalize_mega_moe_weights` 在每次 load_weights 调用尾部执行（被守卫变 no-op，掩盖了时序问题） |

## 特殊发现

1. **MegaMoE 是本模型第一优先级风险**，且失效有两层独立机制：幂等守卫（model.py:318-319）锁死 R3c 重算；`finalize_weights` 主动销毁 loader 侧参数（model.py:352-355）使 layerwise reload 的 `kernel_tensors` 捕获为空、新权重在 `_place_kernel_tensors` 被丢弃。修复需同时：reload 前清 `_transformed_*`（解锁守卫）+ 保留（或重建）loader 参数存储、并对变换产物做地址保持写回（graph 捕获的是 `_transformed_*` 张量地址）。注意 `get_expert_weights`（EPLB，model.py:400-427）返回的是 `_transformed_*` 的视图，EPLB 重排与 reload 修复必须共用同一份存储。
2. **`hc_attn_fn_broadcast` 是"隐性首层门槛"**：只在 PP first rank 的第一层存在，任何输入为 2D 的 forward 都 assert 依赖它（model.py:882）。它是全库少数"裸属性 + 派生 + 无守卫"的 R3b，恰好每次 load_weights 都重算所以 reload-only 场景安全——但 sleep-L2 wake 不 reload 即损坏，且 P2 结构性照不到。
3. **attn_sink 的 raw copy_ 加载模式**（不挂 weight_loader）在本文件出现两处（主模型 + MTP），是 layerwise reload 契约（"一切经 weight_loader"）的违约点；即使不睡眠，纯 RL reload 也会异常或静默丢弃 sink 更新。
4. hash-MoE 的 `tid2eid` 是"非 trainable 大表 R1"新类别：功能上等价 e_score_correction_bias 的路由角色，但不在 SKIP_TENSORS——两个同角色状态走了两条不同的生命周期路径,重设计时应统一。
5. expert_dtype="fp8" 变体（Flash-Base）走 `Fp8MoEMethod`（quant_config.py:153-155），其 PWAL/replace_parameter 生命周期与 #10 审计一致（RESTORABLE/RECOMPUTE），无 MegaMoE 风险——**MegaMoE 风险仅在 `deep_gemm_mega_moe` kernel_config 下激活**（fp4 experts + EP + SM100）。
6. `_mtp_hidden_buffer` 与 `topk_indices_buffer` 展示了正确的 R4 模式：pool 内地址 + 每步全量重写，天然 sleep 安全，无需备份。

## 结论

- **今日会腐蚀（sleep-L2 + RL reload）**，按严重度：
  1. **MegaMoE `_transformed_l1/l2_weights`（R3c 幂等守卫 + 参数销毁）** —— CRITICAL：reload-only = 静默旧权重；sleep-L2 = 全零垃圾输出。主模型与 MTP draft 双份存在。
  2. **MTP draft 全部 buffers（cos_sin_cache 为主）** —— CRITICAL：P2 不覆盖 draft，wake 后接受率崩塌。
  3. **`hc_attn_fn_broadcast`** —— HIGH：wake 未 reload 即服务必损坏；reload 时序上脆弱但最终正确。
  4. **`attn_sink`** —— HIGH：sleep 清零（含 -inf padding 槽）+ reload 路径违约。
  5. `tid2eid` / `e_score_correction_bias` —— 条件性（取决于权重流键集合）。
- **安全**：常规 FP8 线性/attention R1 参数（P3/P4）、主模型 R2 buffers（P2）、compressor/SWA/indexer cache（P7）、host 常量、`_symm_buffer_cache`（pool 外，仅占显存不损坏）。
