# 模型角色审计 #45 — nvidia/GLM-5.2-NVFP4

## 基本信息

- **实现文件**: 与 #23 完全同架构 —— `GlmMoeDsaForCausalLM` → `deepseek_v2.py DeepseekV2ForCausalLM` (deepseek_v2.py:1920-1921, registry.py:116)。MLA + DSA indexer + noaux_tc 路由。arch 级状态清单与 #23 一致，本报告只列差异项与 NVFP4 特有状态，其余引用 #23。
- **量化**: NVIDIA 官方 NVFP4 重打包 → `ModelOptNvFp4Config`：MoE `ModelOptNvFp4FusedMoE` (modelopt.py:1395)，linear `ModelOptNvFp4LinearMethod` (modelopt.py:1115)。Blackwell 自动选 `FLASHINFER_TRTLLM` → `TrtLlmNvFp4Experts*` (oracle/nvfp4.py:179-186; trtllm_nvfp4_moe.py:172-180 仅支持 SM100 家族)。GLM 为门控 silu MoE → `is_act_and_mul=True` 分支全程生效。
- **KV/注意力**: MLA 的 kv_b_proj 若为 NVFP4 量化，`get_and_maybe_dequant_weights` 在 MLA PWAL 中反量化后吸收出 W_UK_T/W_UV (mla_attention.py:908-995)。
- **MTP draft**: 同 #23，`glm_moe_dsa` → `DeepSeekMTPModel` → deepseek_mtp.py (config/speculative.py:331-342)；draft MoE 同样走 NVFP4 trtllm 路径。

## 状态角色清单

（R1 checkpoint 参数、MLA W_UK_T/W_UV、rope cache、attention scales、e_score_correction_bias、expert-map buffers、indexer/topk buffer、MTP draft 缺口 —— 与 #23 各行完全一致，风险判定不变；下表为 NVFP4 差异项。）

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| MoE NVFP4 checkpoint 参数：`w13/w2_weight`(packed u8)、`w13/w2_weight_scale`(e4m3 block)、`weight_scale_2`、`input_scale` | modelopt.py:1441-1556 | R1 | P4+P3 layerwise 生命周期 (reload/layerwise.py:385-429,445-461) | RESTORABLE | 低 |
| PWAL 派生：swizzled block scales + `w13_weight_scale_2[:,0]` 缩并 + 8×`replace_parameter`（无 prefer_copy） | modelopt.py:1557-1602; oracle/nvfp4.py:295-363 | R3a + **R3c**（`[:,0]` 对二跑的 1-D 张量崩溃） | P4 rematerialize 后重跑；copy-back 恢复原地址 | RECOMPUTE | 中：裸重跑 PWAL 立即锁死；layerwise 全链下安全 |
| experts PWAL 原地融合 `w13_weight_scale_2.data.mul_(w13_input_scale)`、`w2_weight_scale_2.data.mul_(w2_input_scale)` | trtllm_nvfp4_moe.py:121-123 | **R3c 非幂等** | 同上（重载还原原始 scale 后融合才正确） | RECOMPUTE | 中 |
| `g1_scale_c = g1_alphas * a2_gscale`（PWAL 中 register_parameter，且 self 引用换到 layer 参数） | trtllm_nvfp4_moe.py:124-137 | R5→参数化（已知 HIGH 常量之一） | 首载后为 layer 参数 → reload kernel_tensors 捕获 + copy-back 保地址 | RECOMPUTE | 低-中（保护成立的前提是 reload 走 layerwise 且形状不变） |
| `gemm1_alpha/gemm1_beta/gemm1_clamp_limit`（silu 无 swiglu 参数） | trtllm_nvfp4_moe.py:94-111 | R5 | GLM silu → quant_config 与 moe_config 均无 alpha/beta/limit → 全 None | N/A | 不触发（本模型）；若未来 GLM 变体带 clamp，注意 145-163 的"除以 g1_alphas 折叠"是又一处 R3c |
| **quant_config `a1_gscale=(1.0/a13_scale)`、`a2_gscale=(1.0/a2_scale)`（计算张量，非参数非 buffer）** | oracle/nvfp4.py:504-522 (507-508) | **R5（已知 HIGH 模式核心）** | 无：P2 不见、copy-back 不见；PWAL 重跑只造新张量 | RECOMPUTE | **高**：sleep-L2 后旧张量同址清零，已捕获 CUDA graph 静默读 0 |
| `moe_kernel` 对象（PWAL 无条件重建 + `fused_experts.process_weights_after_loading` 链式调用） | modelopt.py:1604-1615 | R5 | P4 重建（eager 正确） | RECOMPUTE | 高（同上一条同根；graph 捕获旧 kernel 引用集） |
| NVFP4 Linear PWAL：`input_global_scale`/`weight_global_scale`/`alpha`/`input_global_scale_inv` + `del input_scale/weight_scale_2` | modelopt.py:1207-1238 | R3a + **R3c**（原始参数被删除） | P4 rematerialize + copy-back | RECOMPUTE | 中 |
| MLA kv_b_proj (NVFP4) → W_UK_T/W_UV 反量化吸收 | mla_attention.py:908-935 | R3a（跨层依赖：linear PWAL 先行） | attention 层在 finalize 阶段最后处理 (reload/layerwise.py:280-283,309-312)，顺序正确 | RECOMPUTE | 低 |
| DSA indexer `wk_weights_proj`（FP8 wk 融合路径） | deepseek_v2.py:820-860 | R1+P5 | 同 #23；NVFP4 checkpoint 中 indexer 通常保留高精度/FP8，融合逻辑仍适用 | RESTORABLE | 低 |
| MTP draft（deepseek_mtp.py 全套 + draft 侧 NVFP4 trtllm R5 常量） | deepseek_mtp.py:83-118 | R1/R2/R3/R5 | **P2 不覆盖 draft**；expert-map 在 SKIP_TENSORS 永不恢复 (reload/meta.py:25-32) | RESTORABLE(需 draft reload) | **高** |

## 特殊发现

1. **本模型同时踩中已知 HIGH 模式的两个半元素**: `g1_scale_c` 与 `_g1_alphas`（此处名为 quant_config.g1_alphas，别名 `w13_weight_scale_2` 参数，oracle/nvfp4.py:500-505）都因"参数化 + copy-back"获得了地址保全；真正的孤儿是 `a1_gscale/a2_gscale` 计算张量（oracle/nvfp4.py:507-508）—— trtllm 每次 apply 都从 kernel 对象取它们做激活量化。sleep-L2 丢弃 weights 池后，pre-sleep 捕获的 CUDA graph 中烘焙的旧 `a1_gscale` 地址读出全零 → NVFP4 激活缩放为 0 → 静默全错。`gemm1_alpha` 系对 silu-GLM 为 None，不触发。
2. **NVFP4 是四模型中 R3c 密度最高的路径**: 三处非幂等（MoE `[:,0]`、experts `.mul_`、Linear `del`）意味着"PWAL 可重入"假设完全不成立；layerwise reload 的 rematerialize-from-meta（record_metadata 在模型 init 后立即快照检查点格式，model_loader/utils.py:64）是唯一使 P4 成立的机制。权重生命周期重构中，任何绕开该机制的路径（如 kernel-format 直写 `param.copy_`, gpu_model_runner.py:5538-5542）对本模型必须禁用或补齐融合逻辑。
3. **draft 双重暴露**: GLM-5.2 的 MTP draft 含完整 MoE 层 → draft 侧同样存在 a1_gscale 孤儿 + expert-map 清零两个问题，且 draft 连 P2 都没有。与 #23 结论相同但严重度更高（NVFP4 的清零缩放直接产 NaN，而非仅 fp8 权重残破）。
4. `replace_parameter` 在 modelopt PWAL 中未用 `prefer_copy`（modelopt.py:1595-1602 vs utils.py:82-90），单看会造成地址漂移；实际由 layerwise copy-back (`_copy_and_restore_kernel_tensors`) 兜底恢复初始地址。重构时若想去掉 copy-back，需把这批 replace_parameter 全部改为 prefer_copy 语义并验证形状稳定。

## 结论

GLM-5.2-NVFP4 = #23 的架构状态面 + #28 的 NVFP4 量化状态面，是四个模型中今日风险最高者。**当下即腐坏**（sleep-L2 + RL reload + 图不重捕）：trtllm nvfp4 experts 的 `a1_gscale/a2_gscale` 图捕获孤儿（HIGH，静默错误）；MTP draft 无 P2、expert-map 永不恢复（HIGH）。**有条件安全**：NVFP4 的三处 R3c 非幂等 PWAL 仅在完整 layerwise 生命周期（rematerialize→原始 loader→PWAL→copy-back）下可重入，任何简化路径都会锁死或崩溃。修复优先级建议：把 quant_config 的激活缩放张量参数化（纳入 copy-back/P2 语义）> draft 纳入 P2 与 SKIP_TENSORS 复审 > PWAL 幂等化。
