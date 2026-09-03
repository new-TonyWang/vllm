# 模型角色审计 #28 — nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

## 基本信息

- **实现文件**: `vllm/model_executor/models/nemotron_h.py` — `NemotronHForCausalLM` (registry.py:178)。混合架构：Mamba2 ("M") + Attention ("*") + MoE ("E") 层按 `hybrid_override_pattern` 排布 (nemotron_h.py:527-534, 582)。
- **注意力**: GQA、**无 rope**（NemotronHAttention 无 rotary_emb, nemotron_h.py:408-482），无 cos_sin_cache。
- **MoE**: 非门控 (`is_non_gated_moe=True`, nemotron_h.py:712; `activation_without_mul`, nemotron_h.py:226)，sigmoid 路由 + `e_score_correction_bias` (nemotron_h.py:157-159)，可选 latent MoE (`fc1/fc2_latent_proj`, nemotron_h.py:190-209)。
- **量化**: NVFP4 官方 → `ModelOptNvFp4Config` → MoE 走 `ModelOptNvFp4FusedMoE` (modelopt.py:1395)，linear 走 `ModelOptNvFp4LinearMethod` (modelopt.py:1115)。Blackwell 自动优先 `FLASHINFER_TRTLLM` → `TrtLlmNvFp4Experts*` (oracle/nvfp4.py:179-186, trtllm_nvfp4_moe.py:41)，该 kernel 显式支持非门控 relu2 (trtllm_nvfp4_moe.py:182-207)。ModelOpt 通常将 mamba in_proj/out_proj 排除在量化外（exclude_modules，视 checkpoint 而定）。
- **MTP draft**: `NemotronHMTPModel` → `nemotron_h_mtp.py` (registry.py:639)。draft 仅含 Attention("*") 与 MoE("E") MTP 层（nemotron_h_mtp.py:268-275），**无 mamba 层**。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| Mamba `A`（checkpoint 存 A_log，loader 内 `-exp` 变换） | mamba_mixer2.py:449-463; hf 映射 A_log→A nemotron_h.py:717 | R1+P5 | P5: layerwise reload 重放原始 weight_loader (reload/layerwise.py:404-411) | RESTORABLE | 低（checkpoint-format reload）；kernel-format 直写 (`param.copy_`) 会跳过 -exp 变换 → 需 trainer 侧预变换 |
| Mamba `conv_weights`（conv1d.weight 的非持久 view buffer） | mamba_mixer2.py:441-445 | R3b/R7(参数别名) | P2 覆盖（named_buffers 含它，写回等于写 conv1d.weight 存储）；reload 侧有专门的 alias-buffer 排除逻辑 (reload/meta.py:104-111) + `_place_kernel_tensors` 恢复原 view (layerwise.py:464-474) | PRESERVE | 低 |
| Mamba `_decode_state_offsets`（num_spec>0 时的 arange buffer, persistent=False） | mamba_mixer2.py:504-510 | R2 | P2 覆盖 | RESTORABLE | 低（主模型） |
| Mamba `dt_bias`/`D`/norm 权重 | mamba_mixer2.py:455-456,475-477 | R1 | reload 重写 | RESTORABLE | 低 |
| Mamba `kv_cache` 元组（conv/ssm state 绑定） | mamba_mixer2.py:498 | R7（kv_cache tag） | wake kv_cache tag + 重绑定 | SCRATCH | 低 |
| Attention `_q/_k/_v/_prob_scale` buffers | attention.py:124-135,184 | R2/R1' | P2（主模型）+ reload PWAL 重置/重载 (layerwise.py:343-383) | RESTORABLE | 低（主模型）；draft 上见特殊发现 3 |
| `gate.e_score_correction_bias` (fp32 Param) | nemotron_h.py:157-159 | R1 | SKIP_TENSORS 直接原地加载 (reload/meta.py:31)；FusedMoE 持别名 (nemotron_h.py:225) | RESTORABLE | 低 |
| MoE checkpoint 参数 w13/w2 + `w13/w2_weight_scale`、`weight_scale_2`、`input_scale` | modelopt.py:1441-1556 | R1 | P4+P3 layerwise 生命周期 | RESTORABLE | 低 |
| NVFP4 PWAL 派生：swizzle 后的 scale、缩并后的 `w13_weight_scale_2[:,0]` | modelopt.py:1557-1602; oracle/nvfp4.py:295-363 | R3a + **R3c 幂等锁死** | P4: 必须先 rematerialize 原始 (E,2) 形状才能重跑（`[:,0]` 索引对已处理的 1-D 张量直接崩溃, modelopt.py:1563-1570）；copy-back 保地址 | RECOMPUTE | 中（依赖完整 layerwise 生命周期；裸重跑 PWAL 立即锁死） |
| trtllm nvfp4 experts PWAL 原地融合 `w13_weight_scale_2.mul_(w13_input_scale)` | trtllm_nvfp4_moe.py:121-123 | **R3c 非幂等** | 同上，靠 rematerialize+重载原始值兜底 | RECOMPUTE | 中 |
| trtllm nvfp4 `g1_scale_c`（PWAL 中 register_parameter） | trtllm_nvfp4_moe.py:124-137 | R5→参数化 | 首载后进入 layer._parameters → reload 时被 kernel_tensors 捕获并 copy-back 保地址 | RECOMPUTE | 低-中（地址保全依赖 copy-back 分支形状匹配） |
| trtllm nvfp4 `gemm1_alpha/beta/clamp_limit` | trtllm_nvfp4_moe.py:94-111,145-170 | R5 | relu2 非门控 → 全为 None，不生成 | N/A | 不触发（本模型） |
| **quant_config `a1_gscale`/`a2_gscale`（`1.0/input_scale` 计算张量，非 layer 参数）** | oracle/nvfp4.py:504-522 (507-508) | **R5（已知 HIGH 模式）** | 无：不在 named_buffers（P2 不救）、不在 layer 参数（copy-back 不救）；PWAL 重跑只造新张量，旧地址被 cudagraph 咬死 | RECOMPUTE | **高**：初始张量在 weights 池 → L2 丢弃 → wake 后 VA 同址清零 → 已捕获的图静默读 0 缩放 |
| `moe_kernel` 对象整体（PWAL 无条件重建） | modelopt.py:1604-1615 | R5 | P4 重建（eager 正确）；captured graph 仍引用旧 kernel 的张量 | RECOMPUTE | 高（与上一条同根） |
| `RoutedExperts` expert-map buffers | routed_experts.py:235-245 | R2 | P2（主模型）；reload 时 SKIP_TENSORS 完全跳过 | PRESERVE | 低（主模型）；draft 上 **高** |
| latent MoE `fc1/fc2_latent_proj` | nemotron_h.py:190-209 | R1 | reload 重写 | RESTORABLE | 低 |
| NVFP4 Linear PWAL 派生：`input_global_scale`/`weight_global_scale`/`alpha`/`input_global_scale_inv`（并 `del input_scale/weight_scale_2`） | modelopt.py:1207-1238 | R3a + **R3c**（原始参数被 del，裸重跑必炸） | P4 rematerialize + copy-back | RECOMPUTE | 中 |
| MTP draft 全部状态（enorm/hnorm/eh_proj/final_layernorm/lm_head + attention/MoE 层） | nemotron_h_mtp.py:60-90,145-175,340-350 | R1/R2/R5 | **P2 不覆盖 draft** (gpu_worker.py:270-274)；draft 无 `_build_fused_kv_buffers` 钩子 | RESTORABLE(需 draft reload) | **高** |

## 特殊发现

1. **NVFP4 trtllm 的 R5 常量正是已知 HIGH 模式**: `make_nvfp4_moe_quant_config` 把 `g1_alphas` 直接别名到 `w13_weight_scale_2` 参数（受 copy-back 保护，oracle/nvfp4.py:500-505 注释也点明 EPLB 同步意图），但 `a1_gscale=(1.0/a13_scale)`、`a2_gscale=(1.0/a2_scale)` 是新计算张量（oracle/nvfp4.py:507-508），存活于初始 PWAL 时的 weights 池中，既不是 buffer 也不是参数。sleep-L2 丢页后 wake 同址清零；PWAL 重跑生成的新张量只被新 kernel 对象引用，**已捕获的 CUDA graph 仍读旧地址上的零** → 输出静默全错。eager/重新 capture 则正确。
2. **NVFP4 MoE/Linear PWAL 双重非幂等（R3c）**: MoE 侧 `w13_weight_scale_2[:,0]`（modelopt.py:1570，二跑时张量已是 1-D）+ experts 侧 `.data.mul_(input_scale)`（trtllm_nvfp4_moe.py:122-123）+ Linear 侧 `del layer.input_scale/weight_scale_2`（modelopt.py:1223,1227）。这三处使"只重跑 PWAL"完全不可行，重载必须走 rematerialize→原始 loader→PWAL→copy-back 全链（reload/layerwise.py:86-122, 385-429）。
3. **MTP draft 缺口**: draft 与主模型同池加载 (gpu_worker.py:506-513)，但 `_sleep_saved_buffers` 只遍历主模型。wake 后 draft 的 attention `_k/_v_scale`（NVFP4+FP8-KV 时是真实标定值）、MoE expert-map 全部清零；expert-map 在 SKIP_TENSORS (reload/meta.py:25-32)，即使随后对 draft 做权重重传也不会恢复。draft 无 mamba 层，故无 conv_weights/A 问题；draft MoE 走同一 trtllm nvfp4 路径，特殊发现 1 的 R5 风险在 draft 上同样存在且更无人管。
4. **主模型 mamba 层是本模型的独特状态面，但保护充分**: `conv_weights` 别名 buffer 与 `-exp` loader 变换都被 P2/P5 + reload 的 alias 处理覆盖；`_decode_state_offsets` (R2, arange) 在 named_buffers 中由 P2 恢复。唯一注意点：`_ssd_kernels_warmed_up` 等 python 标志与 host 端状态不受影响。
5. 非门控 MoE 使 `gemm1_alpha/_beta/_clamp` 全部为 None，`g1_scale_c = a2_gscale.clone()` (trtllm_nvfp4_moe.py:129-132) —— clone 出的张量注册为参数受 copy-back 保护，但其源 `a2_gscale` 本身即特殊发现 1 的孤儿张量。

## 结论

该模型在"sleep-L2 + RL reload + CUDA graph 不重捕"组合下**今日会静默腐坏**，根因即已知 HIGH 模式的 NVFP4 变体：`ModelOptNvFp4FusedMoE.process_weights_after_loading` 无条件重建 kernel（modelopt.py:1604-1615），quant config 中 `a1_gscale/a2_gscale` 等计算张量成为图捕获孤儿，sleep-L2 后旧地址内容清零。次级风险：(a) NVFP4 的三处非幂等 PWAL（R3c）使任何绕过 layerwise rematerialization 的重载路径必然锁死或崩溃；(b) MTP draft 完全无 P2 保护，expert-map 类 buffer 清零后连 draft 重载都无法恢复。mamba 混合层的派生状态（conv view、-exp A、arange offsets）在主模型上保护完备，不构成新增风险。
