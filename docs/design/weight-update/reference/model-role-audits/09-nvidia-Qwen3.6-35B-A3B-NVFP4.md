# 模型角色审计 #09 — nvidia/Qwen3.6-35B-A3B-NVFP4

基本信息
- 审计 HEAD: c7ce03bcbd
- **架构映射不确定性**: 仓库中不存在 Qwen3.6 架构(`grep -ri qwen3_6/qwen36` 于 `vllm/model_executor/models/registry.py` 无命中)。最接近的在树架构为 `qwen3_5.py`(`Qwen3_5MoeForCausalLM`,qwen3_5.py:381,继承 `Qwen3NextModel`/`Qwen3NextDecoderLayer` 的混合 GDN+全注意力结构)与 `qwen3_next.py`。Qwen3.6 大概率沿用 Qwen3.5 的 hybrid-GDN MoE 骨架,本报告按 **qwen3_5.py(骨架)+ ModelOptNvFp4(量化路径,确定无疑)** 审计;若 Qwen3.6 实为纯 MoE(非 hybrid),GDN 部分作废,其余结论不变。
- 量化: NVFP4 → `quantization/modelopt.py` `ModelOptNvFp4Config`/`ModelOptNvFp4FusedMoE`(modelopt.py:1017/1395)+ `ModelOptNvFp4LinearMethod`(modelopt.py:1115)。
- MoE 后端: Blackwell 上默认 `FLASHINFER_TRTLLM`(oracle/nvfp4.py:179-188),experts 实现 `trtllm_nvfp4_moe.py`。
- MTP: Qwen3.5 系列有官方 MTP(`Qwen3_5MTP`,registry.py:648-649,qwen3_5_mtp.py);qwen3_next_mtp.py:66-74 的注释证实 NVFP4 checkpoint + MTP 是真实部署形态(`mtp.fc` 在 NVFP4 ckpt 中为 BF16,强制不量化)。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens / lm_head / RMSNorm 权重 / layer_scale | qwen3_next.py:619,810; qwen3_5.py:175-178 | R1 | P3+P4 (reload) | RESTORABLE | 低(reload 覆盖) |
| GDN: `conv1d.weight`(init 时 unsqueeze(1))、`A_log`、`dt_bias`、`norm(RMSNormGated).weight` | qwen_gdn_linear_attn.py:390-396,439-450,459-466 | R1(fp32 A_log/dt_bias;自定义 sharded loader 425-433,449-450) | P3+P4 | RESTORABLE | 低;注意 unsqueeze 发生在 __init__ 而非 loader(record_metadata 在 init 之后,utils.py:64,故 restore 形状正确)。**无 decode_conv1d_weight 式 R3b 派生副本**(forward 全部用 `.view`,1245-1246,1519-1520) |
| GDN in_proj_qkvz / in_proj_ba / out_proj | qwen_gdn_linear_attn.py:403-419,468-476 | R1 | P3+P4;disable_tp 场景由 layerwise.py:420-421 `update_param_tp_status` 修复 | RESTORABLE | 低 |
| router `gate`(ReplicatedLinear, quant_config=None) | qwen3_next.py:140-146 | R1 | P3+P4 | RESTORABLE | 低 |
| rotary `cos_sin_cache`(non-persistent buffer,经 `_ROPE_DICT` 全局共享) | rotary_embedding/base.py:59-71; __init__.py:30,83-84,383 | R2 + R6 | P2(gpu_worker.py:270-279 保存全部 named_buffers) | RECOMPUTE(或 PRESERVE) | 低(P2 覆盖);fused_qk_rmsnorm_rope_gate 直接取该 buffer(qwen3_next.py:352-364),P2 的 copy_ 保地址,graph 安全 |
| `_expert_map` / `expert_mask`(+EPLB 路由表)registered buffers | routed_experts.py:235-236,243-245 | R2 | P2 + SKIP_TENSORS(reload/meta.py:25-32) | PRESERVE | 低 |
| MoE ckpt 参数: `w13_weight`(uint8 packed)、`w2_weight`、`w13/w2_weight_scale`(fp8)、`w13_weight_scale_2`(E,2)、`w13/w2_input_scale` | modelopt.py:1441-1555 | R1 | P3+P4(restore_metadata 为 ckpt 格式,utils.py:64) | RESTORABLE | 低 |
| PWAL 后 kernel 格式参数(convert+replace_parameter×8,无 prefer_copy) | modelopt.py:1572-1602 | R3a | P3 copy-back 回写原 storage(layerwise.py:445-461),graph 指针保持 | RESTORABLE | 低 |
| `w13_weight_scale_2.data.mul_(w13_input_scale)` 原位融合 | trtllm_nvfp4_moe.py:122-123 | R3b/非幂等 | 仅因 reload 先恢复 ckpt 值才安全;裸重跑 PWAL 会二次相乘 | RECOMPUTE(须以 ckpt 值为输入) | 中(流程耦合脆弱) |
| `g1_scale_c`、`gemm1_clamp_limit`、`gemm1_beta`、`gemm1_alpha` — experts PWAL 中 register_parameter 到 layer | trtllm_nvfp4_moe.py:127-170 | R5→已参数化 | P3 copy-back(在 kernel_tensors 里)| RESTORABLE | 低 — **这是已知 rebuild-orphan HIGH 模式的已修复形态**(注册为参数使 copy-back 保地址);Qwen(silu)下 gemm1_* 为 None,仅 g1_scale_c 生效 |
| `quant_config.a1_gscale = 1.0/a13_scale`、`a2_gscale = 1.0/a2_scale` — 派生的裸张量 | oracle/nvfp4.py:504-522(507-508) | **R5 未保护** | 无(非参数、非 buffer;PWAL 每次重建新张量) | RECOMPUTE + 需保地址 | **高** — 见特殊发现 1 |
| `self.moe_kernel = make_nvfp4_moe_kernel(...)` 无条件重建 | modelopt.py:1607-1614 | R3c 反面(无守卫) | 无 | RECOMPUTE | 高(与上一条组合成孤儿模式) |
| Linear NVFP4: PWAL 派生 `input_global_scale`、`weight_global_scale`、`alpha`、`input_global_scale_inv`(注册为 Parameter;del 原 input_scale/weight_scale_2) | modelopt.py:1207-1238 | R3a | P3 copy-back + P4 | RESTORABLE | 低 |
| flashinfer workspace / permute 缓存 | oracle/unquantized.py:338(局部 dict);workspace 经 current_workspace_manager(modular_kernel.py:1105) | R4 scratch | 不在 weights pool | SCRATCH | 低 |
| MTP(Qwen3_5MTP / Qwen3NextMTP)全部权重与 buffer | qwen3_next_mtp.py:43-239 | R1/R2(独立 draft 模型) | **无** — 见特殊发现 3 | RESTORABLE(需接入) | **高(若启用 MTP)** |

## 特殊发现

1. **a1_gscale/a2_gscale 孤儿(本报告最高风险)**。`make_nvfp4_moe_quant_config` 用 `1.0 / a13_scale` 现场生成新张量(oracle/nvfp4.py:507-508),既非 layer 参数也非 buffer。它在每个 forward 的输入量化中被使用(TRTLLM 静态全局 scale 路径),因此指针会被 CUDA graph 捕获。首次 PWAL 在权重加载上下文内执行 → 该张量落在 CuMem weights pool;sleep-L2 后页面被清零;reload 重跑 PWAL 只会创建**新的** quant_config/新张量(modelopt.py:1605-1614),旧张量无 copy-back 通道 → graph 回放读到全 0 的输入全局 scale,静默输出损坏。同文件的 g1_alphas/g2_alphas 走 `w13_weight_scale_2` 参数引用(oracle/nvfp4.py:500-506 注释明确此设计意图),说明修复模式已存在,只有 a1/a2_gscale 漏网。
2. **trtllm_nvfp4 常量已参数化 = 已知 HIGH 模式的修复样板**。trtllm_nvfp4_moe.py:124-170 把 g1_scale_c/gemm1_* 逐一 `register_parameter`,注释直言"Register on the layer so EPLB rearranges"——同时使其进入 reload 的 kernel_tensors copy-back(layerwise.py:452-453),graph 指针保持有效。对比 trtllm_fp8_moe.py:286-292 的 `_g1_alphas/_g1_scale_c` 仍是裸属性(FP8 per-tensor 路径未修复)。
3. **MTP 双重缺口**:(a) 主模型 `load_weights` skip `mtp.`(qwen3_next.py:884),`reload_weights` 只对 `self.get_model()` 生效(gpu_model_runner.py:5505-5530),draft 模型权重不随 RL reload 更新;(b) sleep-L2 的 P2 只备份主模型 `named_buffers`(gpu_worker.py:271-273),draft 模型 buffer 不在册(唯一例外是 dflash 的 `_build_fused_kv_buffers` 手写钩子 P6,gpu_worker.py:275-279,318-324)。MTP rotary 因 `_ROPE_DICT` 共享同一模块实例而**意外**受 P2 保护(R6 副作用),但 `mtp.fc`、MTP 层权重在 sleep-L2 后为零且不被 reload 触达。需运行时验证 drafter 是否有独立 reload 通道。
4. **非幂等 PWAL**:`w13_weight_scale_2[:, 0]` 索引(modelopt.py:1563-1570)要求 (E,2) ckpt 形状,PWAL 后变 (E,);加上 in-place `mul_`,PWAL 只有在"先恢复 ckpt 格式参数"后才可重跑——当前 reload 流程满足,但任何绕过 restore 的手动 PWAL 重跑(如 IPC 权重直写后触发)会崩溃或双倍缩放。

## 结论

- 架构未进树(Qwen3.6),day-0 支持需先落地模型文件;量化路径(ModelOptNvFp4FusedMoE + TrtLlmNvFp4Experts)是确定的,审计结论可直接复用。
- 参数/buffer 侧保护已闭环:R1 由 P3+P4,R2 由 P2+SKIP_TENSORS,trtllm 派生常量已参数化(P3)。
- **今日 sleep-L2 + RL reload 会损坏的点**:① `a1_gscale/a2_gscale` 派生裸张量(oracle/nvfp4.py:507-508)——建议照 g1_scale_c 模式注册为 layer 参数,或在 quant_config 重建时对旧张量做原位 copy;② `moe_kernel` 无条件重建(modelopt.py:1607)——建议改为 unquantized 方法的守卫模式(unquantized_fused_moe_method.py:175-199);③ 启用 MTP 时整个 draft 模型缺 P2/P3 覆盖。
- 终态声明汇总:MoE ckpt 参数 RESTORABLE;expert map/mask PRESERVE;kernel 格式转换 RECOMPUTE(以 ckpt 值为输入、结果回写原地址);a1/a2_gscale 应改为 RESTORABLE(参数化)或 RECOMPUTE+保地址;workspace SCRATCH。
