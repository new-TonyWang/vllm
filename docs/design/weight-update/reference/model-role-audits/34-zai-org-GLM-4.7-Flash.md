# 模型角色审计 #34 — zai-org/GLM-4.7-Flash

## 基本信息

- **实现文件**: `vllm/model_executor/models/glm4_moe_lite.py` — `Glm4MoeLiteForCausalLM` (registry.py:115, glm4_moe_lite.py:518)。lite 架构是"胶水层"：注意力直接复用 `DeepseekV2MLAAttention`/`DeepseekV2Attention` (glm4_moe_lite.py:95-100)，MoE 复用 glm4_moe.py 的 `Glm4MoE` (glm4_moe_lite.py:87-88)。
- **注意力**: `glm4_moe_lite` 在 `is_deepseek_mla` 列表 (transformers_utils/model_arch_config_convertor.py:263-271) → 有 `kv_lora_rank` 即走 MLA (glm4_moe_lite.py:133-136)。带 `is_v32` 分支：若 config 有 `index_topk` 则建 DSA `topk_indices_buffer` (glm4_moe_lite.py:224-234)——GLM-4.7-Flash 目前无 DSA，但代码路径存在，两分支均审计。若退化为 GQA 分支（无 kv_lora_rank）则用 `DeepseekV2Attention`（partial rope 见 deepseek_v2.py rope 配置）。
- **MoE**: `Glm4MoE`：plain `nn.Linear` fp32 gate (glm4_moe.py:141-146) + `e_score_correction_bias` (glm4_moe.py:147-149)，sigmoid 路由。
- **量化（官方变体）**: BF16 主发布 → `UnquantizedFusedMoEMethod`；官方 FP8 变体 → block-fp8 `Fp8MoEMethod`（同 #23 审计路径）。两者 PWAL 都无条件 `_setup_kernel` 重建 kernel（unquantized_fused_moe_method.py:155,260-266; fp8.py:761-763）。
- **MTP draft**: `Glm4MoeLiteMTPModel` → `glm4_moe_lite_mtp.py` (registry.py:642)。draft 的 `mtp_block` 是完整 `DeepseekV2DecoderLayer`（MLA + MoE，glm4_moe_lite_mtp.py:108-120 附近），并自建 `topk_indices_buffer` (glm4_moe_lite_mtp.py:99-106)。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| 全部 checkpoint 参数（qkv/o、MLA q/kv lora、MoE w13/w2、shared_experts、embed/lm_head/norm） | glm4_moe_lite.py / glm4_moe.py / deepseek_v2.py | R1 | reload 重写（layerwise 生命周期） | RESTORABLE | 低 |
| `gate.weight`（plain nn.Linear, fp32） | glm4_moe.py:141-146 | R1 | reload 重写 | RESTORABLE | 低 |
| `gate.e_score_correction_bias` (fp32 Param) | glm4_moe.py:147-149 | R1 | SKIP_TENSORS 原地加载 (reload/meta.py:31)；FusedMoE 持别名 (glm4_moe.py:202) | RESTORABLE | 低（别名依赖原地写；layerwise 对该名字不换对象，成立） |
| MLA `W_UV`/`W_UK_T`（kv_b_proj 吸收） | mla_attention.py:908-995 | R3a | P4 attention finalize 重跑 (reload/layerwise.py:343-357)；`prefer_copy=True` 保地址 (mla_attention.py:993-995) | RECOMPUTE | 低 |
| MLA rope `cos_sin_cache`（persistent=False buffer；GQA 分支则为 partial rope 同款） | rotary_embedding/base.py:59-63; 挂载 deepseek_v2.py:1059 / glm4_moe.py:283-288 | R2 (+R6 `_ROPE_DICT` 共享实例, rotary_embedding/__init__.py:30) | P2 (gpu_worker.py:272-274) | RESTORABLE | 低（主模型） |
| Attention `_q/_k/_v/_prob_scale` buffers | attention.py:124-135,184 | R2/R1' | P2 + reload PWAL 重置 (mla_attention.py:1005-1006) | RESTORABLE | 低（主模型） |
| `RoutedExperts` `_expert_map`/`expert_mask`/routing tables | routed_experts.py:235-245 | R2 | P2（主模型）；reload SKIP_TENSORS 跳过 (reload/meta.py:25-32) | PRESERVE | 低（主模型）；draft 上 **高** |
| BF16: `UnquantizedFusedMoEMethod.moe_kernel`（PWAL 重建） | unquantized_fused_moe_method.py:155-291 | R5 | P4 重建；权重张量本身是 layer 参数，copy-back 保地址 | RECOMPUTE | 低-中（BF16 quant config 无派生缩放张量，孤儿面小） |
| FP8 变体: `Fp8MoEMethod` PWAL 全链（kernel 重排 + `replace_parameter` 无 prefer_copy + kernel 重建） | fp8.py:674-763 | R3a+R5, **R3c**（重排非幂等） | P4+P3 copy-back（reload/layerwise.py:445-461） | RECOMPUTE | 中（同 #23：依赖完整 layerwise 生命周期；block 路径无 per-tensor `_g1_alphas` 孤儿） |
| `topk_indices_buffer`（仅 is_v32 config；plain attr） | glm4_moe_lite.py:224-234 | R4 | VA 保留、每步重写 | SCRATCH | 低 |
| MTP draft: `enorm`/`hnorm`/`eh_proj`/`shared_head`/`embed_tokens` | glm4_moe_lite_mtp.py:90-115,162 | R1 | draft 权重更新时可重写；`shared_weight_names=["embed_tokens"]` 与主模型共享 (glm4_moe_lite_mtp.py:447-455) | RESTORABLE | **高**（见特殊发现 2） |
| MTP draft: MLA W_UV/W_UK_T、attention scales、expert-map buffers、moe_kernel | 同主模型各行 | R3a/R2/R5 | **P2 不覆盖 draft** (gpu_worker.py:270-279) | RECOMPUTE/RESTORABLE | **高** |
| draft 自建 `topk_indices_buffer` | glm4_moe_lite_mtp.py:99-106 | R4 | 无需恢复 | SCRATCH | 低 |

## 特殊发现

1. **lite 架构没有自己的持久状态**：所有类都是 `pass` 子类 (glm4_moe_lite.py:83-100)，状态面 = deepseek_v2 MLA + glm4_moe MoE 的并集。审计结论可随上游两文件的结论联动；lite 特有的只有 `is_v32` 分支的 scratch buffer 和 decoder 组装逻辑 (glm4_moe_lite.py:103-177)。
2. **draft 与主模型共享 embed_tokens（R7 跨模型别名）**：`Glm4MoeLiteMTP.load_weights` 将 `embed_tokens` 声明为 shared (glm4_moe_lite_mtp.py:447)，draft 引用主模型的 Parameter 对象。layerwise reload 的 copy-back 保留原对象 → 别名安全；但任何"换对象不换存储"的自定义重载会让 draft 静默用旧 embedding。sleep-L2 下该共享反而有利：主模型 reload 修复的同时 draft 侧同步修复。其余 draft 独有状态（eh_proj/enorm/hnorm/shared_head.head + 整个 mtp_block）在 wake 后清零，仅当 RL 流程显式做 draft 权重更新时恢复；expert-map buffers 因 SKIP_TENSORS 连 draft 更新都无法恢复。
3. **glm4_moe.py 的 load_weights 有 P5 副作用**：`maybe_fuse_shared_experts` + `skip_spec_layers` (glm4_moe.py:490-497)。reload 走 `model.load_weights` 会重放；直写路径需自行处理 shared-expert 融合命名。lite 的 load_weights (glm4_moe_lite.py:346-370 附近) 同样跳过 `rotary_emb.inv_freq`。
4. **FP8 变体与 #23 同构**：block-128 fp8 下 trtllm/triton/cutlass 的 R5 常量均为受 copy-back 保护的 layer 参数，不触发 per-tensor `_g1_alphas`（trtllm_fp8_moe.py:286-291）已知 HIGH 模式；风险集中在 R3c 非幂等 kernel 重排上。
5. GQA 分支（若最终 config 无 kv_lora_rank）会走 `DeepseekV2Attention`+partial rope；此时无 W_UK_T/W_UV，状态面更小。当前证据（MLA 列表含 glm4_moe_lite）指向 MLA 分支为默认。

## 结论

GLM-4.7-Flash（BF16 主发布）是四个模型中今日风险最低的主模型：R1 全量可重载、R2 buffer 被 P2 覆盖、MLA R3a 由 attention finalize 重算且保地址、BF16 MoE 无派生缩放孤儿。**今日会腐坏的组合**：(a) 启用 MTP draft + sleep-L2 —— draft buffer 无备份、expert-map 永不恢复（与 #23 同根，HIGH）；(b) 若部署 FP8 官方变体，则继承 #23 的 R3c 非幂等 kernel 重排约束（必须走完整 layerwise 生命周期，禁止裸重跑 PWAL）。建议把 lite 审计与 deepseek_v2/glm4_moe 上游文件绑定跟踪，避免上游改动后 lite 结论过期。
