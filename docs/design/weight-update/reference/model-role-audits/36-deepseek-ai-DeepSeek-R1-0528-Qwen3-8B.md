# 模型角色审计 #36 — deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

基本信息:
- HF 排名: 36
- 量化变体: 无(bf16)
- draft: 无官方 draft。该模型是 DeepSeek-R1-0528 向 Qwen3-8B 的蒸馏版,**架构即 Qwen3ForCausalLM**,不含 DeepSeek 的 MTP/MLA 结构;`SupportsEagle`/`SupportsEagle3` 声明(qwen3.py:272),EAGLE3 头仅社区提供
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)——与 Qwen/Qwen3-8B 走完全相同的代码路径
- 架构参数: 同 Qwen3-8B:36 层,hidden 4096,32 Q 头 / 8 KV 头,head_dim=128,`tie_word_embeddings=false`(tokenizer/chat template 换成 DeepSeek 风格,不影响权重生命周期)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371 | R1 | P1/P3 | RESTORABLE | 低 |
| `lm_head.weight`(独立) | qwen3.py:301-306 | R1 | P3 | RESTORABLE | 低 |
| `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | qwen3.py:105-120,qwen2.py:90-103;linear.py:200-211 | R1 | P3;PWAL 空操作(linear.py:214-218) | RESTORABLE | 低 |
| `q_norm/k_norm.weight`(per-head RMSNorm,dim=128) | qwen3.py:150-151;layernorm.py:63-65 | R1 | P3 | RESTORABLE | 低;RL 蒸馏/微调场景自定义同步易遗漏这些小张量 |
| `input_layernorm/post_attention_layernorm/model.norm.weight` | qwen3.py:221-224;qwen2.py:390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache` | rotary_embedding/base.py:59-63(persistent=False) | R2 + R6(_ROPE_DICT 共享) | P2(gpu_worker.py:272-273) | RESTORABLE(理想 RECOMPUTE) | 中:仅 P2;reload-only 不重算 |
| `attn._q/_k/_v/_prob_scale` | attention/attention.py:127-130,184 | R2 | P2 + P4(attention.py:604-616) | RESTORABLE | 低 |
| `attn.q_range/k_range/v_range`(裸 tensor) | attention/attention.py:148-150 | R2 | 无 | RECOMPUTE | 中:wake 后清零;默认路径 benign;fp8 kv 动态 scale 时除零(attention.py:585-587) |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_float` | attention.py:140-145 | R2 | CPU 侧 | PRESERVE | 无 |

## 特殊发现

1. **"DeepSeek"只是权重内容,不是代码路径**:HF config `architectures=["Qwen3ForCausalLM"]`,vLLM 注册直接命中 qwen3.py;不存在 DeepSeek 系特有状态(无 MLA kv_a/kv_b、无 e_score_correction_bias、无 MTP 层)。审计其他 DeepSeek 模型的结论不适用于本模型,反之亦然。
2. 本模型是 RL/蒸馏社区最常用的 rollout 基座之一,重点提醒:自定义权重广播若按"linear 大矩阵"白名单同步,会漏掉 `q_norm/k_norm.weight`(每层仅 128 元素)与各 layernorm——sleep-L2 后这些被清零,模型输出立即全面崩坏且难定位。
3. `_ROPE_DICT` 全局单例(rotary_embedding/__init__.py:30,83-84,383)、`cos_sin_cache` 非持久 buffer、`q_range` 系裸张量缺口:与 Qwen3-8B 完全一致,详见 #02。
4. bf16 无量化 → 无 R3/R4/R5 状态。

## 结论

DeepSeek-R1-0528-Qwen3-8B 在权重生命周期意义上就是 Qwen3-8B:纯 R1 参数 + cos_sin_cache/attention-scale 两组 R2(P2 保护)+ `q_range` 系裸张量缺口。无 DeepSeek 特有状态;唯一值得单独强调的是它作为热门 RL 基座,外部权重同步必须覆盖 q_norm/k_norm 及 layernorm 等小 R1 张量。
