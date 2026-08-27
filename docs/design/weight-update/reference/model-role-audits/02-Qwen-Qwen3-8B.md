# 模型角色审计 #02 — Qwen/Qwen3-8B

基本信息:
- HF 排名: 02
- 量化变体: 无(bf16)
- draft: 无官方 draft。`SupportsEagle`/`SupportsEagle3` 已声明(qwen3.py:272),EAGLE3 头仅有社区版本,非官方
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)
- 架构参数: 36 层,hidden 4096,32 Q 头 / 8 KV 头,head_dim=128,`tie_word_embeddings=false`(独立 `lm_head.weight` 在 checkpoint 中)

## 状态角色清单

与 Qwen3 dense 通用清单一致(详细推导见 01 报告,以下为完整独立表):

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371 | R1 | P1/P3 | RESTORABLE | 低 |
| `lm_head.weight`(独立 ParallelLMHead) | qwen3.py:301-306 | R1 | P3(checkpoint 含 lm_head.weight) | RESTORABLE | 低 |
| `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | qwen3.py:105-120,qwen2.py:90-103;linear.py:200-211 | R1 | P3;PWAL 空操作(linear.py:214-218) | RESTORABLE | 低 |
| `q_norm/k_norm.weight`(per-head RMSNorm,dim=128) | qwen3.py:150-151;layernorm.py:63-65 | R1 | P3 | RESTORABLE | 低;极小张量,自定义同步易遗漏 |
| `input_layernorm/post_attention_layernorm/model.norm.weight` | qwen3.py:221-224;qwen2.py:390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache` | rotary_embedding/base.py:59-63(persistent=False) | R2 + R6(_ROPE_DICT 单例共享 36 层) | P2(gpu_worker.py:272-273) | RESTORABLE(理想 RECOMPUTE) | 中:仅 P2 保护;reload-only 不重算 |
| `attn._q/_k/_v/_prob_scale`(registered buffer) | attention/attention.py:127-130,184 | R2 | P2 + P4(attention.py:604-616) | RESTORABLE | 低 |
| `attn.q_range/k_range/v_range`(裸 tensor 属性) | attention/attention.py:148-150 | R2 | 无;仅 PWAL 重跑重建 | RECOMPUTE | 中:wake 后清零;默认 bf16 kv 路径不读 → benign;fp8 kv + `calculate_kv_scales` 时 attention.py:585-587 除零 |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_float` | attention.py:140-145 | R2 | CPU 侧,池外 | PRESERVE | 无 |

## 特殊发现

1. `_ROPE_DICT`(rotary_embedding/__init__.py:30,83-84,383)全局缓存:全部层共享一个 `cos_sin_cache`,`named_buffers()` 去重后 P2 存一份即可;跨引擎实例共享是 R6 风险点。
2. `cos_sin_cache` 非持久 buffer,不进 `state_dict()`;P2 目前用 `named_buffers()` 能覆盖。若 P2 实现方式变化会漏。config 可重算(base.py:94-103),终态应 RECOMPUTE。
3. `q_range/k_range/v_range` 是今日唯一无保护 GPU 状态(默认无害),建议注册为 buffer 或声明 RECOMPUTE 并在 wake 钩子重建。
4. `tie_word_embeddings=false`:`lm_head.weight`(约 151936×4096 bf16 ≈ 1.2 GiB)是独立 R1 参数,reload 必须包含它;`load_weights` 不做 skip(qwen3.py:341 条件为 False)。
5. bf16 无量化 → 无 R3/R4/R5 状态;PWAL 除 attention scale 重置外为空操作。

## 结论

Qwen3-8B(bf16)全部大张量为 R1,P3 覆写即恢复;非参数状态只有 `cos_sin_cache`(P2)与 attention scale buffer(P2+P4),缺口仅 `q_range/k_range/v_range` 裸张量(默认路径无害)。与 0.6B 的差别仅是独立 lm_head(R1,常规处理)与更多层数;无本模型特有风险。
