# 模型角色审计 #31 — TinyLlama/TinyLlama-1.1B-Chat-v1.0

基本信息:
- 架构: `LlamaForCausalLM` → `vllm/model_executor/models/llama.py`(registry.py:143);bf16,无量化
- 特点: `tie_word_embeddings=False`;GQA(num_kv_heads=4, num_heads=32);rope 默认型(无 rope_scaling)→ 基类 `RotaryEmbedding`(base.py:139),theta=10000,max_position=2048 → cos_sin_cache 极小(2048×64×2)
- 22 层;`head_dim = hidden/heads = 64`
- 审计基线: HEAD c7ce03bcbd;sleep-L2 备份 gpu_worker.py:270-274,恢复 311-316

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | llama.py:376-380 | R1 | 权重更新/reload 重写;Embedding PWAL no-op(vocab_parallel_embedding.py:61) | RESTORABLE | 低 |
| `lm_head.weight`(独立) | llama.py:484-490 | R1 | 权重更新重写 | RESTORABLE | 低 |
| `layers.*.self_attn.{qkv_proj,o_proj}.weight`(GQA:kv_size=4×64/tp) | llama.py:144-178 | R1 | 权重更新重写;UnquantizedLinearMethod PWAL GPU no-op(linear.py:214-218)→ 幂等 | RESTORABLE | 低 |
| `layers.*.mlp.{gate_up_proj,down_proj}.weight` | llama.py:92-108 | R1 | 同上 | RESTORABLE | 低 |
| RMSNorm 权重(input/post_attention/model.norm) | llama.py:305-308, 388-389 | R1 | 同上 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(register_buffer, persistent=False) | base.py:58-63,由 llama.py:233-245 `get_rope` 创建 | **R2** | **P2**(named_buffers 备份 gpu_worker.py:272-274 / 回填 311-316) | RECOMPUTE(理想) | 低:cache 在 weights pool(load 上下文 gpu_worker.py:506-513)→ L2 归零,P2 回填;cache 仅 ~0.5MB,备份开销可忽略。运行期 `_match_cos_sin_cache_dtype`(base.py:105-131)在 bf16+同设备下不会换 buffer 对象 |
| `_ROPE_DICT`(全局,22 层共享同一 rope 实例) | rotary_embedding/__init__.py:30, 83-84, 383 | R6 | 无 | PRESERVE(host) | 无;named_buffers 去重后 P2 一次覆盖等价类 |
| `attn._{k,v,q,prob}_scale`(registered buffer) | attention.py:124-130, 184 | R2/R3a | P2 | RESTORABLE-via-P2 | 低(bf16 + kv auto,恒 1.0) |
| `attn._{k,v,q,prob}_scale_float` / `_k_scale_cpu` / `_v_scale_cpu` | attention.py:140-145 | R2 | host 侧幸存 | PRESERVE | 无 |
| `attn.{q_range,k_range,v_range}`(普通张量属性) | attention.py:148-150 | R2 | **无**(非 buffer,P2 盲区;构建于 GPU weights pool) | RECOMPUTE | 低(仅 calculate_kv_scales 用);系统性空洞记录在案 |
| `attn.kv_cache` | attention.py:463 | R7 | `post_kv_cache_wake_up`(gpu_worker.py:326-329) | SCRATCH | 低 |
| `logits_processor` | llama.py:494-497 | — | host 标量 | PRESERVE | 无 |

## 特殊发现

1. TinyLlama 是本批次里最小/最标准的 untied Llama:没有 llama3 rope、没有 tie、没有量化,状态面 = R1 参数 + rope cache + attention scale buffers,全部落在现有 P2/权重更新覆盖内。
2. GQA(4 kv heads)只改张量形状,不引入额外生命周期状态;TP>4 时走 kv 复制分支(llama.py:149-153),同样无新状态。
3. 与全系 Llama 相同的两个结构性提醒:(a) `q_range/k_range/v_range` 为非 buffer 张量属性,P2 覆盖不到(今日无害);(b) cos_sin_cache 无重算路径,依赖 P2 备份,终态应声明 RECOMPUTE。

## 结论

今日 sleep-L2 无已知风险:R1 全量 RESTORABLE(权重更新),R2 buffers 被 P2 完整覆盖,PWAL 为 no-op 因而 reload 重跑天然幂等。该模型可作为 weight-lifecycle 重构的"最小正确性基线"回归用例。
