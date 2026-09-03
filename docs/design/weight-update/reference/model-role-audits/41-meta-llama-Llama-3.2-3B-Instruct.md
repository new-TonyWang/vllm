# 模型角色审计 #41 — meta-llama/Llama-3.2-3B-Instruct

基本信息:
- 架构: `LlamaForCausalLM` → `vllm/model_executor/models/llama.py`(registry.py:143);bf16,无量化
- 特点: `tie_word_embeddings=True`;rope_scaling type=llama3(factor=32)→ `Llama3RotaryEmbedding`(llama3_rope.py:11-54);max_position=131072;28 层,GQA 24/8 heads
- 审计基线: HEAD c7ce03bcbd;sleep-L2 备份 gpu_worker.py:270-274,恢复 311-316

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | llama.py:376-380 | R1 | 权重更新/reload 重写 | RESTORABLE | 低 |
| `lm_head.weight`(tied,同一 Parameter) | llama.py:491-492 → vocab_parallel_embedding.py:555-557, 80-84 | R1(别名) | 加载 skip `lm_head.`(llama.py:536-539) | RESTORABLE(经 embed_tokens) | 中:tie 对象同一性依赖 reload 原地写;named_parameters 去重后无 `lm_head.weight` 名,按名寻址的权重更新协议须显式处理 |
| 各层 `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | llama.py:162-178, 92-108 | R1 | 权重更新重写;PWAL GPU no-op(linear.py:214-218) | RESTORABLE | 低 |
| RMSNorm 权重 | llama.py:305-308, 388-389 | R1 | 同上 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(persistent=False buffer,131072×128,bf16 ≈ 64MB 单实例、全 28 层共享) | base.py:58-63;Llama3 inv_freq 变换 llama3_rope.py:33-54 | **R2** | **P2**(named_buffers → CPU 备份 gpu_worker.py:272-274;wake 回填 311-316) | RECOMPUTE(理想) | 中低:L2 归零由 P2 回填;131072 长上下文使该 cache 是本模型 P2 备份的主要体积,`.cpu().clone()` 在 sleep 路径上是同步 D2H 拷贝——属于成本问题而非正确性问题 |
| `_ROPE_DICT` | rotary_embedding/__init__.py:30, 83-84, 383 | R6 | 无 | PRESERVE | 无;但注意其等价类语义(与 draft/多模型共享时决定 P2 是否顺带覆盖,详见 #11 审计) |
| `attn._{k,v,q,prob}_scale` buffers | attention.py:124-130, 184 | R2/R3a | P2 | RESTORABLE-via-P2 | 低(bf16, auto kv) |
| `attn._*_scale_float` / `_k_scale_cpu` / `_v_scale_cpu` | attention.py:140-145 | R2 | host 侧 | PRESERVE | 无 |
| `attn.{q_range,k_range,v_range}`(非 buffer 张量属性,GPU) | attention.py:148-150 | R2 | **无**(P2 盲区) | RECOMPUTE | 低(calculate_kv_scales 未启用时不参与计算) |
| `attn.kv_cache` | attention.py:463 | R7 | post_kv_cache_wake_up(gpu_worker.py:326-329) | SCRATCH | 低 |
| `logits_processor` | llama.py:494-497 | — | host 标量 | PRESERVE | 无 |

## 特殊发现

1. 与 #07 完全同型(tied + llama3 rope),只是规模更大;所有结论直接迁移。
2. **P2 成本随 max_position 放大**:cos_sin_cache 是按 `max_position_embeddings` 分配的(base.py:94-103),131072 上下文使单个 buffer 达数十 MB。当前 P2 是"无差别全量 CPU clone"(gpu_worker.py:272-274),对 3B 模型该 clone 的主要字节数其实来自这个**纯可重算**的 cache——是终态把 R2 声明为 RECOMPUTE(P4/P6 重建)而非 PRESERVE-via-backup 的量化论据。
3. `Llama3RotaryEmbedding` 与基类的差异仅在 `_compute_inv_freq`(llama3_rope.py:33-54),重算是纯 CPU/确定性的,重建钩子实现成本低。

## 结论

今日 sleep-L2 安全:R1 靠权重更新,R2 靠 P2。无模型特有风险;贡献的主要审计信号是 P2 备份体积问题(大 max_position 的 rope cache 占大头),支持终态将 cos_sin_cache 归类 RECOMPUTE。tied lm_head 的 reload 不变量要求与 #07/#35 一致。
