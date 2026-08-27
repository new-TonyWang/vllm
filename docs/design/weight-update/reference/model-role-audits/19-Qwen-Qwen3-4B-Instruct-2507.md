# 模型角色审计 #19 — Qwen/Qwen3-4B-Instruct-2507

基本信息:
- HF 排名: 19
- 量化变体: 无(bf16)
- draft: 无官方 draft;`SupportsEagle`/`SupportsEagle3` 声明(qwen3.py:272),社区 EAGLE3 头非官方
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)
- 架构参数: 与 Qwen3-4B 同构(36 层,hidden 2560,32/8 头,head_dim=128),`tie_word_embeddings=true`;差异点:原生长上下文 `max_position_embeddings=262144`,`rope_theta=5,000,000`(2507 版取消 thinking 混合模式,权重结构不变)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371 | R1 | P1/P3 | RESTORABLE | 低 |
| `lm_head`(tie → embed_tokens 别名) | qwen3.py:297-299;skip_prefixes qwen3.py:341 | R1(别名) | P3(单份) | RESTORABLE | 低 |
| `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | qwen3.py:105-120,qwen2.py:90-103;linear.py:200-211 | R1 | P3;PWAL 空操作(linear.py:214-218) | RESTORABLE | 低 |
| `q_norm/k_norm.weight`(per-head RMSNorm,dim=128) | qwen3.py:150-151;layernorm.py:63-65 | R1 | P3 | RESTORABLE | 低;极小张量,自定义同步易遗漏 |
| `input_layernorm/post_attention_layernorm/model.norm.weight` | qwen3.py:221-224;qwen2.py:390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(**262144×128,bf16 ≈ 64 MiB**) | rotary_embedding/base.py:59-63(persistent=False);max_position 传入 qwen3.py:122-127,201 | R2 + R6 | P2(gpu_worker.py:272-273,整份 `.cpu().clone()`) | RESTORABLE(理想 RECOMPUTE) | **中偏高**:本变体 cos_sin_cache 显著偏大,P2 每次 sleep 都往 CPU 拷贝 ~64 MiB 并常驻;reload-only 路径不重算 |
| `attn._q/_k/_v/_prob_scale` | attention/attention.py:127-130,184 | R2 | P2 + P4(attention.py:604-616) | RESTORABLE | 低 |
| `attn.q_range/k_range/v_range`(裸 tensor) | attention/attention.py:148-150 | R2 | 无 | RECOMPUTE | 中:wake 后清零;默认路径 benign;fp8 kv 动态 scale 时除零(attention.py:585-587) |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_float` | attention.py:140-145 | R2 | CPU 侧 | PRESERVE | 无 |

## 特殊发现

1. **长上下文放大 cos_sin_cache 成本**:`RotaryEmbeddingBase.__init__` 按 `max_position_embeddings` 预计算整表(base.py:58-63,94-103)。262144 位置 × head_dim 128 → 约 64 MiB bf16。这使"R2 用 P2 保存"从可忽略变成实际内存/带宽开销,是把 `cos_sin_cache` 终态改为 RECOMPUTE 的最有力论据(本审计各模型中最典型的案例)。
2. `_ROPE_DICT` 全局单例(rotary_embedding/__init__.py:30,83-84,383),R6:36 层共享一份大表,幸而 `named_buffers()` 去重后 P2 只存一份。
3. `q_range/k_range/v_range` 裸张量缺口:同系列通用,默认无害。
4. tie 词嵌入:同 4B 处理(qwen3.py:299,341)。
5. bf16 无量化 → 无 R3/R4/R5 状态。

## 结论

Qwen3-4B-Instruct-2507 的状态角色与 Qwen3-4B 完全同构,唯一实质差异是 262144 上下文使 `cos_sin_cache` 达到 ~64 MiB——今日靠 P2 整份 CPU 备份可用但浪费,重设计中应将其声明为 RECOMPUTE(纯 config 派生,base.py:94-103 可在 wake/reload 后重算),其余状态沿用 R1/P3 与 R2/P2+P4 结论。
