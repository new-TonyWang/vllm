# 模型角色审计 #16 — Qwen/Qwen3-4B

基本信息:
- HF 排名: 16
- 量化变体: 无(bf16)
- draft: 无官方 draft;`SupportsEagle`/`SupportsEagle3` 声明(qwen3.py:272),社区 EAGLE3 头非官方
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)
- 架构参数: 36 层,hidden 2560,32 Q 头 / 8 KV 头,head_dim=128,`tie_word_embeddings=true`

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371 | R1 | P1/P3 | RESTORABLE | 低 |
| `lm_head`(tie → embed_tokens 别名) | qwen3.py:297-299;skip_prefixes qwen3.py:341 | R1(别名) | P3(单份) | RESTORABLE | 低 |
| `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | qwen3.py:105-120,qwen2.py:90-103;linear.py:200-211 | R1 | P3;PWAL 空操作(linear.py:214-218) | RESTORABLE | 低 |
| `q_norm/k_norm.weight`(per-head RMSNorm,dim=128) | qwen3.py:150-151;layernorm.py:63-65 | R1 | P3 | RESTORABLE | 低;极小张量,自定义同步易遗漏 |
| `input_layernorm/post_attention_layernorm/model.norm.weight` | qwen3.py:221-224;qwen2.py:390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache` | rotary_embedding/base.py:59-63(persistent=False) | R2 + R6(_ROPE_DICT 共享) | P2(gpu_worker.py:272-273) | RESTORABLE(理想 RECOMPUTE) | 中:仅 P2;reload-only 不重算 |
| `attn._q/_k/_v/_prob_scale` | attention/attention.py:127-130,184 | R2 | P2 + P4(attention.py:604-616) | RESTORABLE | 低 |
| `attn.q_range/k_range/v_range`(裸 tensor) | attention/attention.py:148-150 | R2 | 无 | RECOMPUTE | 中:wake 后清零;默认路径 benign;fp8 kv 动态 scale 时除零(attention.py:585-587) |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_float` | attention.py:140-145 | R2 | CPU 侧 | PRESERVE | 无 |

## 特殊发现

1. `_ROPE_DICT` 全局单例(rotary_embedding/__init__.py:30,83-84,383),R6:36 层共享一个 `cos_sin_cache`;跨引擎实例共享需注意。
2. `cos_sin_cache` persistent=False:P2 经 `named_buffers()` 可存;state_dict 路线会漏;终态 RECOMPUTE。
3. `q_range/k_range/v_range` 裸张量缺口:默认无害。
4. tie 词嵌入:同 0.6B/1.7B 处理,`load_weights` skip `lm_head.`(qwen3.py:341)。
5. bf16 无量化 → 无 R3/R4/R5 状态。

## 结论

Qwen3-4B(bf16)与 0.6B/1.7B 状态结构完全一致(tie 词嵌入、纯 R1 参数 + cos_sin_cache/attention-scale 两组 R2);sleep-L2 + P2/P3/P4 组合下无未覆盖的大张量,唯一缺口为 `q_range/k_range/v_range`(默认无害)。
