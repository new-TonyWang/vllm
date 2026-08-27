# 模型角色审计 #22 — Qwen/Qwen3-14B

基本信息:
- HF 排名: 22
- 量化变体: 无(bf16);AWQ 变体见 #32 报告
- draft: 无官方 draft;`SupportsEagle`/`SupportsEagle3` 声明(qwen3.py:272),社区 EAGLE3 头非官方
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)
- 架构参数: 40 层,hidden 5120,40 Q 头 / 8 KV 头,head_dim=128,`tie_word_embeddings=false`

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371 | R1 | P1/P3 | RESTORABLE | 低 |
| `lm_head.weight`(独立) | qwen3.py:301-306 | R1 | P3 | RESTORABLE | 低 |
| `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | qwen3.py:105-120,qwen2.py:90-103;linear.py:200-211 | R1 | P3;PWAL 空操作(linear.py:214-218) | RESTORABLE | 低 |
| `q_norm/k_norm.weight`(per-head RMSNorm,dim=128) | qwen3.py:150-151;layernorm.py:63-65 | R1 | P3 | RESTORABLE | 低;极小张量,自定义同步易遗漏 |
| `input_layernorm/post_attention_layernorm/model.norm.weight` | qwen3.py:221-224;qwen2.py:390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache` | rotary_embedding/base.py:59-63(persistent=False) | R2 + R6(_ROPE_DICT 共享 40 层) | P2(gpu_worker.py:272-273) | RESTORABLE(理想 RECOMPUTE) | 中:仅 P2;reload-only 不重算 |
| `attn._q/_k/_v/_prob_scale` | attention/attention.py:127-130,184 | R2 | P2 + P4(attention.py:604-616) | RESTORABLE | 低 |
| `attn.q_range/k_range/v_range`(裸 tensor) | attention/attention.py:148-150 | R2 | 无 | RECOMPUTE | 中:wake 后清零;默认路径 benign;fp8 kv 动态 scale 时除零(attention.py:585-587) |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_float` | attention.py:140-145 | R2 | CPU 侧 | PRESERVE | 无 |

## 特殊发现

1. `_ROPE_DICT` 全局单例(rotary_embedding/__init__.py:30,83-84,383),R6:跨引擎实例共享风险。
2. `cos_sin_cache` persistent=False:P2 经 `named_buffers()` 可存;state_dict 路线会漏;终态 RECOMPUTE。
3. `q_range/k_range/v_range` 裸张量缺口:默认无害。
4. 独立 `lm_head.weight`(151936×5120 bf16 ≈ 1.5 GiB)为 R1,reload 必须覆盖。
5. bf16 无量化 → 无 R3/R4/R5 状态。

## 结论

Qwen3-14B(bf16)状态结构与 8B/32B 一致:纯 R1 参数(含独立 lm_head)+ 两组 R2 buffer(P2 保护)+ `q_range` 系裸张量缺口(默认无害)。sleep-L2 + wake(P2)+ reload(P3/P4)完整可恢复,无本模型特有风险。
