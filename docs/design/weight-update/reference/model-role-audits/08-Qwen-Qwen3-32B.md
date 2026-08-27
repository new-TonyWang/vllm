# 模型角色审计 #08 — Qwen/Qwen3-32B

基本信息:
- HF 排名: 08
- 量化变体: 无(bf16);AWQ 变体见 #38 报告
- draft: 无官方 draft。`SupportsEagle`/`SupportsEagle3` 声明于 qwen3.py:272;EAGLE3 头仅社区提供,非官方
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)
- 架构参数: 64 层,hidden 5120,64 Q 头 / 8 KV 头,head_dim=128,`tie_word_embeddings=false`;大尺寸下通常 TP≥2 部署(TP 切分逻辑 qwen3.py:85-101,对状态角色无影响)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371 | R1 | P1/P3 | RESTORABLE | 低 |
| `lm_head.weight`(独立) | qwen3.py:301-306 | R1 | P3 | RESTORABLE | 低 |
| `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | qwen3.py:105-120,qwen2.py:90-103;linear.py:200-211 | R1 | P3;PWAL 空操作(linear.py:214-218) | RESTORABLE | 低 |
| `q_norm/k_norm.weight`(per-head RMSNorm,dim=128) | qwen3.py:150-151;layernorm.py:63-65 | R1 | P3 | RESTORABLE | 低;64 层×2×128 元素,自定义同步易遗漏 |
| `input_layernorm/post_attention_layernorm/model.norm.weight` | qwen3.py:221-224;qwen2.py:390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache` | rotary_embedding/base.py:59-63(persistent=False) | R2 + R6(_ROPE_DICT 共享 64 层) | P2(gpu_worker.py:272-273) | RESTORABLE(理想 RECOMPUTE) | 中:仅 P2;reload-only 不重算 |
| `attn._q/_k/_v/_prob_scale` | attention/attention.py:127-130,184 | R2 | P2 + P4(attention.py:604-616) | RESTORABLE | 低 |
| `attn.q_range/k_range/v_range`(裸 tensor) | attention/attention.py:148-150 | R2 | 无 | RECOMPUTE | 中:wake 后清零;默认路径不读 → benign;fp8 kv 动态 scale 时除零(attention.py:585-587) |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_float` | attention.py:140-145 | R2 | CPU 侧 | PRESERVE | 无 |

## 特殊发现

1. `_ROPE_DICT` 全局单例(rotary_embedding/__init__.py:30,83-84,383):64 层共用一个 `cos_sin_cache`;P2 经 `named_buffers()` 去重保存一份,跨引擎实例共享是 R6 风险。
2. `cos_sin_cache` 非持久 buffer:当前 P2 可覆盖,state_dict 路线会漏;config 可重算(base.py:94-103)→ 终态 RECOMPUTE。
3. `q_range/k_range/v_range` 裸张量:今日唯一无保护 GPU 状态,默认无害。
4. TP 部署提示:reload 路径的参数 TP 状态由 `update_param_tp_status` 重新校准(model_loader/utils.py:119、reload/layerwise.py:417-421),Qwen3 dense 无 disable_tp/replicated 特例,常规。
5. bf16 无量化 → 无 R3/R4/R5 状态。

## 结论

Qwen3-32B(bf16)与其他 Qwen3 dense bf16 变体状态结构完全一致:R1 参数 + 两组 R2(cos_sin_cache、attention scales)+ 一个裸张量缺口(q_range 系)。规模差异(64 层、TP 部署)不引入新的状态类别;sleep-L2 + wake(P2)+ reload(P3/P4)即可完整恢复。
