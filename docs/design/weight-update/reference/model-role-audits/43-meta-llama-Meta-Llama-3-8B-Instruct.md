# 模型角色审计 #43 — meta-llama/Meta-Llama-3-8B-Instruct

基本信息:
- 架构: `LlamaForCausalLM` → `vllm/model_executor/models/llama.py`(registry.py:143);bf16,无量化
- 特点: `tie_word_embeddings=False`;**无 rope_scaling**(Llama-3 初代)→ 基类 `RotaryEmbedding`(base.py:139),theta=500000,max_position=8192;32 层,GQA 32/8
- 审计基线: HEAD c7ce03bcbd;sleep-L2 备份 gpu_worker.py:270-274,恢复 311-316

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | llama.py:376-380 | R1 | 权重更新/reload 重写 | RESTORABLE | 低 |
| `lm_head.weight`(独立,untied) | llama.py:484-490 | R1 | 权重更新重写 | RESTORABLE | 低(无 tie 问题,按名寻址直达) |
| 各层 `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | llama.py:162-178, 92-108 | R1 | 权重更新重写;UnquantizedLinearMethod PWAL GPU no-op(linear.py:214-218)→ reload 重跑 PWAL 幂等 | RESTORABLE | 低 |
| RMSNorm 权重 | llama.py:305-308, 388-389 | R1 | 同上 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(persistent=False buffer,8192×128) | base.py:58-63;llama.py:233-245 | **R2** | **P2**(named_buffers 备份/回填,gpu_worker.py:272-274, 311-316) | RECOMPUTE(理想) | 低:L2 归零 → P2 回填;8192 上限使备份体积可忽略 |
| `_ROPE_DICT` | rotary_embedding/__init__.py:30, 83-84, 383 | R6 | 无 | PRESERVE | 无;32 层共享单实例 |
| `attn._{k,v,q,prob}_scale` buffers | attention.py:124-130, 184 | R2/R3a | P2 | RESTORABLE-via-P2 | 低(bf16, auto kv,恒 1.0) |
| `attn._*_scale_float` / `_k_scale_cpu` / `_v_scale_cpu` | attention.py:140-145 | R2 | host 侧 | PRESERVE | 无 |
| `attn.{q_range,k_range,v_range}`(非 buffer 张量属性) | attention.py:148-150 | R2 | **无**(P2 盲区;张量建在 GPU weights pool) | RECOMPUTE | 低(仅 calculate_kv_scales 路径使用) |
| `attn.kv_cache` | attention.py:463 | R7 | post_kv_cache_wake_up(gpu_worker.py:326-329) | SCRATCH | 低 |
| `logits_processor` | llama.py:494-497 | — | host 标量 | PRESERVE | 无 |

## 特殊发现

1. 本模型是 #11(Llama-3.1-8B)的"无 rope_scaling、无 EAGLE draft"简化版:untied、纯 bf16、默认 rope。状态面即"Llama 核心集",没有任何模型特有增量。
2. 若该模型被配 EAGLE/EAGLE3 draft(社区存在 Llama-3-8B 的 EAGLE 权重),#11 报告的全部 draft 侧结论(draft buffer 不在 P2、`draft_id_to_target_id` 无恢复、独立 rope cache 归零)原样适用——draft 风险由 speculative 配置引入,与目标模型选型无关。
3. 复核 llama.py:183-201:config 无 `layer_types`,滑窗分支不触发;llama.py:277-280:`is_causal` 缺省 True,走 DECODER Attention,无 encoder-only 特例。

## 结论

今日 sleep-L2 安全,是 untied Llama 的标准形态:R1 → RESTORABLE(权重更新),cos_sin_cache/scale buffers → P2 覆盖,PWAL 幂等。无需模型特有处理;终态诉求与全系一致(rope cache 改 RECOMPUTE,消除对全量 CPU 备份的依赖)。
