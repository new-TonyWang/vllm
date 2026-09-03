# 模型角色审计 #07 — meta-llama/Llama-3.2-1B-Instruct

基本信息:
- 架构: `LlamaForCausalLM` → `vllm/model_executor/models/llama.py`(registry.py:143)
- 精度: bf16,无量化(UnquantizedLinearMethod)
- 特点: `tie_word_embeddings=True`;rope_scaling type=llama3 → `Llama3RotaryEmbedding`(llama3_rope.py:11)
- 审计基线: HEAD c7ce03bcbd;sleep-L2 备份路径 gpu_worker.py:270-274(named_buffers → CPU),恢复 gpu_worker.py:311-316(copy-back)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | llama.py:376-380 | R1 | 权重更新/reload 重写 | RESTORABLE | 低(wake 后必经权重重载) |
| `lm_head.weight`(tied,与 embed_tokens.weight 同一 Parameter 对象) | llama.py:491-492 → vocab_parallel_embedding.py:555-557 → 80-84(`layer.weight = embed_tokens.weight`) | R1(别名) | 加载时 skip `lm_head.`(llama.py:536-539) | RESTORABLE(经 embed_tokens) | 中:reload/权重更新若按名字发送 `lm_head.weight` 会 miss(named_parameters 去重后只剩 embed_tokens 名);若任何路径**替换** embed_tokens 的 Parameter 对象而非原地写,tie 断裂 → lm_head 保持旧/脏张量 |
| `layers.*.self_attn.{qkv_proj,o_proj}.weight` | llama.py:162-178 | R1 | 权重更新重写;PWAL(UnquantizedLinearMethod)GPU 上为 no-op(linear.py:214-218,仅 CPU 分支有副作用) | RESTORABLE | 低;PWAL 幂等 |
| `layers.*.mlp.{gate_up_proj,down_proj}.weight` | llama.py:92-108 | R1 | 同上 | RESTORABLE | 低 |
| `layers.*.{input_layernorm,post_attention_layernorm}.weight`、`model.norm.weight` | llama.py:305-308, 388-389 | R1 | 权重更新重写 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(register_buffer, persistent=False) | rotary_embedding/base.py:58-63;Llama3 变体只改 `_compute_inv_freq`(llama3_rope.py:33-54) | **R2** | **P2**:在 named_buffers 中(persistent=False 不影响 named_buffers)→ 备份/回填 | RECOMPUTE(理想);今日实际 RESTORABLE-via-P2 | 中低:cache 在 weights pool 中分配(load 期,gpu_worker.py:506-513),L2 归零;当前只靠 P2 救。reload 路径不重建该 buffer(无 P4 等价物);若 P2 失效(如运行期 `_match_cos_sin_cache_dtype` base.py:105-131 换出 buffer 对象——bf16+同设备下不触发)则静默错 rope |
| `_ROPE_DICT` 全局字典 | rotary_embedding/__init__.py:30, 83-84, 383 | **R6** | 无 | PRESERVE(host 对象) | 无直接风险;副作用:所有 16 层共享同一 rope 模块实例,named_buffers 去重后 cache 只出现一次,P2 备份/回填对等价类一次生效即可 |
| `attn.{_k_scale,_v_scale,_q_scale,_prob_scale}`(registered buffer) | attention/attention.py:124-130, 184 | R2/R3a | **P2** | RESTORABLE-via-P2(值恒 1.0,亦可 RECOMPUTE) | 低(bf16 + kv_cache_dtype=auto,值未参与计算) |
| `attn._{k,v,q,prob}_scale_float`、`_k_scale_cpu/_v_scale_cpu` | attention.py:140-145 | R2 | host/CPU 侧,天然幸存 | PRESERVE | 无 |
| `attn.{q_range,k_range,v_range}`(普通 tensor 属性,非 buffer) | attention.py:148-150 | R2 | **无**(不在 named_buffers → P2 覆盖不到;模型在 target_device 上下文里构建,张量落 GPU weights pool) | RECOMPUTE | 低但属守护空洞:仅 `calculate_kv_scales` 路径使用;bf16 默认不启用。若开启 fp8 kv + 动态 scale,wake 后为 0 → 静默错 scale |
| `attn.kv_cache`(占位 → 绑定 KV pool) | attention.py:463 | R7 | wake 后 `post_kv_cache_wake_up`(gpu_worker.py:326-329) | SCRATCH/外部 | 低(有专门重绑) |
| `logits_processor`(scale=1.0) | llama.py:494-497 | — | 纯 host 标量 | PRESERVE | 无 |

## 特殊发现

1. **tie 依赖对象同一性**:`ParallelLMHead.tie_weights` 直接 `layer.weight = embed_tokens.weight`(vocab_parallel_embedding.py:80-84)。sleep-L2 本身不破坏 tie(VA 不变、对象不变),但任何 reload 实现若对 embed_tokens 做 "新建 Parameter 再赋值" 而非 `param.data.copy_()`,lm_head 会静默指向旧存储。终态设计需把 tie 关系声明为不变量(P6 钩子校验或 reload 后重新 tie)。
2. **cos_sin_cache 无重算路径**:整个 rope 家族只在 `__init__` 里算一次 cache(base.py:58-63);PWAL 不涉及。今天 L2 的唯一保护是 worker 级 P2 全量 CPU 备份。1B 模型 cache 小(131072×64×2×2B ≈ 33MB,bf16),备份成本可接受,但终态应声明 RECOMPUTE(config 派生、确定性)而非常驻备份。
3. **P2 的覆盖边界=named_buffers**:`q_range/k_range/v_range` 这类普通张量属性是系统性盲区(attention.py:148-150),本模型 bf16 下无害,但同一 llama.py 路径换 fp8 kv-cache 即踩雷。

## 结论

Llama-3.2-1B(bf16, tied)是最"干净"的目标模型形态:R1 全部 RESTORABLE(靠权重更新),唯一非参数 GPU 状态是 rope cos_sin_cache 与 attention scale buffers,均在 named_buffers 内、被 P2 完整覆盖。今日 sleep-L2 无已知静默腐化;主要终态诉求:(a) cos_sin_cache 改声明 RECOMPUTE,摆脱全量 CPU 备份;(b) tied lm_head 的对象同一性需要 reload 侧不变量保护。
