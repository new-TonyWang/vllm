# 模型角色审计 #15 — Qwen/Qwen2.5-0.5B-Instruct

基本信息: HF 排名 15(下载量榜)。bf16 无量化,**tie_word_embeddings=true**(本审计批次中被点名的 tied 代表;实际 0.5B/1.5B/3B 均绑定)。官方无 draft model(0.5B 自身常被用作别家模型的 draft,但 Qwen2.5 官方未发布配套 draft)。vLLM 实现文件: `vllm/model_executor/models/qwen2.py`(Qwen2ForCausalLM)。24 层,hidden 896,14 attn头/2 KV头,**head_dim 64**,标准 attention(无 MLA),RoPE theta 1e6。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371; vocab_parallel_embedding.py:49-58 | R1 | P3(reload copy-back) | RESTORABLE | 低 — wake 后必须 reload |
| `lm_head` — 模块别名 `self.lm_head = self.model.embed_tokens` | qwen2.py:462-464;qwen2.py:503-508(load_weights skip_prefixes=`lm_head.`) | R1(共享 embed_tokens 存储;词表权重占全模型比例在 0.5B 上最高,约 1/3 参数) | P3 | RESTORABLE | 低 — 一次拷回同时修复 embedding 与 logits 两个用途 |
| `qkv_proj.weight` / `qkv_proj.bias`(bias=True) | qwen2.py:156-164; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `o_proj.weight`、`gate_up_proj.weight`、`down_proj.weight` | qwen2.py:90-103,165-171; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `input_layernorm.weight` / `post_attention_layernorm.weight` / `model.norm.weight` | qwen2.py:284-287,390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(register_buffer, persistent=False;head_dim=64 → 形状 [32768, 64]) | rotary_embedding/base.py:58-63 | R2;R6:`_ROPE_DICT` 全局缓存共享单实例(rotary_embedding/__init__.py:30,83-84,383) | P2(gpu_worker.py:270-274,311-316,原地 copy) | RECOMPUTE(理想);今日靠 P2 | 中 — 单一保护点 |
| `attn._q_scale/_k_scale/_v_scale/_prob_scale`(register_buffer) | attention/attention.py:124-130,184 | R2(常数 1.0) | P2 + attention 层 PWAL 重置(attention.py:604-616;reload/layerwise.py:357)+ P7(量化 kv 才生效,gpu_model_runner.py:976-991) | RECOMPUTE | 低 |
| `attn.q_range/k_range/v_range`(普通张量属性) | attention/attention.py:148-150 | R2 | **无** | RECOMPUTE(应声明) | 低(auto kv-cache);`calculate_kv_scales` 开启时踩雷 |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_scale_float` | attention/attention.py:140-145 | R2(host 侧) | CPU 常驻 | PRESERVE | 无 |
| `attn.kv_cache` 占位符 | attention/attention.py:461-463 | R4 | P7 + kv 池独立 tag | SCRATCH | 无 |

无 R3(GPU 上 UnquantizedLinearMethod PWAL 为 no-op,linear.py:214-218)、无 R5、无 R7。

## 特殊发现
- **tied 别名是本模型的核心特征**:`Qwen2ForCausalLM.__init__` 直接做模块级别名(qwen2.py:463-464),而非 `ParallelLMHead.tie_weights()`(vocab_parallel_embedding.py:555-557)那条参数级路径 —— 恢复器不能假设 lm_head 一定有自己的参数条目;`load_weights` 端配合 skip_prefixes 丢弃检查点里的 `lm_head.*`(qwen2.py:503-508)。
- **R6 全局缓存** `_ROPE_DICT`(rotary_embedding/__init__.py:30)。若同进程中 0.5B 作为其他 Qwen2 模型(head_dim 64 一致时)的 draft,可能与目标模型命中同一 RotaryEmbedding 实例 —— 共享状态跨模型,契约上值得显式登记。
- **lazy dtype 迁移** `_match_cos_sin_cache_dtype`(base.py:105-131):当前无害。
- 幂等守卫/graph 捕获常量/prefill-decode 双态:无。

## 结论
今天在 sleep-L2 + reload 生命周期下**安全**。三块拼图:P2(cos_sin_cache + scale buffers)、P3(R1 参数,tied 存储一次拷回)、attention PWAL/P7。0.5B 的特点是 tied 词表占比最大,使"reload 必须完整覆盖 embed_tokens"成为正确性最敏感的一步——任何按名字过滤 `lm_head.*` 的增量权重推送方(RL 训练侧)必须确保推送 `model.embed_tokens.weight`,否则 logits 头静默用零权重。终态契约的改变:把 lm_head 声明为别名(非独立状态)、cos_sin_cache 声明 RECOMPUTE、`q_range` 等散装张量纳入声明,即可让该模型的生命周期完全由契约推导,无需 P2 全量备份兜底。
