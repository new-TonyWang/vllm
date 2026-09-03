# 模型角色审计 #14 — Qwen/Qwen2.5-3B-Instruct

基本信息: HF 排名 14(下载量榜)。bf16 无量化。tie_word_embeddings=true(HF config;3B 与 0.5B/1.5B 同为绑定词表)。官方无 draft model。vLLM 实现文件: `vllm/model_executor/models/qwen2.py`(Qwen2ForCausalLM)。36 层,hidden 2048,16 attn头/2 KV头,head_dim 128,标准 attention(无 MLA),RoPE theta 1e6。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371; vocab_parallel_embedding.py:49-58 | R1 | P3(reload copy-back) | RESTORABLE | 低 — wake 后必须 reload |
| `lm_head` — 模块别名 `self.lm_head = self.model.embed_tokens` | qwen2.py:462-464;load_weights skip_prefixes=`lm_head.` qwen2.py:503-508 | R1(共享存储,无独立权重) | P3(随 embed_tokens 拷回一次即可) | RESTORABLE | 低 |
| `qkv_proj.weight` / `qkv_proj.bias`(bias=True) | qwen2.py:156-164; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `o_proj.weight`、`gate_up_proj.weight`、`down_proj.weight` | qwen2.py:90-103,165-171; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `input_layernorm.weight` / `post_attention_layernorm.weight` / `model.norm.weight` | qwen2.py:284-287,390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(register_buffer, persistent=False) | rotary_embedding/base.py:58-63 | R2;R6:`_ROPE_DICT` 全局缓存,36 层共享单实例(rotary_embedding/__init__.py:30,83-84,383) | P2(gpu_worker.py:270-274,311-316,原地 copy 保地址) | RECOMPUTE(理想);今日靠 P2 | 中 — 单一保护点 |
| `attn._q_scale/_k_scale/_v_scale/_prob_scale`(register_buffer) | attention/attention.py:124-130,184 | R2(常数 1.0) | P2 + attention 层 PWAL 重置(attention.py:604-616;reload/layerwise.py:357)+ P7(量化 kv 才生效) | RECOMPUTE | 低 |
| `attn.q_range/k_range/v_range`(普通张量属性) | attention/attention.py:148-150 | R2 | **无**(P2 不覆盖非 buffer 属性) | RECOMPUTE(应声明) | 低(auto kv-cache 不读);`calculate_kv_scales` 开启时踩雷 |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_scale_float` | attention/attention.py:140-145 | R2(host 侧) | CPU 常驻 | PRESERVE | 无 |
| `attn.kv_cache` 占位符 | attention/attention.py:461-463 | R4 | P7 + kv 池独立 tag | SCRATCH | 无 |

无 R3(bf16 线性层 PWAL 在 GPU 为 no-op,linear.py:214-218)、无 R5、无 R7。

## 特殊发现
- **R6 全局缓存** `_ROPE_DICT`(rotary_embedding/__init__.py:30):共享 RotaryEmbedding 实例;P2 借 named_buffers 去重恢复一次,语义正确。
- **tied lm_head 别名**:同一 nn.Module 对象在模型树中出现两次;reload 的 LAYERWISE_INFO 用 WeakKeyDictionary 按对象去重(reload/layerwise.py:50-52),`_sleep_saved_buffers` 用 named_buffers(默认去重),两套机制都不会双写。
- **lazy dtype 迁移** `_match_cos_sin_cache_dtype`(base.py:105-131):运行期可能替换 buffer 对象,compile 中禁替换;当前无害。
- 幂等守卫/graph 捕获常量:无。

## 结论
今天在 sleep-L2 + reload 生命周期下**安全**,与 1.5B 完全同构(仅层数/宽度不同):P2 救 R2 buffer、P3 重写 R1、attention PWAL/P7 兜底 scale。层数更多(36 层)只是线性放大 P2 备份体积(每层 4 个标量 buffer + 一份共享 cos_sin_cache,可忽略)。无保护的 `q_range/k_range/v_range` 属潜伏项。终态契约落地后:cos_sin_cache 显式 RECOMPUTE,tied 别名应在契约里声明为"alias of embed_tokens"(避免恢复器把它当独立状态重复处理或漏判),散装 GPU 张量属性纳入声明范围。
