# 模型角色审计 #06 — Qwen/Qwen2.5-7B-Instruct

基本信息: HF 排名 06(下载量榜)。bf16 无量化。tie_word_embeddings=false(7B 及以上不绑定,lm_head 为独立 ParallelLMHead)。官方无 draft model(Qwen2.5 系列未发布配套 draft 模型)。vLLM 实现文件: `vllm/model_executor/models/qwen2.py`(Qwen2ForCausalLM)。28 层,hidden 3584,28 attn头/4 KV头,head_dim 128,标准 attention(无 MLA),RoPE theta 1e6。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371; vocab_parallel_embedding.py:49-58 | R1 | P3(reload copy-back) | RESTORABLE | 低 — wake 后必须 reload |
| `lm_head.weight`(独立 ParallelLMHead) | qwen2.py:466-471; vocab_parallel_embedding.py:505-553 | R1 | P3(untied,load_weights 不跳过 lm_head,qwen2.py:503-508) | RESTORABLE | 低 |
| `qkv_proj.weight` / `qkv_proj.bias`(bias=True) | qwen2.py:156-164; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `o_proj.weight`、`gate_up_proj.weight`、`down_proj.weight` | qwen2.py:90-103,165-171; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `input_layernorm.weight` / `post_attention_layernorm.weight` / `model.norm.weight` | qwen2.py:284-287,390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(register_buffer, persistent=False) | rotary_embedding/base.py:58-63 | R2;叠加 R6:`_ROPE_DICT` 全局缓存,28 层共享同一实例(rotary_embedding/__init__.py:30,83-84,383) | P2(gpu_worker.py:270-274 备份,311-316 原地 copy 回,地址不变 → graph 安全) | RECOMPUTE(理想);今日靠 P2 | 中 — 单一保护点,P2 被绕过(如自定义 wake 流程)即静默归零、位置编码全坏 |
| `attn._q_scale/_k_scale/_v_scale/_prob_scale`(register_buffer) | attention/attention.py:124-130,184 | R2(常数 1.0) | P2;reload 时 attention 层 PWAL 重置 1.0(attention.py:604-616;reload/layerwise.py:357);P7(仅量化 kv 生效,gpu_model_runner.py:976-991) | RECOMPUTE | 低 |
| `attn.q_range/k_range/v_range`(普通张量属性) | attention/attention.py:148-150 | R2(env 常数) | **无**(非 buffer 非参数,P2/P3 均不覆盖) | RECOMPUTE(应声明) | 低(auto kv-cache 下 calc_kv_scales 不执行);开启 `calculate_kv_scales` 后为踩雷点 |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_scale_float` | attention/attention.py:140-145 | R2(host 侧) | CPU 常驻 | PRESERVE | 无 |
| `attn.kv_cache` 占位符 | attention/attention.py:461-463 | R4 | P7 + kv_cache 独立池 | SCRATCH | 无 |

无 R3(GPU 上 `UnquantizedLinearMethod.process_weights_after_loading` 为 no-op,linear.py:214-218)、无 R5、无 R7。

## 特殊发现
- **R6 全局缓存** `_ROPE_DICT`(rotary_embedding/__init__.py:30):跨层(乃至同进程跨模型)共享 RotaryEmbedding;named_buffers 去重后 P2 只备份一次共享 cos_sin_cache,恢复语义正确。
- **lazy dtype 迁移**:`_match_cos_sin_cache_dtype`(base.py:105-131)可能在运行期以新张量替换 buffer(base.py:125-131),compile 路径内禁止替换 —— 轻微双态,当前无害。
- 幂等守卫/graph 捕获常量/prefill-decode 双态:无。bf16 线性层无任何 PWAL 变换,是"最干净"的生命周期形态。

## 结论
今天在 sleep-L2 + reload 生命周期下**安全**。拼图:P2(cos_sin_cache + attention scale buffers 整备份原地还原)、P3(全部 R1 参数 reload 重写,含独立 lm_head)、attention 层 PWAL/P7 兜底。与 1.5B 的唯一结构差异是 lm_head 独立存储:reload 必须实际提供 `lm_head.weight`(untied 检查点自带,无风险)。无保护的 `q_range/k_range/v_range` 在本配置下不被读取。终态契约会带来的改变:cos_sin_cache 从"靠 P2 备份"转为声明式 RECOMPUTE(重算而非备份),消除对整备份的强依赖;所有普通 GPU 张量属性(q_range 等)必须显式声明归属,否则契约审计应报未声明状态。
