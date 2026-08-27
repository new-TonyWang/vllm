# 模型角色审计 #05 — Qwen/Qwen2.5-1.5B-Instruct

基本信息: HF 排名 05(下载量榜)。bf16 无量化。tie_word_embeddings=true(HF config;0.5B/1.5B/3B 均绑定,7B 及以上不绑定)。官方无 draft model(Qwen2.5 系列未发布配套 speculative/draft 模型)。vLLM 实现文件: `vllm/model_executor/models/qwen2.py`(Qwen2ForCausalLM,registry.py 中 "Qwen2ForCausalLM")。28 层,hidden 1536,12 attn头/2 KV头,head_dim 128,标准 attention(无 MLA),RoPE theta 1e6。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371; vocab_parallel_embedding.py:49-58 | R1 | P3(reload copy-back);P1(L1 时 offload) | RESTORABLE | 低 — wake 后必须走 reload,否则全零 |
| `qkv_proj.weight` / `qkv_proj.bias`(bias=True) | qwen2.py:156-164; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `o_proj.weight`、`gate_up_proj.weight`、`down_proj.weight` | qwen2.py:90-103,165-171; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `input_layernorm.weight` / `post_attention_layernorm.weight` / `model.norm.weight` | qwen2.py:284-287,390 | R1 | P3 | RESTORABLE | 低 |
| `lm_head` — **模块别名**:`self.lm_head = self.model.embed_tokens` | qwen2.py:462-464;load_weights skip_prefixes=`lm_head.` qwen2.py:503-508 | R1(无独立存储,共享 embed_tokens 权重) | P3(随 embed_tokens 一次拷回) | RESTORABLE | 低 — 别名共享同一 storage,reload 一次覆盖即两处生效 |
| `rotary_emb.cos_sin_cache`(register_buffer, persistent=False) | rotary_embedding/base.py:58-63 | R2(config 派生);叠加 R6:`_ROPE_DICT` 全局缓存使全部 28 层共享同一 RotaryEmbedding 实例(rotary_embedding/__init__.py:30,83-84,383) | P2(gpu_worker.py:270-274 备份 named_buffers,gpu_worker.py:311-316 原地 copy 回) | RECOMPUTE(理想契约);今日实际靠 P2 恢复 | 中 — **唯一保护是 P2**;named_buffers 去重后共享 buffer 只备份/恢复一次,原地 copy 保住 CUDA graph 捕获的地址 |
| `attn._q_scale/_k_scale/_v_scale/_prob_scale`(register_buffer) | attention/attention.py:124-130,184 | R2(bf16+auto kv-cache 下为常数 1.0) | P2;且 reload 时 Attention 层级 PWAL 重置为 1.0(attention.py:604-616;reload/layerwise.py:357);P7(post_kv_cache_wake_up,gpu_model_runner.py:976-991,量化 kv 才生效) | RECOMPUTE | 低 — 双重保护(P2+P4) |
| `attn.q_range/k_range/v_range`(普通张量属性,非 buffer) | attention/attention.py:148-150 | R2(env 常数派生) | **无** — 不在 named_buffers,P2 不覆盖;也不是参数 | RECOMPUTE(应声明) | 低(本模型 auto kv-cache 不走 calc_kv_scales,attention.py:585-587 不执行);若开 `calculate_kv_scales` 则 wake 后被零除线 — 潜在雷 |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_scale_float` | attention/attention.py:140-145 | R2(host 侧副本) | CPU/python 标量,不受 L2 影响 | PRESERVE | 无 |
| `attn.kv_cache` 占位符 → bind_kv_cache 后指向 KV 池 | attention/attention.py:461-463 | R4 | P7 + kv_cache 池独立 tag 管理 | SCRATCH | 无 |

无 R3(bf16 路径 `UnquantizedLinearMethod.process_weights_after_loading` 在 GPU 上是 no-op,linear.py:214-218)、无 R5、无 R7。

## 特殊发现
- **R6 全局缓存**:`_ROPE_DICT`(rotary_embedding/__init__.py:30)按 config 键缓存 RotaryEmbedding 实例,全模型 28 层共享一份 cos_sin_cache;若同进程再建同构模型也会命中同一实例。对 sleep/reload 无害(P2 原地恢复共享 storage),但属于"模块级全局 CUDA 张量缓存",终态契约需显式登记。
- **lazy dtype 迁移**:`_match_cos_sin_cache_dtype`(base.py:105-131)在 dtype/device 不匹配时会用新张量整体替换 `self.cos_sin_cache`(base.py:125-131,compile 中不替换);替换发生在 P2 备份之前则无影响(named_buffers 反映当前对象),属可容忍双态。
- **tie_word_embeddings 别名**:lm_head 与 embed_tokens 是同一 nn.Module 对象;reload 框架 LAYERWISE_INFO 按模块弱引用去重(reload/layerwise.py:50-52),不会重复处理。
- 幂等守卫/prefill-decode 双态/graph 捕获常量:无。

## 结论
今天在 sleep-L2 + reload 生命周期下**安全**,前提是严格执行"wake(weights) → reload 权重 → wake(kv_cache)"顺序。依赖三块拼图:① P2 整备份 named_buffers(救 cos_sin_cache 和 4 个 attention scale buffer,并以原地 copy 保住 graph 地址);② P3 reload copy-back 重写全部 R1 checkpoint 参数(含 tied lm_head 别名);③ P7/attention 层 PWAL 兜底 scale 重置。唯一无保护状态是 `q_range/k_range/v_range` 普通 GPU 张量属性,本模型配置下不被读取,属潜伏项。终态契约落地后:cos_sin_cache 声明 RECOMPUTE(config 派生可重算)即可让 P2 从"全量兜底备份"退化为可选优化;`q_range` 等散装张量必须被纳入 RECOMPUTE 声明,否则任何开启 kv-scale 动态计算的组合都会踩零值雷。
