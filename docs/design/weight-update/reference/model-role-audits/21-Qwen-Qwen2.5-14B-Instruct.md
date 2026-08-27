# 模型角色审计 #21 — Qwen/Qwen2.5-14B-Instruct

基本信息: HF 排名 21(下载量榜)。bf16 无量化。tie_word_embeddings=false(独立 lm_head)。官方无 draft model。vLLM 实现文件: `vllm/model_executor/models/qwen2.py`(Qwen2ForCausalLM)。48 层,hidden 5120,40 attn头/8 KV头,head_dim 128,标准 attention(无 MLA),RoPE theta 1e6。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371; vocab_parallel_embedding.py:49-58 | R1 | P3(reload copy-back) | RESTORABLE | 低 — wake 后必须 reload |
| `lm_head.weight`(独立 ParallelLMHead) | qwen2.py:466-471; vocab_parallel_embedding.py:505-553 | R1 | P3 | RESTORABLE | 低 |
| `qkv_proj.weight` / `qkv_proj.bias`(bias=True) | qwen2.py:156-164; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `o_proj.weight`、`gate_up_proj.weight`、`down_proj.weight` | qwen2.py:90-103,165-171; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `input_layernorm.weight` / `post_attention_layernorm.weight` / `model.norm.weight` | qwen2.py:284-287,390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(register_buffer, persistent=False) | rotary_embedding/base.py:58-63 | R2;R6:`_ROPE_DICT` 全局缓存,48 层共享单实例(rotary_embedding/__init__.py:30,83-84,383) | P2(gpu_worker.py:270-274 备份;311-316 原地 copy 回 → 地址稳定,graph 安全) | RECOMPUTE(理想);今日靠 P2 | 中 — 单一保护点 |
| `attn._q_scale/_k_scale/_v_scale/_prob_scale`(register_buffer,48 层 × 4) | attention/attention.py:124-130,184 | R2(常数 1.0) | P2 + attention 层 PWAL 重置(attention.py:604-616;reload/layerwise.py:357)+ P7(量化 kv 才生效,gpu_model_runner.py:976-991) | RECOMPUTE | 低 |
| `attn.q_range/k_range/v_range`(普通张量属性) | attention/attention.py:148-150 | R2(env 常数派生) | **无**(非 buffer 非参数) | RECOMPUTE(应声明) | 低(auto kv-cache 不读);`calculate_kv_scales` 开启时踩雷 |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_scale_float` | attention/attention.py:140-145 | R2(host 侧) | CPU 常驻 | PRESERVE | 无 |
| `attn.kv_cache` 占位符 | attention/attention.py:461-463 | R4 | P7 + kv 池独立 tag | SCRATCH | 无 |

无 R3(GPU 上 `UnquantizedLinearMethod.process_weights_after_loading` 为 no-op,linear.py:214-218)、无 R5、无 R7。

## 特殊发现
- **R6 全局缓存** `_ROPE_DICT`(rotary_embedding/__init__.py:30):48 层共享一份 cos_sin_cache buffer;P2 去重备份一次。
- **lazy dtype 迁移** `_match_cos_sin_cache_dtype`(base.py:105-131):运行期可替换 buffer 对象(compile 中禁替换),当前无害。
- 14B 常配 TP2 部署:TP 只改变各 rank 上 R1 分片大小,不引入新的状态角色;`update_param_tp_status` 在 reload PWAL 后由框架统一处理(reload/layerwise.py:417-421)。
- 幂等守卫/graph 捕获常量/prefill-decode 双态:无。

## 结论
今天在 sleep-L2 + reload 生命周期下**安全**,与 7B 报告同构:P2(R2 buffers)、P3(全部 R1 含独立 lm_head)、attention PWAL/P7 三块拼图。规模放大到 14B/TP2 不改变角色清单,只提高了"wake 后未 reload 就服务"这一误操作的代价(权重全零输出乱码)。无保护项仍是 `q_range/k_range/v_range`。终态契约的改变:cos_sin_cache → RECOMPUTE 声明、散装 GPU 张量属性纳入声明、R1 依赖 reload 完整性门(get_incomplete_layerwise_reload,reload/layerwise.py:321-340)从"日志警告"升级为"契约校验失败"。
