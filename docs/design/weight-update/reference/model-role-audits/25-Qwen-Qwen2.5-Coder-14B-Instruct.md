# 模型角色审计 #25 — Qwen/Qwen2.5-Coder-14B-Instruct

基本信息: HF 排名 25(下载量榜)。bf16 无量化。tie_word_embeddings=false。官方无 draft model(Coder 系列亦未发布配套 draft;社区常拿 Coder-0.5B/1.5B 自行组 speculative,非官方)。vLLM 实现文件: `vllm/model_executor/models/qwen2.py`(Qwen2ForCausalLM)—— 与 Qwen2.5-14B-Instruct 完全同一实现与超参骨架(48 层,hidden 5120,40/8 头,head_dim 128),仅训练数据/权重值不同,状态角色清单逐条一致。标准 attention(无 MLA),RoPE theta 1e6。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371; vocab_parallel_embedding.py:49-58 | R1 | P3(reload copy-back) | RESTORABLE | 低 — wake 后必须 reload |
| `lm_head.weight`(独立 ParallelLMHead) | qwen2.py:466-471; vocab_parallel_embedding.py:505-553 | R1 | P3 | RESTORABLE | 低 |
| `qkv_proj.weight` / `qkv_proj.bias`(bias=True) | qwen2.py:156-164; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `o_proj.weight`、`gate_up_proj.weight`、`down_proj.weight` | qwen2.py:90-103,165-171; linear.py:200-211 | R1 | P3 | RESTORABLE | 低 |
| `input_layernorm.weight` / `post_attention_layernorm.weight` / `model.norm.weight` | qwen2.py:284-287,390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(register_buffer, persistent=False) | rotary_embedding/base.py:58-63 | R2;R6:`_ROPE_DICT` 全局缓存共享单实例(rotary_embedding/__init__.py:30,83-84,383) | P2(gpu_worker.py:270-274,311-316,原地 copy) | RECOMPUTE(理想);今日靠 P2 | 中 — 单一保护点 |
| `attn._q_scale/_k_scale/_v_scale/_prob_scale`(register_buffer) | attention/attention.py:124-130,184 | R2(常数 1.0) | P2 + attention 层 PWAL 重置(attention.py:604-616;reload/layerwise.py:357)+ P7(量化 kv 才生效) | RECOMPUTE | 低 |
| `attn.q_range/k_range/v_range`(普通张量属性) | attention/attention.py:148-150 | R2 | **无** | RECOMPUTE(应声明) | 低(auto kv-cache);`calculate_kv_scales` 开启时踩雷 |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_scale_float` | attention/attention.py:140-145 | R2(host 侧) | CPU 常驻 | PRESERVE | 无 |
| `attn.kv_cache` 占位符 | attention/attention.py:461-463 | R4 | P7 + kv 池独立 tag | SCRATCH | 无 |

无 R3(GPU 上 UnquantizedLinearMethod PWAL 为 no-op,linear.py:214-218)、无 R5、无 R7。

## 特殊发现
- 与 #21 Qwen2.5-14B-Instruct 完全同构;Coder 变体不引入任何额外模块或状态(vLLM 侧按 `Qwen2ForCausalLM` 架构名路由,registry 不区分 Coder)。
- **R6 全局缓存** `_ROPE_DICT`(rotary_embedding/__init__.py:30)与 **lazy dtype 迁移**(base.py:105-131)同前。
- RL/agent 训练场景(Coder 常用于 SWE-RL)高频 sleep-L2→wake→reload 循环:reload 完整性门 `get_incomplete_layerwise_reload`(reload/layerwise.py:321-340)是防"部分推送权重后带脏状态服务"的关键闸门。
- 幂等守卫/graph 捕获常量/prefill-decode 双态:无。

## 结论
今天在 sleep-L2 + reload 生命周期下**安全**,结论与 #21 相同:P2 救 R2 buffer(cos_sin_cache 是唯一"只有一块拼图"的状态)、P3 重写全部 R1、attention PWAL/P7 兜底 scale。作为 RL day-0 高频目标,该模型的风险不在角色清单本身(极干净的 bf16 形态),而在生命周期编排:每轮权重更新必须完整覆盖所有 R1(含 lm_head 与各层 bias),部分更新时依赖框架的按层完整性核算。终态契约的改变:cos_sin_cache → RECOMPUTE、`q_range` 等散装张量纳入声明、完整性门变为契约级校验。
