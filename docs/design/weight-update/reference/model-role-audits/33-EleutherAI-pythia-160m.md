# 模型角色审计 #33 — EleutherAI/pythia-160m

基本信息
- 审计基线: vLLM main @ c7ce03bcbd
- 模型实现: `vllm/model_executor/models/gpt_neox.py`(318 行,GPTNeoXForCausalLM)
- 架构要点: RoPE(partial rotary,pythia rotary_pct=0.25)、parallel residual、LayerNorm、无 MoE、无量化特殊路径
- 官方审计草稿: 无(本模型没有官方 draft 可对照,本文为独立审计)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `embed_in`(VocabParallelEmbedding) | gpt_neox.py:208-211 | R1 | P3 copy-back | RESTORABLE | 无 |
| `query_key_value` / `dense` / `dense_h_to_4h` / `dense_4h_to_h` 权重与 bias | gpt_neox.py:75-89, 127-138 | R1,QKV **加载时需重排**(见特殊发现 1) | P3 copy-back(须为重排后形态) | RESTORABLE(带 load-transform 约束) | 低(仅当 reload 绕过 `load_weights`) |
| `input_layernorm` / `post_attention_layernorm` / `final_layer_norm` | gpt_neox.py:158-163, 219-221 | R1 | P3 copy-back | RESTORABLE | 无 |
| `embed_out`(ParallelLMHead;tie_word_embeddings 时共享 embed_in.weight) | gpt_neox.py:281-288 | R1(pythia 默认不 tie,独立参数) | P3 copy-back | RESTORABLE | 无 |
| **rotary `cos_sin_cache`**(partial rotary_dim = head_size × 0.25) | gpt_neox.py:91-95(get_rope);rotary_embedding/base.py:59-63 `register_buffer("cos_sin_cache", cache, persistent=False)` | **R2 config 派生 buffer**(本模型的代表性状态) | **P2** `_sleep_saved_buffers`:gpu_worker.py:270-275 sleep-L2 前 `named_buffers()` 全量 CPU 备份;311-316 wake 后 `buffer.data.copy_` 回填(persistent=False 不影响 named_buffers 可见性) | RESTORABLE(经 P2);亦可 RECOMPUTE(`_compute_cos_sin_cache` base.py:94-103 纯函数) | 无 |
| `cos_sin_cache_bf16`(仅 ROCm AITER 路径) | base.py:66-74 | R2 | P2(同为 named buffer) | RESTORABLE | 无 |
| `_ROPE_DICT` 全局 rope 实例缓存 | rotary_embedding/__init__.py:30, 83-84, 383 | R6 全局缓存 | 无直接保护;但缓存的实例即各层 `self.rotary_emb` 子模块,其 buffer 已被 P2 覆盖 | 现状可接受;重构时应明确其跨 model 实例存活语义 | 低 |
| `_match_cos_sin_cache_dtype` 对 buffer 的运行期替换 | base.py:105-131(`self.cos_sin_cache = cos_sin_cache.to(...)` base.py:130) | R2 buffer 的惰性 dtype/device 迁移 | 替换后仍是同名 buffer,P2 备份的是替换后张量 | RESTORABLE | 无(compile/cudagraph tracing 期间已显式跳过原地替换,base.py:126-128) |
| Attention `_q/_k/_v/_prob_scale` buffers(公共层) | attention/attention.py:127-130, 184 | R2 | P2 | RESTORABLE | 无 |
| HF checkpoint `attention.bias` / `attention.masked_bias` | gpt_neox.py:264-267(skip_substrs) | 不落地(未注册) | N/A | N/A | 无 |

## 特殊发现

1. **QKV load-time 重排**:`GPTNeoXModel._repack_qkv`(gpt_neox.py:250-262)把 HF 的 `(num_heads, 3, head_size)` 布局 view/transpose/reshape 成 vLLM 的 `(3, num_heads, head_size)` 布局。与 GPT-2 的 Conv1D 转置同类:GPU 参数与磁盘 checkpoint 不同构,且**重排后 shape 完全不变**——绕过 `load_weights` 的裸 copy_ 会产生静默数值错误(比 GPT-2 更隐蔽,连非方阵 shape 报错的机会都没有)。P3 备份必须取自 GPU 上已重排的参数。
2. **cos_sin_cache 是本模型的教科书 R2 案例**:创建于模型构造期(即 CuMem weights 池上下文内)→ sleep-L2 丢页归零;由于 `register_buffer(persistent=False)` 仍出现在 `named_buffers()`,现有 P2 备份/回填机制完整覆盖。验证过 base.py:63 与 gpu_worker.py:272-273 的链路无缝。若未来有人把它改成普通属性(如某些 rope 变体直接 `self.xxx = tensor`),将逃逸 P2——重构时建议为 R2 状态加 lint/注册约束。
3. **`_ROPE_DICT`(R6)导致跨层甚至跨模型共享同一 RotaryEmbedding 实例**:同一进程内二次建模型(相同 rope 参数)会复用旧实例及其旧 buffer。当前 sleep/reload 不重建模型所以无害;但 worker 内"卸载后重建模型"的路径会拿到指向已释放/已归零池页的 cache,属重构需登记的全局状态。
4. partial rotary(rotary_dim=16 for pythia-160m, head_size=64)只影响 cache 尺寸计算(get_rope `partial_rotary_factor`,rotary_embedding/__init__.py:70-73 附近),无额外状态。
5. 无 register_buffer 于模型文件本身、无 PWAL 自定义、无 kernel 常量、无全局缓存(除 R6 rope)。

## 结论

pythia-160m(gpt_neox)对今日 sleep-L2 无已知风险:唯一的 GPU 派生状态 `cos_sin_cache` 已被 P2 完整覆盖,其余全为 R1。审计价值在于两点通用教训:(a) `_repack_qkv` 证明存在 shape 不变的 load-transform,reload 必须经 `load_weights` 或以 GPU 侧转换后参数为备份源;(b) `_ROPE_DICT` 是被 P2 意外兜住的 R6 全局缓存,新框架应将其显式登记而非依赖巧合。建议作为 "R1 + R2(P2 覆盖)" 的标准回归用例,与 OPT(纯 R1)、gpt-oss(R4/R5 风险)构成梯度。
