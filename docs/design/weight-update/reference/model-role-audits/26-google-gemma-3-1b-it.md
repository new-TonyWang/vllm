# 模型角色审计 #26 — google/gemma-3-1b-it

基本信息
- 审计日期: 2026-07-26,repo HEAD c7ce03bcbd
- 实现文件: `vllm/model_executor/models/gemma3.py`(`Gemma3ForCausalLM`,registry.py:107;1b-it 为纯文本模型,不走 gemma3_mm.py)
- 关键依赖层: `vllm/model_executor/layers/rotary_embedding/base.py`、`vllm/model_executor/layers/layernorm.py`(GemmaRMSNorm)
- 结构特点: tie_word_embeddings=True;sliding/full 交错注意力 → **每层一个 rotary,但全模型有两种 rope(local sliding θ=rope_local_base_freq + global θ)**;query_pre_attn_scalar;final_logit_softcapping

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens / qkv/o_proj / gate_up/down_proj 权重 | gemma3.py:313,132,141,73,80 | R1 | reload 重写 checkpoint 参数 | RESTORABLE | 低 |
| GemmaRMSNorm 权重(q_norm/k_norm/各层 4 个 LN/final norm) | gemma3.py:149-150,254-263,326 | R1(**无 loader 变换**,+1 偏移在 forward 时做:layernorm.py:157 `weight.float()+1.0`) | reload 重写 | RESTORABLE | 低 |
| lm_head(tied:`tie_weights` → `layer.weight = embed_tokens.weight`) | gemma3.py:411-412; vocab_parallel_embedding.py:80-84 | R1(参数别名) | named_parameters 去重;load_weights skip_prefixes=["lm_head."](gemma3.py:447);reload 重写 embed 即同步 | RESTORABLE | 低 |
| `model.normalizer`(sqrt(hidden_size),非持久 buffer) | gemma3.py:332-333 | R2 | P2 `_sleep_saved_buffers` 存全部 named_buffers(gpu_worker.py:271-274,**含非持久 buffer**)+ wake copy-back(gpu_worker.py:311-316) | RESTORABLE(可 RECOMPUTE) | 低 |
| `rotary_emb.cos_sin_cache` ×2 种(sliding 层:rope_type=default+rope_local_base_freq,gemma3.py:166-169;global 层:config.rope_parameters,gemma3.py:158-164) | base.py:58-63(register_buffer persistent=False) | R2 | P2 copy-back;注意 `_ROPE_DICT` 全局缓存使同 key 层共享同一 module 实例(rotary_embedding/__init__.py:30,83-84,383) | RESTORABLE(可 RECOMPUTE) | 低,但见特殊发现 1/2 |
| `scaling = query_pre_attn_scalar**-0.5` | gemma3.py:130 | R2(python 标量,烧进 Attention 构造) | 非张量 | PRESERVE(自动) | 无 |
| `final_logit_softcapping`(LogitsProcessor.soft_cap) | gemma3.py:414-415; logits_processor.py:55,76-79 | R2(python 标量) | 非张量 | PRESERVE(自动) | 无 |

## 特殊发现

1. **双 rope 是"两个内容不同的 cos_sin_cache buffer"**:sliding 层与 global 层 get_rope key 不同(base 频率不同)→ `_ROPE_DICT` 产生两个实例,分别挂在各层 `self_attn.rotary_emb` 下。P2 按 named_buffers 全量备份,两个都覆盖。1b-it 的 global rope 无 scaling(rope_scaling=null),用标准 `RotaryEmbedding`。
2. **cos_sin_cache 有一个惰性再赋值路径**:`_match_cos_sin_cache_dtype`(base.py:105-131)在 device/dtype 不匹配时执行 `self.cos_sin_cache = cache.to(...)`,把 buffer 替换成 forward 期分配的新 tensor(在 CuMem weights pool 之外)。正常路径 init 时已 `.to(dtype)`(base.py:61)且在目标设备上,不会触发;但一旦触发(如 flashinfer fp32 cache + bf16 query 组合),该 buffer 的 storage 就脱离 weights pool——内容仍由 P2 copy-back 恢复,不构成清零风险,只是地址寿命与池不一致,终态设计做地址断言时要留意。
3. **`_ROPE_DICT` 全局缓存(R6)只在 shutdown 清理**(gpu_model_runner.py:6497),sleep/reload 均不清。对本模型无害(rope 实例同时是模型子模块,P2 覆盖),但它意味着"同进程重建模型"会复用旧 rope 对象,是 R6 类的潜伏点,列入全局缓存清单。
4. **GemmaRMSNorm 的 +1 偏移不构成 R3b**:checkpoint 值原样入参,偏移在 forward 侧完成(layernorm.py:157),reload 重灌天然幂等。
5. 无 PWAL 派生参数(未量化)、无 R4/R5 状态;`attn_logits_soft_cap=None`(gemma3.py:243,Gemma3 文本层不用 attn softcap)。

## 结论

gemma-3-1b-it 状态面非常干净:R1 权重 + 3 类 R2(normalizer、两种 cos_sin_cache、若干 python 标量)。当前 P2(named_buffers CPU 备份 + wake copy-back)已完整覆盖所有 R2 张量,今日 sleep-L2 无已知静默清零点。终态建议:normalizer 与 cos_sin_cache 声明为 RECOMPUTE(纯 config 派生,可免去 CPU 备份开销);tied lm_head 依赖"参数别名 + named_parameters 去重"这一隐式契约,建议在终态显式声明别名关系。
