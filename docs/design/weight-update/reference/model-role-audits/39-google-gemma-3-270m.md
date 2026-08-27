# 模型角色审计 #39 — google/gemma-3-270m

基本信息
- 审计日期: 2026-07-26,repo HEAD c7ce03bcbd
- 实现文件: `vllm/model_executor/models/gemma3.py`(`Gemma3ForCausalLM`,registry.py:107)—— 与 #26(gemma-3-1b-it)同一实现类,本报告为 270m 独立结论,状态角色机制详版见 `26-google-gemma-3-1b-it.md`
- 结构特点: 极小模型(~270M,embedding 占比过半),tie_word_embeddings=True(embedding 复用对小模型尤其关键);sliding/full 交错 → 双 rope(local + global);GemmaRMSNorm +1 偏移;final_logit_softcapping

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens.weight(模型最大参数,兼任 lm_head) | gemma3.py:313-318,411-412 | R1(tied 别名) | reload 重写;load_weights skip_prefixes=["lm_head."](gemma3.py:447) | RESTORABLE | 低 |
| 各层 qkv/o_proj、gate_up/down_proj | gemma3.py:132,141,73,80 | R1 | reload 重写 | RESTORABLE | 低 |
| GemmaRMSNorm 全部权重(q/k_norm、层内 4 LN、final norm) | gemma3.py:149-150,254-263,326 | R1(+1 偏移在 forward,layernorm.py:157,无 loader 变换) | reload 重写 | RESTORABLE | 低 |
| `model.normalizer`(sqrt(hidden_size),非持久 buffer) | gemma3.py:332-333 | R2 | P2 named_buffers 备份(gpu_worker.py:271-274)+ wake copy-back(gpu_worker.py:311-316) | RESTORABLE(可 RECOMPUTE) | 低 |
| `cos_sin_cache` ×2(sliding θ=rope_local_base_freq;global θ=rope_theta) | gemma3.py:158-176; base.py:58-63(persistent=False) | R2 | P2 copy-back | RESTORABLE(可 RECOMPUTE) | 低 |
| `scaling`(query_pre_attn_scalar^-0.5)、`final_logit_softcapping` | gemma3.py:130,414-415 | R2(python 标量) | 非张量 | PRESERVE(自动) | 无 |

## 特殊发现

1. **tied embedding 在 270m 上是主导契约**:`lm_head = lm_head.tie_weights(embed_tokens)`(gemma3.py:411-412)使 lm_head.weight 与 embed_tokens.weight 为同一 tensor 对象(vocab_parallel_embedding.py:80-84)。sleep-L2 discard 后 reload 只重写 embed_tokens(checkpoint 也只有一份),别名自动同步;但任何"按 named_parameters 逐个替换 tensor 对象"的新 reload 设计都会打断该别名——终态必须保证 tied 权重要么就地写、要么写后重新 tie。
2. **双 rope**:270m 与 1b 同样是 sliding/full 交错,两个 `_ROPE_DICT` 实例、两份 cos_sin_cache buffer,P2 均覆盖;`_ROPE_DICT`(rotary_embedding/__init__.py:30)只在 shutdown 清(gpu_model_runner.py:6497),属 R6 潜伏类但对本模型无害(详见 #26 特殊发现 2/3,包括 `_match_cos_sin_cache_dtype` 的惰性再赋值路径 base.py:105-131)。
3. 无量化、无 PWAL 派生态、无 R4/R5;270m 的整个可变状态就是"一份 checkpoint 参数 + 3 个 config 派生 buffer + 若干 python 标量",是全审计集合里状态面最小的模型,适合作为终态声明机制(RESTORABLE/RECOMPUTE 標注)的最小验证样例。

## 结论

gemma-3-270m 今日 sleep-L2 无已知风险:R1 全部由 reload 覆盖,R2 buffer 全部由 P2 覆盖。终态建议与 #26 相同——normalizer/cos_sin_cache 声明 RECOMPUTE,tied embedding 别名关系显式声明;并推荐把该模型纳入 sleep-L2/reload 回归测试的最小冒烟集(加载快、覆盖 tied-embedding 与双 rope 两个通用契约)。
