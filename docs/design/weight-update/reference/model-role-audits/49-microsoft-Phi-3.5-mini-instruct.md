# 模型角色审计 #49 — microsoft/Phi-3.5-mini-instruct

基本信息
- 审计日期: 2026-07-26,repo HEAD c7ce03bcbd
- 实现文件: `vllm/model_executor/models/phi3.py`(`Phi3ForCausalLM(LlamaForCausalLM)`,仅 18 行,全部结构继承 `vllm/model_executor/models/llama.py`;registry.py:191)
- 关键依赖层: `vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.py`(LongRoPE)
- 结构特点: longrope 缩放 —— **short/long 两组 rescale 因子**(4k 原生 / 131k 扩展),qkv/gate_up 在 checkpoint 已打包(phi3.py:11-18 packed_modules_mapping 恒等映射),tie_word_embeddings=False

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens / qkv_proj / o_proj / gate_up / down_proj / RMSNorm / lm_head | llama.py(标准 Llama 栈) | R1 | reload 重写 checkpoint 参数 | RESTORABLE | 低 |
| `long_short_cos_sin_cache`(**short 4k 段 + long 131k 段拼接为单 buffer**,persistent=False) | phi3_long_rope_scaled_rope.py:82-95 | R2(双因子 rope 缓存,含 short_mscale/long_mscale 缩放:111-123) | P2 `_sleep_saved_buffers` 备份全部 named_buffers 含非持久(gpu_worker.py:271-274)+ wake copy-back(gpu_worker.py:311-316) | RESTORABLE(可 RECOMPUTE) | 低 |
| `use_long_rope`(python bool,**init 时**由 `max_model_len > original_max_position_embeddings` 定死) | phi3_long_rope_scaled_rope.py:54-55 | R2(config 派生标量) | 非张量;决定 forward 用 short 段还是 long 段(索引偏移 +original_max_position,136-142 行) | PRESERVE(自动) | 无 |
| `short_mscale` / `long_mscale` / `short_factor` / `long_factor`(python float/list) | phi3_long_rope_scaled_rope.py:48-49,79-80 | R2 | 非张量 | PRESERVE(自动) | 无 |
| `logit_scale`(Phi-3.5 无该字段,默认 1.0) | llama.py:494-496 | R2(python 标量) | 非张量 | PRESERVE(自动) | 无 |
| `_ROPE_DICT` 中缓存的 rope 实例(所有层共享同一 Phi3LongRoPE 对象) | rotary_embedding/__init__.py:30,83-84,315-335,383 | R6(全局模块级缓存) | 仅 shutdown 清(gpu_model_runner.py:6497);对 sleep/reload 无影响(实例同时是每层子模块,buffer 被 P2 覆盖) | PRESERVE | 低(见特殊发现 3) |

## 特殊发现

1. **LongRoPE 没有 lazy 切换 —— 这是好消息**。审计前假设"短/长两组因子可能懒加载、长上下文时才建 long 缓存(R6 风险)";实际实现是 `__init__` **eager 计算两份缓存并 cat 成一个 buffer**(phi3_long_rope_scaled_rope.py:82-95),运行期只做索引偏移(136-143:`idx = positions + original_max_position_embeddings`),不存在任何运行期重建/替换路径。`use_long_rope` 在 init 时按 `get_current_vllm_config().model_config.max_model_len` 一次定死(54-55),之后不变。**结论:无 lazy-init rotary 缓存风险。**
2. **该类是普通 nn.Module 而非 CustomOp/RotaryEmbeddingBase 子类**,没有 base.py:105-131 那种 `_match_cos_sin_cache_dtype` 的惰性 buffer 再赋值路径 —— buffer 对象自 init 后地址稳定(在 weights pool 内),sleep-L2 discard 后由 P2 copy-back 恢复到同一 storage。比标准 rope 更"安分"。
3. 需要注意的唯一契约:`use_long_rope` 依赖 init 时的 max_model_len。若终态设计走"重算 R2"(RECOMPUTE)而非 P2 备份,重算函数拿到的 vllm_config 必须与 init 时一致(该值不随 sleep/reload 改变,目前满足);且 long 缓存高达 131072×rotary_dim,fp16 下约 24MB/实例(全层共享 1 份),P2 的 CPU 备份成本可接受但 RECOMPUTE 更优。
4. Phi-3.5-mini rotary_dim = head_dim(无 partial rotary);qkv/gate_up 在 checkpoint 已合并,load 无 shard 拼接变换,reload 幂等性无特殊点。无量化、无 PWAL 派生态、无 R4/R5。

## 结论

Phi-3.5-mini-instruct 今日 sleep-L2 无已知静默清零点:R1 全由 reload 覆盖,唯一的 R2 张量 `long_short_cos_sin_cache` 是非持久 buffer,P2 完整覆盖。重点排查的 LongRoPE"双因子 + 懒切换"假设被证伪 —— 实现是 init 期 eager 双缓存拼接 + 静态索引偏移,属于 R2 中最安全的形态。终态建议:该 buffer 声明 RECOMPUTE(重算入参:head/rotary_dim、max_position、original_max_position、base、两组 factor、两 mscale、dtype,全部来自 config),并把 "rope 类禁止 forward 期重建/再赋值 buffer" 作为 lint 规则的正面样板。
