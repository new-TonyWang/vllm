# 模型角色审计 #35 — HuggingFaceTB/SmolLM2-135M-Instruct

基本信息:
- 架构: `LlamaForCausalLM` → `vllm/model_executor/models/llama.py`(registry.py:143);bf16,无量化
- 特点: `tie_word_embeddings=True`(与 #07 同型);GQA(9 heads / 3 kv heads,head_dim=64);rope 默认型(无 rope_scaling)→ 基类 `RotaryEmbedding`(base.py:139),max_position=8192
- 30 层,hidden=576(超小模型)
- 审计基线: HEAD c7ce03bcbd;sleep-L2 备份 gpu_worker.py:270-274,恢复 311-316

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | llama.py:376-380 | R1 | 权重更新/reload 重写 | RESTORABLE | 低 |
| `lm_head.weight`(tied → 同一 Parameter 对象) | llama.py:491-492 → vocab_parallel_embedding.py:555-557, 80-84 | R1(别名) | 加载 skip `lm_head.`(llama.py:536-539) | RESTORABLE(经 embed_tokens) | 中:与 #07 相同的 tie 脆弱性——权重更新按名寻址会 miss `lm_head.weight`;替换(而非原地写)embed_tokens Parameter 即断 tie。135M 模型 embed 占比极高(~28% 参数),tie 正确性对该模型尤其关键 |
| 各层 `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | llama.py:162-178, 92-108 | R1 | 权重更新重写;PWAL GPU no-op(linear.py:214-218) | RESTORABLE | 低 |
| RMSNorm 权重 | llama.py:305-308, 388-389 | R1 | 同上 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(persistent=False buffer,8192×64) | base.py:58-63;llama.py:233-245 | **R2** | **P2**(named_buffers 备份/回填,gpu_worker.py:272-274, 311-316) | RECOMPUTE(理想) | 低:L2 归零后由 P2 回填;无重算路径是全系共性 |
| `_ROPE_DICT` | rotary_embedding/__init__.py:30, 83-84, 383 | R6 | 无 | PRESERVE | 无;30 层共享单实例,P2 一次覆盖 |
| `attn._{k,v,q,prob}_scale` buffers | attention.py:124-130, 184 | R2/R3a | P2 | RESTORABLE-via-P2 | 低(bf16, auto kv) |
| `attn._*_scale_float` / `_k_scale_cpu` / `_v_scale_cpu` | attention.py:140-145 | R2 | host 侧 | PRESERVE | 无 |
| `attn.{q_range,k_range,v_range}`(非 buffer 张量属性) | attention.py:148-150 | R2 | **无**(P2 盲区) | RECOMPUTE | 低(calculate_kv_scales 未启用) |
| `attn.kv_cache` | attention.py:463 | R7 | post_kv_cache_wake_up(gpu_worker.py:326-329) | SCRATCH | 低 |
| `logits_processor` | llama.py:494-497 | — | host 标量 | PRESERVE | 无 |

## 特殊发现

1. 状态面与 #07(Llama-3.2-1B)完全同构,仅 rope 类不同(默认 `RotaryEmbedding` 而非 `Llama3RotaryEmbedding`)——两者都只在 `_compute_inv_freq` 上有差异,cache 生命周期一致。
2. tied-embedding 在超小模型上放大 tie 断裂的后果:若 reload 替换了 embed_tokens 的 Parameter 对象,lm_head 静默持有旧存储,在 L2 之后旧存储是被丢页的 weights-pool VA,重新 wake 后内容为零 → logits 全零倾向而非"旧权重",症状更隐蔽(argmax 恒定)。
3. 无 GQA/rope/量化方面的特例;仔细核对了 llama.py:183-201 的 `layer_types` 滑窗分支——SmolLM2 config 无 `layer_types`/`sliding_window`,不触发。

## 结论

今日 sleep-L2 下安全,保护链 = P2(rope cache + scale buffers)+ 权重更新(R1)。与 #07 共享同一终态诉求:cos_sin_cache 声明 RECOMPUTE、tie 关系作为 reload 不变量显式校验。适合作为 tied-embedding 路径的最快回归模型(135M,秒级加载)。
