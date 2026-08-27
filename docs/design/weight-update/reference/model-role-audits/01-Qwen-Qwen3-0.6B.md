# 模型角色审计 #01 — Qwen/Qwen3-0.6B

基本信息:
- HF 排名: 01
- 量化变体: 无(bf16);FP8 变体见 #42 报告
- draft: 无官方 draft/MTP 头。`Qwen3ForCausalLM` 声明 `SupportsEagle`/`SupportsEagle3`(qwen3.py:272),社区存在 EAGLE3 头(如 AngelSlim)但非官方
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 `vllm/model_executor/models/qwen2.py` 的 `Qwen2Model`/`Qwen2MLP`)
- 架构参数: 28 层,hidden 1024,16 Q 头 / 8 KV 头,head_dim=128(config 显式给出,非 hidden/heads),`tie_word_embeddings=true`

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | qwen2.py:366-371 | R1 | P1/P3 | RESTORABLE | 低:reload 覆写 |
| `lm_head`(tie → 与 embed_tokens 同对象) | qwen3.py:297-299;skip_prefixes qwen3.py:341 | R1(别名) | P3(按 embed_tokens 单份加载) | RESTORABLE | 低;注意自定义权重同步不要对同一存储写两次 |
| `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | qwen3.py:105-120,qwen2.py:90-103;linear.py:200-211 | R1 | P3;PWAL 在 CUDA 上为空操作(linear.py:214-218) | RESTORABLE | 低 |
| `q_norm.weight`/`k_norm.weight`(per-head RMSNorm,dim=head_dim=128) | qwen3.py:150-151;layernorm.py:63-65(nn.Parameter) | R1 | P3 | RESTORABLE | 低;张量极小(128 元素/层),自定义 RL 同步脚本容易遗漏 → 清零后注意力立即崩坏 |
| `input_layernorm/post_attention_layernorm/model.norm.weight` | qwen3.py:221-224;qwen2.py:390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache` | rotary_embedding/base.py:59-63(`register_buffer(..., persistent=False)`) | R2(叠加 R6,见特殊发现 1) | P2(gpu_worker.py:272-273 `named_buffers()` 含非持久 buffer) | RESTORABLE(理想终态 RECOMPUTE) | 中:唯一保护是 P2;reload-only 路径(不 wake 恢复 buffer)不会重算它 |
| `attn._q_scale/_k_scale/_v_scale/_prob_scale` | attention/attention.py:127-130,184(registered buffer) | R2 | P2 + P4(attention.py:604-616 PWAL 重置为 1.0) | RESTORABLE | 低 |
| `attn.q_range/k_range/v_range` | attention/attention.py:148-150(裸 tensor 属性,未注册) | R2(env 常量派生) | 无(P2 覆盖不到);仅 attention PWAL 重跑时重建 | RECOMPUTE | 中:sleep-L2 wake 后被清零且不恢复;默认路径(bf16 kv cache、`calculate_kv_scales=False`)不读它 → 无害;若启用 fp8 kv + 动态 scale 计算则 attention.py:585-587 除零 |
| `attn._k_scale_cpu/_v_scale_cpu`、`*_scale_float` | attention.py:140-145 | R2 | CPU 侧/Python float,不在 CuMem 池内 | PRESERVE | 无 |
| `LogitsProcessor`/`SiluAndMul` | logits_processor.py;activation.py | — | 无张量状态 | — | 无 |

## 特殊发现

1. **`_ROPE_DICT` 全局单例(R6)**:`get_rope` 用模块级 dict 缓存 RotaryEmbedding 实例(rotary_embedding/__init__.py:30,83-84,383),全部 28 层共享同一个 `cos_sin_cache`。`named_buffers()` 默认去重,P2 只存/恢复一份,行为正确;但同进程再建第二个引擎实例(相同 rope 配置)会拿到同一对象——一个模型 sleep-L2 清零会连带影响另一个。
2. **`cos_sin_cache` 是非持久 buffer**:不进 `state_dict()`,当前 P2 用 `named_buffers()` 所以能存到;若未来把 P2 改成基于 `state_dict()` 的实现会静默漏掉它。终态建议声明为 RECOMPUTE(纯 config 派生,`_compute_cos_sin_cache` base.py:94-103 可重算)。
3. **运行时 buffer 替换路径**:`_match_cos_sin_cache_dtype`(base.py:105-131)在 dtype/device 不匹配时会用新张量替换 `self.cos_sin_cache`(在池外分配)。Qwen3 bf16 初始化即匹配,正常不触发;但这是一个潜在的"池内→池外漂移"入口。
4. **`q_range/k_range/v_range` 裸张量缺口**:见清单;这是本模型今日唯一真正无保护的 GPU 端状态,默认配置下 benign。
5. **tie_word_embeddings**:0.6B 绑定词嵌入,`lm_head` 与 `embed_tokens` 是同一 Parameter(qwen3.py:299),`load_weights` 跳过 `lm_head.`(qwen3.py:341)。P2/P3/reload 元数据(model_loader/utils.py:64 在 init 后立即 `record_metadata_for_reloading`)都按单份存储处理,无双写问题。
6. **PWAL 近似空操作**:bf16 无量化路径,`UnquantizedLinearMethod.process_weights_after_loading` 仅 CPU 平台有动作(linear.py:214-218);Attention 层 PWAL 重置默认 scale(attention.py:604-616)。因此 R3 类状态在本变体不存在。

## 结论

Qwen3-0.6B(bf16)对 sleep-L2 + reload 生命周期基本安全:所有大张量均为 R1 checkpoint 参数,由 P3 覆写恢复;非参数 GPU 状态只有 `cos_sin_cache`(R2,靠 P2)、attention scale buffers(R2,P2+P4)和 `q_range/k_range/v_range` 裸张量(唯一缺口,默认路径无害)。重设计时建议:`cos_sin_cache` 声明 RECOMPUTE(config 可重算),attention scale 声明 RESTORABLE,`q_range` 等纳入注册 buffer 或 PWAL 强制重建;并注意 `_ROPE_DICT` 的跨实例共享语义。
