# 模型角色审计 #48 — mistralai/Mistral-7B-Instruct-v0.2

基本信息:
- 架构: `MistralForCausalLM` → **`vllm/model_executor/models/mistral.py`**(registry.py:168),不再直接映射到 llama.py,但实现整体继承 Llama:`MistralForCausalLM(LlamaForCausalLM)`(mistral.py:230)、`MistralModel(LlamaModel)`(mistral.py:207)、`MistralDecoderLayer(LlamaDecoderLayer)`(mistral.py:144)、`MistralAttention(LlamaAttention)`(mistral.py:77)
- bf16,无量化;`tie_word_embeddings=False`;rope 默认型(theta=1e6,无 rope_scaling)→ 基类 `RotaryEmbedding`(base.py:139);`max_position_embeddings=32768`
- **滑窗**: v0.2 起 `sliding_window=null` → 全局注意力。滑窗判定走 llama.py:182-201:仅当 config 有 `layer_types` 且某层为 `"sliding_attention"` 时把 `sliding_window` 传给 Attention(llama.py:199-201, 216);v0.2 config 不含滑窗层型 → 不触发。滑窗本身只是 Attention 构造参数(host int),**不引入额外 GPU 生命周期状态**
- 审计基线: HEAD c7ce03bcbd;sleep-L2 备份 gpu_worker.py:270-274,恢复 311-316

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | llama.py:376-380(经 MistralModel 继承) | R1 | 权重更新/reload 重写 | RESTORABLE | 低 |
| `lm_head.weight`(独立) | llama.py:484-490 | R1 | 权重更新重写 | RESTORABLE | 低 |
| 各层 `qkv_proj/o_proj.weight`(GQA 32/8) | llama.py:162-178;MistralAttention 仅加 llama_4_scaling 逻辑(mistral.py:106-115),v0.2 config 无该字段 → `do_llama_4_scaling=False`(mistral.py:109),无新状态 | R1 | 权重更新重写;PWAL GPU no-op(linear.py:214-218) | RESTORABLE | 低 |
| 各层 `gate_up_proj/down_proj.weight`(MistralMLP,自有实现但状态面与 LlamaMLP 相同) | mistral.py:32-74 | R1 | 同上 | RESTORABLE | 低 |
| RMSNorm 权重 | llama.py:305-308, 388-389 | R1 | 同上 | RESTORABLE | 低 |
| `ada_rms_norm_t_cond`(可选 t-cond 子网络) | mistral.py:162-178 | R1(条件) | v0.2 config 无 `ada_rms_norm_t_cond` → 为 `None`(mistral.py:178),不存在 | — | 无 |
| `rotary_emb.cos_sin_cache`(persistent=False buffer,32768×128,32 层共享单实例) | base.py:58-63;经 llama.py:233-245 `get_rope` 创建 | **R2** | **P2**(named_buffers 备份 gpu_worker.py:272-274 / 回填 311-316) | RECOMPUTE(理想) | 低:L2 归零 → P2 回填;32768 长度下备份约 16MB,可接受 |
| `_ROPE_DICT` | rotary_embedding/__init__.py:30, 83-84, 383 | R6 | 无 | PRESERVE | 无 |
| `attn._{k,v,q,prob}_scale` buffers | attention.py:124-130, 184 | R2/R3a | P2 | RESTORABLE-via-P2 | 低(bf16, auto kv) |
| `attn._*_scale_float` / `_k_scale_cpu` / `_v_scale_cpu` | attention.py:140-145 | R2 | host 侧 | PRESERVE | 无 |
| `attn.{q_range,k_range,v_range}`(非 buffer 张量) | attention.py:148-150 | R2 | 无(P2 盲区) | RECOMPUTE | 低 |
| `attn.kv_cache` | attention.py:463 | R7 | post_kv_cache_wake_up(gpu_worker.py:326-329) | SCRATCH | 低 |
| `mistral_mapping` / `layer_idx` 等 | mistral.py:236-260, 158 | host 常量 | 常驻 | PRESERVE | 无 |

## 特殊发现

1. **文件归属**:当前树中 MistralForCausalLM 有独立文件 mistral.py(registry.py:168),但其全部权重生命周期行为(load_weights、tie、PWAL、rope 创建)都继承自 llama.py——本审计结论与 untied bf16 Llama(#43)等价。Mistral 特有增量(llama_4_scaling、ada_rms_norm_t_cond、MistralMLP 的 gate_up bias 分离 mistral.py:46-53)在 v0.2 上全部不激活。
2. **滑窗状态确认**:若换 v0.1(sliding_window=4096)或带 `layer_types` 的新 config,滑窗只影响 Attention 构造参数与 KV spec,不新增设备端状态;sleep-L2 视角无差异。
3. `mistral_mapping`(mistral.py:236-260)支持 consolidated.safetensors 加载,包含 `k_fake_quantizer.qscale_act → k_scale` 等映射——若使用 Mistral 官方 fp8 checkpoint 会引入 KV scale 参数并走 kv_cache.py:74+ 的搬运+`del` PWAL(非幂等),v0.2 bf16 不触发,记录为变体风险。
4. HF 权重经 `AutoWeightsLoader` + stacked mapper(llama.py:345-354)加载,q/k/v 与 gate/up 融合;权重更新协议需按融合后名字(`qkv_proj`,`gate_up_proj`)或依赖参数子类的分片 loader——与全系 Llama 相同的注意点。

## 结论

Mistral-7B-Instruct-v0.2 在权重生命周期意义上就是一个 untied、theta=1e6 的标准 Llama:R1 全量 RESTORABLE,R2(rope cache、attention scale buffers)由 P2 覆盖,PWAL 幂等,今日 sleep-L2 无已知风险。需要留意的只有变体开关(滑窗、llama_4_scaling、ada_rms_norm、fp8 kv scales),它们在本 checkpoint 上全部关闭。
