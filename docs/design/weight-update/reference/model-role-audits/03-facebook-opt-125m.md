# 模型角色审计 #03 — facebook/opt-125m

基本信息
- 审计基线: vLLM main @ c7ce03bcbd(本地 checkout,含未提交的 K3 工作区改动,但模型/量化文件以 main 状态为准)
- 模型实现: `vllm/model_executor/models/opt.py`(394 行)
- 架构要点: decoder-only,**learned positional embeddings(无 RoPE)**,LayerNorm,无 MoE,无量化特殊路径
- 官方审计草稿: 无(本模型没有官方 draft 可对照,本文为独立审计)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `embed_tokens`(VocabParallelEmbedding) | opt.py:211-214 | R1 checkpoint 参数 | P3 copy-back + P4 | RESTORABLE | 无 |
| `embed_positions`(OPTLearnedPositionalEmbedding,nn.Embedding 子类) | opt.py:59-68, 216-218 | **R1 checkpoint 参数**(learned pos emb 从 `decoder.embed_positions.weight` 加载,不是 R2 派生 buffer) | P3 copy-back | RESTORABLE | 无 |
| `embed_positions.offset = 2` | opt.py:64 | 纯 Python int(非张量) | 不需要 | N/A | 无 |
| `qkv_proj` / `out_proj` / `fc1` / `fc2` 权重与 bias | opt.py:90-104, 149-163 | R1 | P3 copy-back + P4(UnquantizedLinearMethod 的 PWAL 幂等) | RESTORABLE | 无 |
| `self_attn_layer_norm` / `final_layer_norm`(decoder 级与层级) | opt.py:146-148, 164-166, 247-253 | R1 | P3 copy-back | RESTORABLE | 无 |
| `project_in` / `project_out`(仅 word_embed_proj_dim != hidden_size,125m 不存在) | opt.py:220-241 | R1 | P3 copy-back | RESTORABLE | 无(125m 为 None) |
| `lm_head`(tie_word_embeddings=True 时与 embed_tokens 共享同一 Parameter 对象) | opt.py:352-359 | R1(别名) | 别名共享存储,写 embed_tokens 即同步;load_weights skip `lm_head.weight`(opt.py:387-394) | RESTORABLE | 无(前提:reload 走原位 copy,不做参数对象替换,否则 tie 断裂) |
| Attention `_q_scale`/`_k_scale`/`_v_scale`/`_prob_scale`(公共层) | vllm/model_executor/layers/attention/attention.py:127-130, 184 | R2 config 派生 buffer(register_buffer) | P2 `_sleep_saved_buffers`(gpu_worker.py:270-275 named_buffers 备份;311-316 wake 时 copy_ 回填) | RESTORABLE | 无 |
| Attention `_k_scale_cpu`/`_v_scale_cpu`、`*_float` | attention.py:140-145 | CPU 张量 / Python float | 不在 GPU 池内 | PRESERVE(天然安全) | 无 |

## 特殊发现

1. **本模型是理想的 null-case 基线**:全部持久状态都是 R1 checkpoint 参数或已被 P2 覆盖的公共 attention buffer。没有 register_buffer(模型文件内 0 处)、没有 RoPE、没有权重派生转换、没有 PWAL 自定义逻辑、没有 kernel 常量。
2. learned positional embedding 容易被误分类为 R2:它在 HF checkpoint 中有实体权重(`decoder.embed_positions.weight`),经 `hf_to_vllm_mapper`(opt.py:328-338)正常走 AutoWeightsLoader,属 R1,reload 会重写。
3. tied lm_head 依赖"参数对象同一性"(opt.py:353 直接赋引用)。任何 reload 实现若用 `replace_parameter`/重建参数而非 `param.data.copy_`,tie 会静默断开——这是对 reload 机制的通用约束,而非 OPT 特有 bug。
4. `get_act_fn(config.activation_function)`(opt.py:156)返回无状态激活模块,无风险。

## 结论

OPT-125m 在今日 main 上对 sleep level-2 + reload 无已知风险状态。所有 GPU 驻留状态可由 P2(公共 attention buffer)+ P3/P4(checkpoint 参数)完整恢复。可作为权重生命周期重构的回归基线(最小对照组):任何在 OPT 上出现的 sleep-L2 失效都指向公共基础设施(CuMem 池、P2 备份、reload copy 路径)而非模型代码。
