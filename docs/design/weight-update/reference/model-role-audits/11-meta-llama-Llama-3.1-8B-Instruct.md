# 模型角色审计 #11 — meta-llama/Llama-3.1-8B-Instruct(含 EAGLE3 draft)

基本信息:
- 架构: `LlamaForCausalLM` → `vllm/model_executor/models/llama.py`(registry.py:143);bf16,无量化;`tie_word_embeddings=False`
- rope_scaling type=llama3 → `Llama3RotaryEmbedding`(llama3_rope.py:11-54),max_position 131072
- 官方 EAGLE3 draft 常配:yuhuili/EAGLE3-LLaMA3.1-Instruct-8B → `Eagle3LlamaForCausalLM`(llama_eagle3.py:272)
- 审计基线: HEAD c7ce03bcbd;sleep-L2 备份 gpu_worker.py:270-274,恢复 311-316。**PR 49774(draft named_buffers 备份)不在本树中**(git log 无此提交)

## 状态角色清单

### 目标模型(与 #07 同构,差异:untied、32 层)

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `embed_tokens.weight` | llama.py:376-380 | R1 | 权重更新重写 | RESTORABLE | 低 |
| `lm_head.weight`(独立,untied) | llama.py:484-490 | R1 | 权重更新重写 | RESTORABLE | 低 |
| 各层 `qkv_proj/o_proj/gate_up_proj/down_proj.weight` | llama.py:162-178, 92-108 | R1 | 同上;PWAL GPU no-op(linear.py:214-218) | RESTORABLE | 低 |
| RMSNorm 权重 | llama.py:305-308, 388-389 | R1 | 同上 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(persistent=False buffer) | base.py:58-63 | R2 | P2(named_buffers 备份/回填) | RECOMPUTE(理想) | 中低,同 #07 |
| `attn._{k,v,q,prob}_scale` buffers | attention.py:124-130,184 | R2/R3a | P2 | RESTORABLE-via-P2 | 低(bf16, auto kv) |
| `attn.{q,k,v}_range` 普通张量属性 | attention.py:148-150 | R2 | **无**(非 buffer) | RECOMPUTE | 低(未启用 calculate_kv_scales) |
| `attn.kv_cache` | attention.py:463 | R7 | post_kv_cache_wake_up(gpu_worker.py:326-329) | SCRATCH | 低 |
| `_ROPE_DICT` | rotary_embedding/__init__.py:30,383 | R6 | 无 | PRESERVE | 见"特殊发现 1" |

### EAGLE3 draft(llama_eagle3.py)——重点

draft 在 `model_runner.load_model` 内加载(gpu_model_runner.py:5273-5276),而该调用整体处于 **CuMem "weights" pool 上下文**(gpu_worker.py:506-513)→ draft 的全部 GPU 参数与 buffer 同样被 L2 丢页。但 P2 备份只遍历 `self.model_runner.model.named_buffers()`(gpu_worker.py:271-274)= **仅目标模型**;draft 侧唯一的 wake 钩子是 `_build_fused_kv_buffers`(gpu_worker.py:277-279, 318-324),Eagle3 Llama draft 没有该方法,等于**零覆盖**。

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| draft `midlayer` 各线性层/RMSNorm 权重(含加宽 qkv_proj,输入 2×hidden) | llama_eagle3.py:56-64, 66, 43 | R1 | 仅当启用 draft 权重更新(Runtime Draft Weight Update, #46725 已在树内,gpu_worker.py:1017+ `_set_draft_weight_update_target`)时被重写 | RESTORABLE(条件) | **高**:RL 流程若只更新目标模型权重,draft 参数 wake 后全零 → 接受率崩塌(与 k3 已有 dspark L2 acceptance collapse 根因报告同型) |
| draft `fc`(ReplicatedLinear, 3×hidden→hidden)、`fc_norm`/`input_norm`(可选) | llama_eagle3.py:208-216, 189-206 | R1 | 同上 | RESTORABLE(条件) | 高,同上 |
| draft `embed_tokens.weight` | llama_eagle3.py:158-162 | R1 或别名 | 若 checkpoint 无自带 embed 或与目标一致,`_maybe_share_embeddings` 直接 `del` 后**绑定目标模块**(llm_base_proposer.py:1421-1503,尤其 1501-1503);判定依赖 `process_eagle_weight` 设置的 `has_own_embed_tokens`(models/utils.py:1028-1047) | RESTORABLE(共享时随目标恢复) | 共享时低;**独立时高**(不被目标权重更新覆盖) |
| draft `lm_head.weight` | llama_eagle3.py:295-300;共享逻辑 llm_base_proposer.py:1510-1546 | R1 或别名 | 同上(`has_own_lm_head`) | 同上 | 同上 |
| `draft_id_to_target_id`(nn.Parameter, requires_grad=False, 从 d2t 加载) | llama_eagle3.py:304-307, 389-391 | R1(checkpoint 派生查找表) | 无 P2;仅 draft 权重更新可能重写 | RESTORABLE(须确认 d2t 在更新流中) | **高**:wake 后全零 → `compute_logits` 的 d2t 重映射(llama_eagle3.py:347-357)退化为恒等偏移 0,draft 词表→目标词表映射静默全错 |
| `mask_hidden`(register_buffer, persistent=False;仅 parallel_drafting) | llama_eagle3.py:311-316, 401 | R1(checkpoint 加载的 buffer!) | **无**:不在目标 named_buffers,P2 不覆盖 | RESTORABLE(应纳入 draft P2) | 高(仅在 parallel_drafting 开启时);且 proposer 在 load_model 时把它投影进 `parallel_drafting_hidden_state_tensor`(llm_base_proposer.py:1406-1419),wake 后无人重投影 |
| **draft `rotary_emb.cos_sin_cache`** | base.py:58-63;draft 经 llama.py:233-245 `get_rope` 创建 | **R2** | 取决于 `_ROPE_DICT` key 是否与目标相同(__init__.py:83-84):相同 → 与目标共享同一模块实例,cache 在目标 named_buffers 里 → P2 顺带救活;**不同(EAGLE3 draft config 常见 max_position=2048 / 无 llama3 rope_scaling,而目标 131072+llama3)→ draft 专属模块,P2 覆盖不到** | RECOMPUTE | **高(独立 rope 时)**:wake 后 cache 全零 → draft q/k 旋转全错,无异常、纯接受率/输出质量退化 |
| draft `attn._{k,v,q,prob}_scale` buffers | attention.py:124-130(draft 层同样走 Attention 构造) | R2 | **无**(不在目标 named_buffers) | RECOMPUTE | 低(bf16 值 1.0,但同为覆盖空洞) |
| `config.target_layer_count`(写入 hf_config) | llama_eagle3.py:287 | host 配置 | 常驻 | PRESERVE | 无 |

## 特殊发现

1. **`_ROPE_DICT`(R6)决定 draft rope 是否"蹭到"P2**:get_rope 按 (head_size, rotary_dim, max_position, rope_parameters, dtype) 全局缓存(rotary_embedding/__init__.py:30, 83-84, 383)。目标 32 层共享一个 Llama3RotaryEmbedding;draft 层若 key 一致会复用同一实例(named_buffers 去重后备份一次、回填一次,全体受益);key 不一致则 draft 有独立 cache,处于 P2 盲区。**保护是否生效取决于两份 HF config 的 rope 字段是否逐项相等,这是隐式且脆弱的**。
2. **draft 状态保护整体缺位**:本树内 sleep-L2 对 draft 的唯一照顾是 `_build_fused_kv_buffers` 重建钩子(gpu_worker.py:275-279, 318-324),这是为特定 draft 类型打的补丁,恰说明 draft buffer 恢复目前是 ad hoc 的。PR 49774 将 draft.named_buffers 纳入 P2 属于正确方向,但本 HEAD 未包含;即便包含,`draft_id_to_target_id` 是 **Parameter** 不是 buffer,named_buffers 方案仍覆盖不到,须依赖 draft 权重更新流重发 d2t。
3. **embed/lm_head 共享的方向性**:共享是 draft→目标的模块引用(llm_base_proposer.py:1501-1503),因此目标权重更新天然同步 draft;但共享判定基于加载时 CPU 逐元素比较(llm_base_proposer.py:1452-1463),reload 后不再重判。若 RL 更新使目标 embed 漂移,而 draft 本来是"值相同但独立"被判共享,行为仍正确(引用同一存储);反之独立 embed 的 draft 在只更新目标的流程里会用**旧** embed 继续 draft——不属于 L2 问题,但属于权重生命周期设计要点。
4. draft 的 `qkv_proj` 在 `LlamaDecoderLayer.__init__` 后被**整体替换**(llama_eagle3.py:56-64),原 qkv_proj 的参数成为孤儿并被 GC;审计时按替换后的模块计。

## 结论

目标模型侧与 #07 同级:P2+权重更新即可安全过 L2。风险集中在 EAGLE3 draft:HEAD c7ce03bcbd 下 draft 的参数、`draft_id_to_target_id`、`mask_hidden`、以及(rope key 不一致时的)draft cos_sin_cache 在 sleep-L2 后**全部无人恢复**,唯一救济是显式启用 draft 权重更新且其覆盖面包含 d2t/mask_hidden——否则表现为无异常的接受率崩塌。终态设计必须:(a) 把 draft 纳入 P2 或声明 draft 全部 R2 状态为 RECOMPUTE 并提供重建钩子(P6);(b) 把 `draft_id_to_target_id`、`mask_hidden` 显式声明 RESTORABLE 并纳入 draft 权重更新契约;(c) 消除对 `_ROPE_DICT` key 相等的隐式依赖。
