# 模型角色审计 #29 — nvidia/Gemma-4-31B-IT-NVFP4

基本信息
- 审计日期: 2026-07-26,repo HEAD c7ce03bcbd
- 实现文件: `vllm/model_executor/models/gemma4.py`(`Gemma4ForCausalLM`,registry.py:110);官方 MTP 草稿模型 `vllm/model_executor/models/gemma4_mtp.py`(`Gemma4MTPModel` → `Gemma4MTP`,registry.py:635)
- 量化路径: ModelOpt NVFP4 —— linear:`ModelOptNvFp4LinearMethod`(modelopt.py:1115)+ nvfp4 linear kernel(kernels/linear/nvfp4/*);MoE(若 config 开 `enable_moe_block`,gemma4.py:638-651):`ModelOptNvFp4FusedMoE`(modelopt.py:1395)+ trtllm/cutlass 后端(fused_moe/oracle/nvfp4.py,experts/trtllm_nvfp4_moe.py)
- 结构特点: sliding/full 双 rope(full 层用 proportional `Gemma4RotaryEmbedding`)、YOCO KV 共享 + fast-prefill 静态张量、PLE(per-layer embedding)、`layer_scalar` **checkpoint 加载的 buffer**、可选 MoE(router.scale/per_expert_scale)

## 状态角色清单

### 主模型(gemma4.py)

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens、qkv/o_proj、MLP、各 RMSNorm、PLE 投影(NVFP4 或 bf16) | gemma4.py:980,410,419,617,626-635,1009 | R1 | reload 重写 + PWAL 重跑 | RESTORABLE | 低(依赖 P4 正确性,见下) |
| `layer_scalar`(**每层,register_buffer persistent=True,从 checkpoint 加载**) | gemma4.py:701;load_weights 把 named_buffers 并入 params_dict:gemma4.py:1413-1414 | **R1-in-buffer**(checkpoint 态住在 buffer 里) | 双保险:P2 named_buffers 备份恢复;reload 走 load_weights 也会重写 | RESTORABLE | 低;但对"只重写 named_parameters"的传输型更新(IPC/NCCL per-tensor)是盲区,需确认权重发送端包含它 |
| `router.scale`(MoE 时) / `moe.per_expert_scale` | gemma4.py:274,324 | R1 | reload 重写(checkpoint 名 `.router.per_expert_scale` 重映射:gemma4.py:1643-1646) | RESTORABLE | 低 |
| `router.root_size`(hidden_size^-0.5,非持久 buffer) | gemma4.py:276-280 | R2 | P2 copy-back(gpu_worker.py:271-274,311-316) | RECOMPUTE 可 | 低 |
| `normalizer` / `embed_scale_per_layer` / `per_layer_input_scale` / `per_layer_projection_scale`(均非持久 buffer) | gemma4.py:1061-1068,1002-1006,1026-1037 | R2 | P2 copy-back | RECOMPUTE 可 | 低 |
| `cos_sin_cache` ×2 种(sliding:标准 rope;full:proportional `Gemma4RotaryEmbedding`,零填充 nope 维) | gemma4.py:441-490; gemma4_rope.py:16-77; base.py:58-63 | R2 | P2 copy-back;`_ROPE_DICT` 全局缓存共享实例(rotary_embedding/__init__.py:83-84,144-154) | RECOMPUTE 可 | 低 |
| fast-prefill 静态张量 `self.positions` / `self.hidden_states` / `self.per_layer_inputs`(**普通属性,非 buffer**) | gemma4.py:1110-1136 | R4/R5(cudagraph 捕获的静态输入,地址烧进 graph) | **不在 named_buffers → P2 不备份**;内容每步 forward 先 copy_ 再读(gemma4.py:1219,1240-1247) | SCRATCH(内容)+ PRESERVE(地址,由 CuMem 同 VA remap 保证) | 低(wake 清零无害,首次 forward 即覆盖);终态需保证不重新分配对象 |
| NVFP4 linear PWAL 派生:`input_global_scale`、`weight_global_scale`、`alpha`、`input_global_scale_inv`;**并 del 掉 `input_scale`/`weight_scale_2`** | modelopt.py:1207-1235(del:1223,1227) | **R3c(删旧建新,重跑非幂等)** | P4 + reload 预态恢复:restore_metadata 在构造后、加载前捕获(model_loader/utils.py:62-64;reload/meta.py:92-112),layerwise 先恢复 checkpoint 形态再重放 load+PWAL,拷回原 storage(layerwise.py:90-123) | RECOMPUTE(经 PWAL) | 中:纯 wake 后、reload 完成前模型不可用(设计如此);若 PWAL 在"未恢复预态"的层上重跑会 AttributeError(fail-loud,尚可) |
| NVFP4 linear kernel swizzle:`weight_scale` 被替换为 swizzle_blockscale 结果(pad+重排) | kernels/linear/nvfp4/cutlass.py:35-37;flashinfer.py:48-51,125-127,261-264,327-329 | **R3b(参数自变换,双跑=双 swizzle 损坏)** | 同上,依赖 reload 预态恢复后整层重放 | RECOMPUTE(经 PWAL) | 中(与上行同一保护链) |

### MoE-NVFP4 追加(仅当 enable_moe_block)

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `w13/w2_weight(_scale/_scale_2/_input_scale)` → `convert_to_nvfp4_moe_kernel_format` 后 replace_parameter | modelopt.py:1557-1602; fused_moe/oracle/nvfp4.py:295 | R3b/R3c(kernel 格式重排) | P4 + 预态恢复 + P3 拷回原 storage | RECOMPUTE(经 PWAL) | 中 |
| trtllm 后端:`layer.w13_weight_scale_2.data.mul_(w13_input_scale)`(**就地自乘**) | trtllm_nvfp4_moe.py:122-123 | **R3b 非幂等典型**(重跑=平方) | 仅当 reload 先恢复 checkpoint 值再重放才安全(现有机制满足);裸重跑 PWAL 即损坏 | RECOMPUTE(锁死:必须防裸重跑) | **高**(已知 HIGH 类) |
| trtllm `g1_scale_c` / `gemm1_clamp_limit` / `gemm1_beta` / `gemm1_alpha`(PWAL 中 register_parameter **新建** Parameter,并回挂到 kernel 对象属性 `self.g1_scale_c=layer.g1_scale_c` 等) | trtllm_nvfp4_moe.py:121-170(init 期先有临时版:76-78,94-111) | **R5(graph 捕获 kernel 常量)+ R3c** | reload 重放会 register_parameter 生成**新 tensor 对象/新地址**;若 CUDA graph 已按旧地址捕获,或 kernel 对象属性未重绑,读到 discard 后的旧 storage | RECOMPUTE + PRESERVE 地址(须拷回原 storage 而非换对象) | **高**(prompt 指名的 known-HIGH:gemm1_*/g1_scale_c) |
| `moe_kernel` / `moe_quant_config`(PWAL 里重建的 python 对象,内部持权重/scale 引用) | modelopt.py:1604-1615 | R5/R6(对象级缓存,持 tensor 引用) | PWAL 重跑时重建;但旧 graph 若捕获了旧对象引用的地址则失效 | RECOMPUTE | 中-高 |

### 草稿模型(gemma4_mtp.py)

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| draft 权重(embed/q_proj/o_proj/MLP/pre/post_projection/lm_head/centroids) | gemma4_mtp.py:372-407,496-503,87 | R1 | draft reload(`_set_draft_weight_update_target`,gpu_worker.py:1017-1034) | RESTORABLE | 低 |
| `masked_embedding.token_ordering`(persistent buffer,**checkpoint 加载**) | gemma4_mtp.py:88-91 | R1-in-buffer | **P2 只备份主模型 named_buffers(gpu_worker.py:271-274 `model_runner.model`),draft 的 buffer 不在其中**;依赖 draft reload 经 load_weights 重写 | RESTORABLE | 中:若 draft 在 weights pool 内分配且 RL 更新不重发 token_ordering(静态映射,训练不更新),wake 后为零 → 稀疏 logits 全错 |
| draft `layer_scalar`(persistent buffer,checkpoint 加载) | gemma4_mtp.py:323 | R1-in-buffer | 同上,不在 P2 内 | RESTORABLE | 中(同上) |
| draft `normalizer`(非持久 buffer,backbone_hidden_size^0.5) | gemma4_mtp.py:412-416 | R2 | **不在 P2 内,也不在 checkpoint 里** → 若 draft 权重在 pool 内,wake 后被清零且无人重算 | RECOMPUTE(必须补保护) | **高**(静默清零:embed 输出恒 0) |
| draft `cos_sin_cache`(每层 rotary) | gemma4_mtp.py:212-217 | R2 | 同上不在 P2;**但** `_ROPE_DICT` 若与主模型 key 相同则共享实例,主模型 P2 恢复顺带修好;head_dim/rope 参数不同的层无此运气 | RECOMPUTE | **高**(依赖 rope 共享的偶然性,不可靠) |
| `_stable_full_lm_head_weight`(**惰性 all-gather 缓存,普通属性**,TP>1 时首次 compute_logits 生成) | gemma4_mtp.py:485,556-570 | **R6(lazy-init 缓存)+ R3(lm_head 派生)** | 唯一失效钩子在 `load_weights` 开头置 None(gemma4_mtp.py:599);**绕过 load_weights 的就地权重更新(IPC/NCCL copy_ 进 param storage)不会触发** → 缓存里是旧策略权重 | RECOMPUTE(需注册失效钩子 P6) | **高**(RL 场景 lazy-init 定时炸弹) |
| `_suppress_token_ids`(generation_config 派生 python list) | gemma4_mtp.py:530-532 | R2(非张量) | 无需 | PRESERVE | 无 |

## 特殊发现

1. **`layer_scalar` 把 checkpoint 态放进 buffer**(gemma4.py:701 / gemma4_mtp.py:323),load_weights 特意把 named_buffers 并进 params_dict(gemma4.py:1414)。这打破"buffer=派生, param=checkpoint"的默认分类;任何按角色自动分类的终态机制必须支持 R1-in-buffer 标注,否则权重传输(trainer→rollout)会漏发。
2. **trtllm NVFP4 MoE 是全模型风险最高的簇**:`w13_weight_scale_2.mul_` 就地自变换(trtllm_nvfp4_moe.py:122-123,幂等锁死必需)+ PWAL 里 register_parameter 新建 gemm1_*/g1_scale_c(新地址,R5)+ kernel 对象属性回挂。现有 reload(预态恢复+重放+P3 拷回)在参数级是对的,但 **register_parameter 新建的对象是否被 P3 按名拷回原 storage、以及 CUDA graph 捕获的地址是否因此保住,是终态验收必测点**。
3. **草稿模型完全游离在 P2 之外**:gpu_worker.sleep 只备份 `model_runner.model` 的 buffers(gpu_worker.py:271-274);wake 只额外处理有 `_build_fused_kv_buffers` 的 draft(gpu_worker.py:275-279,318-324)—— Gemma4MTP 没有该钩子。draft 的 R2 buffer(normalizer、cos_sin_cache)在 sleep-L2 后无人恢复、无人重算。若 draft 与主模型同在 weights pool(默认 load_model 全程在 pool 上下文),这是**确定性静默清零**。
4. **`_stable_full_lm_head_weight` 是 prompt 所指 R6 lazy-init 时间炸弹的实例**:失效仅挂在 load_weights(P5),对 copy_ 式权重更新和 sleep-L2(缓存 tensor 在 forward 期分配、不在 pool、内容"幸存"但对应旧权重)都失效。tie_word_embeddings=True 时它缓存的还是 embed_tokens 的旧副本。
5. fast-prefill 静态张量(gemma4.py:1114-1136)与 YOCO 双编译子图(self/cross_decoder,gemma4.py:1078-1106)意味着本模型 cudagraph 捕获面大,所有 R3/R5 派生态的**地址不变性**比内容正确性更难验证。

## 结论

主模型侧:R1/R2 由 reload+P2 覆盖;NVFP4 的 R3b/R3c(swizzle、del+create、mul_ 自变换)依赖"预态恢复→重放→原 storage 拷回"链条,机制存在但幂等锁死与地址保持需要专项测试,MoE-trtllm 的 gemm1_*/g1_scale_c 为已知 HIGH。草稿模型侧发现两个未覆盖点:(a) Gemma4MTP 的 R2 buffer(normalizer/cos_sin_cache)不在 P2 备份范围内,sleep-L2 后静默清零;(b) `_stable_full_lm_head_weight` lazy 缓存仅靠 load_weights 失效,绕过该入口的权重更新会用旧权重出 draft logits。建议:P2 扩展到 draft named_buffers(一行改动:合并 `get_draft_model().named_buffers()`),并给 lazy 缓存加权重版本失效钩子(P6)。
