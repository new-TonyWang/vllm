# 模型角色审计 #46 — RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic

基本信息:
- 架构: `LlamaForCausalLM`(llama.py,registry.py:143)+ compressed-tensors 量化
- 量化方案: weight FP8 **per-channel static** + activation FP8 **per-token dynamic** → `CompressedTensorsW8A8Fp8`(compressed_tensors.py:823-826;scheme 文件 compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py),`is_static_input_scheme=False`(w8a8_fp8.py:61-66, 80-84:activation key=`kFp8DynamicTokenSym`)
- `ignore=["lm_head"]`(checkpoint config;compressed_tensors.py:249, 258)→ lm_head/embed 走非量化路径;`tie_word_embeddings=True`(Llama-3.2-1B)
- kv_cache_scheme=None → KV cache 不量化,attention scale 走默认 1.0
- 审计基线: HEAD c7ce03bcbd;sleep-L2 备份 gpu_worker.py:270-274,恢复 311-316

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `layers.*.{qkv_proj,o_proj,gate_up_proj,down_proj}.weight`(fp8_e4m3,PWAL 前 (N,K) `ModelWeightParameter`,PWAL 后被替换为**转置视图**的裸 `nn.Parameter`(K,N)) | 创建 w8a8_fp8.py:120-124;PWAL 替换 w8a8_fp8.py:160-165, 183-187 | R1 + **R3a**(loader/PWAL 副作用改形态)+ **R3c**(PWAL 非幂等:对已转置权重重跑会把方向转回去,形状仍合法 → 静默错) | 权重更新须以 PWAL 后形态原地写,或走 reload 的"恢复原始布局→重放 load→重跑 PWAL"流水 | RESTORABLE(需 P4/P5 配合) | **中**:L2 丢页后 VA 复原、张量对象不变(`.t().data` 仍指向 weights-pool 原存储),cudagraph 指针安全;风险在恢复协议——若更新流发 bf16 权重则需在线量化,若直接重放 checkpoint fp8 则必须先回到 PWAL 前布局再重跑 PWAL,直接重跑 = R3c 踩雷 |
| `layers.*.*.weight_scale`(float32;PWAL 前 `ChannelQuantScaleParameter` (N,1),PWAL 后被替换为裸 `nn.Parameter`) | 创建 w8a8_fp8.py:126-134 → fp8_utils.py:1277-1282;PWAL 替换 w8a8_fp8.py:188 | R1 + R3a(丢失参数子类与 weight_loader 属性) | 同上 | RESTORABLE | 中:同 weight;per-channel 策略 PWAL **无 `.max()` 合并**(fp8_utils.py:1353-1368 仅 ROCm fnuz 归一化)——`.max()` 合并只在 TENSOR 策略(fp8_utils.py:1325-1350 `requantize_with_max_scale`:1343-1347)与 static input scale(w8a8_fp8.py:193-194)出现,本模型均不触发 |
| `layers.*.*.input_scale` | 动态激活 → **不创建**(w8a8_fp8.py:137-139 条件不成立);PWAL 显式置 `layer.input_scale = None`(w8a8_fp8.py:195-196) | — | — | SCRATCH(不存在) | 无 |
| `scheme.fp8_linear` kernel 对象(`FP8ScaledMMLinearKernel` 子类,持 `QuantFP8` 算子与 config,无 GPU 张量) | 创建 w8a8_fp8.py:141-148 → kernels/linear/__init__.py:580+;其 PWAL 为 no-op(ScaledMMLinearKernel.py:123-124) | R5(host 侧 kernel 选择/配置) | 常驻 Python 对象 | PRESERVE | 无:不持设备内存;`expose_input_quant_key`(w8a8_fp8.py:150)只写 layer 属性 |
| `layer.logical_widths` / `weight_block_size` / `orig_dtype` | w8a8_fp8.py:103-105 | R2(config 派生 host 元数据) | 常驻 | PRESERVE | 无 |
| `model.embed_tokens.weight`(bf16,未量化) | llama.py:376-380 | R1 | 权重更新重写 | RESTORABLE | 低 |
| `lm_head.weight`(ignore 列表 → 未量化;tied → 与 embed_tokens 同一 Parameter) | llama.py:491-492;vocab_parallel_embedding.py:80-84, 555-557 | R1(别名) | 加载 skip `lm_head.`(llama.py:536-539) | RESTORABLE(经 embed_tokens) | 中:tie 对象同一性问题同 #07;FP8 更新流里 embed 是 bf16、layers 是 fp8,**混精度更新协议**须区分对待 |
| RMSNorm 权重(bf16) | llama.py:305-308, 388-389 | R1 | 权重更新重写 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(persistent=False buffer;Llama3RotaryEmbedding,131072 max_position) | base.py:58-63;llama3_rope.py:33-54 | R2 | **P2**(gpu_worker.py:272-274, 311-316) | RECOMPUTE(理想) | 中低,同 #07/#41 |
| `attn._{k,v,q,prob}_scale` buffers(恒 1.0,kv 不量化) | attention.py:124-130, 184 | R2 | P2 | RESTORABLE-via-P2 | 低 |
| `attn.{q_range,k_range,v_range}` 非 buffer 张量 | attention.py:148-150 | R2 | 无(P2 盲区) | RECOMPUTE | 低 |
| `attn.kv_cache` | attention.py:463 | R7 | post_kv_cache_wake_up(gpu_worker.py:326-329) | SCRATCH | 低 |

## 特殊发现

1. **PWAL 副作用清单(channel+dynamic 分支)**,w8a8_fp8.py:152-199:
   - 161-165:ROCm 下 `normalize_e4m3fn_to_e4m3fnuz`(改 weight 位型+scale ×2,**非幂等**);CUDA 下 channel 分支为直通;
   - 165:`weight = weight.t()` → 布局从 (N,K) 变 (K,N)(视图,不拷贝);
   - 183-188:`layer.weight`/`layer.weight_scale` 被**新 `nn.Parameter` 对象替换**——丢失 `ModelWeightParameter`/`ChannelQuantScaleParameter` 子类及其 `weight_loader`,并补写 `input_dim/output_dim` 标签(186-187)。这意味着 PWAL 之后无法直接用原 weight_loader 语义再次加载 checkpoint 分片(qkv/gate_up 的 stacked-shard 加载逻辑依赖参数子类)——reload 必须先恢复 PWAL 前参数对象/布局(P5),否则 R3c;
   - 195-196:`input_scale = None`;
   - 198-199:kernel PWAL no-op(ScaledMMLinearKernel.py:123-124)。
2. **`.max()` 合并不在本模型路径上**:tensor 策略的 `requantize_with_max_scale`(fp8_utils.py:1343-1347)会把 fused 模块(qkv/gate_up)的多个 per-tensor scale 合并重量化——是不可逆的 R3b 信息丢失点;channel 策略天然按行保 scale,无信息丢失,fused shard 只是行拼接。审计其他 compressed-tensors FP8-**static**(per-tensor)模型时须把该项列为高风险,本模型豁免。
3. **sleep-L2 的指针稳定性成立**:PWAL 的转置是视图,最终 weight 仍驻原 weights-pool 分配;CuMem wake 恢复同 VA,故 cudagraph/compiled 图捕获的地址有效,危险只来自"重建张量"的恢复方式。
4. Attention 的 `_k_scale` 等虽为 buffer(P2 覆盖),但 compressed-tensors 的 KV-scale PWAL 路径(kv_cache.py:74-91, 197-198:从 checkpoint `k_scale/v_scale` 参数搬运后 `del`)在本模型不触发(无 kv_cache_scheme);若换用带 kv fp8 的变体,该 `del` 使 PWAL 非幂等且 reload 无法二次按名加载 k_scale——记录为家族性风险。

## 结论

FP8-dynamic 变体把 Llama 审计从"纯 R1+R2"升级为"R1+R3a/R3c":参数本体仍 RESTORABLE,但 PWAL 在**参数对象、布局(转置)、参数子类/加载器属性**三个维度留下副作用,且重跑不幂等。今日 sleep-L2 本身不腐化状态(VA 复原+P2 覆盖 buffers);风险窗口全部在 wake 后的权重恢复协议:必须(a)以 PWAL 后形态原地写入,或(b)完整执行"恢复 PWAL 前状态 → 重放加载 → 重跑 PWAL"。终态声明建议:weight/weight_scale = RESTORABLE(标注 R3a 形态差),PWAL = 必须经由 P5 恢复点重入,fp8_linear kernel 对象 = PRESERVE。
