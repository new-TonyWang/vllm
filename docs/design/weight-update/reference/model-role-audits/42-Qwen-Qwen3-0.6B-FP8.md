# 模型角色审计 #42 — Qwen/Qwen3-0.6B-FP8

基本信息:
- HF 排名: 42
- 量化变体: FP8(官方 checkpoint:`quant_method=fp8`,`activation_scheme=dynamic`,**`weight_block_size=[128,128]` 块量化**——注意不是 per-tensor;`weight_scale_inv` 存于 checkpoint)。量化路径 `vllm/model_executor/layers/quantization/fp8.py` `Fp8LinearMethod`(fp8.py:194-196,267-489),kernel 为 `Fp8BlockScaledMMLinearKernel` 家族(BlockScaledMMLinearKernel.py / deep_gemm.py / cutlass.py)
- draft: 无官方 draft;`SupportsEagle`/`SupportsEagle3` 声明(qwen3.py:272)
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)
- 架构参数: 同 Qwen3-0.6B:28 层,hidden 1024,16/8 头,head_dim=128,`tie_word_embeddings=true`;embed_tokens/lm_head 不量化(`Fp8Config.get_quant_method` 仅对 LinearBase/RoutedExperts/Attention 返回方法,fp8.py:175-220)

## 状态角色清单

基座通用状态(embed_tokens、tie 的 lm_head、RMSNorm 系、q_norm/k_norm、cos_sin_cache、q_range 系)与 #01 报告一致,不重复;FP8 路径特有状态:

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `weight`(fp8 e4m3,checkpoint 格式) | fp8.py:352-355(create_fp8_weight_parameter) | R1 | P3 | RESTORABLE | 低 |
| `weight_scale_inv`(BlockQuantScaleParameter,fp32,checkpoint) | fp8.py:367-379 | R1 | P3 | RESTORABLE | 低 |
| PWAL 变换后的 `weight`/`weight_scale_inv`(基础块路径:fnuz 归一化+padding) | fp8.py:398-444 → BlockScaledMMLinearKernel.py:75-95;fp8_utils.py:1371-1405 | R3a | P3+P4(reload 回灌 checkpoint 格式后重跑,layerwise.py:395-421) | RECOMPUTE | 低(reload 路径正确) |
| DeepGEMM 后处理版 `weight`/`weight_scale_inv`(布局变换 + Blackwell 上 UE8M0 requant) | deep_gemm.py:84-107(`deepgemm_post_process_fp8_weight_block`) | R3a(**非幂等**:scale dtype 可能 fp32→e8m0) | P3+P4 | RECOMPUTE | 中:对已处理张量直接重跑 PWAL 会算错,必须先回灌 checkpoint 格式(今日 P3 已保证;绕过机制的路径危险) |
| Cutlass 16 对齐 padding 后 `weight` + **weight_loader 被换成 `padded_weight_loader`** | cutlass.py:204-240 | R3a + R3b(loader 副作用) | P3(reload 元数据保留原 loader,meta.py:35-40) | RECOMPUTE | 低-中:同 AWQ 的 loader 替换问题,直接 re-load 语义改变(padded loader 尚能工作,风险低于 _noop) |
| `layer.input_scale = None`(dynamic act,无静态输入 scale) | fp8.py:439-442 | — Python None | — | SCRATCH | 无 |
| `attn.q_scale/k_scale/v_scale/prob_scale`(KVCacheScaleParameter 哨兵 -1,加载期) | fp8.py:218-219 → kv_cache.py:57-69 | R3b(PWAL 后被删除,值折叠进 `_k_scale` 等 buffer 与 `*_float`,kv_cache.py:74 起) | P6(reload 专用钩子 `_reload_attention_scales` 重跑 create_weights + PWAL,layerwise.py:360-382) | RECOMPUTE | 低 |
| `attn._q/_k/_v/_prob_scale`(buffer,checkpoint 无 kv scale → 恒 1.0) | attention/attention.py:127-130,184 | R2 | P2 + P6 | RESTORABLE | 低 |
| `fp8_linear` kernel 对象、`quant_fp8` 等 | fp8.py:387-396;BlockScaledMMLinearKernel.py:49-59 | — Python 对象 | 不在 GPU | PRESERVE | 无 |

## 特殊发现

1. **任务提示中的 "scale 合并 .max()/requant" 路径本模型不走**:`requantize_with_max_scale` 合并 N 个 shard scale(fp8.py:416-437;fp8_utils.py:1325-1350)只在 **per-tensor** FP8 checkpoint(`block_quant=False`)触发,属 R3a。Qwen3-0.6B-FP8 是 block [128,128] 量化,PWAL 走 fp8.py:412-413 的块路径,`weight_scale_inv` 逐块加载、无跨 shard 合并;qkv/gate_up 融合层的块 scale 由 `BlockQuantScaleParameter` 分片装载(fp8.py:370-379)。若审计其它 per-tensor FP8 模型需另行核对该路径。
2. **PWAL 幂等性守卫(R3c 风味)**:attention 的 `BaseKVCacheMethod.process_weights_after_loading` 用 `hasattr(layer, "q_scale")` 守卫(kv_cache.py:74-80)——PWAL 后参数被 `del`,二次调用直接 return。reload 靠 `_reload_attention_scales` 重新 `create_weights` 造出哨兵参数(layerwise.py:373)绕开守卫。同类守卫机制 `_already_called_process_weights_after_loading` 由 reload 在重跑前显式删除(layerwise.py:400-401)。
3. **Marlin FP8 回退**:无原生 FP8 硬件(<SM89)时 `use_marlin=True`(fp8.py:287-290,396,399-408),weight 会走 marlin repack,引入与 #32/#38 相同的 workspace(R4+R5)问题;H100/B200 等主流部署不触发。
4. **DeepGEMM UE8M0 requant 只在特定硬件发生**(deep_gemm.py:88-107,`use_deep_gemm_e8m0`),使同一 checkpoint 在不同 GPU 上的 R3a 产物 dtype 不同——重设计的终态声明应按"kernel 决定的处理后格式"参数化,而非写死。
5. tie 词嵌入 + FP8:embed_tokens/lm_head 保持 bf16 不量化,别名与 skip 逻辑同 #01(qwen3.py:299,341)。
6. 基座通用缺口(`q_range` 系裸张量、`cos_sin_cache` 仅 P2、`_ROPE_DICT` R6)同 #01;本变体若配 fp8 kv cache + `calculate_kv_scales`,`q_range` 清零缺口会实际触发除零(attention.py:585-587)。

## 结论

Qwen3-0.6B-FP8 的 weight/weight_scale_inv 是 R1,但 PWAL 会按 kernel 后端做一次非幂等的 R3a 变换(fnuz/UE8M0/padding/布局),cutlass 路径还伴随 R3b 的 loader 替换;今日 reload(P3 回灌 + P4 重跑)链路完整,attention kv-scale 参数的删除+守卫模式由专用钩子(P6,layerwise.py:360-382)兜住。风险集中在:绕过 reload 机制直接对已处理参数重放 PWAL 或 weight_loader、以及 fp8-kv 场景下 `q_range` 系裸张量清零;重设计应将"处理后格式"声明为 RECOMPUTE 且以 checkpoint 格式为唯一合法重放起点。
