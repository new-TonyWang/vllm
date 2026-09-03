# 模型角色审计 #32 — Qwen/Qwen3-14B-AWQ

基本信息:
- HF 排名: 32
- 量化变体: AWQ(4bit,group_size=128,zero_point=true,GEMM 格式)。本仓库(HEAD c7ce03bcbd)中 `awq.py`/`awq_marlin.py` 已合并重构为 `vllm/model_executor/layers/quantization/auto_awq.py`,CUDA 上经 `choose_mp_linear_kernel` 选择 Marlin 后端(auto_awq.py:414-536;kernel 实现 `vllm/model_executor/kernels/linear/mixed_precision/marlin.py`)
- draft: 无官方 draft;`SupportsEagle`/`SupportsEagle3` 声明(qwen3.py:272)
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)
- 架构参数: 40 层,hidden 5120,40/8 头,head_dim=128,`tie_word_embeddings=false`;lm_head 与 embed_tokens 不量化(保持 fp16/bf16)

## 状态角色清单

基座(qwen3/qwen2)通用状态与 #22 报告一致(embed_tokens、lm_head、各 RMSNorm、q_norm/k_norm、cos_sin_cache、attention scales、q_range 系),此处不重复;以下为 AWQ-Marlin 路径特有状态:

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `qweight/qzeros`(checkpoint AWQ 打包格式,加载期) | auto_awq.py:478-518(PackedvLLMParameter) | R1 | P3(reload 用 init 时记录的元数据重建,model_loader/utils.py:64 → reload/meta.py:92-134) | RESTORABLE | 低 |
| `scales`(group scale,checkpoint) | auto_awq.py:506-519 | R1 | P3 | RESTORABLE | 低 |
| PWAL 第一步:AWQ→标准 GPTQ 格式转换后的 `qweight/qzeros` | auto_awq.py:93-168(`_convert_awq_to_standard_format`,528-535 调用) | R3b(整参数替换,**新参数 weight_loader=_noop_loader**,auto_awq.py:129-140/160-168) | P3+P4(reload 先回灌 checkpoint 格式再重跑 PWAL,layerwise.py:395-421) | RECOMPUTE | 中:见特殊发现 1 |
| PWAL 第二步:marlin repack 后的 `qweight` | marlin.py:124-137,210(`gptq_marlin_repack`) | R3a | P3+P4 | RECOMPUTE | 低(reload 路径正确) |
| marlin permute 后的 `scales` | marlin.py:139-170,211(`marlin_permute_scales`) | R3a | P3+P4 | RECOMPUTE | 低 |
| `w_zp`(由 qzeros 派生的 marlin 零点) | marlin.py:182-207(`marlin_zero_points`) | R3a | P3+P4 | RECOMPUTE | 低 |
| `layer.g_idx` / `layer.g_idx_sort_indices`(AWQ 无 act-order → 空张量) | marlin.py:178-180;marlin_utils.py:423-426(numel=0 Parameter) | R3a(空) | 无需 | SCRATCH | 无(numel=0,清零无义) |
| `kernel.workspace`(marlin 全局归约锁 buffer,int32 zeros,SM 数大小) | marlin.py:114-115;marlin_utils.py:399-407 | R4(挂在 MPLinearKernel Python 对象上,**非 module 参数/buffer**)+ R5(地址被 CUDA graph 捕获) | P2/P3 均不可见;P4 重跑时**新建**(换地址) | SCRATCH(内容期望恒为零) | 中:见特殊发现 2 |
| `kernel.is_k_full`、`self.quant_config` 等 | marlin.py:104-105;auto_awq.py:426-429 | — Python 标量/对象 | 不在 GPU | PRESERVE | 无 |
| `lm_head.weight`(未量化,独立) | qwen3.py:301-306;auto_awq 对非量化层返回 unquantized 方法 | R1 | P3 | RESTORABLE | 低 |

## 特殊发现

1. **`_noop_loader` 锁死直接重载(R3c 风味)**:`_convert_awq_to_standard_format` 用 `weight_loader=_noop_loader` 的新参数整体替换 `qweight/qzeros`(auto_awq.py:129-140,160-168)。PWAL 之后,任何**绕过 layerwise-reload 机制**、直接对现存参数调 `weight_loader` 的权重同步(某些 RL 框架的 naive 二次 `load_weights`)会被静默吞掉——不报错、不更新。官方 reload 路径之所以安全,是因为 `record_metadata_for_reloading` 在模型 init 后、任何加载/PWAL 之前捕获了原始参数(含原 weight_loader,meta.py:35-40 保留 `__dict__`),reload 时 `restore_layer_on_meta` 用它重建 checkpoint 格式参数(layerwise.py:119;meta.py:115-134)。重设计需把"PWAL 后参数不可再直接 load"显式化。
2. **marlin workspace 的双重身份(R4+R5)**:内容上它是"期望恒为零"的锁 buffer(marlin_utils.py:405-407),sleep-L2 把它清零反而回到静息态——**内容无风险**;但它 (a) 不是 module 状态,P2 存不到、named_buffers 看不到;(b) 每次 PWAL 重跑都 `marlin_make_workspace_new` 新建换地址(marlin.py:115),而 `apply_gptq_marlin_linear` 的 workspace 指针已被捕获进 CUDA graph(marlin.py:230-245)——reload 后 graph 若不重捕获,将继续写旧地址(CuMem 池 VA 仍映射,不会立刻炸,属静默隐患)。终态建议:workspace 声明 SCRATCH 但要求**地址稳定**(注册为 persistent=False buffer 或复用旧张量)。
3. **PWAL 非幂等**:对已 repack 的 qweight 再跑一次 `_convert_awq_to_standard_format`+`gptq_marlin_repack` 会产出垃圾(位序/形状全变),属 R3 全家的共性——只有"先回灌 checkpoint 格式再 PWAL"(P3 语义)是合法的重放方式。
4. **shape 变化**:PWAL 后 qweight 从 (K, N/8) 变 marlin tile 布局、qzeros 变 w_zp,storage 大小与 checkpoint 不同;reload 的 copy-back(layerwise.py:445-461)按"处理后格式 → 原 kernel 张量"逐名拷贝,保住 CUDA graph 的 data_ptr。基座通用缺口(`q_range` 系裸张量、`cos_sin_cache` 仅 P2)同 #22。

## 结论

Qwen3-14B-AWQ 的 checkpoint 参数(qweight/qzeros/scales)经历两级不可逆变换(AWQ→GPTQ 格式、marlin repack/permute),全部属 R3a/R3b,今日依赖"init 期元数据 + reload 回灌 + PWAL 重跑"(P3+P4)恢复,链路完整但脆弱点有二:PWAL 后参数的 `_noop_loader` 会静默吞掉绕过机制的直接重载;marlin workspace 是 P2/P3 都不可见的 kernel 私有 GPU 张量,内容清零无害但 PWAL 重建换地址与已捕获 CUDA graph 存在指针漂移隐患。重设计时建议将 workspace 升格为地址稳定的注册 buffer(SCRATCH),并对 R3b 参数替换强制保留可重载语义。
