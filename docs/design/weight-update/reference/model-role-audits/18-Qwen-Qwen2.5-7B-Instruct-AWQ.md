# 模型角色审计 #18 — Qwen/Qwen2.5-7B-Instruct-AWQ

基本信息: HF 排名 18(下载量榜)。AWQ 4bit(group_size 128, zero_point=true),tie_word_embeddings=false。官方无 draft model。vLLM 实现文件: `vllm/model_executor/models/qwen2.py`(Qwen2ForCausalLM)。**量化路径注意**:本 HEAD(c7ce03bcbd)已无 `awq_marlin.py`;HF `quant_method="awq"` 经 `AutoAWQConfig.override_quantization_method`(auto_awq.py:259-283)收敛到 `auto_awq`,CUDA 上命中 `AutoAWQMarlinLinearMethod`(auto_awq.py:308-331),底层经 `choose_mp_linear_kernel` 选 `MarlinLinearKernel`(vllm/model_executor/kernels/linear/mixed_precision/marlin.py)。lm_head 不量化(`lm_head_quantized` 默认 False,auto_awq.py:242,288-297),embed_tokens 走 UnquantizedEmbeddingMethod。28 层,hidden 3584,28/4 头,head_dim 128,标准 attention。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `qweight/qzeros/scales`(创建时为 AWQ checkpoint 布局) | auto_awq.py:478-519(create_weights) | R1(载入态) | P3+P4:reload 用 `record_metadata_for_reloading` 在**首次 PWAL 前**存下的 checkpoint 形状元数据重建参数(model_loader/utils.py:62-64;reload/meta.py:92-134),重载原格式权重后重跑 PWAL,再把 kernel 格式结果拷回原存储(reload/layerwise.py:113-122,396-426,445-461) | RESTORABLE | 低 — 但**强依赖 checkpoint-format reload 路径** |
| PWAL 后的 kernel 态 `qweight`(AWQ→GPTQ 布局转换 + marlin repack)、`scales`(marlin permute)、`qzeros`(marlin_zero_points) | auto_awq.py:528-536 + auto_awq.py:93-168;marlin.py:88-211(transform_w_q:124-137, transform_w_s:139-170, w_zp:182-207) | R3a(检查点确定性重算) | P4(经 reload 框架的"元数据还原→重载→重跑"三步) | RECOMPUTE | 中 — 见特殊发现:转换**非幂等**且转换后 weight_loader 被换成 no-op |
| `qkv_proj.bias`(fp16,marlin 原地 permute/pad) | qwen2.py:161; marlin.py:213-216 | R1+R3a(PWAL 原地变换) | P3+P4 | RESTORABLE | 低 |
| `layer.g_idx` / `layer.g_idx_sort_indices`(AWQ 无 act_order → 空张量参数) | marlin.py:178-180; marlin_utils.py:423-426 | R3a(内容为空,numel=0) | PWAL 重建;首次 PWAL 后才存在,不在 restore_metadata 中,reload 时被删再重建(reload/meta.py:120-123 + marlin.py:179-180) | RECOMPUTE | 无(零元素) |
| `kernel.workspace`(marlin 全局同步计数器,`sms*1` 个 int 零值) | marlin.py:114-115; marlin_utils.py:399-407 | **R4(要求恒零的 scratch)+ R5(地址随 apply_weights 传入 kernel,marlin.py:230-244,可被 CUDA graph 按址捕获)**;挂在 MPLinearKernel python 对象上,不是模块参数/buffer | **无 P2/P3**(不在 named_buffers/参数里)。sleep-L2 归零后内容恰好等于合法状态(需零) | SCRATCH(需零初始化) | 中 — sleep 本身无害(零即合法);**reload 重跑 PWAL 会 `marlin_make_workspace_new` 重新分配**,旧 workspace 被释放;若 CUDA graph 已按旧地址捕获,replay 写入已释放/复用内存 |
| `self.kernel`(MPLinearKernel 实例:config、is_k_full、名字映射) | auto_awq.py:521-526; MPLinearKernel.py:37-54 | R4(python 侧状态,无 GPU 数据除 workspace) | 对象常驻,reload 只重跑其 PWAL(is_k_full/workspace 被刷新,marlin.py:105,115) | PRESERVE | 低 |
| `model.embed_tokens.weight`(不量化) | qwen2.py:366-371; vocab_parallel_embedding.py:49-58 | R1 | P3 | RESTORABLE | 低 |
| `lm_head.weight`(不量化,独立 ParallelLMHead) | qwen2.py:466-471; auto_awq.py:288-297(不命中量化) | R1 | P3 | RESTORABLE | 低 |
| RMSNorm 权重(input/post_attention/model.norm) | qwen2.py:284-287,390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(persistent=False buffer) | rotary_embedding/base.py:58-63 | R2;R6 `_ROPE_DICT` 共享实例(rotary_embedding/__init__.py:30,83-84,383) | P2(gpu_worker.py:270-274,311-316) | RECOMPUTE(理想);今日靠 P2 | 中 — 单一保护点 |
| `attn._q/_k/_v/_prob_scale` buffers、`q_range/k_range/v_range`、`*_cpu/_float` | attention/attention.py:124-150,184 | R2 | 同 bf16 版:P2 + attention 层 PWAL(attention.py:604-616)/P7;q_range 等普通属性**无保护** | RECOMPUTE / PRESERVE(host) | 低(auto kv-cache) |
| `attn.kv_cache` 占位符 | attention/attention.py:461-463 | R4 | P7 + kv 池 | SCRATCH | 无 |

## 特殊发现
- **PWAL 非幂等(但被框架化解)**:`_convert_awq_to_standard_format`(auto_awq.py:93-168)无条件假定输入是 AWQ 位序/输出维打包;对已转换的 kernel 态权重重跑会产出垃圾。reload 框架靠"首次 PWAL 前记录的 checkpoint 元数据"(model_loader/utils.py:64)每次从原格式起步,规避了幂等问题 —— 属 R3a 而非 R3c,但**任何绕过 reload 框架、直接对现有参数重跑 PWAL 的设计都会写坏权重**。
- **weight_loader 被替换为 no-op**:转换后新建的 qweight/qzeros `PackedvLLMParameter` 带 `_noop_loader`(auto_awq.py:129-139,160-168)。这意味着 PWAL 之后若不经元数据还原、直接对层调 weight_loader 推送权重,会**静默丢弃**。kernel-format 直拷路径(gpu_model_runner.py:5532-5542 else 分支,`param.copy_`)倒是可用,但要求调用方自己给出 marlin 打包格式。
- **workspace 重分配 vs CUDA graph(R5)**:每次 PWAL 重跑都 `self.workspace = marlin_make_workspace_new(device)`(marlin.py:115),而 reload 框架只保证**模块上的参数/buffer** 地址稳定(reload/layerwise.py:445-461 拷回原存储);kernel 对象上的 workspace 不在此列。eager 调用用新地址无碍;已捕获的 graph 持旧地址,存在悬空写风险。今日实际能跑通多依赖"reload 后不重新 capture 但 workspace 旧内存未被复用"的运气,或 piecewise 编译把 marlin GEMM 留在 graph 外。
- `_kernel_backends_being_used` 类级 set(auto_awq.py:424):仅日志用途,无 GPU 状态。
- R6 `_ROPE_DICT` 同 bf16 版。

## 结论
今天在 sleep-L2 + reload 生命周期下**基本安全,但拼图比 bf16 多且更脆**:① P2 救 R2 buffer;② reload 的"checkpoint 元数据还原 → 原格式重载 → P4 重跑 PWAL → 拷回原 kernel 存储"四步是 AWQ 权重的唯一恢复通道(P3 单独不够,因为 kernel 态形状/布局与检查点不同);③ marlin workspace 靠"归零即合法"的巧合躲过 sleep-L2。风险点:PWAL 非幂等 + no-op loader 使一切非标准更新路径(RL 侧按参数名直推 AWQ 原格式张量而不走 layerwise 框架)静默失败;workspace 的 R5 悬空地址问题在"reload 后继续用已捕获 graph"的场景下是真实隐患。终态契约的改变:qweight/qzeros/scales 声明 RESTORABLE(经 R3a 重算),workspace 显式声明 SCRATCH(需零初始化、地址须稳定或 reload 后强制 re-capture),并要求 PWAL 声明自身幂等性 —— 这三条会把当前靠框架惯例维持的正确性变成可校验的契约。
