# 模型角色审计 #40 — Qwen/Qwen2.5-Coder-32B-Instruct-AWQ

基本信息: HF 排名 40(下载量榜)。AWQ 4bit(group_size 128, zero_point=true),tie_word_embeddings=false,**大模型多卡 TP 场景**(典型 TP2/TP4)。官方无 draft model。vLLM 实现文件: `vllm/model_executor/models/qwen2.py`(Qwen2ForCausalLM)。64 层,hidden 5120,40 attn头/8 KV头,head_dim 128,intermediate 27648,标准 attention。量化路径与 #18 相同:本 HEAD 已无 `awq_marlin.py`,HF `quant_method="awq"` → `AutoAWQConfig`(auto_awq.py:259-283)→ `AutoAWQMarlinLinearMethod`(auto_awq.py:308-331)→ `MarlinLinearKernel`(kernels/linear/mixed_precision/marlin.py)。lm_head/embed_tokens 不量化(auto_awq.py:242,288-297)。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `qweight/qzeros/scales`(AWQ checkpoint 布局创建,TP 下按 input/output 维分片) | auto_awq.py:478-519 | R1(载入态) | P3+P4:reload 经 checkpoint 元数据还原(model_loader/utils.py:62-64;reload/meta.py:92-134)→ 原格式重载 → 重跑 PWAL → 拷回原 kernel 存储(reload/layerwise.py:113-122,396-426,445-461) | RESTORABLE | 低 — 强依赖 checkpoint-format reload 路径 |
| kernel 态 `qweight/scales/qzeros`(AWQ→GPTQ 转换 + marlin repack/permute/zero_points) | auto_awq.py:528-536,93-168; marlin.py:88-211 | R3a | P4(经 reload 框架) | RECOMPUTE | 中 — 转换非幂等 + no-op loader(见特殊发现) |
| `qkv_proj.bias`(marlin 原地 permute/pad) | qwen2.py:161; marlin.py:213-216 | R1+R3a | P3+P4 | RESTORABLE | 低 |
| `layer.g_idx` / `g_idx_sort_indices`(空张量,无 act_order) | marlin.py:178-180; marlin_utils.py:423-426 | R3a(numel=0) | PWAL 重建 | RECOMPUTE | 无 |
| `kernel.workspace`(每个量化 Linear 一份,SM 数 × int32 零值) | marlin.py:114-115; marlin_utils.py:399-407 | **R4(须恒零 scratch)+ R5(地址经 apply_weights 进 kernel,marlin.py:230-244,可被 graph 按址捕获)**;kernel 对象属性,非模块状态 | 无 P2/P3;sleep-L2 归零 = 恰为合法内容 | SCRATCH(需零初始化) | 中 — reload 重跑 PWAL 重新分配 workspace(marlin.py:115),旧地址若被 graph 捕获则悬空;64 层 × 3 个量化投影 → 暴露面比 7B-AWQ 更大 |
| `self.kernel`(MPLinearKernel:config/is_k_full/参数名映射) | auto_awq.py:521-526; MPLinearKernel.py:37-54 | R4(python 状态) | 常驻;PWAL 重跑刷新 is_k_full/workspace(marlin.py:105,115) | PRESERVE | 低 |
| `model.embed_tokens.weight`、`lm_head.weight`(均不量化) | qwen2.py:366-371,466-471 | R1 | P3 | RESTORABLE | 低 |
| RMSNorm 权重(64×2 + model.norm) | qwen2.py:284-287,390 | R1 | P3 | RESTORABLE | 低 |
| `rotary_emb.cos_sin_cache`(persistent=False buffer) | rotary_embedding/base.py:58-63 | R2;R6 `_ROPE_DICT` 共享(rotary_embedding/__init__.py:30,83-84,383) | P2(gpu_worker.py:270-274,311-316) | RECOMPUTE(理想);今日靠 P2 | 中 |
| `attn._q/_k/_v/_prob_scale`、`q_range/k_range/v_range`、`*_cpu/_float` | attention/attention.py:124-150,184 | R2 | P2 + attention 层 PWAL(attention.py:604-616)/P7;q_range 等普通属性无保护 | RECOMPUTE / PRESERVE(host) | 低(auto kv-cache) |
| `attn.kv_cache` 占位符 | attention/attention.py:461-463 | R4 | P7 + kv 池 | SCRATCH | 无 |

## 特殊发现
- **TP 分片与 marlin 约束**:RowParallel 层(o_proj、down_proj)分片输入维 K,`MarlinLinearKernel.can_implement` 要求每 rank K 可被 group_size=128 整除(marlin.py:71-80);32B 的 down_proj K=27648 在 TP2/4/8 下均整除,qkv/o_proj K=5120 亦然 → 走 marlin 快路径而非回退 `AutoAWQLinearMethod`(auto_awq.py:318-333 的回退分支)。tile 不对齐由 PWAL 零填充解决(marlin.py:82-83,112)——**padding 属于 R3a 重算的一部分**,reload 元数据是未 pad 的 checkpoint 形状,重跑 PWAL 后再次 pad,与 kernel 存储形状一致(reload/layerwise.py:453 直接 `copy_` 要求形状吻合,该不变量由变换确定性保证)。
- **PWAL 非幂等**:`_convert_awq_to_standard_format`(auto_awq.py:93-168)只能吃 AWQ 原布局;对 kernel 态重跑会产出垃圾。reload 框架靠首次 PWAL 前记录的元数据规避(R3a 而非 R3c)。
- **no-op weight_loader**:转换后 qweight/qzeros 挂 `_noop_loader`(auto_awq.py:129-139,160-168)→ 绕过 layerwise 框架直接按参数名调 weight_loader 推权重会被静默丢弃;RL 权重直推方案必须走 `reload_weights(is_checkpoint_format=True)`(gpu_model_runner.py:5527-5530)。
- **每 rank 独立 workspace + 独立 _ROPE_DICT**:TP 各进程各有全局缓存与 workspace,无跨 rank 共享状态;NCCL/权重传输引擎注册的是参数存储指针(R7 归属权重传输侧,模型侧无 data_ptr 外泄)。
- `_kernel_backends_being_used` 类级 set(auto_awq.py:424):仅日志。

## 结论
今天在 sleep-L2 + reload 生命周期下**基本安全**,但正确性由四块拼图串联:P2(R2 buffers)、reload 元数据还原 + P4 重跑(AWQ R3a 链,唯一能重建 kernel 态权重的通道)、P3 拷回原存储(保 CUDA graph 地址)、marlin workspace"归零即合法"的巧合。TP 大模型场景把两类风险放大:一是任何 rank 的 reload 不完整都会留下静默零权重分片(完整性门 reload/layerwise.py:321-340 必须全 rank 通过);二是 64 层 × 3 投影的 workspace 在 PWAL 重跑后全部换新地址,R5 悬空写风险面显著大于 7B-AWQ,"reload 后不重新 capture graph"的隐含假设应被显式化。终态契约的改变:kernel 态权重声明 RESTORABLE(经 R3a),workspace 声明 SCRATCH(零初始化 + 地址稳定性要求或强制 re-capture),PWAL 幂等性成为可声明、可校验属性 —— 契约化后,当前"框架惯例 + 巧合"支撑的 AWQ-TP 生命周期才有可审计的安全边界。
