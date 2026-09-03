# 模型角色审计 #38 — Qwen/Qwen3-32B-AWQ

基本信息:
- HF 排名: 38
- 量化变体: AWQ(4bit,group_size=128,zero_point=true,GEMM 格式)→ 本仓库 `auto_awq.py` + Marlin 后端(`AutoAWQMarlinLinearMethod`,auto_awq.py:414-544;kernel `vllm/model_executor/kernels/linear/mixed_precision/marlin.py`)
- draft: 无官方 draft;`SupportsEagle`/`SupportsEagle3` 声明(qwen3.py:272)
- vLLM 实现文件: `vllm/model_executor/models/qwen3.py`(继承 qwen2.py)
- 架构参数: 64 层,hidden 5120,64/8 头,head_dim=128,`tie_word_embeddings=false`;lm_head/embed_tokens 不量化;大尺寸下常 TP≥2(marlin `can_implement` 对 group 跨 TP rank 有校验,marlin.py:71-83)

## 状态角色清单

基座(qwen3/qwen2)通用状态与 #08 报告一致(embed_tokens、lm_head、RMSNorm 系、q_norm/k_norm、cos_sin_cache、attention scales、q_range 系),不重复;AWQ-Marlin 特有状态(与 #32 同路径,64 层规模放大):

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `qweight/qzeros/scales`(checkpoint AWQ 格式) | auto_awq.py:478-519 | R1 | P3(init 期元数据 model_loader/utils.py:64 + reload 回灌 layerwise.py:115-122) | RESTORABLE | 低 |
| AWQ→GPTQ 格式转换后的 `qweight/qzeros` | auto_awq.py:93-168(调用点 528-535) | R3b(参数整体替换,新 weight_loader=_noop_loader,auto_awq.py:129-140/160-168) | P3+P4 | RECOMPUTE | 中:绕过 reload 机制的直接重载被静默吞掉(R3c 风味) |
| marlin repack 后 `qweight` | marlin.py:124-137,210 | R3a | P3+P4 | RECOMPUTE | 低 |
| marlin permute 后 `scales` | marlin.py:139-170,211 | R3a | P3+P4 | RECOMPUTE | 低 |
| `w_zp`(marlin 零点,由 qzeros 派生) | marlin.py:182-207 | R3a | P3+P4 | RECOMPUTE | 低 |
| `g_idx`/`g_idx_sort_indices`(空张量) | marlin.py:178-180;marlin_utils.py:423-426 | R3a(numel=0) | 无需 | SCRATCH | 无 |
| `kernel.workspace`(per-layer,int32 zeros) | marlin.py:114-115;marlin_utils.py:399-407 | R4 + R5(CUDA graph 捕获地址;非 module 状态,P2 不可见) | P4 重跑新建(换地址) | SCRATCH(期望恒为零) | 中:内容清零无害;PWAL 重建导致 graph 指针漂移隐患(详见 #32 特殊发现 2) |
| `lm_head.weight`(未量化,151936×5120) | qwen3.py:301-306 | R1 | P3 | RESTORABLE | 低 |

## 特殊发现

1. 与 #32(Qwen3-14B-AWQ)共享全部 AWQ-Marlin 结论:`_noop_loader` 锁死直接重载、workspace 双重身份(R4+R5)、PWAL 非幂等、qweight shape 在 PWAL 后与 checkpoint 不同(reload copy-back 按处理后格式拷回原 storage,layerwise.py:445-461,保 CUDA graph data_ptr)。
2. 64 层 × 每层 4 个 marlin linear(qkv/o/gate_up/down)→ 每层各持一个 `MPLinearKernel` 实例及 workspace;这些 GPU 张量总量不大(每个 = SM 数 × 4 字节)但数量多(256 个),全部游离于 P2/P3 视野之外——重设计时适合统一为进程级共享 SCRATCH 分配。
3. TP 部署注意:AWQ group_size=128,TP 切分要求 `input_size_per_partition % 128 == 0`(marlin.py:71-80);reload 后 `update_param_tp_status` 重新校准(layerwise.py:417-421),无 Qwen3 特例。
4. 基座通用缺口同 #08:`q_range/k_range/v_range` 裸张量、`cos_sin_cache` 仅 P2、`_ROPE_DICT` R6 共享。

## 结论

Qwen3-32B-AWQ 与 14B-AWQ 状态结构完全一致,只是规模放大:checkpoint 量化参数经两级不可逆变换(R3a/R3b),依赖 init 期元数据 + reload 回灌 + PWAL 重跑恢复;两个系统性脆弱点(PWAL 后参数 `_noop_loader` 静默吞写、marlin workspace 游离于 P2/P3 且 PWAL 重建换地址)在 64 层规模下影响面更大,应在重设计中优先解决。
