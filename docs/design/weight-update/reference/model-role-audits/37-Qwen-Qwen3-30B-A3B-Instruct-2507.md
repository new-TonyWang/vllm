# 模型角色审计 #37 — Qwen/Qwen3-30B-A3B-Instruct-2507

基本信息
- 审计 HEAD: c7ce03bcbd
- 架构: `Qwen3MoeForCausalLM` → `qwen3_moe.py`(registry.py:198),与 #27(Qwen3-30B-A3B)**同一模型文件、同一状态集合**;本报告为独立审计,不依赖 #27 结论成立。
- 精度: 本仓库条目为 bf16。官方另发布 `Qwen3-30B-A3B-Instruct-2507-FP8` 兄弟变体——**FP8 变体不在本条目审计范围**(其路径特征 = Fp8MoEMethod block 量化,见 #47 报告 FP8 段与 #30 报告),本报告聚焦 bf16 且显式标注两者差异点。
- 结构: 128 专家 top-8,无 shared expert(config 无 shared_expert_intermediate_size → qwen3_moe.py:180-202 走 else 分支),无 e_score_correction_bias,router 为 ReplicatedLinear gate(qwen3_moe.py:172-178)。262k 上下文(rope_parameters 差异只影响 cos_sin_cache 内容,不改变角色)。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens / lm_head / RMSNorm(input/post_attention/q_norm/k_norm/final) | qwen3_moe.py:466-477,333-334,404-407,577-585 | R1 | P3+P4 | RESTORABLE | 低 |
| qkv_proj / o_proj / 稠密 MLP(如有 mlp_only_layers) | qwen3_moe.py:293-309,85-127 | R1 | P3+P4 | RESTORABLE | 低 |
| router `gate` | qwen3_moe.py:172-178 | R1 | P3+P4 | RESTORABLE | 低 |
| rotary `cos_sin_cache`(non-persistent buffer;`_ROPE_DICT` R6 共享;2507 的长上下文使该 buffer 显著更大) | qwen3_moe.py:311-316; rotary_embedding/base.py:59-71; rotary_embedding/__init__.py:30,83,383 | R2+R6 | P2(gpu_worker.py:270-279/310-316,copy_ 保地址) | RECOMPUTE 或 PRESERVE | 低;注意 P2 将其整份克隆到 CPU——长上下文模型的 sleep 内存/时延成本项 |
| MoE `w13_weight`/`w2_weight`(bf16) | unquantized_fused_moe_method.py:88-138 | R1 | P3+P4 | RESTORABLE | 低 |
| PWAL 权重 shuffle → kernel 布局(FI TRTLLM/AITER;oracle/unquantized.py:312-355) | unquantized_fused_moe_method.py:155-177 | R3a | 守卫 + `prefer_copy=True` 原位回写(:175-177;model_executor/utils.py:47-90),另有 layerwise copy-back 双保险(layerwise.py:445-461) | RECOMPUTE(保地址,已实现) | 低 |
| `moe_kernel`(仅首次构建,`is_weight_update` 守卫) | unquantized_fused_moe_method.py:184-199 | R3c 守卫(正确用法) | 注释论证 bias/SwiGLU 原位更新无需重建 | PRESERVE | 低 |
| `_expert_map` / `expert_mask`(+EPLB 表,若启用) | routed_experts.py:235-245 | R2 | P2 + SKIP_TENSORS(reload/meta.py:25-32) | PRESERVE | 低 |
| MoERunner `_combined_gate_weight` | moe_runner.py:275,332-344 | R3c 锁死隐患 | 无失效机制 | RECOMPUTE(应在 reload 后置 None) | 本模型不触发(`_fse_fuse_gate=False`,moe_runner.py:274,因不传 shared_expert_gate,qwen3_moe.py:204-217) |
| workspace / permute 缓存 | modular_kernel.py:1075-1112; oracle/unquantized.py:338 | R4 scratch | 集中管理 / 调用级局部 | SCRATCH | 无 |

## 特殊发现

1. 与 #27 状态集合逐项一致(同文件同代码路径);2507 差异(rope theta/长上下文、训练数据)不引入任何新状态角色。唯一量变:cos_sin_cache 随 max_position 增大,P2 的 CPU 备份体积与 sleep/wake 时延随之增大(gpu_worker.py:272-273 是全量 `.cpu().clone()`)。
2. **bf16 与 FP8 变体的分叉点**(用户若换 `-FP8` 仓库,以下立即生效):量化方法从 UnquantizedFusedMoEMethod(守卫+prefer_copy,unquantized_fused_moe_method.py:175-199)切换为 Fp8MoEMethod(每次 PWAL 无条件 `self.moe_kernel = make_fp8_moe_kernel`,fp8.py:711-718;replace_parameter 无 prefer_copy,fp8.py:698-701)。参数值由 layerwise copy-back 兜底,但 kernel/prepare_finalize 重建的孤儿面暴露。风险从"低"升到"中"(详见 #47 FP8 段)。
3. `Qwen3MoeModel.load_weights` 的 `ignore_unexpected_suffixes`(qwen3_moe.py:526-538)允许 bf16 引擎静默忽略 FP8 ckpt 的 scale 张量——这意味着**错把 FP8 ckpt 喂给 bf16 配置不会报错**,RL 权重通道搭建时需在上游校验 dtype,防止 reload 时 scale 被静默丢弃。
4. 该文件支持 EPLB(enable_eplb → 冗余专家 + 路由表 buffers,qwen3_moe.py:157-170,616-631);RL 场景若启用 EPLB,`update_expert_map`(routed_experts.py:262-274)会重建 buffers 并重新 register——与 SKIP_TENSORS 名单耦合(名字不变故仍受保护),但 P2 备份的是 sleep 时刻的映射,wake 后若 EPLB 继续重排属正常流程。

## 结论

- bf16 的 Qwen3-30B-A3B-Instruct-2507 在今日代码下 sleep-L2 + RL reload **安全**:所有 R1 走 P3+P4,R2 走 P2+SKIP_TENSORS,R3a 守卫且保地址,无 R5 派生 GPU 常量。
- 需要声明的终态:ckpt 参数 RESTORABLE;expert map/mask PRESERVE;shuffle RECOMPUTE(保地址);workspace SCRATCH;`_combined_gate_weight` 应全局声明为 RECOMPUTE(本模型未触发但同路径)。
- 行动项(供权重生命周期重设计):① 上游校验 ckpt 精度与引擎精度一致(防静默丢 scale);② 长上下文模型评估 P2 全量 CPU 克隆的成本,cos_sin_cache 更适合 RECOMPUTE 而非 PRESERVE。
