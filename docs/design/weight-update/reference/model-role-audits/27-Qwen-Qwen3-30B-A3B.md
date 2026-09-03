# 模型角色审计 #27 — Qwen/Qwen3-30B-A3B

基本信息
- 审计 HEAD: c7ce03bcbd
- 架构: `Qwen3MoeForCausalLM` → `vllm/model_executor/models/qwen3_moe.py`(registry.py:198)。
- 精度: bf16(无 quant_config)→ MoE 走 `UnquantizedFusedMoEMethod`(routed_experts.py:196-200 兜底),线性层走 UnquantizedLinearMethod。
- 结构要点: 128 专家 top-8;`shared_expert_intermediate_size` 在该 config 中为 0/缺省 → **无 shared expert、无 shared_expert_gate**(qwen3_moe.py:180-202);**无 e_score_correction_bias**(FusedMoE 调用未传,qwen3_moe.py:204-217;该项在 SKIP_TENSORS 中但本模型不存在此张量)。
- MoE 后端: Blackwell 优先 `FLASHINFER_TRTLLM`/`FLASHINFER_CUTLASS`(oracle/unquantized.py:69-93),否则 Triton。

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens、lm_head(可 tie,584-585)、各 RMSNorm、qkv/o_proj、gate_up/down_proj | qwen3_moe.py:466-477,578-585,293-334,404-407 | R1 | P3+P4(reload;record_metadata 于 init 后 utils.py:64) | RESTORABLE | 低 |
| router `gate`(ReplicatedLinear,注意此处传入了 quant_config——bf16 下无效) | qwen3_moe.py:172-178 | R1 | P3+P4 | RESTORABLE | 低 |
| rotary `cos_sin_cache`(non-persistent buffer;`_ROPE_DICT` 全局共享实例) | qwen3_moe.py:311-316; rotary_embedding/base.py:59-71; __init__.py:30,83,383 | R2 + R6 | P2(gpu_worker.py:270-279 备份全部 named_buffers,310-316 copy_ 回写保地址) | RECOMPUTE 或 PRESERVE | 低 |
| MoE `w13_weight` / `w2_weight`(bf16,3D per-expert) | unquantized_fused_moe_method.py:88-138 | R1 | P3+P4 | RESTORABLE | 低 |
| PWAL 权重 shuffle(FLASHINFER_TRTLLM/AITER 需要,oracle/unquantized.py:312-355) | unquantized_fused_moe_method.py:155-177 | R3a | **守卫良好**: `is_weight_update = self.moe_kernel is not None`(:175),reload 时 `replace_parameter(..., prefer_copy=True)` 原位 copy 保 data_ptr(model_executor/utils.py:47-90) | RECOMPUTE(已实现保地址) | 低 — 这是全仓的**正确样板** |
| `self.moe_kernel = make_unquantized_moe_kernel(...)` | unquantized_fused_moe_method.py:184-199 | R3c 守卫(仅首次构建) | 注释明确 bias/SwiGLU 参数会原位更新故无需重建 | PRESERVE(kernel 对象)| 低 |
| TrtLlmBf16Experts 内部状态 | trtllm_bf16_moe.py:31-96 | — | 只有标量配置,**无派生 GPU 常量**(无 gemm1_alpha 等) | — | 无 |
| `_expert_map` / `expert_mask` buffers | routed_experts.py:235-236 | R2 | P2 + SKIP_TENSORS(reload/meta.py:25-32,reload 永不 meta 化) | PRESERVE | 低 |
| EPLB 路由表 buffers(expert_global_to_physical 等,仅 enable_eplb) | routed_experts.py:239-245 | R2 | P2 + SKIP_TENSORS;EPLB 重排走 update_expert_map(262-274) | PRESERVE | 低(默认不启用) |
| MoERunner `_combined_gate_weight`(gate+shared_expert_gate 融合副本,首次 forward 生成、`if None` 守卫) | runner/moe_runner.py:275,332-344 | R3c 锁死隐患 | 无失效机制 | RECOMPUTE(reload 后须置 None) | **本模型不触发**(无 shared_expert_gate → `_fse_fuse_gate=False`,moe_runner.py:274);ROCm FSE 模型有效风险,记录备查 |
| ROCm `_maybe_pad_weight` padding | unquantized_fused_moe_method.py:204-212 | R3a | 注释论证幂等且保 data_ptr | RECOMPUTE | 低 |
| permute 索引缓存(FI 权重 shuffle 用) | oracle/unquantized.py:338(每次调用新建局部 dict) | R4 scratch | 天然无状态 | SCRATCH | 无 |
| MoE workspace | modular_kernel.py:1075-1112(current_workspace_manager) | R4 scratch | 集中管理,非 weights pool | SCRATCH | 低 |

## 特殊发现

1. **Unquantized 路径是权重生命周期的"金标准"实现**:`_setup_kernel` 用 `moe_kernel is not None` 判定 weight-update 再用 `prefer_copy=True` 原位回写(unquantized_fused_moe_method.py:168-199),连注释都点名 "CUDA graphs may have captured the parameter addresses"。其余量化方法(fp8.py:698-711、modelopt.py:1595-1607)尚未跟进此模式,可作为重构目标形态引用。
2. reload 双保险:即使 `_setup_kernel` 重新 register 参数,layerwise 的 `_copy_and_restore_kernel_tensors`(layerwise.py:445-461)也会把处理后的值 copy 回初始 kernel 张量 storage 并重挂,graph 指针双重安全。
3. bf16 无任何 loader 副作用(R3b)与派生 scale(R5);模型文件本体零 register_buffer、零裸张量属性——状态几乎全部集中在 FusedMoE 工厂产物(MoERunner/RoutedExperts)内。
4. `AutoWeightsLoader(..., ignore_unexpected_suffixes=[...scale...])`(qwen3_moe.py:526-538)使同一模型文件可服务 FP8 变体 ckpt;本 bf16 报告不涉及,FP8 路径见 #47/#30 报告。

## 结论

- Qwen3-30B-A3B(bf16)在今日代码下 **sleep-L2 + RL reload 基本安全**:R1 全走 P3+P4,R2 全走 P2(+SKIP_TENSORS),R3a 有守卫且保地址,无 R5 派生常量,R4 均为 scratch。
- 唯一的结构性隐患是 MoERunner._combined_gate_weight 的 `if None` 守卫永不失效(moe_runner.py:339)——本模型因无 shared_expert_gate 不触发,但属于同一工厂代码路径,重构时应把它declared 为 RECOMPUTE 并在 reload 完成钩子中置 None。
- 终态声明:全部 ckpt 参数 RESTORABLE;expert map/mask/EPLB 表 PRESERVE;权重 shuffle RECOMPUTE(保地址);workspace/permute 缓存 SCRATCH。
