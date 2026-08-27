# 模型角色审计 #20 — ibm-granite/granite-4.1-8b

基本信息
- 审计日期: 2026-07-26,repo HEAD c7ce03bcbd
- 实现文件: `vllm/model_executor/models/granitemoehybrid.py`(`GraniteMoeHybridForCausalLM`,registry.py:123)。Granite 4.x "h" 系列为 mamba2+attention 混合架构;若 granite-4.1-8b 实为纯 dense(`GraniteForCausalLM`,granite.py),见文末附录,风险面是本报告的严格子集。
- 关键依赖层: `vllm/model_executor/layers/mamba/mamba_mixer2.py`(MambaMixer2)、`vllm/model_executor/layers/rotary_embedding/base.py`
- 量化: 无(bf16)

## 状态角色清单

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| embed_tokens / qkv/o_proj / MLP / MoE(w13_/w2_) / RMSNorm 权重 | granitemoehybrid.py:343,255,265; granitemoe.py:104 | R1 | reload 重写 checkpoint 参数 | RESTORABLE | 低 |
| lm_head.weight(tie 时直接别名 `= embed_tokens.weight`) | granitemoehybrid.py:665-666 | R1(参数别名) | named_parameters 去重,reload 重写 embed 即覆盖 | RESTORABLE | 低 |
| `mamba.A`(checkpoint A_log → loader 内 `-torch.exp(x.float())`) | mamba_mixer2.py:449-463 | R3b(loader 变换,源→参非自变换) | P5:变换在 weight_loader 内,reload 重灌 A_log 时自动重放,幂等 | RESTORABLE | 低 |
| `mamba.D` / `mamba.dt_bias` | mamba_mixer2.py:455-456,459,464 | R1 | reload 重写 | RESTORABLE | 低 |
| `mamba.conv1d.weight`(init 时一次性 `unsqueeze(1)` 就地改形) | mamba_mixer2.py:441 | R3-view(结构变换只在 __init__ 做一次) | mamba_v2_sharded_weight_loader 就地 `param.data[...].copy_`(mamba_mixer2.py:216),reload 不重复 unsqueeze | RESTORABLE | 低 |
| `mamba.conv_weights`(**conv1d.weight 的非持久 buffer 视图**) | mamba_mixer2.py:442-445 | R3-view / 参数别名 buffer | P2 会把它当 buffer 存回(copy_ 写入同一 storage,无害冗余);reload 侧 `reload/meta.py:59-71,108` 显式识别参数别名 buffer 并从 capture 集剔除,layerwise 拷回原 storage 保住别名(layerwise.py:115-116) | PRESERVE(地址别名必须保持) | 低(有专门处理);若未来 loader 改为替换 `.data` 而非就地 copy,别名断裂 → 高 |
| `mamba._decode_state_offsets`(num_spec>0 时,arange) | mamba_mixer2.py:506-510 | R2(config 派生,非持久 buffer) | P2 `_sleep_saved_buffers`(gpu_worker.py:271-274 named_buffers 含非持久 buffer)+ wake copy-back(gpu_worker.py:311-316) | RESTORABLE(可 RECOMPUTE) | 低 |
| attention `rotary_emb.cos_sin_cache`(仅 `position_embedding_type=="rope"` 时存在;Granite-4 h 系多为 NoPE → rotary_emb=None) | granitemoehybrid.py:273-281; base.py:58-63 | R2(非持久 buffer) | P2 copy-back | RESTORABLE | 低 |
| `embedding_multiplier` / `residual_multiplier` / `attention_multiplier` / `logits_scaling` | granitemoehybrid.py:347,68,155,236,667-671 | R2(python 标量,非张量) | 不占显存,不受 sleep 影响 | PRESERVE(自动) | 无 |
| `mamba.kv_cache`(占位 tuple,运行期绑定 conv/ssm state) | mamba_mixer2.py:498 | R7(外部 kv-cache 地址) | kv_cache wake tag 路径管理,模型外 | SCRATCH/外部 | 低(超出模型职责) |
| `mamba._ssd_kernels_warmed_up`(python bool) | mamba_mixer2.py:479 | R4(warmup 标志) | 无需保护 | PRESERVE | 无 |
| MoE gate(ReplicatedLinear)+ FusedMoE 权重(bf16 未量化) | granitemoe.py:95-104 | R1 | reload 重写;UnquantizedFusedMoEMethod PWAL 中 ROCm padding 已注明幂等且保 data_ptr(unquantized_fused_moe_method.py:204-212) | RESTORABLE | 低 |

## 特殊发现

1. **conv_weights 参数别名 buffer 是本模型最"结构性"的状态**(mamba_mixer2.py:442-445)。它不是独立存储,而是 `conv1d.weight` 的 2D 视图。三条路径都必须保持"别名不断裂":
   - sleep-L2/P2:wake 时 `buffer.data.copy_()` 写入的是 conv1d.weight 的同一 storage,先于 reload 执行,随后 reload 重写参数,视图内容自动一致 —— 顺序上无冲突;
   - reload:`reload/meta.py:59-71` 的 `_is_non_persistent_parameter_alias_buffer` 已把这类 buffer 从 restore 元数据剔除,layerwise 拷回原 storage(P3),别名保持有效;
   - 风险边界:任何把 `conv1d.weight` 换成新 tensor 对象(而非就地 copy)的加载路径都会让 `conv_weights` 指向旧(已 discard)storage,forward(mamba_mixer2.py:812,994)直接读脏数据。
2. **A 参数的 `-exp` 变换是幂等安全的样板**:变换封装在 weight_loader(P5),输入永远是 checkpoint 的 A_log,不存在"参数自变换"(R3b 重跑翻倍)问题。load_weights 中 `A_log→A` 改名在模型侧(granitemoehybrid.py:501-502)。
3. **Granite-4 的 attention 层可能整体无 RoPE**(NoPE,granitemoehybrid.py:280-281 rotary_emb=None),此时全模型唯一的 R2 张量只剩 `_decode_state_offsets`(且仅在开 spec-decode 时)。
4. 四个乘子(embedding/residual/attention/logits_scaling)均为 python float,不是 tensor,天然免疫 sleep-L2;新终态设计无需为其建 buffer。

## 结论

granite-4.1-8b(hybrid 假设)在当前 P2+P3+P5 组合下没有已知的 sleep-L2 静默清零点;核心关注项是 **mamba2 conv_weights 参数别名视图的地址不变性契约**,现有 reload 机制已显式处理,但应在终态设计中把"conv1d.weight 只允许就地写"固化为不变式并加测试。若实为 dense GraniteForCausalLM(granite.py:137,195,313,370-375),状态面只剩 R1 权重 + 标准 rotary R2 buffer + 标量乘子,风险更低。
