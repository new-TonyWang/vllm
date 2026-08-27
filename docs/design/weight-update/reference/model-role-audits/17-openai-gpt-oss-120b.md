# 模型角色审计 #17 — openai/gpt-oss-120b

基本信息
- 审计基线: vLLM main @ c7ce03bcbd(K3 debug worktree 中的 mxfp4 修复不在本 checkout)
- 模型实现: 与 gpt-oss-20b 共用 `vllm/model_executor/models/gpt_oss.py` 与 `vllm/model_executor/layers/quantization/mxfp4.py`(GptOssMxfp4MoEMethod);状态角色结构与 #12 报告完全同源,本报告聚焦 120b 的规模/EP 差异
- 规格差异: 36 层、128 experts / top-4(20b 为 24 层、32 experts),多机 EP / 大 TP 是主流部署形态
- 官方审计草稿: 无(本文为独立审计)

## 状态角色清单

(角色结构与 #12 相同,下表仅列出 120b/EP 场景下评估有变化或需强调的条目;未列条目结论同 #12。)

| 状态 | file:line | 角色 | 现有保护 | 终态声明 | 今日 sleep-L2 风险 |
|---|---|---|---|---|---|
| `_cache_permute_indices`(每层 method 实例一份 dict) | mxfp4.py:145;oracle/mxfp4.py:756-870;flashinfer core.py:170-190 | R4 有状态缓存 | 无(P2/P3/P4 均不触及) | RECOMPUTE(PWAL 前失效) | **高 — root-cause-1,main 未修复**。120b 目标硬件即 SM100(TRTLLM backend 默认命中),且 36 层 × 每层独立 dict,污染面更大;EP 切分后 per-rank shape 不同于 20b,但 key 按 shape 命中的机制相同 |
| `gemm1_alpha` / `gemm1_beta` / `gemm1_clamp_limit`(每层 experts 对象,`[val] * local_num_experts` float32 CUDA 张量) | trtllm_mxfp4_moe.py:55-80;常量源 mxfp4.py:416-418 | R5 graph 捕获常量 | 无 | PRESERVE(池外)或原位刷新 + 重捕获 | **高 — root-cause-2,main 未修复**。张量长度 = local_num_experts(EP 下 128/ep_size),graph 捕获后地址固化;sleep-L2 归零 → replay 读零 |
| MoE 权重 EP 切分加载(`ep_rank_start:ep_rank_end`) | gpt_oss.py:1114-1119(切分参数);422, 440, 470, 493, 512, 532(mxfp4 分支各处) | R1(EP 分片) | P3 copy-back(须按同一 EP 布局)+ P4 | RESTORABLE | 低——但 reload 权重传输层必须重放同一 EP/TP 切分逻辑;`_load_weights_mxfp4` 的切分参数(tp_size 经 `flatten_tp_across_dp_and_pcp`,gpt_oss.py:394-400)是纯函数,可重入 |
| `w2_bias` 非 EP 分支 `weight.zero_()` 污染源张量 | gpt_oss.py:534-536 | 加载路径副作用 | 无 | 应 clone-then-zero | 中(EP 部署不走此分支;混合 TP 部署走) |
| TRITON backend `del` 参数 + 非 Parameter 赋回、`w13/w2_precision_config` | oracle/mxfp4.py:1136-1174;mxfp4.py:346-355 | R3a/R3c | 无 | RESTORABLE(先恢复 checkpoint 形态) | 高(Hopper/TRITON 部署时);SM100 TRTLLM 部署不走此路径 |
| 其余(sinks R1、YaRN cos_sin_cache R2、_ROPE_DICT R6、RMSNorm/线性层 R1、attention scale buffers R2) | 同 #12 报告各行 | — | 同 #12 | 同 #12 | 同 #12 |

## 特殊发现

1. **风险与 20b 同源但暴露面更大**:120b 的默认推荐部署(B200/SM100 + FlashInfer TRTLLM mxfp4 backend,EP≥8)恰好同时激活 root-cause-1(R4 permute cache)与 root-cause-2(R5 gemm1_* graph-orphan)两条路径;20b 在单卡小配置上有时落入 TRITON/Marlin backend 而只暴露 R3c。对 120b 做 RL rollout(sleep-L2 循环)时两个 bug 模式必现。
2. **每层一个 quant-method 实例 → 128-expert 张量 × 36 层**:`_cache_permute_indices` 虽 per-layer,但各层 shape 相同,首层 PWAL 后其余层命中同 shape 也各自缓存一份(dict 独立)。失效钩子必须遍历所有层的 method 实例,不能只清一个"全局"缓存——这是与历史版本(全局 `_cache_permute_indices`)的关键差异,重构时容易漏。
3. **EP 下 experts_per_rank = 128 // ep_size**(gpt_oss.py:1116-1119),整除假设无守护;`gemm1_*` 张量长度随 local_num_experts 变化,EP resize/弹性扩缩时旧 graph 一定要作废(不仅是数值归零问题,还有 shape 失配问题)。
4. Quark 变体(amd/gpt-oss-120b-* W-MXFP4-A-FP8)走 `_load_weights_quark`(gpt_oss.py:570-980),内部依赖 `layer.mlp.experts._quant_method.weight_dtype`(gpt_oss.py:615-617)做分支——reload 时若 quant method 已被 PWAL 改变内部状态,该探测仍安全(weight_dtype 是构造时常量,mxfp4.py:140)。
5. 无 120b 专属代码路径:所有差异均由 config 驱动(num_hidden_layers/num_local_experts/intermediate_size),因此对 #12 的修复自动覆盖 120b;但验证矩阵必须包含 EP>1 + cudagraph 的组合,单卡 20b 复现不了 EP 侧的 graph-orphan 形态。

## 结论

gpt-oss-120b 与 20b 共享同一实现,main 现状同样携带**未修复的 root-cause-1(mxfp4.py:145 R4 permute cache)与 root-cause-2(trtllm_mxfp4_moe.py:55-80 R5 gemm1_* graph-orphan)**;且因其默认部署(SM100 TRTLLM + EP + cudagraph)恰是两个 bug 的激活条件,120b 是该 bug 类的**最高风险实例**。修复验证应以 120b EP≥2 + sleep-L2 + reload + cudagraph replay 为准入用例;同时注意失效钩子需按层遍历 quant-method 实例,并在 EP 弹性场景下强制 graph 重捕获。
