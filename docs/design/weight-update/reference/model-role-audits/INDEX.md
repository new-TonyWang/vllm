# HF Top-50 官方模型 vLLM 状态角色审计 · 索引

日期：2026-07-26　代码基线：/home/inf-aoshen/vllm @ c7ce03bcbd（fork main）
目的：为权重生命周期终态设计（状态契约 RESTORABLE/PRESERVE/RECOMPUTE/SCRATCH）
提供现实依据。榜单来源：HF API `sort=downloads`、`pipeline_tag=text-generation`
（2026-07-26 快照），筛选官方组织模型（含官方量化变体与 draft model），
排除个人账号/finetune/测试仓库/GGUF-only。

角色分类：R1 checkpoint 参数 / R2 config 派生 buffer / R3 权重派生
（a=PWAL 可重算, b=loader 副作用, c=幂等守卫锁死）/ R4 转换缓存与
workspace / R5 graph 捕获 kernel 常量 / R6 全局缓存 / R7 外部地址。
现有保护 P1–P7、终态声明四类的定义见
`../k3-sleep2-state-census-20260724.md`。

## 模型清单（按下载排名）

| # | 模型 | vLLM 实现 | 审计批次 |
|---|------|-----------|----------|
| 01 | Qwen/Qwen3-0.6B | qwen3.py | B |
| 02 | Qwen/Qwen3-8B | qwen3.py | B |
| 03 | facebook/opt-125m | opt.py | F |
| 04 | openai-community/gpt2 | gpt2.py | F |
| 05 | Qwen/Qwen2.5-1.5B-Instruct | qwen2.py | A |
| 06 | Qwen/Qwen2.5-7B-Instruct | qwen2.py | A |
| 07 | meta-llama/Llama-3.2-1B-Instruct | llama.py | D |
| 08 | Qwen/Qwen3-32B | qwen3.py | B |
| 09 | nvidia/Qwen3.6-35B-A3B-NVFP4 | qwen3 系（agent 确认）+ modelopt | C |
| 10 | deepseek-ai/DeepSeek-R1 | deepseek_v2.py + deepseek_mtp.py | E |
| 11 | meta-llama/Llama-3.1-8B-Instruct | llama.py + EAGLE3 draft | D |
| 12 | openai/gpt-oss-20b | gpt_oss.py + mxfp4 | F |
| 13 | Qwen/Qwen3-1.7B | qwen3.py | B |
| 14 | Qwen/Qwen2.5-3B-Instruct | qwen2.py | A |
| 15 | Qwen/Qwen2.5-0.5B-Instruct | qwen2.py | A |
| 16 | Qwen/Qwen3-4B | qwen3.py | B |
| 17 | openai/gpt-oss-120b | gpt_oss.py + mxfp4 | F |
| 18 | Qwen/Qwen2.5-7B-Instruct-AWQ | qwen2.py + awq_marlin | A |
| 19 | Qwen/Qwen3-4B-Instruct-2507 | qwen3.py | B |
| 20 | ibm-granite/granite-4.1-8b | granite 系（agent 确认） | G |
| 21 | Qwen/Qwen2.5-14B-Instruct | qwen2.py | A |
| 22 | Qwen/Qwen3-14B | qwen3.py | B |
| 23 | zai-org/GLM-5.2-FP8 | glm4_moe 系 + fp8 + MTP | H |
| 24 | deepseek-ai/DeepSeek-V4-Flash | deepseek_v4（fork 目录） | E |
| 25 | Qwen/Qwen2.5-Coder-14B-Instruct | qwen2.py | A |
| 26 | google/gemma-3-1b-it | gemma3.py | G |
| 27 | Qwen/Qwen3-30B-A3B | qwen3_moe.py | C |
| 28 | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | nemotron 系 + modelopt | H |
| 29 | nvidia/Gemma-4-31B-IT-NVFP4 | gemma4.py + modelopt (+gemma4_mtp) | G |
| 30 | Qwen/Qwen3-Coder-Next-FP8 | qwen3_next.py（混合 GDN）+ fp8 | C |
| 31 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | llama.py | D |
| 32 | Qwen/Qwen3-14B-AWQ | qwen3.py + awq_marlin | B |
| 33 | EleutherAI/pythia-160m | gpt_neox.py | F |
| 34 | zai-org/GLM-4.7-Flash | glm4_moe_lite 系 + MTP | H |
| 35 | HuggingFaceTB/SmolLM2-135M-Instruct | llama.py | D |
| 36 | deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | qwen3.py | B |
| 37 | Qwen/Qwen3-30B-A3B-Instruct-2507 | qwen3_moe.py | C |
| 38 | Qwen/Qwen3-32B-AWQ | qwen3.py + awq_marlin | B |
| 39 | google/gemma-3-270m | gemma3.py | G |
| 40 | Qwen/Qwen2.5-Coder-32B-Instruct-AWQ | qwen2.py + awq_marlin | A |
| 41 | meta-llama/Llama-3.2-3B-Instruct | llama.py | D |
| 42 | Qwen/Qwen3-0.6B-FP8 | qwen3.py + fp8 | B |
| 43 | meta-llama/Meta-Llama-3-8B-Instruct | llama.py | D |
| 44 | deepseek-ai/DeepSeek-V4-Pro | deepseek_v4（fork 目录） | E |
| 45 | nvidia/GLM-5.2-NVFP4 | glm 系 + modelopt nvfp4 | H |
| 46 | RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic | llama.py + compressed-tensors | D |
| 47 | Qwen/Qwen3-Coder-30B-A3B-Instruct | qwen3_moe.py（含官方 FP8 变体） | C |
| 48 | mistralai/Mistral-7B-Instruct-v0.2 | llama 兼容 | D |
| 49 | microsoft/Phi-3.5-mini-instruct | phi3.py（LongRoPE） | G |
| 50 | deepseek-ai/DeepSeek-V3 | deepseek_v2.py + deepseek_mtp.py | E |

批次：A=Qwen2.5(8) B=Qwen3-dense(11) C=Qwen3-MoE/Next(5) D=Llama 系(8)
E=DeepSeek(4) F=GPT 系(5) G=Gemma/Granite/Phi(5) H=GLM/Nemotron(4)，共 50。

## 映射修正（审计中确认，与初始假设不同处）

- #23/#45 GLM-5.2 → `deepseek_v2.py` 空子类 `GlmMoeDsaForCausalLM`（:1920），
  MTP 被 `config/speculative.py:331-342` 重写为 `deepseek_mtp.py`；
- #09 Qwen3.6 树内无实现，按 `qwen3_5.py` 骨架 + ModelOpt NVFP4 路径审计
  （量化路径结论不受映射影响，报告内已注明）；
- #20 granite-4.1 → `granitemoehybrid.py`；#28 Nemotron-3-Super →
  `nemotron_h.py`（混合 Mamba2+Attn+非门控 MoE，无 rope）；
- #34 GLM-4.7-Flash → `glm4_moe_lite.py`（全 pass 子类，无自有持久状态）。

## 汇总发现（50 模型审计完成，2026-07-26）

### 今日 sleep-L2 + RL reload 下会腐坏的模型（按严重度）

| 级别 | 模型 | 腐坏点 |
|------|------|--------|
| CRITICAL | #24/#44 DeepSeek-V4 系（MegaMoE） | `_transformed_l1/l2` R3c **双重失败**：幂等守卫挡重算（不睡也端旧策略）+ 源参数置 None 使 reload 新权重被 `_place_kernel_tensors` 丢弃；3 入口共享缺陷 |
| CRITICAL | 所有带官方 MTP/EAGLE draft 的模型（#10/11/23/24/28/29/30/34/44/45/50 + #09） | **draft 三不管**：P2 只备份主模型、`reload_weights` 只管主模型、主模型 load_weights 跳过 `mtp.`；draft rope 靠 `_ROPE_DICT` key 碰撞意外幸存；表现为接受率静默塌陷 |
| HIGH | #12/#17 gpt-oss（mxfp4） | 已知双根因在 main 未修：permute 缓存 R4 + gemm1 常量 R5 + TRITON 路径 R3c |
| HIGH | #09/#28/#29/#45 NVFP4 | `a1_gscale/a2_gscale`（oracle/nvfp4.py:507-508）裸张量 graph 孤儿——注意 `gemm1_*/g1_scale_c` 在本分支已用 layer-Parameter 方式修复，**修了三个漏第四个** |
| HIGH | #29 Gemma4-MTP | `_stable_full_lm_head_weight` 惰性 all-gather 副本只在 `load_weights` 失效，原地 RL 更新绕过 → draft logits 过期 |
| 条件 HIGH | 全部 MoE（bias 型路由） | `e_score_correction_bias` = Parameter（P2 不覆盖）∩ SKIP_TENSORS（copy-back 豁免）→ 仅当 RL 流含该 key 才存活；V4 `tid2eid` 同角色却不同生命周期 |
| MEDIUM | AWQ 变体（#18/32/38/40）、FP8 变体（#42/46）、NVFP4 全部 | PWAL 非幂等（R3c 家族）：`_noop_loader` 静默丢弃直接 push / 二次转置 / in-place `mul_`——安全性完全寄生于 layerwise 的先恢复再重跑 |
| LOW/安全 | 全部 bf16 dense（约 24 个）+ #3/4/33 基线 + #20/26/34/39/49 | 纯 R1+R2 形态，P2/P3 全兜住 |

### 对终态设计的现实依据（六条）

1. **干净形态是主流**（约 30/50 纯 R1+R2）：契约对它们零成本——只是把
   "恰好安全"变成"声明安全"。风险集中在量化 MoE + draft 两个维度。
2. **draft 模型是横跨所有家族的最大保护真空**（12/50 受影响）：
   终态"凡分配必有记录"天然消除 main/draft 特判。
3. **打地鼠不收敛的直接证据**：本分支 trtllm_nvfp4 修了 `gemm1_*` 三个
   常量、漏了 `a1/a2_gscale`；FP8 孪生 `_g1_alphas` 未修;
   marlin workspace 每次 PWAL 重分配 vs graph 捕获旧地址。
4. **R3c（幂等守卫/非幂等 PWAL）是跨方案普遍模式**（MegaMoE、ModelOpt、
   AWQ `_noop_loader`、compressed-tensors 二次转置、NVFP4 in-place mul、
   ROCm `_combined_gate_weight`）——影子层重构（PWAL 跑在影子上）整类消除。
5. **成本声明需要分级**：`cos_sin_cache` 在 262k 上下文可达 ~64 MiB——
   PRESERVE 不总是廉价，RECOMPUTE 声明有真实收益。
6. **正确样板在库内已存在**：MLA `prefer_copy=True`、
   `UnquantizedFusedMoEMethod` 守卫重建、mamba view+param-alias 检测、
   flashinfer sinks 原地 refresh——终态契约 = 把这些孤例升格为强制。

### 新发现的细分角色（补充 R1–R7）

- R1 带加载变换（GPT-2 Conv1D 转置、gpt_neox `_repack_qkv`）：绕过
  weight_loader 的直写会静默腐坏;
- R1 内容住 R2 容器（Gemma4 `layer_scalar` buffer）：param-only 传输协议盲区;
- 惰性派生副本 + 失效钩子挂错入口（Gemma4MTP lm_head all-gather）;
- 裸 `copy_` 绕过 loader 的加载（DeepSeek-V4 `attn_sink`）:与 layerwise
  meta 状态不兼容;
- tied embedding 的对象同一性依赖：替换式 reload 静默拆 tie。
