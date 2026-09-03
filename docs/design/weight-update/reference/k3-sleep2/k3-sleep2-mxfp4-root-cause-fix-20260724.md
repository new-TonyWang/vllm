# K3 sleep/wake level-2 输出损坏：根因定位与最小修复（dummy-9L 验证通过）

更新时间：2026-07-24
作者：Claude / Codex（AI 辅助；所有代码修改仅在 claude-code 调试
worktree，未提交）

## 1. Worktree 与分支

- 新调试 worktree（所有修改仅发生于此，保持未提交）：
  `/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/k3/worktrees/infra-change-capsule-stack-claude-code`
- 分支：`infra-change-capsule-stack-claude-code`，HEAD = `4d3148da816f4b748660d1911f7ca258b4a7b863`
  （与源 worktree 同一 baseline commit）。

## 2. 基线与 dirty 修改一致性

- 创建方式：`git worktree add -b … 4d3148da…` + 应用源 worktree 未提交 diff。
- 快照时刻（2026-07-23）源 dirty diff（7 文件、785 行）SHA256
  `4714a64ecc9f40b32f0265eb74c9175803c805521d637469f8a4c6178c9c96e0`，
  新 worktree 应用后 `git diff` SHA256 **完全一致** → bit 级复制。
- 源 worktree
  `/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/k3/worktrees/infra-change-capsule-stack`
  全程只读，未被本任务修改。注：快照之后源 worktree 被**外部并行工作**继续
  修改（现为 9 个 dirty 文件，含其自己的 mxfp4.py 改动），与本任务无关。
- 新 worktree 当前 dirty = 快照的 7 个文件 + 本任务修复的
  `vllm/model_executor/layers/quantization/mxfp4.py`（+59/−2），共 8 个文件，
  全部未提交。

## 3. 实验矩阵（Kimi-K3-dummy-9L, TP4/DP1/EP4, max_num_seqs=1, max_model_len=1024, 默认 CUDA Graph, 无 MTP, `K3_WEIGHT_LIFECYCLE_OBSERVABILITY=1`）

| 组 | Job | 配置 | fixed-token oracle | 结果目录（绝对路径，含 serve-srun.log / lifecycle.log / oracle-*.json） |
|---|---|---|---|---|
| A | 10171 | update-only（无修复） | **PASS** | `/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/k3/agent_run/results/k3-dummy-cc-update-only-L2-10171` |
| B | 10172 | level-1 + update（无修复） | **PASS**（prompt/sampled logprobs 全 bit-exact） | `/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/k3/agent_run/results/k3-dummy-cc-full-L1-10172` |
| C | 10173 | level-2 + update（无修复） | **FAIL**（错误 token 与历史 10168 逐 token 相同） | `/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/k3/agent_run/results/k3-dummy-cc-full-L2-10173` |
| D1 | 10176 | level-2 + Fix#1 | FAIL→部分修复：prompt_logprobs 变为 bit-exact、step0 正确、decode step1+ 仍偏 | `/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/k3/agent_run/results/k3-dummy-cc-full-L2-10176` |
| D2 | 10177 | level-2 + Fix#1 + Fix#2 | **PASS**（token/prompt_logprobs/sampled logprobs 全 bit-exact；`pre_post_same=true`） | `/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/k3/agent_run/results/k3-dummy-cc-full-L2-10177` |
| E | 10180 | level-1 + Fix#1 + Fix#2 | **PASS**（`pre_post_same=true`、`post_repeat_same=true`；修复后 level-1 无回归） | `/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/k3/agent_run/results/k3-dummy-cc-full-L1-10180` |

历史对照（源 worktree、无 observability）：10170（L1，PASS）
`…/results/k3-dummy-p2-mla-full-10170`；10168（L2，FAIL）
`…/results/k3-dummy-p2-mla-full-10168`。
实验提交脚本：`…/k3/agent_run/scripts/run_k3_dummy_9l_cc_obs.sbatch`
（指向 claude-code worktree + observability，其余参数与 10168/10170 相同）。

## 4. 第一个异常的生命周期阶段

fingerprint 五阶段（initial_load → before_sleep_level_2 → after_wake_weights →
after_finish_weight_update → after_wake_kv_cache）逐张量对比：

- 内容**破坏**发生于 `sleep level 2`（cumem discard；wake 后所有 weights-pool
  张量 digest 变为全零指纹）。
- 第一处**错误值产生**于 `finish_weight_update` 内 MoE
  `process_weights_after_loading` 重跑（Fix#1 前）：转换输出本身就错。
- 第二处错误值暴露于 **resume 后 decode 的 FULL CUDA graph 回放**（Fix#2 前）。
- checkpoint 参数本身全部正确恢复（如 `conv1d.weight`、`o_norm.weight`、
  `kv_b_proj.weight`：wake 后归零、finish 后 digest 精确还原）。

## 5. 具体张量与调用路径 + 错误类别

### 根因 #1：MXFP4 shuffle 置换索引缓存内容被清零（runtime value refresh）

- 张量：`Mxfp4MoEMethod._cache_permute_indices` 内 memoize 的 CUDA 索引张量。
  flashinfer `get_w2_permute_indices_with_cache`
  （`flashinfer/fused_moe/core.py:170-192`）把索引 `.to(dst.device)` 后按
  shape 缓存。
- 调用链：初始 `load_model`（`_maybe_get_memory_pool_context("weights")` 内）
  → `Mxfp4MoEMethod.process_weights_after_loading` → `_setup_kernel`
  → `convert_weight_to_mxfp4_moe_kernel_format`
  （`fused_moe/oracle/mxfp4.py`，TRTLLM 分支 12 处使用该缓存）。
  索引页属于 cumem weights pool → L2 丢弃、wake 将同一虚拟地址重新映射到
  新物理页并清零 → reload 重跑转换时 dict 命中返回**地址仍合法但内容为零的
  索引** → shuffle 全取 row 0 → 重建的
  `w13_weight/w2_weight/w13_weight_scale/w2_weight_scale` 内容错误。
- 证据：C 组 fingerprint 四张量 finish 后 digest/sum 与初始不符
  （sum 600→918 / 1001→932 / 2224→2272 / 2304→2336），而 A、B 组全阶段
  bit 一致；影响 prefill+decode（oracle 自 prompt 位置 1 偏移）。
- 类别：**派生 tensor 未正确重建（内容错误）**，对应
  vllm#48312 类别 2（runtime value refresh）。不是悬垂指针；
  `data_ptr`/虚拟地址全程稳定，错误在于新映射页中的内容未被刷新。

### 根因 #2：trtllm expert 激活常量的 CUDA Graph storage identity 失配

- 张量：`TrtLlmMxfp4ExpertsBase.__init__` 创建的
  `gemm1_alpha / gemm1_beta / gemm1_clamp_limit`
  （`fused_moe/experts/trtllm_mxfp4_moe.py:56-110`；K3 用 SITU 激活，
  alpha=situ_beta、beta=situ_linear_beta，per-expert fp32）。
- 机制：expert 实例在初始 `_setup_kernel`（weights pool 内）构建；decode 的
  FULL cudagraph 捕获这些张量的 storage identity。L2 丢弃后同一虚拟地址被
  重新映射、内容变为零。reload 重建 kernel 生成**新** alpha/beta 张量：
  eager prefill 使用新 kernel 实例，因此 D1 的 prompt_logprobs 已实测
  逐浮点 bit-exact、step0 正确；decode graph replay 仍绑定原来的、内容未被
  重写的 storage → SITU 激活常数为 0 → decode 自 step1 起错。
- 类别：**CUDA Graph 捕获的 storage identity 未被 post-load 重建路径保持**，
  对应 vllm#48312 类别 1（storage identity）。旧 storage 的地址本身仍合法，
  失效的是其内容以及新 kernel 与 graph 可见 storage 之间的绑定。
- 证据：D1 与 D2 的唯一差别是 Fix#2；D1 prefill 精确 / decode 偏，D2 全精确。

### 其余排查项结论

- MLA `W_UK_T`/`W_UV`：为 `kv_b_proj.weight` 的存储 view（data_ptr 相同），
  identity update 后自动正确；deferred finalize 对 2 个 MLA 层
  （layers.3/7）均执行（日志确认），fingerprint 精确恢复。
- `_q_scale/_k_scale/_q_scale_inv/_k_scale_inv`：全阶段不变（bf16 KV 下常量 1）。
- GDN `decode_conv1d_weight`/`decode_norm_weight`：sleep 的 named_buffers CPU
  备份在 wake 恢复 + 自定义 loader 重载时刷新，全阶段 bit 一致。
- MegaMoE `_transformed_l1/l2_weights`：未启用（moe_backend=auto），
  fingerprint 报 not-found；该路径存在同类风险（finalize early-return 守卫
  + 原 param 置 None），后续如启用需单独处理。
- checkpoint 参数恢复：全部正确（见 §4）。

## 6. 最小修复（仅 `vllm/model_executor/layers/quantization/mxfp4.py`，+59/−2 行，未提交）

1. **Fix#1** `_CpuPermuteIndicesCache(dict)`：`__setitem__` 强制 `.cpu()`，
   并以其替换 `GptOssMxfp4MoEMethod.__init__`/`Mxfp4MoEMethod.__init__` 的
   `_cache_permute_indices = {}`。memo 落在 CPU（cumem 不可及、睡眠免疫），
   全部 12 个使用点本就 `.to(device)` 后才索引，无语义变化。
2. **Fix#2** `_preserve_expert_activation_constants(previous, new)`：两个
   `_setup_kernel` 在 `make_mxfp4_moe_kernel` 前保存 `previous_kernel`，
   构建后把新 experts 的 `gemm1_alpha/gemm1_beta/gemm1_clamp_limit` 值
   copy 进旧张量（graph 捕获的存储）并把新 experts 绑回旧张量，保证
   graph 回放读到的存储内容始终正确。

该文件在服务启动脚本的容器白名单内（`start_kimi_k3_dp4_tp4_node.sh`），
故实验直接生效；`fused_moe/oracle/mxfp4.py` 与
`fused_moe/experts/trtllm_mxfp4_moe.py` 无需改动。

## 7. 修复前后 fixed-token oracle

```text
修复前 (C, 10173):  FAIL  post=[113502, 28532, 56875, 153277, 81400, 28554, 118958, 8610]
仅 Fix#1 (D1,10176): FAIL  prompt_logprobs bit-exact；post=[128751, 28358, …]（step0 对、step1+ 错）
Fix#1+#2 (D2,10177): PASS  pre==post==[128751, 114111, 69898, 73588, 42191, 6340, 78, 64761]
                     prompt_logprobs、sampled logprobs 全部逐位一致
```

D2 已确认真实执行 L2 丢弃（`sleep discard allocation tag=weights` 日志、
wake 后各 fingerprint 短暂归零、finish 后逐位还原）。

E 进一步确认同一修复在 level-1 下没有回归：Slurm job 10180
`COMPLETED 0:0`，33/33 bucket、`finish_weight_update`、resume 和 KV wake
均成功，strict oracle 的 pre/post 以及 post repeat 全部一致。

静态检查：

```text
pre-commit run ruff-check --files vllm/model_executor/layers/quantization/mxfp4.py
pre-commit run ruff-format --files vllm/model_executor/layers/quantization/mxfp4.py
git diff --check -- vllm/model_executor/layers/quantization/mxfp4.py
```

三项均通过。

## 8. 对大模型 run 的推论

大模型 K3（r4-10097-12 等）同为 `quantization=mxfp4` + trtllm 路径 +
SITU 激活：根因 #1 解释其 prompt 位置 1 起的 prefill 偏移与
"连贯但严重退化"（92 层 MoE 权重被错误 shuffle），根因 #2 解释 decode
恶化。建议以相同两处修复在大模型 L2 lifecycle 复验
（`generic-mla-kv-gc` overlay 需相应包含新版 mxfp4.py，或改用
worktree 挂载方式）。

## 9. 未提交状态确认

新 worktree `git status`：8 个 modified 文件（原 7 + mxfp4.py），
无 staged、无 commit；`git log -1` 仍为 baseline `4d3148da8`。
