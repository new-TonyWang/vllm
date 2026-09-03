# K3 level-2 lifecycle 输出腐蚀根因分析（update-only vs full lifecycle A/B）

更新时间：2026-07-23
作者：Claude（AI 辅助分析，基于日志/源码静态取证，未运行新实验）

## 0. TL;DR

- **破坏发生在 `sleep level 2`（物理丢弃、零备份）；表现为错误的机制是
  "wake + 323 bucket update + finish 未能重建的派生/辅助状态"。**
  即用户问题选项中的 **1 + 6 组合**；选项 3（update 本身）的值级正确性被
  run A 的 bit-exact 结果证明，选项 4（resume）/ 5（wake KV 作为直接原因）
  被 prompt_logprobs 证据排除。
- Run A（update-only）PASS **不证明重建覆盖完备**，只证明"写入的值正确"。
  A 中未被重写的张量保留旧的正确值；B 中同样的张量是 remap 后的未定义页。
  这是 identity update 的**掩蔽效应**，与
  [vllm#48312](https://github.com/vllm-project/vllm/issues/48312)
  第 2 类（runtime value refresh）/ 第 4 类（reload state preservation）
  失败形态一致。

## 1. 对照组

| | A: update-only (PASS) | B: full level-2 lifecycle (FAIL) |
|---|---|---|
| 结果目录 | `agent_run/results/k3-isolate-clean-update-only-r4-10097-8` | `agent_run/results/k3-isolate-clean-full-r4-10097-12` |
| 流程 | abort→pause→reset→323 update→finish→resume | abort→reset→sleep L2→wake weights→pause→reset→323 update→finish→resume→wake kv |
| oracle | PASS（pre/post **逐浮点 bit-exact**，含 prompt_logprobs 与 sampled logprobs） | FAIL（post 确定性、可重复，但从 **prompt 位置 1** 起偏移 0.3–0.7 nat） |
| GSM8K | 128/128 → 128/128 | 128/128 → 3/128 |

镜像（629aec sqsh）、模型 revision、TP4/DP4/EP16、`generic-mla-kv-gc`
overlay、manifest SHA、497,220 tensors / 1.56 TB、323 bucket 内容完全一致。
两个 run 的 reload 拓扑也一致：`2351 mapped + 118 layerwise`
（118 = 92 WNA16 RoutedExperts + 24 MLA + embed + lm_head）。

## 2. 证据链（全部来自现有日志/源码，无新实验）

### 2.1 Run A 证明传输与 reload 链路值级恒等

对两个 run 的 `oracle-sequence-compare.json` 逐字段比对：

- A：`pre.prompt_logprobs == post.prompt_logprobs`（True），
  `pre.token_logprobs == post.token_logprobs`（True）→ **bit-exact**。
- B：两者均 False，首个差异在 **prompt 位置 1**；且
  `post_first_second_same == post_repeat_same == True`（完全确定性）。

推论：
1. publisher→NCCL packed→`load_weights`→layerwise→finish 在
   "旧存储有效" 前提下能做到数值恒等 → **update 机制本身无罪（排除选项 3）**。
2. B 的腐蚀出现在纯 prefill 重算（prefix cache 已 reset）的
   prompt_logprobs 中 → **排除 resume（选项 4）与 wake KV 直接脏读（选项 5）**；
   也排除 kernel 数值噪声（GSM8K 128→3 是崩塌，且 A 证明可 bit-exact）。
3. B 的 GSM8K 输出语法/格式完好但算术检索崩坏（"2\*2=24"、复读循环）
   —— 少数派生状态失效、主体权重完好的典型签名。

### 2.2 sleep L2 是无备份物理丢弃，wake 不恢复任何数据

B 的 `serve-srun.log`：

```text
cumem.py:287  sleep freed 127.04 GiB in total, 0.00 GiB backed up, 127.04 GiB discarded directly
cumem.py:321  wake_up start tags=['weights'] allocations=1118 restore=119.80 GiB
cumem.py:321  wake_up start tags=['kv_cache'] allocations=26  restore=7.24 GiB
```

overlay 版 `cumem.py` 的 `wake_up` 只做 `create_and_map`（映射新物理页），
**页内容未定义、无零值保证**。此后正确性完全依赖三条重建通道：

1. worker 级 buffer 备份：`gpu_worker.sleep`（overlay :196-206）把
   `model.named_buffers()` 全量存 CPU，首次 wake copy 回（:231-236）；
2. 323 bucket 重传：`model.load_weights` 原地写 checkpoint 张量；
3. finish：`finalize_layerwise_reload` = layerwise PWAL 重跑 +
   attention（含 K3 MLA）PWAL 重跑 + `refresh_runtime_weights_after_loading`。

**A/B 唯一语义差别：三条通道都覆盖不到的张量，
A 中 = 旧的正确值，B 中 = 未定义页。**

### 2.3 日志已确认完整/排除的环节

| 环节 | 结论 | 证据 |
|---|---|---|
| 323 bucket 完整送达 | 完整 | 16 rank × 323 `chunk complete`，index=322 @21:45:38 |
| 92 个 WNA16 MoE 层 | layerwise 全 PWAL 重跑 | 1472 条 `Layerwise reload processing`，全部 `loaded==expected==982,646,784`（触发点一致，无静默丢弃） |
| embed / lm_head | 完整 | `loaded==expected==293,601,280` |
| `e_score_correction_bias` | 会被恢复 | ckpt index 含 92 个该张量；publisher 无过滤；`SKIP_TENSORS` 只挡 layerwise 包装，`load_weights` 默认路径原地写 |
| `_expert_map` 等路由表 | 双保险 | 629aec `routed_experts.py:235-245` 注册为 buffer → sleep 备份覆盖 |
| GDN decode 派生 buffer | 双保险 | non-persistent buffer（备份）+ conv/norm loader 重载时刷新（`kimi_gdn_linear_attn.py:507`） |
| rotary cos_sin_cache | 不存在 | K3 MLA 为 NoPE |
| MLA prefill 路径 | 只读 ckpt 参数 | `_forward_prefill_fused` 用 live `kv_b_proj`；prefill backend 只持 workspace |
| 9-token oracle 输出门 | live 路径 | tokens<512 走 multi_stream，`g_proj` 实时计算 |

## 3. 根因候选（按概率排序）

### P1（最可能）：mamba/GDN 状态页 "零假设" 失效 —— 初始化时一次性清零，wake 后无人再清零

- KDA/GDN conv+ssm 状态在 `kv_cache` 池，sleep 同被丢弃，
  wake kv 仅 remap 7.24 GiB、无数据。
- fork 的 kv-zeroing（overlay 名 `generic-mla-kv-gc` 的由来）在
  `v1/core/single_type_kv_cache_manager.py:86`：
  **只有 `FullAttentionSpec/TQFullAttentionSpec/MLAAttentionSpec/HiddenStateCacheSpec`
  组记录 zero-on-allocate 的新块 id，`MambaSpec` 组不记录。**
  任何未被 attention 组记录间接覆盖的 mamba 状态字节区
  （或 wake 时已在 free pool、"曾清零过、此后不再清零"的块），
  其零值依赖启动时的一次性 `zero_()` —— A 中永远成立，B 中被 remap 打破。
- 症状吻合度最高：prefill 自首 token 起**中度**偏移（递归初态污染但受
  门控/衰减约束）、decode **复利恶化**（复读循环、检索错乱）、
  相同请求确定复现（块分配顺序确定 → 同一批垃圾页）。

### P2：24 个 MLA 层的 PWAL 重建链路运行时未按设计生效（W_UK_T / W_UV / `_q_scale_inv`）

- 设计路径：overlay `layerwise.py:50-58` `_is_post_load_attention` 的
  `AttentionLayerBase` 回退分支 → `_finalize_attention_layer` →
  K3 `mla.py:288` PWAL → `replace_parameter(prefer_copy=True)` 原地重灌。
- **finish 全程（~2s）无任何日志可证实该步真的执行**。若 isinstance/
  条件分支在容器实际代码中失配未跑，`W_UK_T/W_UV` 即为零页 →
  decode 吸收路径（`mla.py:484` bmm）整体打瘪。
- 单独无法解释 prompt_logprobs 偏移（prefill 不读这两个张量），
  故排 P1 之后；与 P1 叠加可完整解释 "prefill 中度偏 + decode 崩塌"。

### P3：普通属性张量 / "页内容仍为初值" 假设类（identity 下值中性，真实 RL 更新必炸）

- `models/kimi_k3/nvidia/ops/gate_sigmoid_mul.py:30` `_WEIGHT_RHS_CACHE`：
  模块级 dict，按 `weight.data_ptr()` 缓存 `g_proj.weight.t().contiguous()`，
  **永不失效**；地址跨 sleep/reload 不变 → 真实权重更新后
  ≥512 token 的 prefill 输出门永远用旧权重。identity 下值相同，
  不解释本次 B，但必须与本问题一起修。
- MegaMoE `_transformed_l1/_l2_weights`（普通属性 + finalize early-return
  守卫 + 原始 param 置 None）——本次未启用
  （需 `moe_backend=deep_gemm_mega_moe`），该路径下 L2 sleep 必炸。
- 所有依赖 "remap 后页仍为零/初值" 的位置（padding 区、
  `torch.empty` 的 g_idx 等）：remap 页内容无任何保证。

## 4. 源码调用栈

```text
sleep L2:
  EngineCore.sleep → Worker.sleep                overlay gpu_worker.py:192
    ├─ named_buffers → CPU 备份                  gpu_worker.py:196-206   ← buffer 唯一保险
    └─ SleepModeBackend.suspend → CuMemAllocator.sleep   （discard，0 备份）
wake(weights):
  Worker.wake_up                                 gpu_worker.py:227
    ├─ CuMemAllocator.wake_up → create_and_map   overlay cumem.py:299-370 ← 只映射不恢复
    └─ buffer 备份 copy 回                        gpu_worker.py:231-236
update:
  /start_weight_update → NCCLEngine.start_weight_update
    → initialize_layerwise_reload                overlay layerwise.py:111
  /update_weights ×323 → receive_weights → packed_nccl_broadcast_consumer
    → model.load_weights → wrapped loader → _layerwise_process
      （materialize → 灌权重 → quant PWAL → copy 回原存储） layerwise.py:555-599
finish:
  finish_weight_update → finalize_layerwise_reload       layerwise.py:503
    ├─ finalize_layerwise_processing → deferred_attn
    │    → _finalize_attention_layer → K3 MLA PWAL       mla.py:288
    └─ refresh_runtime_weights_after_loading（GDN 刷新）  kimi_gdn_linear_attn.py:507
wake(kv_cache):
  Worker.wake_up → cumem map(7.24 GiB, 无数据)
    → post_kv_cache_wake_up → init_block_table_layout_tensors
  （mamba 状态零值依赖 zero-on-allocate；MambaSpec 不在记录集合：
    single_type_kv_cache_manager.py:86）
```

## 5. 最小阶段拆分验证方案（按信息量排序）

1. **P0 · 校验和审计（一次 run 定位到张量级，最推荐）**：
   full lifecycle 服务上用 `collective_rpc` 注入只读探针，
   *sleep 前* 与 *finish 后* 各 dump 一次：每个 named param/buffer +
   已知属性张量（`W_UK_T`、`W_UV`、`_q_scale_inv`、`decode_conv1d_weight`、
   `_expert_map`、`e_score_correction_bias`）+ 若干 mamba 状态块的
   SHA/范数，直接 diff。identity update 下任何不相等项即元凶清单。
2. **P1 · sleep level 1 对照**：同一 full 流程仅 level 2 → level 1。
   预期 PASS；PASS 即证明 reload 机制健全、故障特异于 discard 语义。
   最便宜的 "机制无罪证明"，建议最先跑。
3. **P2 · enforce-eager 对照**：level 2 全流程 + 关 CUDA graph。
   PASS → graph 捕获了被 rebind 的旧存储；FAIL → 值级状态缺失。
4. **P3 · 双重 update**：finish 后立刻再做一轮 start→323→finish。
   仍 FAIL → 腐蚀在 reload 机制可达范围之外（普通属性 / mamba 状态 /
   模块级缓存），收窄到 P1/P3 类。
5. **P4 · mamba 状态定点检验**：wake kv 后先跑 oracle（预期 FAIL），
   随后 debug RPC 将所有 MambaSpec 状态区 `zero_()` 再跑同一 oracle。
   回到基线 → P1 成立。P0 与 P4 可合并进一次 job。

## 6. 关键文件索引

- 结果：`agent_run/results/k3-isolate-clean-update-only-r4-10097-8/`、
  `agent_run/results/k3-isolate-clean-full-r4-10097-12/`
- 实际生效代码：629aec 镜像 +
  `/mnt/lustre01/users/inf-aoshen/.cache/k3-overlays/10097/generic-mla-kv-gc/`
- overlay 关键文件：`model_executor/model_loader/reload/{layerwise,meta}.py`、
  `device_allocator/cumem.py`、`v1/worker/gpu_worker.py`、
  `distributed/weight_transfer/nccl_engine.py`
- 629aec 树内（本仓库可 `git show 629aec286d:<path>`）：
  `vllm/models/kimi_k3/nvidia/{mla,linear}.py`、
  `vllm/models/kimi_k3/nvidia/ops/gate_sigmoid_mul.py`、
  `vllm/v1/core/single_type_kv_cache_manager.py`、
  `vllm/model_executor/layers/fused_moe/routed_experts.py`
- 上游背景：vllm#48312（weight reload correctness RFC，
  失败类别 2/4 与本案同形态）
