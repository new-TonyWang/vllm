# K3 sleep-L2 全库状态普查（深化"例子一"：选择性备份的成员判据）

日期：2026-07-24　作者：Claude（4 路并行普查 + 交叉汇总）
代码基线：`worktrees/infra-change-capsule-stack-claude-code` @ `4d3148da`（含 v2 修复）
目的：把"行李/门禁卡/便签"比喻扩展成完整状态分类学，为例子一
（allocator 层打标 + fail-safe 默认备份）提供可执行的普查清单。

---

## 1. 生命周期契约的权威事实（普查线 3）

- Tag 机制：`cumem.py` 中 `use_memory_pool(tag=...)` 期间的分配被打上
  tag（`_python_malloc_callback` → `AllocationData(handle, tag)`）。
  **只有 pool 上下文内的分配被 CuMem 管理**；上下文外走普通 caching
  allocator，sleep 完全不碰。实际只用两个 tag：`weights`
  （`gpu_worker.py:619-625` 包住整个 `load_model`，含 PWAL）和
  `kv_cache`（`gpu_worker.py:913`）。
- Sleep 语义（`sleep_mode_backend.py:117-122`）：
  L1 → `offload_tags=("weights",)`：weights CPU 备份，kv_cache 丢弃；
  L2 → `offload_tags=()`：**weights 和 kv_cache 全部丢弃、无备份**。
- Wake：同一 VA 重映射到新物理页。kv_cache 页显式 `cudaMemset 0`
  （`cumem.py:403`）；**weights 页不 memset**，内容未定义（通常为零，
  无保证）。
- CUDA graph：capture 在 pool 上下文之外、使用独立 graph pool
  （`cuda_graph.py:200`），graph 本体不受 sleep 影响；
  **wake/reload 后从不重新 capture**（全库无调用点）。正确性完全依赖
  "同一 VA + 原地重写存储"；replay 时有 data_ptr 断言
  （`cuda_graph.py:351`）——地址被换会硬崩，**内容被清零则静默**。
- 刻意放在 pool 外（sleep 不碰）：NCCL 通信 buffer、workspace manager、
  model runner 输入 buffer、采样器/logits 预热分配、KV-zero 元数据。

## 2. 现有保护机制矩阵（含 `_sleep_saved_buffers` 专项分析）

现状的安全性 = 7 个各自为政的机制拼图咬合，每块成员判据不同：

| # | 机制 | 成员判据 | 位置 | 盲区 |
|---|------|----------|------|------|
| P1 | L1 pool 级 CPU offload | `tag == "weights"` | `cumem.py:270-278` | 仅 L1 |
| P2 | **L2 buffer 无差别备份** `_sleep_saved_buffers` | `model.named_buffers()` 可达（persistent 与否均含） | `gpu_worker.py:388-397, 421-427` | 裸属性、非 Module 对象、模块级缓存 |
| P3 | reload copy-back 回原存储 | registered param / persistent buffer 且在 checkpoint 加载集内 | `layerwise.py:443-460` | 派生物、SKIP_TENSORS |
| P4 | PWAL 重跑重算 | "PWAL 恰好会重算的东西"（删 `_already_called…` 标志强制重跑，`layerwise.py:398`） | 各 quant/attention 方法 | **被幂等守卫破坏**（§4-CRITICAL）；重算进新存储则 graph 孤儿 |
| P5 | loader 副作用重建 | 挂在某个基权重的 weight_loader 闭包上 | 如 `kimi_gdn:220-234` | 部分 reload 跳过基权重时失效 |
| P6 | 显式 rebuild/refresh 钩子 | 有人记得写钩子（opt-in） | `_build_fused_kv_buffers`、`init_fp8_kv_scales`、v2 `refresh_activation_constants` | fail-dangerous：漏写即 bug |
| P7 | kv_cache memset-0 + scale 重置 | `tag == "kv_cache"` | `cumem.py:398-412`、`gpu_model_runner.py:976-1021` | 仅 kv_cache |

**P2 专项**（用户要求）：`_sleep_saved_buffers` 是例子一的既有雏形——
L2 并非"什么都不存"，而是对主模型全部 registered buffer 做无差别
CPU 备份 + 唤醒 `copy_` 回写（原存储，graph 安全）。其判据
"是否 `register_buffer` 在 Module 树上"是**代码组织谓词**而非
**可恢复性谓词**：两个已修根因（quant method 上的 dict、kernel 对象上
的裸属性）都活在 Module 树之外，结构性漏网。反过来它拯救了一批被
单一"reload 视角"误判为 HIGH 的状态（`_one_scale`、`cos_sin_cache`、
Mamba2 `_decode_state_offsets`——均为 registered buffer，L2 下全部被
P2 兜住）。draft 模型仅特定 metadata 走 P6 重建。

**腐蚀判定公式**：损坏 ⇔ 睡时在被丢弃 pool 内 ∧ 不被 P2 备份 ∧
不被 P3 重写 ∧ 不被 P4/P5/P6 重算 ∧ 之后其内容仍被读取
（graph 维度：即使重算，若进新存储而 graph 捕获旧存储 → 对 graph
路径仍损坏 = rebuild 反模式）。

## 3. 状态分类总表（比喻扩展版）

| 类 | 比喻 | 定义 | 代表 | 保护 | 判定 |
|----|------|------|------|------|------|
| R1 | 🧳 行李 | checkpoint 参数 | 所有 named params | P3（copy-back 保址） | 安全（设计如此） |
| R2 | 🪑 家具 | config 派生、init 时建、registered buffer | `_one_scale`(mla.py:220)、rotary `cos_sin_cache`、expert maps、`e_score_correction_bias`* | P2 | 安全，但**仅因 P2 兜底**；*bias 是 Param 非 buffer，走 P3，需确认 dtype recast 在 reload 复现 |
| R3a | 👕 重叠的衣服 | 权重派生、PWAL 重算 | `W_UK_T/W_UV`、`_q/_k_scale_inv`、marlin repack、mxfp4 shuffle | P4（MLA 路径显式保址，**样板**） | 条件安全："wake 后必有 reload" |
| R3b | 👕 搭便车重建 | 权重派生、靠别人 loader 副作用 | `decode_conv1d_weight`、`decode_norm_weight`（kimi_gdn） | P5 | LOW-MED：部分 reload 会漏 |
| R3c | 👕 **幂等守卫锁死** | 权重派生、PWAL 重跑被 early-return 挡住 | MegaMoE `_transformed_l1/l2_weights`（linear.py:180，源参数已置 None）；ModelOpt Mxfp8 dequant（modelopt.py:2141） | 无 | **CRITICAL**（本次未启用路径；启用必现） |
| R4a | 🗝️ 有状态门禁卡 | 转换期缓存、内容被信任 | mxfp4 permute memo | v2-P1 已改局部 | 已修；其余 flashinfer 同类均为函数局部（安全） |
| R4b | 🗝️ 无状态门禁卡 | scratch、写后读 | marlin `layer.workspace`、moe sort workspace、flashinfer workspace | 无需保护 | 无害（清零不影响语义） |
| R5 | 📝 便签 | kernel 常量、graph 捕获地址 | mxfp4 `gemm1_*`（已修 v2-P2）；**trtllm_fp8 / trtllm_nvfp4 / flashinfer_cutlass 的 `gemm1_*` + `_g1/_g2_alphas/g1_scale_c`（未修）** | 仅 mxfp4 有 P6 | **HIGH**：全库只有 mxfp4 保留 kernel；其余方法 PWAL 无条件 rebuild → graph 孤儿模式通病 |
| R6 | 📻 电台预置表 | 模块级/全局 CUDA 张量缓存（Module 树外） | 采样器 `_TRITON_TABLE_CACHE`（topk_topp_triton.py:18，常量表；`reset_buffer_cache()` 零调用者）；Lamport `_workspace_cache`（MiniMax） | 无 | **需验证**（见 §4 修正）：惰性初始化在 pool 上下文外 → 当前时序下不被 sleep 触碰；安全是**时序巧合**非契约 |
| R7 | 🔑 外配钥匙 | data_ptr 注册给外部实体 | Lamport IPC handles；NCCL symmetric window（独立 pool） | 无 | 当前部署不涉及；weight-transfer 引擎（ipc/nccl_engine）**确认干净**，不跨 sleep 缓存指针 |

**prefill/decode 双态**（K3 共 4 组：kv_b_proj vs W_UK_T/W_UV；
`_one_scale` vs `_q_scale_inv`；conv1d 视图 vs `decode_conv1d_weight`；
o_norm bf16 vs fp32 拷贝）：双态本身不危险，危险在**两态重建路径分属
不同机制**（P4 vs P5 vs P2），审计时必须逐态核对。

## 4. 新发现隐患（按优先级；均为本次普查新增，非 KNOWN）

1. **CRITICAL** MegaMoE `_transformed_l1/l2_weights`
   （`kimi_k3/nvidia/linear.py:180-211`）：幂等 early-return + 源参数置
   None → reload 重跑 PWAL 也不重算，且**不可重建**。
   `moe_backend=deep_gemm_mega_moe` + L2 必现。
2. **CRITICAL** ModelOpt Mxfp8 dequant（`modelopt.py:2141-2143`）：
   PWAL 重跑被 `_already_called_process_weights_after_loading` 守卫
   no-op；copy-back 还会把 MXFP8 字节灌进 BF16 形状 param。
3. **HIGH** trtllm_fp8 / trtllm_nvfp4 / flashinfer_cutlass 的
   `gemm1_alpha/beta/clamp_limit` 及派生 `_g1/_g2_alphas/g1_scale_c`：
   与根因二同构（rebuild 孤儿），无 refresh 机制。v2-P2 的
   build-once + refresh 模式可直接推广。
4. **需验证（原 agent 判 HIGH，交叉修正为 likely-safe-by-accident）**
   `_TRITON_TABLE_CACHE`：采样器常量表，惰性建于首次采样调用——
   即启动预热（`gpu_worker.py:1046-1065`，pool 上下文外）→ 普通
   allocator → sleep 不碰。**但这是初始化时序的巧合**：任何把首次
   调用挪进 pool 上下文的重构都会引爆它，且 `reset_buffer_cache()`
   无人调用。建议防御性接入 wake 路径。Lamport workspace 同理
   （MiniMax-only）。
5. **VERIFY** `e_score_correction_bias` 的 `out_dtype` recast
   （deepseek_v2.py:393）在 reload 路径是否复现。
6. **VERIFY** quark_moe / compressed_tensors 系 rebuild 路径的
   precision_config 重挂与 graph 孤儿排查（普查线 1 标记
   MEDIUM-VERIFY 清单见 agent 输出）。

## 5. 有价值的空类别（扫过且为空）

- `lru_cache/@cache` 返回 CUDA 张量：40+ 处全为类/配置/bool/CPU 张量。
- torch.compile/inductor 权重烘焙：无（权重始终是 graph 输入）。
- cudagraph 静态输入地址：replay 有断言 → 崩溃而非静默（cumem 保 VA
  后此断言恒通过，风险转移到内容维度）。
- 采样器其余张量：全部每调用 CPU→device 临时构造。
- weight-transfer 引擎跨 sleep 指针缓存：无。

## 6. 对例子一的深化结论

1. **判据碎片化是根因的根因**：7 个保护机制各有成员判据（tag /
   named_buffers / checkpoint 加载集 / "PWAL 恰好重算" / loader 闭包 /
   记得写钩子），肇事者与新隐患全部落在判据缝隙里。例子一 =
   把判据统一到**分配层**（分配必经 allocator，无死角），默认方向
   反转为"未声明可重建 → 备份"。
2. **P2 (`_sleep_saved_buffers`) 证明例子一的形态已被接受**：无差别
   备份小状态 + 唤醒回写，成本可忽略。例子一只是把它的扫描面从
   "Module 树"下沉到"分配记录"。
3. **R6 类揭示第二个隐藏变量：初始化时序**。lazy-init 状态是否在
   pool 内取决于首次调用时机——目前靠巧合安全。allocator 打标把
   时序巧合变成显式契约。
4. **graph 维度正交于来源分类**：R3/R5 即使被重算，重算进新存储
   仍孤儿化 graph（rebuild 反模式）。故例子一需配套"内容原地刷新"
   纪律（v2-P2 / MLA `prefer_copy` 样板）才完整。

## 7. 行动项（并入共识文档 §4）

- [ ] R3c 两个 CRITICAL：MegaMoE 与 ModelOpt Mxfp8 的幂等守卫需改为
  reload 感知（或注册为可 copy-back 的 Parameter 且形状一致）。
- [ ] R5 HIGH：把 v2 的 build-once + `refresh_activation_constants`
  推广到 trtllm_fp8 / trtllm_nvfp4 / flashinfer_cutlass（可上游化为
  `FusedMoEKernel.refresh_after_weight_reload()`）。
- [ ] R6：`reset_buffer_cache()` 接入 wake；lazy-init 状态盘点。
- [ ] VERIFY 清单（§4-5/6）。
- [ ] fingerprint 观测扩展：`gemm1_*`、`_g*_alphas`、采样器表。

来源：4 路并行 Explore 普查（quant/MoE、attention/mamba、生命周期契约、
长尾），交叉修正见 §4-4。实验凭据沿用主报告
`k3-sleep2-mxfp4-root-cause-fix-20260724.md`。
