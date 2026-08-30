# Weight Update Correctness

RL 训练场景下 vLLM 权重热更新的完整重构方案。基于 88 个 PWAL、181 个 replace_parameter、13 个在线量化实现的源码审计。

## 设计文档

按阅读顺序排列。每个文档同时发布为在线 Artifact（可交互浏览）。

| 文件 | Artifact | 说明 |
|------|----------|------|
| [weight-update-design-v7.html](weight-update-design-v7.html) | [Design v7](https://claude.ai/code/artifact/7db3d1fe-0cd5-43f8-9bae-824ff61564fc) | 主设计文档：三个扩展点（C restore / A weight_loader / B refresh_derived_state）、场景覆盖与 v1/v2 scope、实施计划 |
| [per-param-copy.html](per-param-copy.html) | [Per-Param Copy](https://claude.ai/code/artifact/d0e57a92-65e7-4466-8c4b-f727ec1cf4d8) | Per-Param 六类分类（①②③ eager 完结 / ④⑤⑥ 需要 finish）、SGLang/Miles 逐类对照 |
| [pwal-pipeline.html](pwal-pipeline.html) | [PWAL 管道](https://claude.ai/code/artifact/b90bc156-90c1-46cb-ba68-08a44bead1ed) | PWAL 管道全景：四个挂载点、契约真空分析、refresh_derived_state 契约设计、88 实现全量普查与派生状态清单 |
| [fused-param-scale-buffer.html](fused-param-scale-buffer.html) | [Fused Param Scale](https://claude.ai/code/artifact/e6463868-a8e0-47e1-9962-7556460cb6e5) | NCCL 到达粒度 vs 量化 scale 粒度分析、逐方法 scale 实证（per-tensor 实际是 per-expert） |
| [rfc-landscape.html](rfc-landscape.html) | [SGLang 全景](https://claude.ai/code/artifact/fd60f803-d9d7-44c5-903d-379f14de5dcb) | SGLang/Miles 生态调研：Miles 五步流程、RFC 汇总、三处独立收敛 |
| [cudagraph-refit.html](cudagraph-refit.html) | [CUDA Graph Refit](https://claude.ai/code/artifact/f79830d1-c0f2-46ae-9bc9-4f8061db3aae) | CUDA Graph 权重更新机制：vLLM 用 copy_() 保地址不变，无需 refit |
| [layerwise-reload.html](layerwise-reload.html) | [Layerwise Reload](https://claude.ai/code/artifact/3561b288-4535-4e37-b825-378bebb4332a) | Layerwise reload 分析（参考，我们不采用此路径） |
| [weight-transforms-primer.html](weight-transforms-primer.html) | [权重变换入门](https://claude.ai/code/artifact/4f9216d8-b1b8-477f-9bcf-987510da8e44) | 背景知识：PWAL 的所有操作（reshape/transpose/repack/量化/scale 融合/派生等）从简单到复杂讲解 |
| [transport-loading-boundary.html](transport-loading-boundary.html) | — | 传输与加载的边界：逐引擎分析（NCCL/NIXL/Mooncake/RDT）、架构边界 |

## 源码审计

| 路径 | 说明 |
|------|------|
| [reference/model-role-audits/](reference/model-role-audits/) | 50 个模型的 weight loading 角色审计（PWAL 行为、派生状态、kernel format） |
| [reference/model-role-audits/INDEX.md](reference/model-role-audits/INDEX.md) | 审计索引 |
| [reference/k3-sleep2/](reference/k3-sleep2/) | Kimi K3 sleep level 2 的状态普查、根因分析、MXFP4 修复 |

## 我们的 PR

| PR | 状态 | 说明 |
|----|------|------|
| [#49201](https://github.com/vllm-project/vllm/pull/49201) | CLOSED (conflicts) | WeightLoadSession lifecycle 统一。核心 PR，需 rebase |
| [#49459](https://github.com/vllm-project/vllm/pull/49459) | OPEN | Reload layout-identical weights directly |
| [#52497](https://github.com/vllm-project/vllm/pull/52497) | OPEN | Rank-local IPC weight updates |
| [#49519](https://github.com/vllm-project/vllm/pull/49519) | MERGED | Defer post-load attention weight processing |
| [#49558](https://github.com/vllm-project/vllm/pull/49558) | MERGED | Filter packed expert weights during EP loading |
| [#51125](https://github.com/vllm-project/vllm/pull/51125) | MERGED | Size w13 by shard count for non-gated MoE |

## 相关 vLLM RFC

| RFC | 状态 | 说明 |
|-----|------|------|
| [#48312](https://github.com/vllm-project/vllm/issues/48312) | OPEN | Weight Reload Correctness — 7 类 failure taxonomy（主 RFC） |
| [#48920](https://github.com/vllm-project/vllm/issues/48920) | OPEN | Unify Weight Loading Lifecycle（#49201 实现的 RFC） |
| [#48478](https://github.com/vllm-project/vllm/issues/48478) | OPEN | Fail-Closed Graph Storage Contract |
| [#49090](https://github.com/vllm-project/vllm/issues/49090) | OPEN | MTP Completeness → Transaction Boundary |
| [#50079](https://github.com/vllm-project/vllm/issues/50079) | OPEN | Kimi K3 RL Support Roadmap |
| [#46439](https://github.com/vllm-project/vllm/issues/46439) | OPEN | NCCL M2N Sharding-Aware Transfer |

## SGLang/Miles 参考

详细调研见 [rfc-landscape.html](rfc-landscape.html)（[Artifact](https://claude.ai/code/artifact/fd60f803-d9d7-44c5-903d-379f14de5dcb)）。

| 参考 | 说明 |
|------|------|
| [sgl#28565](https://github.com/sgl-project/sglang/issues/28565) | Miles restore/PWAL RPC 上游化（CLOSED — inactivity） |
| [sgl#31783](https://github.com/sgl-project/sglang/issues/31783) | 量化 2026 H2 Roadmap（含 PWAL 标准化 checkbox） |
| [sgl#28585](https://github.com/sgl-project/sglang/issues/28585) | load_weights 和 completeness check 分离 |
| [sgl#31796](https://github.com/sgl-project/sglang/pull/31796) | RL weight update 时保持 NPU 参数（merged） |
