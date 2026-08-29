# Weight Update Design Review Notes

## 1. 是否可以用 checkpoint schema metadata 取代显式 restore hook？

在“4. 调用栈设计”中，扩展点 C 要求量化方法实现
`restore_weights_before_loading()`。这里需要进一步讨论：是否可以在首次加载前保存
量化参数对应的 checkpoint schema，例如原始 `dtype`、`shape`、`stride`、参数类型及
`weight_loader` 等信息；reload 时根据这些 metadata 临时恢复 loader 所期望的参数结构，
再调用原生 `weight_loader`？

如果可行，通用 reload 框架便不需要依赖每个量化方法分别实现 restore_weights_before_loading。同时，这种
做法看起来仍可与扩展点 B `refresh_derived_state()` 配合：前者负责恢复 checkpoint
加载目标，后者负责在更新完成后刷新 runtime derived state。需要进一步确认这种方案对
alias、Buffer persistence、tensor subclass，以及不可逆 layout conversion 的覆盖情况。

## 2. Per-Param Copy 分类 文件中，何时调用 `FINISH` ？

需要一套显式的分片描述与 coverage 机制，用来判断当前 inference rank 所需的所有逻辑
分片是否已经到齐。只有相关参数或处理单元的依赖闭包完整后，才能执行后处理并释放对应
的 staging Buffer。这里还需要区分单个参数、fused 参数、单个 expert 和整个 module
各自的完成边界。

## 3. Reload 如何选择 Per-Param Copy 的处理路径？

Per-Param Copy 将量化 reload 分为六类，但通用 reload 流程本身无法自然判断某个参数
应该进入哪一类。需要明确这项知识由谁声明。我认为可以考虑三类方案：

1. 引入 quant-aware loader。在首次加载时记录每个参数实际执行的量化、布局转换和派生
   逻辑，reload 时根据记录选择对应的处理路径。
2. 由各量化方法提供专用 reload 接口。reload 不再统一调用 `model.load_weights()`，而是
   调用模型或量化方法提供的 `reload_weights()`。
3. 设计更通用的 checkpoint-to-runtime staging/commit 协议，以同一套生命周期覆盖六类
   路径，仅将具体转换逻辑留给原有 `weight_loader`、量化方法或 derived-state hook。

需要重点比较三种方案在模型兼容性、改造范围、可审计性、失败恢复以及冷加载逻辑复用
方面的成本。

## 4. 什么时候需要 staging Buffer？

如果 staging Buffer 能够按需分配，并在对应处理单元完成后立即释放，那么可以考虑让
更多路径统一先写入 checkpoint-format Parameter/Buffer，而不是只为少数量化路径提供
专用暂存逻辑。

例如，即使训练端分两次流式发送一个 fused BF16 QKV 参数：第一次只发送 Q，第二次发送 K/V。
reload 可以在首次分片到达时懒创建临时参数，在 Q/K/V 全部到齐后调用原生 loader 或转换
逻辑，将完整结果提交到原 runtime Parameter 的稳定 storage，随后释放临时 Buffer。类似的，训练端可能使用流式方式发送moe的多个专家，只要当前rank 的专家全部收到就可以调用量化逻辑（无论是blockwise，per-channel，per-tensor）

这套方案需要与 Sharded RDT 的 slice plan、owner routing 和 layer-group pipeline 一起
评估。RDT 已经能够根据训练端和推理端的并行布局传输目标 rank 所需的 slice，有机会解决
并行策略不同导致的 shape 与 ownership 映射问题；但其当前接收路径写入 layerwise reload
Buffer。需要确认能否让 RDT slice 直接写入按参数或处理单元懒创建的 staging Buffer，
同时保持以下边界清晰：

- RDT 负责分片发现、owner routing、传输和传输 Buffer 生命周期；
- reload 负责逻辑分片 coverage、转换依赖、runtime storage 提交和失败语义；
- 传输 group 或 chunk 的结束不能被当作参数、unit 或 module 完成的依据。
