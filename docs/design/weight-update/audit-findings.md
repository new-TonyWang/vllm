# Weight Update Correctness — Independent Audit Findings

## Summary Verdict

The proposed restore-copy-refresh (C-A-B) design is architecturally sound and correctly identifies the root cause (PWAL transforms make checkpoint-format weight_loaders incompatible with runtime-format parameters). The design's key insight — that the checkpoint format is the natural interface between "eager in weight_loader" and "deferred in finish" — matches SGLang's proven boundary. However, there are **three gaps that could cause silent data corruption** if not addressed, and the relationship with the existing layerwise reload infrastructure needs clearer specification. The proposal is viable for v1 scope (dtype-matched transfers); the path to v2 (online quantization) is clean but unproven.

---

## 1. Correctness

### 1.1 The restore-copy-refresh sequence is logically correct for the stated categories

The six-way classification (types 1-6) is a correct partition of the problem space. The key invariant — "weight_loader always sees checkpoint-format shapes because restore undoes PWAL transforms before load, and refresh redoes the necessary subset after load" — holds provided the restore is an exact inverse.

### 1.2 GAP: resize_ during restore can change data_ptr even when inference is paused

The brief acknowledges this: "resize_ 可能换地址的问题(推理已暂停)". The brief's answer ("inference is paused") is **necessary but not sufficient**.

The real issue: `_copy_and_restore_kernel_tensors` (line 392 of `layerwise.py`) performs `param.data.copy_(getattr(layer, name))` to write processed values back into the **original** kernel tensor storage. This works because the kernel tensors dict is captured before meta-device restoration. But the proposed design adds a new wrinkle:

- In type-5 layers, restore does `param.resize_(original_checkpoint_shape)` which may change the storage if the new size exceeds the original allocation.
- After load+refresh, the param needs to be resized back to runtime shape and the data written into the **original** data_ptr captured by CUDA graphs.

The existing layerwise reload code handles this through `_copy_and_restore_kernel_tensors`: it saves references to the original kernel tensors, processes on a separate materialized copy, then `copy_`s back. So the design's claim that "resize_ is safe because inference is paused" is correct **only if the final write-back path preserves the original data_ptr**. In the existing code, this is handled. The concern is whether the new "refresh_derived_state" path correctly routes through the same `_copy_and_restore_kernel_tensors` mechanism, or whether it introduces a parallel code path that might bypass it.

**Failure scenario**: If a type-5 PWAL (e.g., FP8 Marlin `weight.t()` at `fp8.py:374`) is "refreshed" by re-transposing in-place but the result is stored on a new allocation (because `resize_` allocated new storage), CUDA graphs would read stale data from the old pointer.

**Recommendation**: The design should explicitly specify that refresh_derived_state must always write through `copy_` into pre-captured kernel tensor storage, never through `resize_` + assignment. This is the contract the existing layerwise reload enforces, and the new extension point should inherit it.

### 1.3 GAP: FP8 MoE _setup_kernel unconditionally rebuilds moe_kernel

Looking at `fp8.py:649-687`, `_setup_kernel` calls `make_fp8_moe_kernel()` on **every** invocation. Unlike `UnquantizedFusedMoEMethod._setup_kernel` (which guards with `is_weight_update = self.moe_kernel is not None` and skips kernel creation on updates), the FP8 path always creates a new `moe_kernel`. The brief claims "kernel 只建一次" but this is not currently true for FP8 MoE.

The `flashinfer_b12x_moe.py` expert also creates derived state (`w1_sf_mma`, `w2_sf_mma`) during its own `process_weights_after_loading` (line 142-157). These are stored as instance attributes on the expert object, not as layer parameters. If the kernel is rebuilt, these are recreated; if not, they become stale.

**Failure scenario**: Under the "no kernel rebuild" contract, if weights change but `w1_sf_mma` is not recomputed, the MMA-layout scale factors used by FlashInfer would correspond to the old weights while the actual weight data is new. This would produce silently wrong outputs.

**Recommendation**: For MoE kernels that store derived state as instance attributes (not layer parameters), the refresh_derived_state contract must explicitly cover these. Either (a) always recompute them (which is cheap — it's just `flashinfer_convert_sf_to_mma_layout`), or (b) treat them as type-6 and include them in the refresh scope.

### 1.4 MLA derived state (W_UV, W_UK_T) is correctly identified

The `mla_attention.py:1016-1118` shows that `process_weights_after_loading` derives `W_UV` and `W_UK_T` from `kv_b_proj` weight. Lines 1116-1118 already use `replace_parameter(self, "W_UV", ..., prefer_copy=True)`, which preserves data_ptr. This is type-6 and the design correctly places it in the refresh_derived_state category. The existing code is already weight-update-aware.

### 1.5 Attention layer deferred processing is a subtle ordering dependency

In the existing layerwise reload (`layerwise.py:281-285`), attention layers are deferred until after all other layers are processed. The design's refresh_derived_state for MLA needs the same ordering guarantee — kv_b_proj must be loaded before W_UV/W_UK_T can be derived. The brief doesn't explicitly address this, but the existing infrastructure handles it.

---

## 2. Completeness

### 2.1 The six categories cover the observed landscape

Walking through the actual PWAL implementations:

| PWAL pattern | Files | Category |
|---|---|---|
| No-op (return) | base_config default, ~70+ methods | type 1 |
| dtype cast only | (handled by copy_ auto-cast) | type 2 |
| FP8 per-tensor requant + transpose | fp8.py:370-416 | type 5 |
| FP8 MoE scale processing | fp8.py:689-738 | type 3/4 |
| Unquantized MoE kernel format | unquantized_fused_moe_method.py:174 | type 5 |
| MLA derived state | mla_attention.py:1016 | type 6 |
| FlashInfer B12x scale layout | flashinfer_b12x_moe.py:93 | type 6 (unlisted) |
| CUTLASS FP4 scale fusion | cutlass_moe.py:695 | type 3 |
| Marlin weight repack | fp8.py:371-379 | type 5 |
| wNa16 Marlin pack | (compressed_tensors) | type 5 |

### 2.2 GAP: Expert-level derived state is a missing sub-category

The FlashInfer B12x `w1_sf_mma` / `w2_sf_mma` pattern (line 93-157 of `flashinfer_b12x_moe.py`) stores derived tensors on the **expert object** (`self.w1_sf_mma`), not on the layer. These are not parameters, not buffers — they're plain Python attributes on a `FusedMoEExpertsModular` instance. The layerwise reload infrastructure's `_copy_and_restore_kernel_tensors` only handles `parameters` and `buffers` via `get_layer_params_buffers`. Expert-object attributes are invisible to it.

This is not covered by any of the six categories. It's closest to type 6, but the design's refresh_derived_state is defined as "per-module" where the module is the layer. The expert is a sub-object of the method, which is itself an attribute of the layer.

**Recommendation**: Add a sub-category 6b for "kernel-object derived state" and specify that refresh_derived_state on a MoE layer must walk into `self.quant_method.moe_kernel.fused_experts` (or equivalent) and recompute its cached scale transforms.

### 2.3 The HPC rope_norm pattern is another derived-state variant

`hpc/rope_norm.py:227` has its own `process_weights_after_loading` that fuses rope parameters into attention weights. This is type 6 (derived state from combining multiple params). The brief's framework handles it, but implementors should be aware.

---

## 3. Feasibility

### 3.1 PWAL refactoring scope is accurately estimated

The brief claims ~10-15 real PWAL implementations need modification. Counting from the grep results: 173 mentions of `process_weights_after_loading` in `quantization/`, 24 in `fused_moe/`. But most are either (a) the base class no-op, (b) calling through to a sub-method (`self.fp8_linear.process_weights_after_loading`), or (c) trivial in-place scale adjustments that are already idempotent.

The truly layout-changing PWALs that need restore + refresh:
1. FP8 linear Marlin path (transpose) — `fp8.py:374`
2. FP8 linear non-Marlin path (transpose) — `fp8.py:405`
3. Unquantized MoE kernel format conversion — `unquantized_fused_moe_method.py:132-137`
4. FP8 MoE kernel format conversion — `fp8.py:660-667`
5. MLA attention derived state — `mla_attention.py:1116-1118`
6. FlashInfer B12x scale layout — `flashinfer_b12x_moe.py:142-157`
7. wNa16 Marlin pack (compressed_tensors) — exists in SGLang, likely exists/will exist in vLLM
8. MXFP4 weight packing — `mxfp4.py` (not checked, but known)

The estimate of 10-15 is reasonable. The key work is extracting pure functions from these, not the number of call sites.

### 3.2 Pure function extraction is straightforward for most cases

The transpose case (`weight.t()`) is trivially extractable. The `convert_to_unquantized_kernel_format` and `convert_to_fp8_moe_kernel_format` are **already** pure functions — they take tensors in, return tensors out. The real extraction work is separating the "transform data" logic from the "register new parameter + build kernel" logic, which the unquantized MoE method already demonstrates with the `is_weight_update` pattern.

### 3.3 Risk: fp8_linear.process_weights_after_loading chain

FP8 linear's PWAL delegates to `self.fp8_linear.process_weights_after_loading(layer)` (line 379, 416), where `fp8_linear` is a kernel-specific object (Marlin, Cutlass, etc.) that may do its own weight repacking. The design needs to handle this two-level delegation: the outer FP8 method decides transpose/requant, then the inner kernel method does format-specific packing. The pure-function extraction must capture both levels.

---

## 4. Risk

### 4.1 Flag-based old/new stack switching is safe with proper fallback

The brief proposes a flag to switch between the new C-A-B stack and the old "rerun full PWAL" fallback. This is sound:
- The old path (existing layerwise reload) is battle-tested
- The new path adds restore (C) and refresh (B) around the same weight_loader (A)
- The flag can be per-quant-method, so unimplemented methods fall back gracefully
- The "fail-closed" policy (reject weight update for un-annotated quant methods) prevents silent corruption

### 4.2 Risk: Partial updates and crash recovery

The brief doesn't address what happens if weight transfer fails mid-stream (e.g., NCCL timeout during receive_weights). Looking at `nccl_engine.py`, there is no rollback mechanism — `start_weight_update` has already moved parameters to meta device via `initialize_layerwise_reload`. If the transfer fails:
- Some layers may have received new weights, others haven't
- `finalize_layerwise_reload` will place kernel_tensors back for unloaded layers (line 266-269 of `layerwise.py`), but the loaded layers will have partial new data

This is a pre-existing risk in the current architecture, not introduced by the design. But it's worth noting that the design doesn't make it worse or better.

### 4.3 Risk: Concurrent access during restore phase

The design says "inference is paused" during the entire C-A-B lifecycle. This is ensured by `start_weight_update` being called from the worker's execute loop (which is single-threaded with respect to inference). The `gpu_model_runner.py` drives this. As long as this serialization is maintained, there is no race condition.

### 4.4 Risk: The is_weight_update pattern in UnquantizedFusedMoEMethod is fragile

The current `is_weight_update = self.moe_kernel is not None` pattern (line 145 of `unquantized_fused_moe_method.py`) works because `moe_kernel` starts as `None` and is set on first PWAL. But this sentinel-based detection is implicit. If any code path sets `moe_kernel` before the first PWAL (unlikely but possible), the first load would incorrectly take the weight-update path and try to `copy_` into non-existent storage. The design should formalize this with an explicit flag rather than relying on sentinel values.

---

## 5. SGLang Comparison

### 5.1 What we do that Miles/SGLang does

- **Same checkpoint-format boundary**: Both designs use checkpoint format as the interface between transfer and processing. SGLang's `update_weights_from_distributed` calls `model.load_weights(weights)` (line 273 of SGLang's `weight_updater.py`), and then (via `load_weights_and_postprocess`) unconditionally reruns PWAL. Our design does the same thing in the old fallback path.

- **restore_weights_before_loading**: SGLang has exactly this interface in `compressed_tensors_wNa16_moe.py:407`. It does `param.resize_(orig_shape)` and sets `is_marlin_converted = False`. Our design proposes the same for type-5 layers.

### 5.2 What we do more than Miles/SGLang

- **Selective refresh instead of unconditional full PWAL**: This is the core value-add. SGLang reruns all of `process_weights_after_loading` including kernel rebuilds. Our design skips kernel rebuilds for layers where the kernel is already correct (types 1-4 in the load phase, kernel reuse in finish). This matters for:
  - FlashInfer B12x MMA layout precomputation
  - CUTLASS expert setup
  - TRTLLM NvFP4 kernel initialization
  - DeepGEMM kernel compilation (if applicable)

- **Fail-closed contract**: SGLang silently proceeds with weight updates for all quant methods. Our design rejects updates for quant methods that haven't declared support. This is strictly safer.

- **Pure function extraction**: SGLang doesn't factor PWAL transforms into reusable pure functions; it just reruns the whole method. Our factoring enables the selective refresh.

### 5.3 What SGLang does that we don't

- **HPC-Ops derived weight cache guard**: SGLang's `_unsupported_derived_weight_cache_error` (line 35 of SGLang's `weight_updater.py`) explicitly rejects weight updates when the HPC bf16xfp32 GEMM has a cached weight split. This is a concrete instance of a type-6 derived state that **cannot** be refreshed because the cache is opaque. Our design should have an equivalent guard for any analogous opaque caches in vLLM.

- **CUDA IPC weight cache guard**: SGLang's `_assert_weight_cache_inactive` prevents updates when weights are shared via CUDA IPC. This is a deployment-level concern that our design doesn't address (though it may not apply to vLLM's architecture).

### 5.4 The differences are justified

SGLang can afford unconditional full PWAL because its primary MoE backend (Triton) has near-zero PWAL cost — the transforms are trivial reshapes that the JIT compiler absorbs. vLLM's backend diversity (DeepGEMM, FlashInfer, TRTLLM, CUTLASS) means PWAL costs are real and heterogeneous. The selective refresh is a necessary adaptation, not premature optimization.

---

## 6. Alternatives

### 6.1 Alternative: Shadow parameter set

Keep a parallel set of parameters in checkpoint format, receive updates into those, then run a focused transform to write into the runtime parameters. This avoids the restore step entirely.

**Pros**: No restore needed, no risk of restore-inverse bugs.
**Cons**: 2x parameter memory. Unacceptable for large models.
**Verdict**: Correctly rejected by the design.

### 6.2 Alternative: Trainer-side transform

Have the trainer apply the same PWAL transforms before sending, so the engine receives runtime-format weights and can just `copy_`.

**Pros**: Zero engine-side transform cost.
**Cons**: Trainer must replicate vLLM's internal weight format knowledge, which varies by quant method, backend, and hardware. This is the coupling nightmare that checkpoint format exists to avoid. Also impossible for derived state (type 6) since the trainer doesn't have the model's attention config.
**Verdict**: Correctly rejected. The checkpoint format is the right abstraction boundary.

### 6.3 Alternative: Incremental PWAL with copy_ (the current layerwise approach)

This is what the code already does: materialize on meta, load into fresh params, run full PWAL, then `copy_` back into original storage. The design's proposal is an optimization of this — skipping unnecessary PWAL steps and avoiding kernel rebuilds.

**Pros**: Already works, battle-tested.
**Cons**: Rebuilds kernels unnecessarily, which has real latency cost.
**Verdict**: This is the fallback path, which the design correctly preserves.

### 6.4 Unconsidered alternative: PWAL idempotency annotation

Instead of extracting pure functions and implementing restore/refresh, annotate each PWAL as idempotent-safe or not. For idempotent PWALs, just rerun them (the common case). For non-idempotent ones (which create new allocations via `replace_parameter`), use `prefer_copy=True` to make them idempotent.

**Analysis**: The `prefer_copy` parameter on `replace_parameter` already exists and does exactly this. The unquantized MoE method already uses it (line 146). The FP8 MoE method does NOT use it (line 673-676) — every `replace_parameter` call in FP8 MoE's `_setup_kernel` lacks `prefer_copy`. If `prefer_copy=True` were added to all `replace_parameter` calls in PWAL methods, many of them would become safe to rerun without restore.

This is a **simpler intermediate step** that could be done before the full C-A-B refactoring: add `prefer_copy=True` to all `replace_parameter` calls in PWAL methods, add the `is_weight_update` guard to skip kernel rebuilds where the unquantized MoE method already demonstrates the pattern, and you get most of the benefit with much less refactoring.

**This does not replace the full design** — type-5 layers with genuine shape changes still need restore, and type-6 derived state still needs refresh — but it could be a valuable stepping stone for v1.

---

## Recommended Changes

1. **Specify the data_ptr preservation contract for refresh_derived_state**: refresh must always write through `copy_` into kernel tensor storage captured before the reload cycle. Document that `resize_` during refresh is forbidden; only restore (before load) may resize, and the layerwise infrastructure handles the write-back.

2. **Add `prefer_copy=True` to FP8 MoE `replace_parameter` calls**: `fp8.py:673-676` should use `prefer_copy=is_weight_update` (following the pattern at `unquantized_fused_moe_method.py:146`). This is a bug fix for the existing layerwise reload path, not just the proposed design.

3. **Address expert-object derived state**: Specify how `flashinfer_b12x_moe.py`'s `w1_sf_mma`/`w2_sf_mma` and similar kernel-object attributes are refreshed. Either include them in the refresh_derived_state scope or document them as requiring kernel rebuild.

4. **Add an opaque-cache guard**: Analogous to SGLang's HPC bf16xfp32 guard, vLLM should reject weight updates when any layer has opaque caches that cannot be refreshed. Check for this in `start_weight_update`.

5. **Formalize the `is_weight_update` detection**: Replace the `self.moe_kernel is not None` sentinel with an explicit flag set by the weight update lifecycle. This flag should be passed through the PWAL call site, not inferred from object state.

6. **Consider the `prefer_copy` stepping stone**: Before the full C-A-B refactoring, add `prefer_copy=True` to all `replace_parameter` calls in PWAL methods as a low-risk first step that unblocks weight updates for many quant methods without the full restore/refresh machinery.
