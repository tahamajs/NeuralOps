# Wave 2 Audit Report: A1 — Models Core Architecture

**Auditor:** Agent A1 (Models Core)  
**Date:** 2026-11-15  
**Target Subsystems:** NeuralOps Models Core (`neuralops/models/modeling*.py`, `configuration.py`, `cache.py`, `protocols.py`, `hybrid_spec.py`, `layer_norm_fix.py`, `gla_hf_model.py`, `cuda_graph.py`, `paged_state.py`, `hub.py`, `__init__.py`)  
*(Note: Explicitly excluded `models/surgery.py` and `models/surgical_io.py` per orchestrator domain boundaries)*  
**Status:** PASS — All 7 targeted P0/P1/P2 issues remediated. 100% test pass on models core regression suite. Ruff linting and formatting passed.

---

## 1. Executive Summary

This deep audit evaluated the core models subsystem of NeuralOps pursuant to Section 3.1 of the Wave 2 Orchestrator plan. Seven (7) high-priority architectural defects (3 P0, 3 P1, 1 P2) were identified, audited, and resolved with rigorous unit tests and backward-compatibility protections.

### Key Metrics
- **Files Audited:** 15 modules in `neuralops/models/`
- **Targeted Bugs Resolved:** 7 (P0-2, P0-5, P0-17, P1-RMSNorm, P1-CUDAGraph, P1-Kwargs, P2-Config)
- **Regression Suite Added:** [`tests/models/test_models_core_wave2_regressions.py`](file:///Users/tahamajs/dev/Project/tests/models/test_models_core_wave2_regressions.py) (7 test functions, 100% pass)
- **Core Models Suite:** 37 passing unit tests across `tests/models/`
- **Lint & Style:** 100% clean under `ruff check`

---

## 2. Issues Investigated & Resolved

### P0-2: `modeling.py::_reorder_cache` Crashes on `UnifiedHybridCache`
- **Root Cause:** `NeuralOpsForCausalLM._reorder_cache` attempted direct iteration over `past_key_values`, assuming a raw tuple-of-tuples layout. When passed a `UnifiedHybridCache` instance, iteration failed with `TypeError` because `UnifiedHybridCache` lacked `__iter__`. Furthermore, rebuilding into a plain tuple discarded `_seen_tokens` and broke polymorphic cache dispatch.
- **Resolution:**
  - Implemented `reorder_cache(self, beam_idx: torch.Tensor) -> UnifiedHybridCache` directly on [`UnifiedHybridCache`](file:///Users/tahamajs/dev/Project/neuralops/models/cache.py#L190-L215), reordering internal states across the batch dimension while preserving `_seen_tokens`.
  - Added `__iter__` to `UnifiedHybridCache` to safely allow layer-level iterations.
  - Updated `NeuralOpsForCausalLM._reorder_cache` in [`modeling.py`](file:///Users/tahamajs/dev/Project/neuralops/models/modeling.py#L278-L305) to check `hasattr(past_key_values, "reorder_cache")` and dispatch seamlessly to the cache's native reordering.
- **Verification:** [`test_p0_2_reorder_cache_unified_hybrid_cache`](file:///Users/tahamajs/dev/Project/tests/models/test_models_core_wave2_regressions.py#L14-L39) tests dense and recurrent layer reordering under index permutation.

### P0-5: `cache.py::UnifiedHybridCache._seen_tokens` Never Updated
- **Root Cause:** `UnifiedHybridCache._seen_tokens` was initialized to 0 and only mutated if external callers invoked `update_seen_tokens`. During standard `NeuralOpsModel.forward` execution with `use_cache=True` and `past_key_values=cache`, `_seen_tokens` remained 0. In autoregressive generation loops, `prepare_inputs_for_generation` calls `cache.get_usable_length()`, which queried `_seen_tokens` (or misidentified recurrent state dims as seq len); returning 0 prevented `input_ids` from being trimmed to the next token.
- **Resolution:**
  - In `UnifiedHybridCache.update_layer`, dynamically computed and added `incoming_tokens` to `_seen_tokens` when updating layer 0. Added full HuggingFace Cache protocol method `update(key_states, value_states, layer_idx, cache_kwargs)`.
  - In `NeuralOpsModel.forward`, when `past_key_values` is an instance of `UnifiedHybridCache`, each layer's present state is populated directly, and `past_key_values.update_seen_tokens(seq_length)` is executed before returning the updated cache as `next_decoder_cache`.
- **Verification:** [`test_p0_5_unified_hybrid_cache_seen_tokens_updated`](file:///Users/tahamajs/dev/Project/tests/models/test_models_core_wave2_regressions.py#L42-L80) confirms sequence length progression across incremental steps and full model forward execution.

### P0-17: `gla_hf_model.py` Broken Generation, Remove GenerationMixin
- **Root Cause:** `GLAModel` explicitly inherited from `GenerationMixin` while lacking proper KV-cache decoding support and returning static full-sequence embeddings in its forward method. Autoregressive generation via `model.generate(...)` crashed due to conflicting parameter signatures and missing recurrent cache bridges.
- **Resolution:**
  - Removed `GenerationMixin` from the module imports and base class list of `GLAModel` in [`neuralops/models/gla_hf_model.py`](file:///Users/tahamajs/dev/Project/neuralops/models/gla_hf_model.py#L76-L95).
  - Explicitly defined `can_generate(self) -> bool: return False` to signal to HuggingFace generation utilities that the model operates as a representation backbone.
  - Removed defunct `prepare_inputs_for_generation` stub.
- **Verification:** [`test_p0_17_gla_hf_model_generation_mixin_removed`](file:///Users/tahamajs/dev/Project/tests/models/test_models_core_wave2_regressions.py#L83-L105) asserts `not issubclass(GLAModel, GenerationMixin)` and verifies causal loss / forward / backward execution.

### P1: `modeling_norm.py::NeuralOpsRMSNorm` Dtype Mismatch
- **Root Cause:** In `NeuralOpsRMSNorm.forward`, the PyTorch reference fallback executed `return self.weight * hidden_states.to(input_dtype)`. When `self.weight` was in `float32` (default initialization) and `hidden_states` was in `float16` or `bfloat16`, PyTorch type promotion promoted the output tensor to `float32`. This caused silent dtype inflation and type mismatches in subsequent layers. Additionally, passing `self.weight` directly to Triton kernels without matching `hidden_states.dtype` caused Triton runtime kernel failures.
- **Resolution:**
  - In Triton execution, converted `weight = self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)` and cast kernel output to `hidden_states.dtype`.
  - In PyTorch reference execution, converted `weight = self.weight.to(device=hidden_states.device, dtype=input_dtype)` and returned `(weight * hidden_states.to(input_dtype)).to(input_dtype)`.
- **Verification:** [`test_p1_rmsnorm_dtype_preservation`](file:///Users/tahamajs/dev/Project/tests/models/test_models_core_wave2_regressions.py#L108-L129) tests `float16`, `bfloat16`, and `float32` tensors against default `float32` parameters.

### P1: `cuda_graph.py` Silent Cache Mismatch (Reject Non-Static Cache)
- **Root Cause:** `CUDAGraphDecodeRunner.capture` accepted any `sample_cache` object without validation. If a dynamic cache (such as HuggingFace `DynamicCache` or dynamically reallocated state) was supplied, CUDA Graph recording and replay would silently reallocate memory addresses, leading to silent memory corruption, wrong decode steps, or driver crashes.
- **Resolution:**
  - Added strict validation in `CUDAGraphDecodeRunner.capture` in [`neuralops/models/cuda_graph.py`](file:///Users/tahamajs/dev/Project/neuralops/models/cuda_graph.py#L63-L84): explicitly rejects caches where `getattr(cache, "is_static", None) is False`, caches with `"Dynamic"` in their class name, or caches that do not conform to `UnifiedHybridCache` or `StaticCache`. Raises explicit `ValueError` or `TypeError`.
  - Marked `UnifiedHybridCache.is_static: bool = True` in [`cache.py`](file:///Users/tahamajs/dev/Project/neuralops/models/cache.py#L27).
- **Verification:** [`test_p1_graph_runner_rejects_non_static_cache`](file:///Users/tahamajs/dev/Project/tests/models/test_models_core_wave2_regressions.py#L132-L151) tests rejection of dynamic caches and acceptance of static `UnifiedHybridCache`.

### P1: `modeling.py::NeuralOpsModel.forward` Kwargs Leak
- **Root Cause:** `NeuralOpsModel.forward` forwarded `**kwargs` indiscriminately to each `decoder_layer`, which in turn passed `**kwargs` directly to `self.self_attn`. Many attention modules in `ATTENTION_REGISTRY` have explicit non-variadic signatures (e.g. `(hidden_states, attention_mask=None, ...)`). Passing standard HuggingFace training/generation arguments (e.g. `labels`, `loss_kwargs`, `cache_position`, `num_items_in_batch`) resulted in unhandled `TypeError: forward() got an unexpected keyword argument` crashes.
- **Resolution:**
  - In `NeuralOpsModel.forward`, filtered out model-level keys (`labels`, `loss_kwargs`, `num_items_in_batch`, `cache_position`, `return_dict`, `output_attentions`, `output_hidden_states`, `use_cache`, `past_key_values`) before dispatching to layers.
  - In `NeuralOpsBlock.__init__` and `forward`, inspected `self.self_attn.forward`'s signature: if variadic kwargs are not accepted, filtered `kwargs` to only the parameters explicitly declared by the attention module.
- **Verification:** [`test_p1_model_forward_kwargs_leak`](file:///Users/tahamajs/dev/Project/tests/models/test_models_core_wave2_regressions.py#L154-L177) validates causal LM forward pass with `cache_position`, `num_items_in_batch`, and custom kwargs.

### P2: `configuration.py::get_layer_attn_spec` $\mathcal{O}(L^2)$ Init
- **Root Cause:** `NeuralOpsConfig.get_layer_attn_spec(layer_idx)` invoked `normalize_hybrid_spec` on every layer invocation. Since `NeuralOpsModel.__init__` instantiates $L$ layers in a loop, `normalize_hybrid_spec` was executed $L$ times for an $L$-length model, causing an $\mathcal{O}(L^2)$ initialization overhead.
- **Resolution:**
  - Implemented cached spec storage (`_cached_attn_specs` and `_cached_comp_specs`) on `NeuralOpsConfig` in [`configuration.py`](file:///Users/tahamajs/dev/Project/neuralops/models/configuration.py#L141-L185). The full specification is normalized once on the first query and reused in $\mathcal{O}(1)$ time for subsequent layers. Automatic cache invalidation occurs if `self.attn` or `self.compression` is modified.
- **Verification:** [`test_p2_config_attn_spec_caching`](file:///Users/tahamajs/dev/Project/tests/models/test_models_core_wave2_regressions.py#L180-L206) tests caching, identical object reuse across a 64-layer model, and cache invalidation upon spec reassignment.

### Architectural Hygiene: Elimination of Import-Time Circular Dependency
- **Root Cause:** `neuralops/models/__init__.py` eagerly imported `ModelSurgery`, `SurgeryTransactionError`, etc. from `neuralops.models.surgery`. In turn, `surgery.py` imported `neuralops.api`, creating an import cycle whenever `neuralops.models` was imported by other packages.
- **Resolution:**
  - Implemented lazy module resolution via `__getattr__` in [`neuralops/models/__init__.py`](file:///Users/tahamajs/dev/Project/neuralops/models/__init__.py#L40-L65), fulfilling Rule 1.3 (no import-time side effects). All symbols in `__all__` remain completely accessible to downstream callers without triggering eager imports of surgical infrastructure.

---

## 3. Regression Test Results

```
tests/models/test_models_core_wave2_regressions.py::test_p0_2_reorder_cache_unified_hybrid_cache PASSED
tests/models/test_models_core_wave2_regressions.py::test_p0_5_unified_hybrid_cache_seen_tokens_updated PASSED
tests/models/test_models_core_wave2_regressions.py::test_p0_17_gla_hf_model_generation_mixin_removed PASSED
tests/models/test_models_core_wave2_regressions.py::test_p1_rmsnorm_dtype_preservation PASSED
tests/models/test_models_core_wave2_regressions.py::test_p1_graph_runner_rejects_non_static_cache PASSED
tests/models/test_models_core_wave2_regressions.py::test_p1_model_forward_kwargs_leak PASSED
tests/models/test_models_core_wave2_regressions.py::test_p2_config_attn_spec_caching PASSED

============================== 7 passed in 15.25s ==============================
```

Existing core models tests:
- `tests/models/test_auto_registration.py`: PASSED
- `tests/models/test_config.py`: PASSED
- `tests/models/test_hybrid_spec.py`: PASSED
- `tests/models/test_kv_cache.py`: PASSED
- `tests/models/test_model_forward.py`: PASSED
- `tests/models/test_modeling_distilbert.py`: PASSED
- `tests/models/test_modeling_gpt2.py`: PASSED
- `tests/models/test_modeling_roberta.py`: PASSED

---

## 4. Conclusion & Hand-Off

All targeted P0, P1, and P2 defects in the Models Core domain are fully resolved, with strict adherence to repository rules (no `main()` in library files, no `torch.cuda.amp` imports, no hardcoded device placement, clean ruff linting, backward compatibility preserved). Models Core is ready for downstream Wave 2 integration.
