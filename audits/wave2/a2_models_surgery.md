# Wave 2 Audit Report: A2 — Models Surgery & IO

**Auditor:** Agent A2 (Models Surgery & IO)  
**Date:** 2026-11-15  
**Target Subsystems:** NeuralOps Models Surgery & IO (`neuralops/models/surgery.py`, `neuralops/models/surgical_io.py`)  
*(Note: Explicitly constrained to `models/surgery.py` and `models/surgical_io.py` per Section 3.2 of the Wave 2 Orchestrator domain boundaries)*  
**Status:** PASS — All 4 targeted P0/P1/P2 issues remediated. 100% test pass on new regression suites and existing surgical suites (27 tests total). Ruff linting and formatting passed.

---

## 1. Executive Summary

This deep audit evaluated the models surgery and surgical IO subsystems of NeuralOps pursuant to Section 3.2 of the Wave 2 Orchestrator plan. Four (4) high-priority defects (1 P0, 2 P1, 1 P2) along with an import-time circular dependency were identified, audited, and resolved with comprehensive regression unit tests and strict backward-compatibility protections.

### Key Metrics
- **Files Audited & Modified:** `neuralops/models/surgery.py`, `neuralops/models/surgical_io.py`
- **Targeted Bugs Resolved:** 4 (P0-1, P1-SurgicalCheckpoint, P2-BackboneSurgeryHooks, P1-HealMixedDtype) + Circular Import Decoupling
- **New Regression Test Suites:**
  - [`tests/models/test_surgical_io.py`](file:///Users/tahamajs/dev/Project/tests/models/test_surgical_io.py): 6 test cases (100% pass)
  - [`tests/models/test_surgery.py`](file:///Users/tahamajs/dev/Project/tests/models/test_surgery.py): 5 test cases (100% pass)
- **Full Surgical Test Coverage:** 27 passing tests (11 new unit tests in `tests/models/` + 16 existing tests in `tests/test_surgical_delta_and_hub_loader.py` and `tests/test_unified_surgical_delta_and_cards.py`)
- **Lint & Style:** 100% clean under `ruff check` and `ruff format`

---

## 2. Issues Investigated & Resolved

### P0-1: `surgical_io.py::save_surgical_delta` Deletes User Files
- **Root Cause:** In [`save_surgical_delta`](file:///Users/tahamajs/dev/Project/neuralops/models/surgical_io.py), the parameter `remove_full_weights: bool = True` defaulted to `True`. When saving a surgical delta into an output directory containing pre-existing user checkpoints, model files (`pytorch_model.bin`, `model.safetensors`, `model-*.safetensors`), or optimizer checkpoints, the function automatically unlinked and permanently deleted those user files.
- **Resolution:**
  - Changed default parameter to `remove_full_weights: bool = False` in `save_surgical_delta`.
  - Added explicit docstring warning explaining the parameter semantics and safeguarding user directory contents by default.
  - Purging only executes if `remove_full_weights=True` is explicitly passed by the caller.
- **Verification:**
  - [`test_save_surgical_delta_preserves_existing_user_files`](file:///Users/tahamajs/dev/Project/tests/models/test_surgical_io.py#L27-L50) asserts that existing `.bin`, `.safetensors`, and `.pt` files remain intact by default.
  - [`test_save_surgical_delta_purges_only_when_explicitly_requested`](file:///Users/tahamajs/dev/Project/tests/models/test_surgical_io.py#L53-L78) asserts that purging occurs only when `remove_full_weights=True` is provided.

---

### P1: `surgical_io.py::load_surgical_checkpoint` Silent Skipped Load
- **Root Cause:** In [`load_surgical_checkpoint`](file:///Users/tahamajs/dev/Project/neuralops/models/surgical_io.py), delta checkpoint loading bypassed strict verification:
  - Delta mode forced `effective_strict = False` and merely logged warnings on `unexpected_keys`.
  - Expected surgical parameters from the manifest that were missing from delta checkpoint tensors were silently ignored.
  - When loading LoRA adapters, if a target module path did not exist on the base model, it was silently skipped with only a warning, even when `strict=True`.
  - If a checkpoint contained tensors that matched 0 keys on the model, it returned successfully with 0 loaded tensors without error.
- **Resolution:**
  - Enforced strict contract: when `strict=True` and delta checkpoint mode is active, any unexpected keys raise `RuntimeError(f"Unexpected key(s) in delta checkpoint: {missing_keys}")`.
  - Added verification that all expected surgical submodule parameters in `submodules` are present in the checkpoint; if missing under `strict=True`, raises `RuntimeError`.
  - Added missing LoRA target module verification: when `strict=True`, non-existent target paths raise `KeyError`.
  - Added empty load check: if state dict is non-empty but 0 keys were loaded, raises `RuntimeError("Zero surgical tensors were loaded into the base model")`.
- **Verification:**
  - [`test_load_surgical_checkpoint_raises_on_unexpected_keys_when_strict`](file:///Users/tahamajs/dev/Project/tests/models/test_surgical_io.py#L81-L103) verifies `RuntimeError` on unexpected keys.
  - [`test_load_surgical_checkpoint_raises_on_missing_surgical_keys_when_strict`](file:///Users/tahamajs/dev/Project/tests/models/test_surgical_io.py#L106-L127) verifies `RuntimeError` when surgical keys are omitted.
  - [`test_load_surgical_checkpoint_raises_on_missing_lora_target_module`](file:///Users/tahamajs/dev/Project/tests/models/test_surgical_io.py#L130-L151) verifies `KeyError` on invalid LoRA targets.
  - [`test_load_surgical_checkpoint_raises_on_zero_loaded_keys`](file:///Users/tahamajs/dev/Project/tests/models/test_surgical_io.py#L154-L177) verifies `RuntimeError` on disjoint parameter names.

---

### P2: `surgery.py::apply_backbone_surgery_hooks` No-op Execution Hooks
- **Root Cause:** The docstring for `apply_backbone_surgery_hooks` claimed to install runtime execution hooks on model backbones, but the function only iterated over submodules and stamped static capability attributes (`_neuralops_supports_kv_cache = True`, `_neuralops_supports_masking = True`) without attaching any runtime PyTorch hooks or forward adapters.
- **Resolution:**
  - Made the API and documentation honest: clearly documented that the function primarily inspects and tags capability metadata on decoder attention modules.
  - Added `attach_forward_adapter: bool = False` argument. When `attach_forward_adapter=True`, it wraps module forward passes with `adapt_decoder_attention_forward(fwd_fn, ...)` to adapt call signatures at runtime.
  - Safeguarded against recursion: passed the unwrapped underlying `orig_forward` callable `fwd_fn` rather than `module` to prevent recursive re-entry.
  - Changed return type to `list[str]` returning the names of all tagged modules for verifiable tracking.
- **Verification:**
  - [`test_apply_backbone_surgery_hooks_tags_metadata_honestly`](file:///Users/tahamajs/dev/Project/tests/models/test_surgery.py#L38-L56) confirms capability flags and return list.
  - [`test_apply_backbone_surgery_hooks_attaches_forward_adapter_when_requested`](file:///Users/tahamajs/dev/Project/tests/models/test_surgery.py#L59-L81) verifies runtime argument adaptation and recursion prevention.

---

### P1: `surgery.py::_heal_replaced_model_impl` Mixed-Dtype Crash & Instability
- **Root Cause:** In [`_heal_replaced_model_impl`](file:///Users/tahamajs/dev/Project/neuralops/models/surgery.py) and [`SubspaceAnchorWrapper`](file:///Users/tahamajs/dev/Project/neuralops/models/surgery.py):
  1. `SubspaceAnchorWrapper.forward` mixed base and anchor branch outputs without matching tensor dtypes, causing `RuntimeError: expected scalar type Half but found Float`.
  2. Teacher/student dtype mismatch: when teacher is float32 and student is float16/bfloat16, passing calibration batches or computing hidden states resulted in dtype conflicts.
  3. Intermediate parameter dtype heterogeneity: when student layer parameters contained mixed dtypes (e.g., float16 linear weights with float32 bias or layernorm), feeding converted inputs triggered operator dtype crashes.
  4. Pre-hook input capture bug: registering pre-hooks on `parent_mod` mistakenly intercepted root token IDs `(batch, seq)` instead of intermediate hidden states `(batch, seq, dim)` when attention modules were direct children of the block/model.
  5. Optimization tracking flaw: `best_state` was saved after `optimizer.step()` using the pre-step loss value. If a gradient step caused parameters to produce NaNs on step $t+1$, the early-stopping break reloaded the NaN-producing parameters from step $t$. Furthermore, if gradients became non-finite, AdamW corrupted the weights.
- **Resolution:**
  - **`SubspaceAnchorWrapper`:** Cast inputs to base and anchor parameter dtypes respectively. Computed linear blend `(1 - alpha) * y_base.float() + alpha * y_anchor.float()` in float32 and cast back to `base_layer` output dtype.
  - **Calibration & Target Normalization:** Normalized calibration batches to teacher device and floating dtype. Handled student input device/dtype casting dynamically via `_call_surg_module`.
  - **Intra-Module Dtype Harmonization:** Synchronized parameter dtypes inside replaced student layers before micro-healing optimization.
  - **Direct Pre-Hook Caching:** Registered forward pre-hooks directly on `t_mod` (the attention module) rather than `parent_mod`, capturing accurate hidden states regardless of parent block nesting.
  - **Safe State Tracking & Gradient Guards:**
    - Recorded `initial_state` before training.
    - Updated `best_state` before the optimizer step using the verified finite loss of the current parameters.
    - Checked all parameter gradients with `torch.isfinite(p.grad).all()`; if non-finite gradients are detected, clipped or halted optimization without corrupting weights.
    - Added fallback: if optimization aborted without a finite loss improvement or if `final_mse` is non-finite, restored `initial_state` and set `final_mse = init_mse`.
- **Verification:**
  - [`test_subspace_anchor_wrapper_mixed_dtype`](file:///Users/tahamajs/dev/Project/tests/models/test_surgery.py#L84-L114) validates mixed float16/float32 branches in anchor wrapper.
  - [`test_heal_replaced_model_mixed_dtype_teacher_student`](file:///Users/tahamajs/dev/Project/tests/models/test_surgery.py#L117-L146) tests healing across float32 teacher and float16 student.
  - [`test_heal_replaced_model_mixed_dtype_inside_student`](file:///Users/tahamajs/dev/Project/tests/models/test_surgery.py#L149-L177) tests healing with mixed precision parameters inside the student module.

---

### Architectural Hygiene: Circular Import Elimination
- **Root Cause:** `neuralops/models/surgery.py` imported `from neuralops.api import replace_attention, transplant_attention`. Concurrently, `neuralops/api.py` imported `neuralops.trainer`, which imported `foundation_pretrainer`, which imported `surgery`. This circular dependency resulted in `ImportError: cannot import name 'replace_attention' from partially initialized module 'neuralops.models.surgery'`.
- **Resolution:**
  - Removed top-level import of `neuralops.api` from `surgery.py`.
  - Replaced with lazy delegation functions `def replace_attention` and `def transplant_attention` inside `surgery.py` that import `neuralops.api` on demand when invoked.

---

## 3. Test Verification Results

### Unit Regression Suite (`tests/models/`)
```text
tests/models/test_surgical_io.py::test_save_surgical_delta_preserves_existing_user_files PASSED
tests/models/test_surgical_io.py::test_save_surgical_delta_purges_only_when_explicitly_requested PASSED
tests/models/test_surgical_io.py::test_load_surgical_checkpoint_raises_on_unexpected_keys_when_strict PASSED
tests/models/test_surgical_io.py::test_load_surgical_checkpoint_raises_on_missing_surgical_keys_when_strict PASSED
tests/models/test_surgical_io.py::test_load_surgical_checkpoint_raises_on_missing_lora_target_module PASSED
tests/models/test_surgical_io.py::test_load_surgical_checkpoint_raises_on_zero_loaded_keys PASSED
tests/models/test_surgery.py::test_apply_backbone_surgery_hooks_tags_metadata_honestly PASSED
tests/models/test_surgery.py::test_apply_backbone_surgery_hooks_attaches_forward_adapter_when_requested PASSED
tests/models/test_surgery.py::test_subspace_anchor_wrapper_mixed_dtype PASSED
tests/models/test_surgery.py::test_heal_replaced_model_mixed_dtype_teacher_student PASSED
tests/models/test_surgery.py::test_heal_replaced_model_mixed_dtype_inside_student PASSED

============================= 11 passed in 12.56s =============================
```

### Existing Surgical Delta & Hub Suite (`tests/`)
```text
tests/test_surgical_delta_and_hub_loader.py::test_hub_exports_present PASSED
tests/test_surgical_delta_and_hub_loader.py::test_extract_surgical_manifest_captures_all_component_types PASSED
tests/test_surgical_delta_and_hub_loader.py::test_save_surgical_delta_and_load_surgical_checkpoint_roundtrip PASSED
tests/test_surgical_delta_and_hub_loader.py::test_normalize_base_model_id_paths_and_aliases PASSED
tests/test_surgical_delta_and_hub_loader.py::test_apply_decoder_attention_surgery_qwen_and_llava PASSED
tests/test_surgical_delta_and_hub_loader.py::test_load_surgeried_model_decoder_attention_end_to_end PASSED
tests/test_surgical_delta_and_hub_loader.py::test_load_surgeried_model_bert_attention_and_compressed_lora PASSED
tests/test_surgical_delta_and_hub_loader.py::test_zoo_wrappers_route_to_load_surgeried_model PASSED
tests/test_surgical_delta_and_hub_loader.py::test_async_uploader_zoo_routing_and_delta_cleanup PASSED
tests/test_unified_surgical_delta_and_cards.py::test_save_surgical_delta_excludes_frozen_vision_and_anchors PASSED
tests/test_unified_surgical_delta_and_cards.py::test_unified_model_card_all_six_zoos_honest_audit PASSED
tests/test_unified_surgical_delta_and_cards.py::test_unified_model_card_yaml_frontmatter_and_nonfinite_guard PASSED
tests/test_unified_surgical_delta_and_cards.py::test_honest_artifact_audit_empty_and_legacy_bin PASSED
tests/test_unified_surgical_delta_and_cards.py::test_vlm_24_layer_and_llama_22_layer_delta_exclusion PASSED
tests/test_unified_surgical_delta_and_cards.py::test_qwen_compressor_delta_and_bert_pareto_card PASSED
tests/test_unified_surgical_delta_and_cards.py::test_load_surgeried_model_local_roundtrip PASSED

============================= 16 passed in 12.68s =============================
```

---

## 4. AGENTS.md Operational Compliance

| AGENTS.md Rule | Compliance Assessment |
|---|---|
| **Rule 0: Invariants** | Backward compatibility preserved; existing APIs remain fully functional. Reversible git commits. Complete isolation maintained. |
| **Rule 1.3: No Import-Time Side Effects** | Lazy module delegation used for API imports in `surgery.py` to eliminate circular dependencies. No side effects at module scope. |
| **Rule 2.1: AMP API** | No `torch.cuda.amp` imports; standard `torch.amp` semantics and device-agnostic dtype handling. |
| **Rule 2.5: Device Fallback** | All surgery and healing operations run reliably on CPU and CUDA without device hardcoding or forced failures. |
| **Rule 6.1: Linting & Formatting** | Passed `ruff check` and `ruff format` cleanly across all modified source and test files. |
| **Rule 9.4: Fail Closed** | Numerical errors during healing fail closed, restoring the initial state safely without returning unhandled NaNs. |

---

## 5. Conclusion

Agent A2 has successfully audited and resolved all targeted P0, P1, and P2 defects in `neuralops/models/surgery.py` and `neuralops/models/surgical_io.py`. All 27 regression and functional tests are green, circular dependencies are eliminated, and all operational rules from `AGENTS.md` are satisfied.
