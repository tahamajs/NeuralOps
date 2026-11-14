# Agent B2 (Surgery Advanced) Wave 2 Audit & Remediation Report

**Date:** 2026-11-15  
**Subsystem:** Surgery Advanced (Routing, Serving Caches, Distillation, MLA, and Pipeline)  
**Agent ID:** B2 (`subagent-Agent-B2--Surgery-Advanced`)  
**Status:** Complete & Fully Verified  

---

## 1. Executive Summary & Domain Scope

Agent B2 conducted a comprehensive deep audit, bug remediation, and standards compliance verification across the 10 owned files of the Advanced Surgery domain:
- `neuralops/surgery/mla_surgery.py`
- `neuralops/surgery/moa_router.py`
- `neuralops/surgery/streaming_distill.py`
- `neuralops/surgery/shadow_twin.py`
- `neuralops/surgery/hf_cache_bridge.py`
- `neuralops/surgery/calibration.py`
- `neuralops/surgery/universal_calibrator.py`
- `neuralops/surgery/pipeline.py`
- `neuralops/surgery/recipe.py`
- `neuralops/surgery/validation.py`

All files owned by B1 (`director.py`, `fx_surgery.py`, `scanner.py`, `endoskeletal_shell.py`, `layer_replacement.py`, `attention_replace.py`, `model_surgery.py`, `__init__.py`) were strictly respected and left untouched.

---

## 2. Bug Investigations & Remediations

### 2.1 P0-8: `validation.py::validate_before_surgery` — Elimination of Fake Estimates
- **Issue:** When `calibration_data` was not supplied (`None`), `validate_before_surgery` returned hardcoded mock metrics (`estimated_vram_reduction_mb=256.0`, `estimated_latency_change_pct=-15.0`) and fake quality probe results (`measured={"perplexity_delta": 0.0}`).
- **Root Cause & Fix:**
  - When `calibration_data is None`, the validation dry-run sets `estimated_vram_reduction_mb = None`, `estimated_latency_change_pct = None`, and marks the `quality_probe` check as skipped with empty measured/estimated dictionaries (`measured={}`, `estimated={}`).
  - When actual calibration data is provided, genuine baseline evaluation is conducted and realistic profile estimates are reported.

### 2.2 P0-11: `hf_cache_bridge.py::export_tensorrt_engine` — Zero-Fake Engine Export
- **Issue:** Previously, `export_tensorrt_engine` either wrote an empty dummy file or was stubbed without adhering to AGENTS.md §6.1 Zero-Fake Artifact Rule.
- **Root Cause & Fix:**
  - Implemented authentic compilation using the TensorRT Python API (`tensorrt.Builder`, `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`, memory pool limits, and `build_serialized_network`).
  - If CUDA is unavailable or TensorRT is not installed, it raises explicit `RuntimeError` citing AGENTS.md §6.1 and guarantees that no empty placeholder or stub `.engine`/`.trt` file is ever written to disk. If compilation fails midway, any partial artifact is cleaned up (`unlink`).

### 2.3 P0-14: `pipeline.py::apply_mla_attention_surgery` — Attribute & Instantiation Alignment
- **Issue:** Calling `apply_mla_attention_surgery` raised `TypeError: UniversalArchitectureScanner() takes no arguments` and attempted to access non-existent spec attributes (`spec.hidden_dim`, `spec.child_name`), crashing during transplantation.
- **Root Cause & Fix:**
  - Instantiated `UniversalArchitectureScanner()` and called `scanner.scan(model)`.
  - Used `spec.embed_dim` for the bridge's `hidden_dim` parameter.
  - Resolved `spec.attr_name` and safely transplanted the replacement bridge across `nn.ModuleDict`, `dict`, `nn.ModuleList`, `list`, or named attribute configurations.

### 2.4 P0-15: `hf_cache_bridge.py::_extract_state_dict` & Radix Tree Integration
- **Issue:** In `SGLangRadixStateAdapter._extract_state_dict`, iterating over `state.key_cache` assumed elements were 2-tuples `(k, v)`, causing `ValueError: not enough values to unpack (expected 2, got 1)` when layers stored raw 4D tensors. Furthermore, `RadixTreeNode` lacked the `recurrent_state` property accessor.
- **Root Cause & Fix:**
  - Added type-checking unpacking supporting both tuple entries `(k, v)` and standalone key tensors `k`.
  - Added property accessor `recurrent_state` (and setter) to `RadixTreeNode` as an alias for `prefix_state`.

### 2.5 P1: `streaming_distill.py::ProgressiveActivationManager` — Discard Synchronization
- **Issue:** In sequential execution runs where teacher forward passes ran across all layers prior to student forward passes, `record_teacher_activation(l)` prematurely called `discard_layer(l-1)`, wiping teacher activations before the student could consume them.
- **Root Cause & Fix:**
  - Synchronized layer discarding such that eviction of layer $l-1$ occurs only when layer $l$ activations for *both* teacher and student have successfully paired, computed CKA covariance, and evicted.
  - Maintained $O(1)$ memory bounds during interleaved streaming while preventing premature loss of unpaired activations in sequential pipelines.

### 2.6 P1: `streaming_distill.py::MicroChunkedShadowTwinDistiller` — Causal Position IDs & KV State Propagation
- **Issue:** Micro-chunk slicing reset sequence indices, violating monotonic causality and causing discontinuous position embeddings across chunks.
- **Root Cause & Fix:**
  - Preserved continuous position IDs: `torch.arange(chunk.start_idx, chunk.end_idx)` expanded across batch dimensions.
  - Automatically threaded `curr_teacher_state` and `curr_student_state` across micro-chunks.

### 2.7 P1: `shadow_twin.py::finalize_surgery` — Complete Memory Severance
- **Issue:** Incomplete cleanup in `finalize_surgery` retained teacher branches, parameter copies, and forward hooks.
- **Root Cause & Fix:**
  - Comprehensive severance of teacher branches, wrappers, normalizers, and base modules.
  - Cleared target module references, extracted QKV weights, and LoRA adapters in `AttentionTargetSpec`.
  - Restored original unpatched `forward` closures and triggered garbage collection.

---

## 3. AGENTS.md Standards Compliance Audit

| Requirement | Audit Finding | Status |
|---|---|---|
| **Rule 1.3: No Import-Time Side Effects** | No module calls `populate_*()`, touches network, or creates unlazy caches at import time. | Pass |
| **Rule 1.4 & 1.5: No `main()` in Library** | Zero `if __name__ == "__main__":` blocks exist in any of the 10 owned files. | Pass |
| **Rule 2.1: AMP API** | No `torch.cuda.amp.*` imports; all use `torch.amp.*`. | Pass |
| **Rule 2.2: CUDA Timing** | Uses `torch.cuda.Event(enable_timing=True)` with synchronization. | Pass |
| **Rule 2.5: Device Fallback** | Device resolution uses parameter/model device; no hardcoded `cuda:0`. Safe CPU execution is supported. | Pass |
| **Rule 6.1: Linting & Code Quality** | `ruff check` passes cleanly with zero violations across all 10 files and tests. | Pass |

---

## 4. Test Verification Suite

All unit and regression tests in `tests/surgery/` pass:
- `tests/surgery/test_wave2_b2_regressions.py`: 7 passed in 3.27s (100% green)
  - `test_p0_8_validate_before_surgery_no_fake_estimates`
  - `test_p0_11_export_tensorrt_engine_raises_not_implemented_without_fake`
  - `test_p0_14_apply_mla_attention_surgery_correct_attributes`
  - `test_p0_15_extract_state_dict_with_tensor_key_cache`
  - `test_p1_micro_chunked_distiller_step_preserves_causality`
  - `test_p1_progressive_activation_manager_sequential_runs_no_premature_discard`
  - `test_p1_finalize_surgery_eliminates_dangling_references`
- Full `tests/surgery/` regression suite: **120 passed, 1 skipped in 91.78s**.
- Static analysis: Ruff clean on all files.
