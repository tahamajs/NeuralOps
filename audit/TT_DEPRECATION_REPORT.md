# Tensor Train (`tt_compressor`) Deprecation Report

## Executive Summary
The Tensor Train compression module (`tt_compressor` / `tt`) has been formally deprecated and excluded from all active benchmark pipelines across the repository and GPU execution server. The module produced catastrophic empirical degradation across transformer architectures (RoBERTa SST-2 at 50.92% random chance baseline, GPT-2 WikiText-2 perplexity diverging to 2,156, DistilBERT degrading to ~58%). These findings have been permanently documented in `MASTER_PROJECT_RESULTS.md` Section 12 as an informative scientific negative result demonstrating the fundamental mismatch between multidimensional tensor network contractions and 2D transformer weight geometry.

In addition, targeted bug fixes and safeguards were applied across the 4 problem modules identified during benchmark execution:
1. `tt_compressor.py`
2. `lillama_compressor.py`
3. `erc_svd.py`
4. `mamba_svd_flash.py`

---

## 1. Codebase Changes & Deprecation Implementation

### 1.1 Canonical Skip List & Pipeline Guards
- Created [`audit/deprecated_modules.txt`](audit/deprecated_modules.txt) containing all alias variants:
  - `tt`, `tt_linear`, `tt_compressor`, `tt_linear_compressor`, `tt_factory`, `tt_compressor_factory`, `tt_linear_factory`
  - `lillama_compressor`
  - `mamba_svd_flash`
- Updated [`scripts/run_autonomous_48h_pipeline.py`](scripts/run_autonomous_48h_pipeline.py) to dynamically parse `audit/deprecated_modules.txt` and filter pending jobs before dispatch.
- Updated [`Codes/framework/run_quality_sweep.py`](Codes/framework/run_quality_sweep.py):
  - Added module-level `DEPRECATED_MODULES` dictionary with descriptive reasons.
  - Added early exit guard in `run_single_module_main()` returning `status: "SKIP_DEPRECATED"`.
  - Pruned active queue on GPU server (`D:\SVD_Project\paper\_PIPELINE_QUEUE.json`), removing all 12 deprecated job entries (0 remaining).

### 1.2 Module 1: `tt_compressor.py` Fixes
- Added `warnings.warn(..., DeprecationWarning)` on import.
- Updated module docstring with empirical failure rates, root cause, and citations.
- Fixed line 261 `out = out` no-op bug -> `out = out * self.output_scale`.
- Removed dead and uninitialized `self.shift` reference.
- Added strict parameter reduction check in `compress()`: falls back to original layer if `tt_layer_params >= orig_layer_params`.
- Registered `tt_deprecated` alias in `COMPRESSOR_REGISTRY`.

### 1.3 Module 2: `lillama_compressor.py` Fixes
- Renamed class to `LiLLaMADistiller(BaseCompressor)` with `category = "weight_rewriting"`.
- Maintained backward-compatibility alias `LiLLaMACompressor = LiLLaMADistiller`.
- Added `MAX_ACTIVATION_SAMPLES = 512` memory cap in `_collect_io_activations()` to prevent GPU/RAM exhaustion during calibration.

### 1.4 Module 3: `erc_svd.py` Fixes
- Added Pre-LayerNorm encoder detection (`is_pre_layernorm_encoder = "roberta" in model_name_or_cls`).
- Capped `effective_max_error = min(self.max_error, 0.15)` for pre-LN encoders to eliminate residual noise amplification.
- Robust forward check with attention mask fallback.

### 1.5 Module 4: `mamba_svd_flash.py` Fixes
- Removed conflicting `@property def self` accessor.
- Added Windows platform guard (`is_windows and L > 128`) and low-rank decoder attention fallback to prevent watchdog timeouts on Windows environments lacking native CUDA selective scan kernels.

---

## 2. Documentation in MASTER_PROJECT_RESULTS.md
Added Section 12: **Deprecated & Failed Modules**:
- Comprehensive table recording Module, Architecture, Metric, Baseline, Status, and Root Cause.
- Theoretical explanation of why Tensor-Train decomposition fails on transformer weights (mode scrambling, lack of grid locality, flattened singular value spectra).

---

## 3. Git Commits & Remote Sync Verification
The changes were committed separately and synchronized across both repositories:
1. `083a5ad37` — `fix(tt): deprecation warning, output_scale no-op, shift cleanup, param check`
2. `71aa7faa8` — `refactor(lillama): rename to distiller, add category, exclude from compression`
3. `ec7a8ac1f` — `fix(erc_svd): pre-layernorm detection, robust forward check, stricter threshold`
4. `7d010d5c7` — `fix(mamba_svd_flash): remove self property, add Windows guard and low-rank fallback`
5. `a7cb96ad6` — `feat(pipeline): add deprecated_modules skip list and sweep guard`
6. `b69c79570` — `docs: document deprecated and failed modules as scientific negative results`

Pushed to GitHub `main` and pulled into remote GPU server `D:\SVD_Project`. Queue verified with 0 pending deprecated jobs.
