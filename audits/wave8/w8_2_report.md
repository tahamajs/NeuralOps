# Wave 8 Agent W8-2 Remediation Report: Hardcoded CUDA:0 & AMP Elimination

## 1. Overview & Objective
Agent W8-2 was tasked with enforcing Rule 2.1 (modern `torch.amp` API usage) and Rule 10 (no hardcoded `'cuda:0'` device strings, resolving devices dynamically via `torch.cuda.current_device()`, device parameters, or `resolve_device()`) across `neuralops/analysis/roofline.py`, `neuralops/config/benchmark_schema.py`, `neuralops/runtime/config.py`, `neuralops/cli/core/devices.py`, `neuralops/evaluation/metrics.py`, and standalone evaluations under `neuralops/evaluation/standalone/`.

## 2. Inventory of Changes

### A. Elimination of Hardcoded `cuda:0` Strings
1. **`neuralops/analysis/roofline.py`**:
   - `get_device_roofline_specs`: Replaced fallback `target_device = torch.device(device if device is not None else "cuda:0")` with dynamic resolution using `torch.cuda.current_device()` when CUDA is available and defaulting cleanly to CPU otherwise.
   - `profile_roofline`: Replaced `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` with dynamic `torch.cuda.current_device()`.

2. **`neuralops/evaluation/metrics.py`**:
   - `measure_peak_vram_mb`: Replaced `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` with `torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")`.

3. **`neuralops/runtime/config.py`**:
   - Updated documentation/docstring attribute from `"cuda:0"` to `"cuda:N"`, ensuring no hardcoded index is prescribed.

4. **`neuralops/evaluation/standalone/`**:
   - `eval_arithmetic_continuation_256steps.py`: Replaced `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` with dynamic current device resolution and device name logging.
   - `eval_concurrent_offloading_jitter.py`: Replaced hardcoded `"cuda:0"` and device ordinal 0 with dynamic current device resolution.
   - `eval_continuous_500token_stress.py`: Replaced hardcoded `"cuda:0"` and device ordinal 0 with dynamic current device resolution.
   - `eval_cot_jitter_128steps.py`: Replaced hardcoded `"cuda:0"` and device ordinal 0 with dynamic current device resolution.
   - `eval_reasoning_and_cot_scaling.py`: Replaced hardcoded `"cuda:0"` and device ordinal 0 with dynamic current device resolution.
   - `eval_reasoning_long_context.py`: Replaced hardcoded `"cuda:0"` and device ordinal 0 with dynamic current device resolution.
   - `eval_speculative_decoding.py`: Replaced hardcoded `"cuda:0"` and device ordinal 0 with dynamic current device resolution.
   - `eval_speculative_sweep_k1_k8.py`: Replaced hardcoded `"cuda:0"` and device ordinal 0 with dynamic current device resolution.
   - `eval_ultra_long_context_scaling.py`: Parameterized default `device_name` to `None`, dynamically resolving to current device if CUDA is available, or CPU.
   - `master_comprehensive_evaluation.py`: Updated CLI argument default for `--device` to dynamically resolve via `torch.cuda.current_device()`.
   - `README.md`: Updated sample command lines to use `--device cuda` instead of hardcoded `cuda:0`.

### B. PyTorch AMP Compliance (Rule 2.1)
- Verified repository conformance to Rule 2.1. Zero prohibited `torch.cuda.amp` imports exist across all modified and audited modules. All mixed precision operations rely cleanly on `torch.amp` or PyTorch native precision types.

## 3. Verification & Test Evidence
- Authoring unit test suite: `tests/benchmarks/test_device_resolution_remediation.py`
  - Validates dynamic CPU fallback in `get_device_roofline_specs` and `profile_roofline`.
  - Validates dynamic `cuda:{index}` current device resolution for `get_device_roofline_specs`.
  - Validates fallback behavior for `measure_peak_vram_mb`.
  - Validates `RuntimeConfig.resolve_device()` auto-resolution.
  - Validates `run_ultra_long_context_scaling` default device resolution.
- Regression testing:
  - `tests/benchmarks/test_cli_devices.py`: 4 passed
  - `tests/benchmarks/test_resource_scaling.py`: 4 passed
  - `tests/benchmarks/test_device_resolution_remediation.py`: 4 passed, 1 skipped (requires hardware CUDA)
  - All 12 unit tests passing, 0 failures.
  - Test evidence captured in `audits/wave8/w8_2_evidence/test_execution.log`.
- Code Hygiene:
  - `python3 -m ruff check`: Clean (All checks passed).
  - `python3 -m ruff format --check`: Clean (All files formatted).
