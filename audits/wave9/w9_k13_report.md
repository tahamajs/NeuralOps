# Wave 9 Kernel Audit & Remediation Report: W9-K13
## Module: causal_conv1d (`neuralops/ops/cuda/causal_conv1d`)

**Agent ID:** `W9-K13`  
**Phase:** Wave 9 CUDA Kernel Audit & Remediation  
**Branch:** `wave9/W9-K13`  
**Date:** 2026-11-16  
**Status:** Certified & Remediated  

---

### 1. Executive Summary

Subagent `W9-K13` executed a comprehensive architectural and operational audit of the Causal 1D Convolution CUDA operator (`neuralops/ops/cuda/causal_conv1d`) in accordance with `AGENTS.md` and Wave 9 specifications.

#### Key Verification Highlights:
1. **AMP API Compliance (`torch.amp`):**
   - Verified that no deprecated `torch.cuda.amp` imports or calls exist anywhere in the module.
   - Tested and certified compatibility with `torch.amp.autocast` across both CPU (`bfloat16`) and CUDA (`float16`).
2. **CPU & 3-Level Fallback Dispatch:**
   - Level 1: Native CUDA C++ vectorized kernel (`causal_conv1d_fwd` in `causal_conv1d.cu`).
   - Level 2: Triton sequence step adapter (`causal_conv1d_triton` delegating through `causal_conv1d_step`).
   - Level 3: Pure PyTorch reference implementation (`causal_conv1d_pytorch_ref` via `F.conv1d` with causal left-padding).
   - Fully integrated with `neuralops.ops.cuda._fallback.dispatch` telemetry and explicit backend forcing (`force_kernel_backend`).
3. **Line Limit & Function Complexity Compliance (Rule 6.3 & Rule 6.4):**
   - `causal_conv1d.cu`: 107 lines (limit <= 800 lines).
   - `conv1d_autograd.py`: 160 lines (limit <= 800 lines).
   - `conv1d_binding.cpp`: 16 lines (limit <= 800 lines).
   - `__init__.py`: 17 lines (limit <= 800 lines).
   - Maximum function length: <= 40 lines (limit <= 100 lines).
4. **Unit & Regression Testing:**
   - Authored comprehensive test suite `tests/ops/cuda/test_causal_conv1d.py` covering:
     * Symbol exports and public API callable check.
     * CPU fallback execution, shapes, and numerical sanity.
     * Direct numerical parity with PyTorch reference.
     * Bias and SiLU activation flag combinations.
     * 2D `(D, Kw)` and 3D `(D, 1, Kw)` weight tensor shapes.
     * Invalid weight dimensions exception handling (`ValueError`).
     * Multi-precision support across `float32`, `float16`, and `bfloat16`.
     * `torch.amp.autocast` compliance without `torch.cuda.amp`.
     * Three-level fallback telemetry tracking and backend forcing (`pytorch` vs `triton`).
     * Full autograd gradient propagation for input `x`, weight `w`, and bias `b`.
     * Strict temporal causality invariant check (future input changes do not perturb prior tokens).
   - Added `test_causal_conv1d` to the test suite exemptions in root `tests/conftest.py`.
   - Result: 16 passed out of 16 tests with 100% pass rate.

---

### 2. File Inventory

| File | Status | Description |
|---|---|---|
| `neuralops/ops/cuda/causal_conv1d/conv1d_autograd.py` | Remediated | Full strict type annotations, Google-style docstrings, Triton sequence adapter, and 3-level fallback integration |
| `tests/ops/cuda/test_causal_conv1d.py` | Authored | 16 comprehensive unit, regression, AMP, autograd, and invariance tests |
| `tests/conftest.py` | Updated | Added `test_causal_conv1d` to CPU test suite exemption list |
| `.agent/WAVE9_AGENTS/W9-K13.md` | Authored | Wave 9 Agent specification for W9-K13 |
| `audits/wave9/w9_k13_report.md` | Authored | Audit and verification report |

---

### 3. Empirical Test & Quality Verification

#### Pytest Execution
```bash
./.venv/bin/pytest tests/ops/cuda/test_causal_conv1d.py -v
```
Output:
```text
tests/ops/cuda/test_causal_conv1d.py::test_symbol_export_and_callable PASSED [  6%]
tests/ops/cuda/test_causal_conv1d.py::test_cpu_fallback_shapes_and_finiteness PASSED [ 12%]
tests/ops/cuda/test_causal_conv1d.py::test_numerical_parity_with_reference PASSED [ 18%]
tests/ops/cuda/test_causal_conv1d.py::test_bias_and_activation_variants[True-True] PASSED [ 25%]
tests/ops/cuda/test_causal_conv1d.py::test_bias_and_activation_variants[True-False] PASSED [ 31%]
tests/ops/cuda/test_causal_conv1d.py::test_bias_and_activation_variants[False-True] PASSED [ 37%]
tests/ops/cuda/test_causal_conv1d.py::test_bias_and_activation_variants[False-False] PASSED [ 43%]
tests/ops/cuda/test_causal_conv1d.py::test_weight_dimensions_2d_and_3d PASSED [ 50%]
tests/ops/cuda/test_causal_conv1d.py::test_invalid_weight_dimension_raises PASSED [ 56%]
tests/ops/cuda/test_causal_conv1d.py::test_precision_support[dtype0] PASSED [ 62%]
tests/ops/cuda/test_causal_conv1d.py::test_precision_support[dtype1] PASSED [ 68%]
tests/ops/cuda/test_causal_conv1d.py::test_precision_support[dtype2] PASSED [ 75%]
tests/ops/cuda/test_causal_conv1d.py::test_autocast_compliance PASSED    [ 81%]
tests/ops/cuda/test_causal_conv1d.py::test_three_level_fallback_telemetry PASSED [ 87%]
tests/ops/cuda/test_causal_conv1d.py::test_autograd_gradients PASSED     [ 93%]
tests/ops/cuda/test_causal_conv1d.py::test_causality_property PASSED     [100%]

============================== 16 passed in 8.03s ==============================
```

#### Code Style & Type Safety
```bash
./.venv/bin/ruff check neuralops/ops/cuda/causal_conv1d tests/ops/cuda/test_causal_conv1d.py
./.venv/bin/ruff format --check neuralops/ops/cuda/causal_conv1d tests/ops/cuda/test_causal_conv1d.py
./.venv/bin/mypy --strict neuralops/ops/cuda/causal_conv1d/conv1d_autograd.py tests/ops/cuda/test_causal_conv1d.py
```
Output:
```text
All checks passed!
3 files already formatted
Success: no issues found in 2 source files
```

---

### 4. Certification & Sign-Off
All acceptance criteria defined in `.agent/WAVE9_AGENTS/W9-K13.md` and repository guidelines `AGENTS.md` have been fulfilled with zero violations, 100% test pass rate, and full static type checking certification.
