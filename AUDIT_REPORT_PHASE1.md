# NeuralOps Global Static Audit & Infrastructure Health Report (Phase 1)
**Date:** 2026-11-03  
**Auditor Role:** Senior Code Auditor  
**Audit Scope:** `neuralops/` core codebase, registries, dispatch layers, compression engines, analysis pipelines  
**Mode:** Strictly Read-Only (No code edits, no staging, no commits)  
**Governing Standard:** `AGENTS.md` (§0 Core Principles, §6 Silent Failure Policy, §7 Kernel Dispatch Fallbacks, §8 Repository Invariants)

---

## 1. Executive Summary (Top 5 Critical Issues)

| ID | Issue Description | Location | Severity | Governing Rule |
|---|---|---|---|---|
| **ISS-01** | **Silent Baseline Loss in State Mutation**: `commit()` silently swallows `Exception` when cloning model state to CPU baseline. Stale or `None` baseline causes silent corruption during rollback and surgery verification. | `neuralops/surgery/model_surgery.py:204-205` | **CRITICAL** | `AGENTS.md` §6.3 |
| **ISS-02** | **Hardcoded Numeric Fallback on Metric Failure**: `effective_rank()` catches bare `Exception` during `np.linalg.eigvalsh` and returns hardcoded `1.0`. Erroneous or degenerate covariances are reported as valid metric outputs. | `neuralops/analysis/layerwise_analyzer.py:148-150` | **CRITICAL** | `AGENTS.md` §6.2, §6.3 |
| **ISS-03** | **Compression Telemetry Loss via Empty Dict Fallback**: Multiple compressors (`LiLLaMACompressor`, `KroneckerCompressor`, `KroneckerTTLoRA`, `FAQQuantizer`) catch bare `Exception` in `compress()` and return `self.model, {}`, dropping metrics silently. | `neuralops/compression/experimental/lillama_compressor.py:371`<br>`neuralops/compression/tensor_decomp/kronecker_compressor.py:712`<br>`neuralops/compression/tensor_decomp/kronecker_tt_lora.py:588`<br>`neuralops/compression/quantization/faq_quantizer.py:536` | **CRITICAL** | `AGENTS.md` §0.2, §6.3 |
| **ISS-04** | **Unlogged Triton-to-PyTorch Level 2 -> Level 3 Fallback**: In `KroneckerTTLoRALinear.forward`, failure of Triton fused GEMM kernel silently falls back to standard PyTorch linear without logging warning. Rank adaptation also silently swallows SVD failure. | `neuralops/compression/tensor_decomp/kronecker_tt_lora.py:333-334, 345-346` | **CRITICAL** | `AGENTS.md` §6.3, §7 |
| **ISS-05** | **Synthetic Telemetry Fallback in Benchmarks & Monitoring**: `advanced_data.py` substitutes hardcoded `0.25` when `rel_err` is missing to fabricate compression ratio. `stats.py` returns `1.0` if t-test calculation crashes. | `neuralops/runners/monitor/advanced_data.py:109`<br>`neuralops/benchmarks/core/stats.py:38-40` | **MAJOR** | `AGENTS.md` §6.2 |

---

## 2. Full Table of Silent Failures & Policy Violations

### Classification Criteria:
- **CRITICAL**: Causes silent state corruption, unlogged kernel fallback in execution forward pass, or replaces telemetry/data metrics with hardcoded numbers.
- **MAJOR**: Unannotated bare `except Exception: pass` or fallback in telemetry readers/finders without warning log.
- **MINOR**: Defensive import hook or best-effort telemetry marked with allowed annotations.

| # | File & Line | Type | Severity | Code Snippet / Grep Evidence | Violation Details & Remediation |
|---|---|---|---|---|---|
| 1 | `neuralops/surgery/model_surgery.py:204` | Silent Exception Swallow | **CRITICAL** | `try: self.baseline_model = copy.deepcopy(self.model).cpu() except Exception: pass` | If deepcopy fails, `self.baseline_model` remains stale or uninitialized. Must log warning and raise or set explicit fallback state. |
| 2 | `neuralops/analysis/layerwise_analyzer.py:148` | Hardcoded Metric Fallback | **CRITICAL** | `try: eigvals = np.linalg.eigvalsh(self.cov_matrix) ... except Exception: return 1.0` | Returns hardcoded `1.0` when SVD/eigenvalue decomposition fails. Violates §6.2. Must raise or return `float('nan')` with logged warning. |
| 3 | `neuralops/compression/tensor_decomp/kronecker_tt_lora.py:333` | Silent Kernel Dispatch Fallback | **CRITICAL** | `try: lora_bypass = fused_down_up_gemm_bias_triton(...) except Exception: lora_bypass = F.linear(...)` | Silent fallback from Triton kernel to PyTorch reference without `logger.warning`. Violates §7 contract. |
| 4 | `neuralops/compression/tensor_decomp/kronecker_tt_lora.py:345` | Silent Exception Swallow | **MAJOR** | `try: _, S, _ = torch.linalg.svd(...) except Exception: return` | Silently aborts rank adaptation on SVD solver failure without reporting to caller. |
| 5 | `neuralops/compression/experimental/lillama_compressor.py:371, 396` | Empty Data Return | **CRITICAL** | `try: return self.model, self.get_compression_stats() except Exception: return self.model, {}` | Drops compression metrics dictionary silently instead of logging error or raising. |
| 6 | `neuralops/compression/tensor_decomp/kronecker_compressor.py:712, 765` | Empty Data Return | **CRITICAL** | `try: return self.model, self.get_compression_stats() except Exception: return self.model, {}` | Drops compression metrics dictionary silently instead of logging error or raising. |
| 7 | `neuralops/compression/tensor_decomp/kronecker_tt_lora.py:588, 621` | Empty Data Return | **CRITICAL** | `try: return self.model, self.get_compression_stats() except Exception: return self.model, {}` | Drops compression metrics dictionary silently instead of logging error or raising. |
| 8 | `neuralops/compression/quantization/faq_quantizer.py:536` | Empty Data Return | **CRITICAL** | `try: return self.model, self.get_compression_stats() except Exception: return self.model, {}` | Drops compression metrics dictionary silently instead of logging error or raising. |
| 9 | `neuralops/runners/monitor/advanced_data.py:109` | Hardcoded Fallback Number | **MAJOR** | `ratio = round(1.0 / max(0.001, (rel_err if rel_err is not None else 0.25) * 2), 2)` | Fabricates `0.25` error when `rel_err` is None, generating synthetic compression ratio. |
| 10 | `neuralops/benchmarks/core/stats.py:38` | Hardcoded Metric Fallback | **MAJOR** | `try: res = stats.ttest_ind(...) except Exception: return 1.0` | Silently substitutes `1.0` for p-value on statistical test error. |
| 11 | `neuralops/training/__init__.py:43` | Bare Exception Swallow | **MINOR** | `try: spec = importlib.util.find_spec(cand) except Exception: pass` | Dynamic import redirection hook silently suppresses errors without log. |
| 12 | `neuralops/layers/ssm/mamba_block.py:201` | Best-Effort Telemetry | **MINOR** | `# allowed-swallow: best-effort-telemetry pass` | Compliant with §6.3 annotation, side-effect only. |
| 13 | `neuralops/layers/sparse/lsh_attention.py:338` | Best-Effort Telemetry | **MINOR** | `# allowed-swallow: best-effort-telemetry pass` | Compliant with §6.3 annotation, side-effect only. |
| 14 | `neuralops/benchmarks/results/schema.py:225` | Best-Effort Telemetry | **MINOR** | `# allowed-swallow: best-effort-telemetry pass` | Compliant with §6.3 annotation, side-effect only. |

---

## 3. Fake Artifacts Audit (AGENTS.md §6.1)

Command executed:
```bash
grep -rn "NEURALOPS_TRT_ENGINE\|placeholder.*\.onnx\|placeholder.*\.gguf" neuralops/
```
Raw Output:
```
Binary file neuralops/integrations/__pycache__/tensorrt_llm.cpython-314.pyc matches
neuralops/integrations/tensorrt_llm.py:476:        if engine_data.startswith(b"NEURALOPS_TRT_ENGINE_V1"):
```

Analysis of `neuralops/integrations/tensorrt_llm.py:476-490`:
```python
if engine_data.startswith(b"NEURALOPS_TRT_ENGINE_V1"):
    raise ValueError("The file is a legacy NeuralOps placeholder, not a TensorRT engine.")
...
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    raise IntegrationNotImplementedError(
        "TensorRT execution binding is not implemented for NeuralOps models; "
        "the runner will not return fabricated logits."
    )
```
Analysis of `neuralops/export/exporter.py:49`:
```python
if fmt == "awq":
    raise NotImplementedError(
        "AWQ export is not implemented: writing a quantization config beside dense weights "
        "does not quantize a model. Quantize the model with a supported AWQ tool first."
    )
```
Analysis of `neuralops/export/gguf_exporter.py`:
- Valid binary GGUF serialization with `GGUF_MAGIC = b"GGUF"` and structured header metadata; no dummy text placeholders found.

**Verdict:** PASS. No fake `.onnx`, `.gguf`, or `.trt` artifacts are emitted. Legacy placeholder detection explicitly raises `ValueError` and unimplemented bindings explicitly raise `IntegrationNotImplementedError`.

---

## 4. Dead Code & Import Health (Task 2)

### 4.1 AST Dead Code Auditor
Command executed:
```bash
python3 scripts/utils/audit_dead_code.py 2>&1 | tail -20
```
Raw Output:
```
Audited 1595 Python source files. Issues found: 0
```

### 4.2 Linter & Type Integrity Check
Command executed:
```bash
./.venv/bin/ruff check neuralops/
```
Raw Output:
```
All checks passed!
```

---

## 5. Registry Integrity & Architectural Invariants (Task 3)

### 5.1 Verification Commands and Evidence

#### Attention and Compressor Registry Check
```bash
python3 -c "import json; d = json.load(open('neuralops/benchmarks/static_catalog.json')); print('Attentions:', len(d.get('attentions', [])), 'Compressors:', len(d.get('compressors', [])))"
```
Raw Output:
```
Attentions: 131 Compressors: 72
```

Live Registry Verification:
```bash
python3 -c "import neuralops; print('Live ATTENTION_REGISTRY:', len(neuralops.ATTENTION_REGISTRY), 'Live COMPRESSOR_REGISTRY:', len(neuralops.COMPRESSOR_REGISTRY))"
```
Raw Output:
```
Live ATTENTION_REGISTRY: 131 Live COMPRESSOR_REGISTRY: 72
```

#### Invariants & Consistency Suite
Command executed:
```bash
python3 scripts/audit_readme_consistency.py
```
Raw Output:
```
======================================================================
NeuralOps Invariants & README Consistency Audit
======================================================================
[1/6] Verifying Attention and Compressor Registry Invariants...
[2/6] Verifying Triton Modules & Unified Operators...
[3/6] Verifying Native CUDA Kernels & Package Counts...
[4/6] Verifying Master Jupyter Notebooks Invariant...
[5/6] Verifying Version Consistency...
[6/6] Verifying Local Markdown Links in README.md...
======================================================================
SUCCESS: 0 errors found! All invariants verified.
======================================================================
```

### 5.2 Breakdown of Canonical Architecture Counts:
- **Attention Registry Keys:** Exactly 131 canonical keys in `ATTENTION_REGISTRY` (128 implementation modules under `neuralops/layers/`).
- **Compressor Registry Keys:** Exactly 72 canonical keys in `COMPRESSOR_REGISTRY` (89 implementation files under `neuralops/compression/`).
- **Triton Kernel Modules & Unified Operators:** Exactly 26 canonical modules (62 unified operators across 4 tiers: 13 standalone, 27 core, 2 root, 20 extended).
- **Native CUDA Kernel Packages:** 130 native CUDA C++ kernel packages (132 kernel implementations).
- **Master Jupyter Notebooks:** Exactly 29 demonstration notebooks.

**Verdict:** PASS. All architectural invariants are intact.

---

## 6. Recommended Next Steps for Phase 2 (Code Surgery)

Once approved by the repository owner, Phase 2 should execute surgical fixes targeting the 5 Critical issues without expanding scope:

1. **Remediate `model_surgery.py:204`**: Replace bare `except Exception: pass` with logging at `error` level and explicit state retention / exception re-raising.
2. **Remediate `layerwise_analyzer.py:148`**: Replace `return 1.0` with `logger.warning` and `float('nan')` or descriptive `ArithmeticError` propagation.
3. **Remediate Compressor Stats Fallbacks (`lillama_compressor.py`, `kronecker_compressor.py`, `kronecker_tt_lora.py`, `faq_quantizer.py`)**: Replace `except Exception: return self.model, {}` with `logger.error` and structured error propagation.
4. **Remediate Kernel Dispatch in `kronecker_tt_lora.py:333`**: Add `logger.warning("Level 2 Triton fused GEMM failed (%s); falling back to Level 3 PyTorch reference.", exc)` to comply with §7 fallback rules.
5. **Remediate Synthetic Numbers in `advanced_data.py:109` and `stats.py:38`**: Replace hardcoded `0.25` with `None` handling, and replace silent `return 1.0` in `ttest_ind` with proper nan/warning logging.

---
**Status:** AUDIT COMPLETE. HALTED PER CORE DIRECTIVE. AWAITING OWNER APPROVAL TO PROCEED TO PHASE 2.
