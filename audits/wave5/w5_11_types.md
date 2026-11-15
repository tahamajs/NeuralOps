# Forensic Static Typing & Mypy Consistency Audit Report (W5-11)

**Subagent ID:** `W5-11`  
**Phase:** 3 — Cross-Cutting Quality  
**Target Subsystems:** `neuralops/benchmarks/cli_v2`, `neuralops/benchmarks/config_v2`, `neuralops/benchmarks/results_v2`, `neuralops/benchmarks/suites_v2`, `neuralops/core`, `neuralops/api.py`, `neuralops/surgery`, `neuralops/models`  
**Date:** 2026-11-15  
**Compliance Authority:** `AGENTS.md` Rule 6.1 (Strict Mypy Compliance) & `PROJECT_MASTER_TRUTH.md`

---

## 1. Executive Summary

Static typing enforcement is a foundational pillar of NeuralOps stability, preventing dynamic runtime attribute exceptions, unhandled `None` dereferences, and signature drifts across internal subsystems. Under Rule 6.1 of `AGENTS.md`, all new and refactored modules must pass `mypy --strict`.

An exhaustive static typing audit was executed using `mypy 2.3.1` across both the strict v2 subsystems and the core legacy public interfaces of `neuralops/`.

### Key High-Level Findings:
1. **Rule 6.1 Scope (`config_v2/`, `results_v2/`, `cli_v2/`):** After addressing minor export and untyped third-party call diagnostics in `config_v2/loader.py` and `results_v2/ledger.py`, all 18 source files across `config_v2`, `results_v2`, and `cli_v2` pass `mypy --strict` with **0 errors** (100% compliance).
2. **Planned Module (`suites_v2/`):** Scheduled for sibling implementation in Wave 6; currently not yet staged on `main`.
3. **Core Public Subsystems (`mypy --strict`):** Found **948 errors in 69 files** across 313 checked source files.
4. **Full Repository (`mypy --ignore-missing-imports`):** Found **897 errors**, with 37.2% (334 errors) stemming from a single dictionary type mismatch (`list` vs `tuple`) in `neuralops/ops/cuda/_map.py`.

---

## 2. Quantitative Typing Health Summary

| Audit Scope | Command Invocations | Total Files Evaluated | Errors Reported | Status / Assessment |
|---|---|---|---|---|
| **Rule 6.1 v2 Scope** | `mypy --strict neuralops/benchmarks/{config_v2,results_v2,cli_v2}/` | 18 | **0** | **PASS (100% Strict Compliant)** |
| **Core Strict Scope** | `mypy --strict neuralops/core neuralops/api.py neuralops/models neuralops/surgery neuralops/benchmarks/cli_v2` | 313 | **948** | Requires Wave 6 Typing Hardening |
| **Full Repository Scope** | `mypy --ignore-missing-imports neuralops/` | 1,808 | **897** | High concentration in legacy mappings |

---

## 3. Error Code Breakdown Table

### 3.1 Strict Mode Diagnostic Breakdown (`mypy_strict_core.txt`)

| Mypy Error Code | Occurrences | Percentage | Primary Root Cause |
|---|---|---|---|
| `[arg-type]` | 252 | 26.6% | Incompatible argument types passed to PyTorch operations and internal class constructors |
| `[operator]` | 123 | 13.0% | Unsupported operands (e.g. `@`, `+`, `*`) between `Tensor` and `nn.Module` or optional types |
| `[assignment]` | 119 | 12.6% | Reassignment of variables to `None` or subtypes without `Optional[...]` annotations |
| `[no-any-return]` | 88 | 9.3% | Functions with explicit return annotations returning un-cast or untyped `Any` expressions |
| `[index]` | 79 | 8.3% | Indexing into types inferred as `Any` or un-indexed containers |
| `[union-attr]` | 76 | 8.0% | Accessing attributes on `Union[T, None]` without explicit `is not None` guards |
| `[no-untyped-def]` | 37 | 3.9% | Functions or methods lacking complete parameter or return annotations |
| `[type-arg]` | 34 | 3.6% | Generic types (e.g. `DataLoader`, `dict`, `tuple`) used without type parameters |
| `[no-untyped-call]` | 33 | 3.5% | Calling unannotated legacy helper functions from strictly typed contexts |
| `[return-value]` | 17 | 1.8% | Returned expression does not match annotated signature |
| *Other codes* | 70 | 7.4% | `[override]`, `[attr-defined]`, `[no-redef]`, `[misc]` |
| **Total** | **948** | **100.0%** | |

### 3.2 Full Repository Breakdown (`mypy_full.txt`)

| Mypy Error Code | Occurrences | Primary Driver / Subsystem |
|---|---|---|
| `[dict-item]` | 334 | `neuralops/ops/cuda/_map.py` defines `dict[str, tuple[str, str]]` but assigns `list[str]` values |
| `[assignment]` | 114 | Dynamic tensor / module assignments in trainer and ops |
| `[operator]` | 106 | Arithmetic on PyTorch modules and union types |
| `[arg-type]` | 105 | Loss function and optimizer parameter passing |
| `[union-attr]` | 88 | Unchecked `None` attributes in dynamic backend dispatch |
| `[index]` | 40 | Tensor slicing and dictionary lookups |
| `[return-value]` | 19 | Return type variance in standalone Triton ops |
| `[attr-defined]` | 19 | Missing module-level registry exports |
| `[unused-ignore]` | 18 | Stale `# type: ignore` comments from earlier refactors |
| `[var-annotated]` | 15 | Unannotated local container initializations |

---

## 4. Top Offending Modules & Forensic Density Analysis

The top 15 modules contributing to static typing errors under strict mode are catalogued below:

| Errors | Module Path | Primary Vulnerability / Pattern |
|---|---|---|
| **113** | `neuralops/models/surgery.py` | Overloaded surgery hook functions with loose polymorphic argument handling |
| **58** | `neuralops/surgery/scanner.py` | Module tree traversal producing `Tensor \| Module` unions without type narrowing |
| **45** | `neuralops/core/peft/loha.py` | Unannotated low-rank factor multiplication and weight unpacking |
| **44** | `neuralops/core/low_rank/svd_linear.py` | SVD decomposition buffers, QR fallback return types, device placement unions |
| **39** | `neuralops/core/peft/lora.py` | Parameter registration and adapter scaling factor typing |
| **39** | `neuralops/core/peft/lokr.py` | Kronecker product decomposition tensor typing |
| **38** | `neuralops/surgery/director.py` | `UniversalSurgeryDirector` module replacement passes without explicit type guards |
| **36** | `neuralops/surgery/streaming_distill.py` | Intermediate activation hooks and streaming buffer types |
| **34** | `neuralops/models/modeling.py` | Pretrained model initialization arguments and forward signature variance |
| **32** | `neuralops/surgery/fx_surgery.py` | PyTorch FX graph node transformations and symbol replacements |
| **32** | `neuralops/api.py` | Public facade functions accepting loose `**kwargs` and unvalidated configuration dicts |
| **31** | `neuralops/core/peft/simple_lora.py` | Lightweight LoRA adapter forwarding |
| **27** | `neuralops/surgery/mla_surgery.py` | Multi-head Latent Attention compression surgery interfaces |
| **20** | `neuralops/core/_attention_injectors.py` | Dynamic attention replacement dispatch |
| **19** | `neuralops/core/architecture.py` | Architecture registry introspection |

---

## 5. Public API Type Completeness Scorecard

| Interface / Module | Strict Mypy Status | Parameter Annotations | Return Annotations | None-Safety Guards |
|---|---|---|---|---|
| `neuralops.benchmarks.config_v2` | **PASS (0 errors)** | 100% | 100% | Complete |
| `neuralops.benchmarks.results_v2` | **PASS (0 errors)** | 100% | 100% | Complete |
| `neuralops.benchmarks.cli_v2` | **PASS (0 errors)** | 100% | 100% | Complete |
| `neuralops.api` | **32 errors** | Partial (loose kwargs) | Annotated | Missing type narrowing |
| `neuralops.surgery.director` | **38 errors** | Complete | Complete | Missing `isinstance(mod, Module)` narrowing |
| `neuralops.core.low_rank` | **44 errors** | Annotated | Annotated | QR/SVD return type unions |
| `neuralops.ops.cuda._map` | **334 errors (full)** | Annotated | Annotated | `list[str]` vs `tuple[str, str]` literal mismatch |

---

## 6. Remediated Related Files

Pursuant to Section 1.3 of W5-11 Operational Specification ("Holistic Verification & Fixes"), the following files in the v2 benchmark subsystems were corrected to achieve 100% clean `mypy --strict` compliance:

1. **`neuralops/benchmarks/config_v2/loader.py`**:
   - Explicitly exported `ConfigValidationError` in `__all__` to satisfy module re-export typing contracts in `config_v2/__init__.py`.
2. **`neuralops/benchmarks/results_v2/ledger.py`**:
   - Annotated untyped pyarrow parquet calls (`pq.read_table`, `pq.write_table`) with `# type: ignore[no-untyped-call]` to prevent strict type check breakage due to missing third-party stubs.

Following these changes, `mypy --strict neuralops/benchmarks/config_v2/ neuralops/benchmarks/results_v2/ neuralops/benchmarks/cli_v2/` passes with **0 errors across 18 source files**.

---

## 7. Prioritized Wave 6 Typing Remediation Roadmap

To achieve repo-wide strict typing compliance without breaking backward compatibility:

### Package 1: Low-Hanging Structural Fixes (Immediate 37% error reduction)
- **Target:** `neuralops/ops/cuda/_map.py`
- **Action:** Convert list literals in `_KERNEL_SYMBOL_MAP` values from `[...]` to `(...)` to match the declared `dict[str, tuple[str, str]]` type, immediately eliminating 334 `[dict-item]` errors.

### Package 2: Type Narrowing for Surgery & Scanner (Eliminating ~250 errors)
- **Target:** `neuralops/surgery/scanner.py`, `neuralops/surgery/director.py`, `neuralops/surgery/fx_surgery.py`
- **Action:** Replace `Tensor | Module` ambiguous unions with explicit type guards (`isinstance(target, torch.nn.Module)`) prior to member dereference and `@` matrix operations.

### Package 3: PEFT & SVD Adapter Typing (Eliminating ~190 errors)
- **Target:** `neuralops/core/peft/{lora, loha, lokr, simple_lora}.py`, `neuralops/core/low_rank/svd_linear.py`
- **Action:** Explicitly annotate adapter forward return signatures and declare `Parameter` types for all low-rank factors.

### Package 4: Public API & v2 Sibling Architecture (Rule 6.1 Enforcement)
- **Target:** `neuralops/api.py`, `suites_v2/`
- **Action:** Complete `suites_v2/` implementation under strict mypy and replace untyped `**kwargs` in `neuralops.api` with typed parameter objects or overloads.

---

## 8. Audit Evidence Verification

The following audit artifacts have been generated and validated:
- `audits/wave5/w5_11_evidence/mypy_strict_output.txt`
- `audits/wave5/w5_11_evidence/mypy_full_output.txt`
- `audits/wave5/w5_11_evidence/mypy_strict_core.txt`
- `audits/wave5/w5_11_evidence/mypy_full.txt`
- `audits/wave5/w5_11_evidence/typing_summary.json`
- `audits/wave5/w5_11_evidence/typing_error_breakdown.json`
- `audits/wave5/w5_11_evidence/top_offending_modules.txt`
