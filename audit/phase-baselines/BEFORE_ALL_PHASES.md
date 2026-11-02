# Baseline State Before 10-Work-Package Fix Sequence

**Date:** 2026-11-02  
**Commit SHA:** `d8c69ba0fb9bfab1090f528c0edeca6ed530d47d`  
**Package Version:** `2.0.1` (`neuralops/__init__.py:75`)  
**Status:** Pre-flight Verification Complete & Working Tree Clean  

---

## 1. Source Document Verification
- **Path:** `audit/CODE_FIX_PROPOSALS_2026-11-02.md`
- **File Metadata:** 54,504 bytes, updated Nov 2 2026.
- **Verification:** Verified present and readable. Contains complete specifications for WP1 through WP10 across the 4 planned execution phases.

---

## 2. Baseline Repository Invariants

| Invariant | Expected Count | Measured Count | Verification Command / Method | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Attention Mechanisms** | 131 | 131 | `len(neuralops.ATTENTION_REGISTRY)` | **PASS** |
| **Model Compressors** | 72 | 72 | `len(neuralops.COMPRESSOR_REGISTRY)` | **PASS** |
| **Canonical Triton Modules** | 26 | 26 | `neuralops.benchmarks.benchmark_all_26_kernels` | **PASS** |
| **Jupyter Notebooks** | 29 | 29 | `find notebooks -maxdepth 1 -name "*.ipynb" \| wc -l` | **PASS** |
| **CUDA C++ Kernel Packages** | 130 | 130 | `scripts/audit_readme_consistency.py` | **PASS** |

Raw shell check outputs:
```bash
$ find notebooks -maxdepth 1 -name "*.ipynb" | wc -l
29

$ ./.venv/bin/python -c "import neuralops; print('ATTENTION_REGISTRY:', len(neuralops.ATTENTION_REGISTRY)); print('COMPRESSOR_REGISTRY:', len(neuralops.COMPRESSOR_REGISTRY))"
ATTENTION_REGISTRY: 131
COMPRESSOR_REGISTRY: 72
```

---

## 3. Tool Quality Gates

### 3.1 Ruff Lint Check (`neuralops/`, `tests/`)
```bash
$ ./.venv/bin/ruff check neuralops/ tests/
All checks passed!
```

### 3.2 Dead Code & AST Audit (`scripts/utils/audit_dead_code.py`)
```bash
$ ./.venv/bin/python scripts/utils/audit_dead_code.py
Audited 1593 Python source files. Issues found: 0
```

### 3.3 README & Repository Invariant Consistency (`scripts/audit_readme_consistency.py`)
```bash
$ ./.venv/bin/python scripts/audit_readme_consistency.py
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

---

## 4. Planned Work Packages Overview

- **Phase 1:** WP 1 (5 Crashing CUDA bindings) + WP 6 (4 Pseudo-Triton module cleanups)
- **Phase 2:** WP 2 (7 NaN/numeric kernel guards) + WP 3 (7 Timeout kernel bounds)
- **Phase 3:** WP 4 (7 Sub-optimal kernels) + WP 5 (71 `FAIL_CORRECTNESS` tolerance & mock improvements) + WP 7 (8 Parameter sync mismatches)
- **Phase 4:** WP 8 (46 Orphan result archives) + WP 9 (87 Purged legacy script shims) + WP 10 (SVD rank bounds & calibration checks)
