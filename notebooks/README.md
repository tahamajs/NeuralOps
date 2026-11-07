# NeuralOps Interactive Notebook Ecosystem

**Canonical Source of Truth:** `notebooks/` (94 unique Jupyter notebooks)
**Compatibility Symlink:** `notebooks/extended -> .`
**Current Release Alignment:** NeuralOps v2.0.6

---

## 1. Architecture & Single Source of Truth

The `notebooks/` directory contains 94 canonical, self-contained demonstration tutorials spanning subquadratic attention operators, model surgery, knowledge distillation, hardware profiling, and high-throughput serving.

To eliminate duplication drift (Finding F-75):
- `notebooks/` is the sole source of truth (188 duplicate files consolidated to 94 unique notebooks).
- `notebooks/extended/` is maintained as a directory symlink pointing to `.` so legacy links and scripts resolve seamlessly without duplicating storage or diffs.

---

## 2. Execution Tiers & Scope

Per project policy (Decision 5):
1. **Executed Subset (29 Notebooks)**:
   - Foundational and core tutorials (`00_...` through `25_...`, including `01a`/`01b`, `02a`/`02b`, and `05a`/`05b`) have physical execution telemetry and plots verified on NVIDIA RTX 3090 (Sharif University HPC cluster).
2. **Structure Validated Subset (65 Notebooks)**:
   - Extended ecosystem notebooks (`26_...` through `90_...`) have syntactically validated cells, correct schema, and valid JSON structure checked on CI. Full execution requires dedicated multi-GPU HPC allocations.

---

## 3. How to Run & Verify

### Interactive Execution
```bash
jupyter lab notebooks/
```

### CI Smoke & Structural Validation (CPU)
Validates notebook JSON structure, parses all code cells with AST, and runs fast CPU smoke tests:
```bash
pytest tests/test_notebooks_execution.py -q -k "smoke or structure"
```

### Catalog Conformance Check
Asserts all 94 notebooks are indexed in the master catalog and that `notebooks/extended` is a symlink:
```bash
pytest tests/test_notebook_catalog.py -q
```

### Full Hardware Execution (HPC / GPU CI)
```bash
pytest tests/test_notebooks_execution.py -v -m "gpu"
```
