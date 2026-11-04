---
name: neuralops-release-hardening-pipeline
description: Multi-phase pre-release validation, secret leakage scanning, license header compliance, invariant counting, and package distribution verification.
---

# NeuralOps Release Hardening Pipeline Protocol

Before publishing an official NeuralOps distribution release (wheel, sdist, GitHub release, or Git tag), all repository invariants, security checks, licensing requirements, and packaging artifacts must pass strict verification gates.

This skill provides an exhaustive 6-phase engineering checklist and execution protocol for release hardening.

---

## Phase 1: Security & Credential Leakage Sweep
1. **Detect Uncommitted Sensitive Data**:
   - Run POSIX inspection to ensure zero API keys, tokens, or SSH credentials exist in repository trees:
     ```bash
     grep -rnE "(ghp_|sk-[a-zA-Z0-9]{20,}|-----BEGIN OPENSSH PRIVATE KEY-----|BEGIN RSA PRIVATE KEY)" . \
       --exclude-dir={.git,.venv,node_modules}
     ```
2. **Scan Ignore Lists**:
   - Verify `.gitignore` rules prevent staging of sensitive or generated directories:
     - `audit/`, `docs/`, `results/`, `academic/`, `paper/`, `thesis_final/`, `logs/`, `scratch/`, `grading.log`.

---

## Phase 2: Top-of-File License Header Compliance
1. **Audit Standard MIT License Header**:
   - Verify that all newly created and modified Python source files begin with the standard license header:
     ```python
     #!/usr/bin/env python3
     # Copyright (c) 2024-2026, Taha Majlesi
     #
     # This source code is licensed under the MIT license found in the
     # LICENSE file in the root directory of this source tree.
     # For a list of all contributors, visit:
     #   https://github.com/tahamajs/NeuralOps/graphs/contributors
     ```
2. **Eliminate Banned Author Docstrings**:
   - Verify that `rg "Author: Taha Majlesi"` returns zero occurrences (authorship managed via Git history).

---

## Phase 3: Canonical Count & Invariant Audit
1. **Execute Invariant Audits**:
   - Run consistency checks:
     ```bash
     python3 scripts/utils/audit_dead_code.py
     ```
2. **Verify Canonical Totals**:
   - Exact count of 131 logical attention keys in `ATTENTION_REGISTRY`.
   - Exact count of 72 logical compression keys in `COMPRESSOR_REGISTRY`.
   - Exact count of 26 canonical Triton kernel modules in `neuralops/core/triton_kernels/`.
   - Exact count of 110 unique CUDA modules and 114 native implementations.
   - Exact count of 29 Jupyter tutorials in `notebooks/`.

---

## Phase 4: Linter & Static Analysis Verification
1. **Execute Ruff Linters**:
   - Run full repository linting check:
     ```bash
     ./.venv/bin/ruff check neuralops/ tests/ scripts/
     ```
   - Guarantee zero fatal errors or unresolved rule violations.

---

## Phase 5: Distribution Packaging & Artifact Inspection
1. **Build Wheel and Source Distribution**:
   - Execute package build:
     ```bash
     python3 -m build --sdist --wheel
     ```
2. **Inspect Wheel Contents**:
   - Inspect package archive to ensure no test fixtures, scratch directories, or C++ build intermediate objects (`.o`, `.so` without architecture tags) are bundled:
     ```bash
     tar -ztvf dist/neuralops-*.tar.gz | grep -E "(\.so|\.o|results|audit)" || true
     ```

---

## Phase 6: Git Tagging & Release Documentation
1. **Enforce Mandatory Commit Attribution**:
   - Ensure all release commits use:
     `--author="taha <tahamaj4@gmail.com>"`
2. **Tag Release**:
   - Create signed or annotated tag:
     ```bash
     git tag -a v<VERSION> -m "NeuralOps release v<VERSION>"
     ```
3. **Generate Release Changelog**:
   - Compile changelog detailing verified performance numbers, bug fixes, and verified hardware targets.
