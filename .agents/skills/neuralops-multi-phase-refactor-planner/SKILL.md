---
name: neuralops-multi-phase-refactor-planner
description: Multi-phase codebase modernization, AST verification, registry synchronization, and legacy construct elimination protocol.
---

# NeuralOps Multi-Phase Refactor Planner Protocol

Refactoring foundational deep learning architectures, operator registries, or dispatch layers across hundreds of source files risks silent regressions, broken public APIs, and registry desynchronization.

This skill governs a strict 6-phase engineering refactor protocol to ensure zero regressions and maintain architectural integrity.

---

## Phase 1: Architectural Blast Radius & Dependency Mapping
1. **Identify Target Symbols & Entry Points**:
   - Trace all public imports and consumers of the symbols slated for modification:
     ```bash
     rg "from neuralops\.<subsystem> import <symbol>" neuralops/ tests/
     ```
2. **Registry Invariant Verification**:
   - Check canonical registry counts:
     - Exactly 131 logical attention keys in `ATTENTION_REGISTRY`.
     - Exactly 72 logical compression keys in `COMPRESSOR_REGISTRY`.
     - Exactly 26 canonical Triton kernel modules.
     - Exactly 110 unique CUDA modules and 114 native implementations.
3. **Formulate Single-Sentence Purpose**:
   - Articulate the refactor's single purpose and exact blast radius before touching any file.

---

## Phase 2: Deprecation & Compatibility Layering
1. **Preserve Public API Contracts**:
   - If a symbol or module path must be relocated or renamed, create a backward-compatible alias in the original location.
   - Attach a standard deprecation warning:
     ```python
     import warnings
     warnings.warn("Symbol 'X' has moved to 'Y'. Please update your imports.", DeprecationWarning, stacklevel=2)
     ```
2. **Prevent Breaking Changes**:
   - Never remove a function parameter without a multi-release deprecation cycle.
   - Ensure `__all__` definitions in `__init__.py` files remain consistent.

---

## Phase 3: Incremental File-by-File Modernization
1. **Read Entire Target File**:
   - Read the target file completely prior to editing. Never edit blindly based on snippets.
2. **Execute Targeted Edits**:
   - Keep diffs minimal and focused. Do not combine stylistic cleanup with structural refactors.
   - Preserve existing explanatory comments and docstrings.
   - Maintain the standard top-of-file MIT license header.

---

## Phase 4: AST Audit & Dead Code Verification
1. **Run AST Dead-Code Auditor**:
   - Parse AST across all modified files using `scripts/utils/audit_dead_code.py`.
   - Ensure zero undefined variables, unused imports, or orphaned syntax trees.
2. **Ruff & Linter Validation**:
   - Run `ruff check <modified_files>` to guarantee adherence to repository linting rules.

---

## Phase 5: Regression & Cross-Subsystem Parity Testing
1. **Demonstrate Pre-Fix / Post-Fix Proof**:
   - For bug fixes or behavioral corrections, write a regression test that fails on the un-refactored code and passes on the refactored code.
2. **Execute Subsystem Test Suite**:
   - Run CPU-only tests locally (under explicit user authorization) or submit to Sharif HPC queue for GPU-accelerated suites.

---

## Phase 6: Atomic Commit & PR Readiness Review
1. **Enforce Commit Boundaries**:
   - Separate refactoring commits from feature commits or documentation commits.
   - Enforce mandatory Git attribution: `--author="taha <tahamaj4@gmail.com>"`.
2. **Verify Ignored Directory Invariants**:
   - Confirm that `audit/`, `docs/`, `results/`, `logs/`, and build artifacts are not staged for commit.
