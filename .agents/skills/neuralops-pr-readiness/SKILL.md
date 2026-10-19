---
name: neuralops-pr-readiness
description: Prepare or update a NeuralOps pull request with strict checklists, dead-code AST audit, shebang verification, and changelog requirements.
---

# NeuralOps Pull Request Readiness and Submission Quality Gate

Use this skill to prepare, audit, and validate any Pull Request before submitting it to NeuralOps. Every pull request must pass the automated repository quality gates, pass the dead-code AST auditor, satisfy universal shebang standards, update the changelog, and present verifiable empirical evidence. Read [CONTRIBUTING.md](../../../CONTRIBUTING.md) for contribution rules.

---

## 1. Strict PR Readiness Checklist

Before marking a PR as ready for review or merging, every item in this checklist must be verified:

```markdown
### NeuralOps Pull Request Quality Gate Checklist
- [ ] **Scope and Atomicity**: PR implements exactly one cohesive architectural improvement, kernel optimization, or bug fix. Unrelated changes or refactors are separated.
- [ ] **Numerical Correctness**: All unit, operator, and matrix tests pass across applicable sequence lengths, head dimensions, and dtypes (FP16, BF16, FP32).
- [ ] **Autograd Gradient Sanity**: Custom forward kernels include analytical or bridged backward paths passing finite difference checks (`torch.autograd.gradcheck` with `atol=1e-3, eps=1e-5`).
- [ ] **Cross-Backend Parity**: Native CUDA C++, Triton JIT, and PyTorch reference outputs match with Cosine Similarity $\ge 0.95$ and absolute difference within tolerance.
- [ ] **Dead Code Elimination**: 100% clean audit from `scripts/utils/audit_dead_code.py` with zero shadowed definitions, unreachable statements, dead Triton kernels, or commented code blocks.
- [ ] **Universal Shebang & Permissions**: All standalone CLI scripts start with `#!/usr/bin/env python3` and carry executable POSIX permissions (`chmod +x`).
- [ ] **Code Hygiene & Typing**: Clean pass on formatting, linting, and type hints with zero unresolved issues (`ruff`, `flake8`, `mypy`).
- [ ] **Changelog & Documentation**: Detailed entry appended to `CHANGELOG.md` under the appropriate semantic version and category; public docstrings updated.
- [ ] **Empirical Benchmarking**: For performance-sensitive PRs, include before/after throughput (TFLOPs/s) and latency (ms) recorded on identical hardware.
```

---

## 2. Dead Code AST Audit (`scripts/utils/audit_dead_code.py`)

NeuralOps enforces automated static analysis to eliminate dead code across both Python AST and CUDA C++ sources. All PRs must achieve a clean exit code (`0`) from `scripts/utils/audit_dead_code.py`.

### 2.1 The 6 Scanned Dead Code Classes

```
+-----------------------------------------------------------------------------------------------+
|                        NeuralOps Dead Code AST Audit Engine                                   |
+---+----------------------------+--------------------------------------------------------------+
| # | Class                      | Violation Description & Detection Logic                      |
+---+----------------------------+--------------------------------------------------------------+
| 1 | Shadowed Definitions       | Duplicate function or class definitions within the same      |
|   |                            | scope overriding an earlier declaration without @overload    |
+---+----------------------------+--------------------------------------------------------------+
| 2 | Unreachable Statements     | Statements physically placed after return, raise, continue,  |
|   |                            | or break inside the same basic block                         |
+---+----------------------------+--------------------------------------------------------------+
| 3 | Dead Triton JIT Kernels    | Kernels decorated with @triton.jit that are never invoked via|
|   |                            | kernel[...] or referenced anywhere in the repository         |
+---+----------------------------+--------------------------------------------------------------+
| 4 | Unused Private Helpers     | Module-private functions (_helper) defined at top level that  |
|   |                            | have zero references across the entire codebase corpus       |
+---+----------------------------+--------------------------------------------------------------+
| 5 | Commented Code Blocks      | Faux-commented Python statements (>3 consecutive lines)      |
|   |                            | left behind from temporary debugging or scratch experiments  |
+---+----------------------------+--------------------------------------------------------------+
| 6 | Dead CUDA C++ Kernels      | __global__ or __device__ functions in CUDA sources that are   |
|   |                            | never dispatched by host C++ or PyTorch C++ bindings         |
+---+----------------------------+--------------------------------------------------------------+
```

### 2.2 Running the Dead Code Audit CLI
Execute the auditor from the repository root:
```bash
python scripts/utils/audit_dead_code.py
```

Expected clean output:
```text
================================================================================
NeuralOps Dead-Code & Unreachable AST Audit
================================================================================
Scanning Python files in repository...
Scanning CUDA C++ files in neuralops/core/cuda_kernels/...
[AUDIT] Shadowed definitions: 0 issues found.
[AUDIT] Unreachable statements: 0 issues found.
[AUDIT] Dead Triton JIT kernels: 0 issues found.
[AUDIT] Unused private helper functions: 0 issues found.
[AUDIT] Commented code blocks: 0 issues found.
[AUDIT] Dead CUDA C++ kernels: 0 issues found.
--------------------------------------------------------------------------------
Audit PASSED: Repository is 100% clean of dead or unreachable code.
```

If the auditor detects any violation, it prints the exact file path and line number, returning exit code `1`. Fix all flagged issues before staging commits.

---

## 3. Universal Shebang & File Permissions Verification

Every executable script in `scripts/`, `tests/`, and CLI tools must adhere to universal Unix execution standards:

1. **Shebang Specification**: The very first line of any executable Python script must be:
   ```python
   #!/usr/bin/env python3
   ```
2. **Standard Copyright Header**: Followed by the NeuralOps MIT copyright notice:
   ```python
   # Copyright (c) 2024-2026, Taha Majlesi
   #
   # This source code is licensed under the MIT license found in the
   # LICENSE file in the root directory of this source tree.
   ```
3. **Executable Bit**: The script must possess POSIX execution permissions (`755` or `+x`).

### 3.1 Verification and Remediation CLI
Run this one-liner to verify and fix shebangs and file permissions across all scripts:
```bash
# Audit scripts missing executable permissions
find scripts -type f -name "*.py" ! -perm -111 -exec chmod +x {} +

# Audit and verify first-line shebangs across all scripts
python -c '
from pathlib import Path
for p in Path("scripts").rglob("*.py"):
    lines = p.read_text().splitlines()
    if not lines or lines[0] != "#!/usr/bin/env python3":
        print(f"Missing universal shebang: {p}")
'
```

---

## 4. Changelog and Release Documentation Requirements

NeuralOps follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and adheres to [Semantic Versioning (SemVer)](https://semver.org/). Every substantive PR must include a documented entry in `CHANGELOG.md` under `## [Unreleased]` or the targeted version.

### 4.1 Standard Changelog Categories
Entries must be classified into one of the following standard sections:
- `### Added`: New attention mechanisms, compressors, CUDA/Triton kernels, or public APIs.
- `### Changed`: Changes in existing functionality, performance refactors, or altered default configurations.
- `### Deprecated`: Features scheduled for future removal.
- `### Removed`: Previously deprecated features or dead code removed.
- `### Fixed`: Bug fixes, numerical stability corrections, or memory leak resolutions.
- `### Security`: Vulnerability patches or input sanitization hardening.
- `### Performance`: Measurable speedups, memory footprint reductions, or throughput gains.

### 4.2 Formatting an Entry
```markdown
### Added
- **Sliding Window Native CUDA Kernel**: Warp-synchronous SM 8.9 implementation with 2.4x speedup over PyTorch baseline (`neuralops.core.cuda_kernels.sliding_window`).
- **Autograd Double-Precision Verification**: Full `gradcheck` suite for chunked linear attention with `atol=1e-3, eps=1e-5`.

### Fixed
- **FP16 Underflow in Delta Net Decay**: Clamped cumulative log-decay exponent at -14.0 to prevent subnormal zero-flush on long sequences (#142).
```

---

## 5. End-to-End PR Verification & Publishing Workflow

Follow this complete step-by-step workflow when submitting or updating a PR:

### Step 1: Discover and Run Dependent Tests
```bash
# Discover test suites affected by the modified files
python scripts/find_dependent_tests.py neuralops/core/cuda_kernels/sparse_attention/sparse_autograd.py

# Execute targeted test suites
pytest tests/cuda_kernels/ -v
pytest tests/test_cuda_triton_3level_complete.py -v
```

### Step 2: Run Repository Quality Auditors
```bash
# Execute dead code AST auditor
python scripts/utils/audit_dead_code.py

# Verify formatting and linting
pre-commit run --all-files
```

### Step 3: Publish or Update PR Description via REST API
To avoid classic-Projects integration issues with `gh pr edit`, write the PR body to a temporary file and update through the GitHub REST API:

```bash
PR_NUMBER=42
cat << 'EOF' > /tmp/pr_body.md
## Summary of Changes
- Implements Native CUDA C++ kernel for sliding window attention under SM 8.0+.
- Integrates with 3-level fallback hierarchy (Level 1 CUDA -> Level 2 Triton -> Level 3 PyTorch).
- Verified mathematical parity with cosine similarity >= 0.99 and autograd gradcheck (atol=1e-3, eps=1e-5).

## Quality Gate Checklist
- [x] All 131 attention matrix tests pass.
- [x] Zero dead code reported by `scripts/utils/audit_dead_code.py`.
- [x] Universal shebang verified on all scripts.
- [x] Updated `CHANGELOG.md`.

## Empirical Benchmarks
- Baseline PyTorch: 14.2 ms (T=4096, D=128, B=4)
- Native CUDA: 3.8 ms (3.7x speedup, 118 TFLOPs/s on RTX 4070 SM 8.9)
EOF

gh api -X PATCH "repos/tahamajs/NeuralOps/pulls/$PR_NUMBER" -F "body=@/tmp/pr_body.md"
```
