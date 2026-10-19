# NeuralOps Guidelines

Guidance for AI coding agents (Claude Code, Codex, Antigravity, etc.) working in this repo.

**Read `CONTRIBUTING.md` first.** It is the authoritative source for all code style, docstring, comment, commit, PR, and testing conventions, and applies to humans and agents alike. This file covers agent-specific operational behavior that doesn't belong in a human contributor doc.

---

## 1. Git Safety & Attribution Laws

- **Mandatory Commit Attribution**: All Git commits MUST strictly use `--author="taha <tahamaj4@gmail.com>"`.
- **Never discard uncommitted work** with `git checkout HEAD -- <file>`, `git reset --hard`, or `git restore` to "get a clean base". Unstaged changes are unrecoverable (no blob, no reflog). Edit in place or `git stash` instead, and confirm with the user when in doubt.
- **On `main`**: never commit or push without explicit per-action approval. Suggest a feature branch first.
- Don't rewrite or amend already-pushed commits unless the user asks.
- **Branch names and attribution**: name local and remote branches after the change. Do not include tool or assistant identities in branch names, commit messages, PR titles/bodies, or review comments. Do not add `Co-Authored-By` trailers or generated-by notices.

---

## 2. Authorship, Header & Comment Discipline

- **Strict Header Standard**: Every source file MUST start with the standard top-of-file MIT license block referencing contributors:
  ```python
  #!/usr/bin/env python3
  # Copyright (c) 2024-2026, Taha Majlesi
  #
  # This source code is licensed under the MIT license found in the
  # LICENSE file in the root directory of this source tree.
  # For a list of all contributors, visit:
  #   https://github.com/tahamajs/NeuralOps/graphs/contributors
  """
  Module Docstring here...
  """
  ```
- **Banned Personal Docstring Headers**: NEVER insert inline `"""Author: Taha Majlesi` or `# Author: Taha Majlesi` docstrings. Codebase authorship is managed via Git history and the official contributors graph.
- **You own the comments your change invalidates.** When a change makes a comment or docstring in the code you touched factually wrong, fix it or delete it in the same commit — "the code you touched" means the same function, kernel, or class, not a repo-wide sweep for stale mentions. If you can't tell whether a comment is still true, keep it and flag it in the PR description; never delete a "why" comment just because you can't verify it.
- **Comments are not your scratchpad.** State design conclusions, not the journey: "not X, it does not compile" is a welcome why comment; "tried X first, then switched to Y" is chain-of-thought noise. This covers docstrings too.
- **Fix the words, not the style.** Correcting stale content is not a license to reformat the surrounding docstring or comment style.

---

## 3. Strict Execution Environment Rules

- **Zero Local macOS Python Execution**: NEVER execute `python` or `pytest` directly on the local macOS host. macOS Python 3.14 lacks typing extensions, PyTorch CUDA extensions, and native accelerator drivers.
- **Static Verification on Host**: Local verification on macOS must be strictly conducted via POSIX inspection (`grep`, `awk`, `find`, `node`), AST parsers (`scripts/utils/audit_dead_code.py`), and lint checkers.
- **Dynamic & GPU Kernel Verification**: Dynamic tests, `torch.autograd.gradcheck`, CUDA C++ compilation, and Triton JIT execution must be run remotely on `taha@192.168.1.100` (`/mnt/d/SVD_Project`).

---

## 4. Repository Invariants & Non-Core Directories

- **Canonical Counts Invariant**:
  - `ATTENTION_REGISTRY` must contain exactly **131** attention mechanisms.
  - `COMPRESSOR_REGISTRY` must contain exactly **72** model compression engines.
  - `neuralops/core/triton_kernels/` must contain exactly **26** canonical Triton kernel modules (housing 62 unified operators across 4 tiers).
  - `neuralops/core/cuda_kernels/` must contain exactly **114** native CUDA C++ kernel packages.
  - `notebooks/` must contain exactly **27** fully executed Jupyter notebooks.
- **Ignored Directories Law**:
  - `audit/`, `docs/`, `results/`, `academic/`, `paper/`, `thesis_final/`, `logs/`, `scratch/`, and `grading.log` are strictly ignored in `.gitignore`.
  - These folders contain academic thesis drafts, build artifacts, raw telemetry, and local experiment outputs. They must NEVER be committed to Git.

---

## 5. Kernel & Architecture Design Standards

- **3-Level Fallback Hierarchy**: Every kernel operation must adhere to the 3-level fallback contract:
  1. *Level 1*: Native CUDA C++ / HIP compiled extension (maximum throughput).
  2. *Level 2*: Triton JIT kernel (autotuned for FP16/BF16/FP32).
  3. *Level 3*: Vectorized PyTorch reference (universal CPU/MPS fallback).
- **Zero-Overhead Dispatch**: Kernel backend selection must use `@lru_cache` or zero-allocation dispatch tables.
- **No Block Pointers in Mainline Triton**: Check the diff for `tl.make_block_ptr` / `tl.advance`; these are banned from mainline Triton code (Huawei Ascend NPU backend exemption only).
- **Numerical Budget & Dtype Guards**: Custom forward kernels must compute or accumulate in FP32 when inputs are FP16 or BF16. Finite difference autograd gradient checks must pass `atol=1e-3, eps=1e-5`.
- **Flame-Style Multi-Dataset & Packing Standards**:
  - Multi-dataset ingestion must support comma-separated paths: `--training.dataset HuggingFaceFW/fineweb-edu,opencsg/Fineweb-Edu-Chinese-V2`.
  - Zero-padding variable-length document packing (`--training.varlen`) must generate valid `cu_seqlens` boundary offsets `[0, len_1, len_1+len_2, ..., seq_len]`.
  - Learning rate schedulers must support Warmup-Stable-Decay (`wsd`).

---

## 6. PR Workflow & Quality Gates

- **No busywork PRs**: don't open a one-off PR for a single typo or isolated style tweak; bundle trivial cleanups into substantive work.
- **Minimal diff**: every changed line serves the PR's stated purpose. Drop drive-by reformats, unrelated cleanups, and speculative generality before pushing.
- **Blast radius audit**: for every shared symbol you touched, verify all callsites hold. State in the PR description which shapes, dtypes, and backends are affected.
- **Mandatory Quality Gates**:
  1. `python3 scripts/utils/audit_dead_code.py` $\implies$ 0 issues found.
  2. `python3 scripts/audit_readme_consistency.py` $\implies$ 0 errors found.
  3. Universal shebang `#!/usr/bin/env python3` and executable permissions on CLI scripts.

---

## 7. Repo-local Skills

Use the [skill index](.agents/skills/README.md) to select task-specific workflows and their execution order. Skills supplement `CONTRIBUTING.md` and this file; they do not define competing contribution policies.

Load a relevant skill when the task matches its scope. In particular, load [neuralops-pr-readiness](.agents/skills/neuralops-pr-readiness/SKILL.md) before opening any PR.
