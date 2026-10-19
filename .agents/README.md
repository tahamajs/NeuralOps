# NeuralOps Agent Workspace

Repository-local agent configuration and technical skill definitions reside here. [CONTRIBUTING.md](../CONTRIBUTING.md) defines contribution rules; [AGENTS.md](../AGENTS.md) defines agent operational behavior and safety constraints.

---

## 1. Skill Index & Quick Navigation

The [Skill Index](skills/README.md) serves as the authoritative catalog of engineering workflows across NeuralOps. Skills provide executable step-by-step procedures, technical guidance, CLI workflows, and mathematical foundations for specialized engineering tasks.

### Core Architecture & Contribution Skills
- **[Design Coverage](skills/neuralops-design-coverage/SKILL.md)**: Floating-point precision budgets (FP16, BF16, FP32, FP8), `input_guard` standards, and verifier contracts.
- **[Correctness Coverage](skills/neuralops-correctness-coverage/SKILL.md)**: Matrix test guidelines for all 131 attention mechanisms and 72 compressors, autograd gradient checks (`atol=1e-3, eps=1e-5`), causal masking perturbation proofs, and cross-backend parity.
- **[Dispatch Backends](skills/neuralops-dispatch-backends/SKILL.md)**: 3-level fallback hierarchy (Level 1 CUDA C++/HIP -> Level 2 Triton JIT -> Level 3 PyTorch), `force_kernel_backend` context manager, and zero-overhead caching.
- **[PR Readiness](skills/neuralops-pr-readiness/SKILL.md)**: Submission quality checklist, 6-class dead code AST auditor (`scripts/utils/audit_dead_code.py`), universal shebang verification, and Keep a Changelog standards.
- **[PR Stacks](skills/neuralops-pr-stacks/SKILL.md)**: 5-layer atomic stack decomposition, branching strategies, `git rebase --onto` workflows, and GitHub native stacked PRs.

### Hardware Performance & Specialization Skills
- **[NVIDIA Performance](skills/neuralops-nvidia-performance/SKILL.md)**: Nsight Compute (NCU) profiling, warp intrinsics, and shared memory bank conflict resolution.
- **[Ascend Performance](skills/neuralops-ascend-performance/SKILL.md)**: DaVinci NPU architecture tuning, Unified Buffer (UB) tiling, and CANN compiler optimization.
- **[Optimization Loop](skills/neuralops-optimization-loop/SKILL.md)**: Automated microbenchmarking and iterative kernel tuning with frozen correctness assertions.
- **[KDA](skills/neuralops-kda/SKILL.md)**: KDA gate activations, safe intra-chunk diagonal block offsets, and recurrent state transitions.
- **[Triton to Gluon](skills/neuralops-triton-to-gluon/SKILL.md)**: Porting Triton JIT memory layouts and barrier synchronizations to Ascend Gluon C++.

---

## 2. Skill Organization Conventions

All skills are maintained under `.agents/skills/<skill-name>/SKILL.md`. Each skill must include standard YAML frontmatter with `name` and `description` attributes. Technical documentation within skills must strictly adhere to NeuralOps architecture, conventions, and Python standards with zero placeholders.
