# NeuralOps Skills Catalog & Architectural Index

[CONTRIBUTING.md](../../CONTRIBUTING.md) defines contribution rules; [AGENTS.md](../../AGENTS.md) defines agent operational behavior. This directory contains specialized technical workflows and development skills for NeuralOps. Use this index as the authoritative catalog for engineering operations across operator design, hardware kernel optimization, verification, and pull request workflows.

---

## 1. Development Lifecycle & Skill Workflow

NeuralOps kernel and module development follows a rigorous four-phase engineering lifecycle:

```
+----------------------------------------------------------------------------------------------------+
|                                    NeuralOps Engineering Lifecycle                                 |
+----------------------------------------------------------------------------------------------------+
| Phase 1: Architecture & Design Contract                                                            |
| -> [neuralops-design-coverage](neuralops-design-coverage/SKILL.md)                                      |
|    Define FP16/BF16/FP32 precision budgets, input_guard contracts, and 3-level dispatch verifiers.   |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
| Phase 2: Hardware Acceleration & Optimization                                                      |
| -> [neuralops-dispatch-backends](neuralops-dispatch-backends/SKILL.md)                                  |
|    Implement Level 1 CUDA C++ / HIP, Level 2 Triton JIT, and Level 3 PyTorch Reference.            |
| -> [neuralops-nvidia-performance](neuralops-nvidia-performance/SKILL.md)                                |
|    Profile NCU/Nsight metrics, warp occupancy, shared memory bank conflicts, and tensor cores.     |
| -> [neuralops-ascend-performance](neuralops-ascend-performance/SKILL.md)                                |
|    Tune DaVinci NPU cube/vector cores, UB tiling, and Ascend CANN constraints.                      |
| -> [neuralops-optimization-loop](neuralops-optimization-loop/SKILL.md)                                  |
|    Iterate through automated microbenchmarking with frozen correctness gates.                       |
| -> [neuralops-kda](neuralops-kda/SKILL.md)                                                              |
|    Tune KDA log-space gates, diagonal block tiling, and recurrent state updates.                    |
| -> [neuralops-triton-to-gluon](neuralops-triton-to-gluon/SKILL.md)                                      |
|    Port Triton memory layouts and barrier synchronizations to Ascend Gluon C++.                     |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
| Phase 3: Comprehensive Correctness & Parity Verification                                           |
| -> [neuralops-correctness-coverage](neuralops-correctness-coverage/SKILL.md)                            |
|    Exhaustive 131 attention & 72 compressor matrices, high-precision autograd gradchecks           |
|    (atol=1e-3, eps=1e-5), causal perturbation proofs, and cross-backend cosine parity (>= 0.95).   |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
| Phase 4: Quality Gate & PR Submission                                                              |
| -> [neuralops-pr-readiness](neuralops-pr-readiness/SKILL.md)                                            |
|    Execute 6-class AST dead code audit, verify universal shebangs, format changelog, run CI.       |
| -> [neuralops-pr-stacks](neuralops-pr-stacks/SKILL.md)                                                  |
|    Decompose features into atomic 5-layer stacks, manage git rebase --onto, link native PR stacks.  |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Comprehensive Skills Catalog

| Skill | Category | Primary Focus & Capabilities |
| :--- | :--- | :--- |
| **[neuralops-design-coverage](neuralops-design-coverage/SKILL.md)** | Architecture & Contracts | Floating-point numerical budgets (FP16 underflow/overflow, BF16 quantization noise, FP32 master states), `input_guard` tensor sanitization, and dispatch verifier contracts. |
| **[neuralops-correctness-coverage](neuralops-correctness-coverage/SKILL.md)** | Verification & Testing | Full matrix test suites for all 131 attention mechanisms and 72 compressors, analytical double-precision `gradcheck` (`atol=1e-3, eps=1e-5`), future-token causal perturbation proofs, and cross-backend parity. |
| **[neuralops-dispatch-backends](neuralops-dispatch-backends/SKILL.md)** | Runtime Dispatch | 3-Level fallback chain (Level 1 CUDA C++/HIP -> Level 2 Triton JIT -> Level 3 Vectorized PyTorch), `force_kernel_backend` context manager, telemetry counters, and `_AutogradBridge`. |
| **[neuralops-pr-readiness](neuralops-pr-readiness/SKILL.md)** | CI & Quality Gates | Strict submission quality checklist, automated 6-class dead code AST audit (`scripts/utils/audit_dead_code.py`), universal shebang verification, and Keep a Changelog formatting. |
| **[neuralops-pr-stacks](neuralops-pr-stacks/SKILL.md)** | Git & Code Review | 5-Layer stack taxonomy, clean branching strategies, `git rebase --onto` surgery workflows, and GitHub native stacked PR verification. |
| **[neuralops-nvidia-performance](neuralops-nvidia-performance/SKILL.md)** | GPU Optimization | NVIDIA NCU roofline analysis, warp-level shuffle intrinsics, shared memory bank conflict elimination, and Tensor Core MMA utilization on Ada Lovelace/Hopper/Ampere. |
| **[neuralops-ascend-performance](neuralops-ascend-performance/SKILL.md)** | NPU Optimization | Huawei Ascend DaVinci architecture tuning, Unified Buffer (UB) memory alignment, Cube/Vector core pipelining, and CANN compiler directives. |
| **[neuralops-optimization-loop](neuralops-optimization-loop/SKILL.md)** | Performance Iteration | Automated benchmarking harness, frozen correctness assertion gates, latency/TFLOPs tracking, and flamegraph profiling. |
| **[neuralops-kda](neuralops-kda/SKILL.md)** | Operator Specialization | KDA gate mechanics, safe intra-chunk diagonal block offsets, log-space activations, and recurrent state transitions. |
| **[neuralops-triton-to-gluon](neuralops-triton-to-gluon/SKILL.md)** | Platform Porting | Translating Triton JIT kernel memory hierarchies, thread-block schedules, and barriers into native Ascend Gluon C++ extensions. |

---

## 3. Maintaining and Adding Skills

Every skill must reside in `.agents/skills/<skill-name>/SKILL.md` with standard YAML frontmatter:
```yaml
---
name: <skill-name>
description: <Comprehensive single-sentence description of the skill's purpose and scope>
---
```

### Skill Authoring Standards:
- **100% NeuralOps Terminology**: Adhere strictly to NeuralOps architecture, conventions, and Python standards without mentioning external or deprecated repositories.
- **Deep Technical Guidance**: Provide concrete code snippets, mathematical definitions, ASCII / Mermaid diagrams, and step-by-step CLI workflows.
- **Zero Placeholders**: Never include empty templates, `TODO` comments, or generic placeholder text.
