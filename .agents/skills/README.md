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
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
| Phase 5: Academic Publication & arXiv Release                                                      |
| -> [paper-codebase-deep-explorer](paper-codebase-deep-explorer/SKILL.md)                            |
|    Systematically explore, index, grep, and cross-reference symbols, registries, and kernels.      |
| -> [paper-abstract-intro-architect](paper-abstract-intro-architect/SKILL.md)                        |
|    Synthesize title, 5-sentence abstract, and surgical motivation against LLM fragmentation.      |
| -> [paper-math-formulation-writer](paper-math-formulation-writer/SKILL.md)                          |
|    Derive chunk-parallel scans, Cholesky SVD whitening, and Stiefel Riemannian manifold gradients. |
| -> [paper-linear-recurrent-associative-scans](paper-linear-recurrent-associative-scans/SKILL.md)    |
|    Author Blelloch scans, chunk-parallel dual recurrences, and delta-rule associative memory.       |
| -> [paper-attention-mechanisms-taxonomy](paper-attention-mechanisms-taxonomy/SKILL.md)              |
|    Synthesize unified taxonomies, operator contracts, and state equations for 131 attentions.      |
| -> [paper-lowrank-svd-factorization](paper-lowrank-svd-factorization/SKILL.md)                  |
|    Author Eckart-Young proofs, Cholesky whitening pullback, SVD spectra, and 72-engine taxonomy.  |
| -> [paper-fisher-information-spectral](paper-fisher-information-spectral/SKILL.md)              |
|    Formulate Fisher Information Matrix curvature, KFAC Kronecker blocks, and Fisher-weighted SVD.   |
| -> [paper-hessian-eigenspectrum-loss](paper-hessian-eigenspectrum-loss/SKILL.md)                  |
|    Analyze Hessian eigenspectra, Lanczos iteration, spectral condition, and flat minima theory.     |
| -> [paper-riemannian-stiefel-optimizer](paper-riemannian-stiefel-optimizer/SKILL.md)              |
|    Formalize Stiefel manifold Riemannian geometry, projections, retractions, and Adam/SGD optim.   |
| -> [paper-structured-pruning-admm](paper-structured-pruning-admm/SKILL.md)                        |
|    Author ADMM channel/head pruning, 2:4 hardware sparsity, and dynamic token shedding.            |
| -> [paper-tensor-quantization-microscaling](paper-tensor-quantization-microscaling/SKILL.md)        |
|    Author FP8 (e4m3fn/e5m2), FP4/INT4 microscaling (MX), SmoothQuant migration, and error bounds.   |
| -> [paper-knowledge-distillation-healing](paper-knowledge-distillation-healing/SKILL.md)          |
|    Formulate 6-pillar composite KD loss, WSD scheduling, and Banach fixed-point contraction proof.  |
| -> [paper-systems-kernel-architect](paper-systems-kernel-architect/SKILL.md)                      |
|    Document 3-level fallback hierarchy, CUDA float4 vectors, and vLLM/SGLang serving bridges.      |
| -> [paper-cuda-cpp-microarchitecture](paper-cuda-cpp-microarchitecture/SKILL.md)                  |
|    Author 128-bit vector loads, warp shuffles, bank-conflict elimination, and register budgets.     |
| -> [paper-triton-kernel-architect](paper-triton-kernel-architect/SKILL.md)                          |
|    Design unified 2D-tiled Triton JIT kernels without block pointer deprecations for MLSys/arXiv.  |
| -> [paper-kv-cache-eviction-policies](paper-kv-cache-eviction-policies/SKILL.md)                  |
|    Author KV-cache dynamics, H2O, StreamingLLM attention sinks, and 32k Needle-in-a-Haystack.       |
| -> [paper-roofline-hardware-profiler](paper-roofline-hardware-profiler/SKILL.md)                  |
|    Ground performance models, empirical rooflines, arithmetic intensity bounds, and cold caches.   |
| -> [paper-serving-radix-paged-attention](paper-serving-radix-paged-attention/SKILL.md)              |
|    Author vLLM PagedAttention virtual blocks, SGLang RadixAttention prefix trees, and zero-fake.   |
| -> [paper-empirical-evaluator](paper-empirical-evaluator/SKILL.md)                                  |
|    Report 203-module census, RTX 4070 telemetry, roofline bounds, and sequence length crossovers.  |
| -> [paper-multi-seed-statistical-rigor](paper-multi-seed-statistical-rigor/SKILL.md)              |
|    Author multi-seed statistical significance, Welch's t-tests, CI, and cross-seed variance.       |
| -> [paper-kernel-postmortem-forensics](paper-kernel-postmortem-forensics/SKILL.md)                  |
|    Author hardware race autopsies, deadlock analyses, mixed-precision overflows, and negative data.|
| -> [paper-arxiv-submission-packager](paper-arxiv-submission-packager/SKILL.md)                      |
|    Gate camera-ready compilation (0 warnings), audit BibTeX provenance, package .tar.gz for cs.LG.  |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Comprehensive Skills Catalog

### 2.1 Core Engineering & Hardware Kernel Skills

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

### 2.2 Academic Research & arXiv Preprint Authoring Suite

| Skill | Category | Primary Focus & Capabilities |
| :--- | :--- | :--- |
| **[paper-codebase-deep-explorer](paper-codebase-deep-explorer/SKILL.md)** | Codebase Exploration | Systematically explore, index, grep, and cross-reference symbols, registries, kernels, and mathematical definitions across NeuralOps for paper authoring. |
| **[paper-abstract-intro-architect](paper-abstract-intro-architect/SKILL.md)** | Manuscript Authoring | Formulate high-signal titles, 5-sentence structured abstracts, surgical paradigm positioning against LLM fragmentation, and 5-point contribution narratives for arXiv/conference submissions. |
| **[paper-math-formulation-writer](paper-math-formulation-writer/SKILL.md)** | Mathematical Theory | Chunk-parallel linear/associative recurrences, Cholesky covariance whitening, Eckart-Young-Mirsky SVD proofs, Stiefel Riemannian optimization, and 6-pillar distillation objectives. |
| **[paper-linear-recurrent-associative-scans](paper-linear-recurrent-associative-scans/SKILL.md)** | Recurrent & Associative Theory | Parallel prefix scans (Blelloch work-efficiency vs Hillis-Steele), chunk-parallel dual recurrences, delta-rule error-correcting associative memory, and comparative taxonomy (GLA, DeltaNet, RetNet, RWKV-v6). |
| **[paper-attention-mechanisms-taxonomy](paper-attention-mechanisms-taxonomy/SKILL.md)** | Attention Taxonomy | Exhaustive mathematical operator contracts, state recurrence equations, complexity bounds ($\mathcal{O}(N^2 d)$ vs $\mathcal{O}(N d^2)$ vs $\mathcal{O}(N W d)$), and TikZ trees for all 131 attention mechanisms across 14 architectural families. |
| **[paper-lowrank-svd-factorization](paper-lowrank-svd-factorization/SKILL.md)** | Spectral Factorization | Eckart-Young-Mirsky optimality, activation covariance whitening, Cholesky factor pullback, condition number bounds, spectral energy decay, and 72-engine taxonomy. |
| **[paper-tensor-quantization-microscaling](paper-tensor-quantization-microscaling/SKILL.md)** | Quantization & Microscaling | Hardware-grounded FP8 (e4m3fn/e5m2), FP4/INT4 microscaling (OCP MX standards), SmoothQuant outlier migration, in-SRAM Triton fusion, and analytical noise/Frobenius error bounds. |
| **[paper-structured-pruning-admm](paper-structured-pruning-admm/SKILL.md)** | Pruning & Sparsity | ADMM structured channel/head pruning, 2:4 hardware sparsity, dynamic token shedding, and non-convex convergence proofs for research papers. |
| **[paper-fisher-information-spectral](paper-fisher-information-spectral/SKILL.md)** | Information Geometry | Fisher Information Matrix curvature, KFAC Kronecker factorization, empirical activation covariance, natural gradient metrics, Fisher-weighted SVD, and Cholesky pullback. |
| **[paper-riemannian-stiefel-optimizer](paper-riemannian-stiefel-optimizer/SKILL.md)** | Riemannian Optimization | Stiefel manifold Riemannian geometry, closed-form orthogonal tangent projection, QR/Polar/Cayley retractions with Woodbury speedup, Riemannian Adam/SGD algorithms, and spectral condition number invariance proofs. |
| **[paper-hessian-eigenspectrum-loss](paper-hessian-eigenspectrum-loss/SKILL.md)** | Loss Landscape & Hessian | Loss landscape geometry, empirical Hessian and Gauss-Newton formulation, matrix-free Lanczos eigenspectrum extraction, spectral norm bounds, condition number $\kappa(\mathbf{H})$, flat minima PAC-Bayes generalization, SVD curvature relaxation, and multi-pillar KD landscape smoothing. |
| **[paper-knowledge-distillation-healing](paper-knowledge-distillation-healing/SKILL.md)** | Distillation & Healing | Comprehensive mathematical and experimental sections on post-surgery 6-pillar knowledge distillation, Warmup-Stable-Decay (WSD) scheduling, and Banach fixed-point contraction proof. |
| **[paper-systems-kernel-architect](paper-systems-kernel-architect/SKILL.md)** | Systems & Dispatch | 3-Level fallback engine architecture (CUDA C++ -> Triton JIT -> PyTorch), zero-cost dispatch tables, 128-bit vectorization, bank conflict elimination, and vLLM/SGLang serving integration. |
| **[paper-cuda-cpp-microarchitecture](paper-cuda-cpp-microarchitecture/SKILL.md)** | Hardware Microarchitecture | Native CUDA C++ kernels: 128-bit vector loads, warp-level shuffle intrinsics, shared memory bank conflict elimination, and register budgeting for systems papers. |
| **[paper-triton-kernel-architect](paper-triton-kernel-architect/SKILL.md)** | Triton Kernel Architecture | Design, autotune, and verify unified 2D-tiled Triton JIT GPU kernels without block pointer deprecations for MLSys and arXiv papers. |
| **[paper-kv-cache-eviction-policies](paper-kv-cache-eviction-policies/SKILL.md)** | KV-Cache Systems | Systems and algorithmic sections on KV-cache memory dynamics, Heavy Hitter Oracle (H2O), StreamingLLM attention sinks, and long-context (32k) needle retrieval evaluation. |
| **[paper-roofline-hardware-profiler](paper-roofline-hardware-profiler/SKILL.md)** | Roofline & Profiling | Hardware-grounded performance modeling sections, empirical roofline diagrams, arithmetic intensity bounds, and cold-cache microbenchmarking on NVIDIA Ada Lovelace. |
| **[paper-serving-radix-paged-attention](paper-serving-radix-paged-attention/SKILL.md)** | Serving & Inference | Systems and deployment sections on high-throughput serving integration (vLLM PagedAttention, SGLang RadixAttention), KV-cache compression, and zero-fake export compliance. |
| **[paper-empirical-evaluator](paper-empirical-evaluator/SKILL.md)** | Empirical Evaluation | 203-module evaluation matrix (42 PASS_FAST, 161 PASS_SLOW), hardware power/thermal profiles (-63.6% power, -6.7°C), sequence scaling horizons, and L2 cold-cache microbenchmarking on RTX 4070 SM 8.9. |
| **[paper-multi-seed-statistical-rigor](paper-multi-seed-statistical-rigor/SKILL.md)** | Statistical Reproducibility | Dual-tier multi-seed protocol (standard vs stress seeds), Welch's two-sample $t$-tests with Welch-Satterthwaite df, parametric & BCa bootstrapped 95% CIs, standardized effect sizes (Cohen's $d$), zero-divergence audits, and publication boxplots. |
| **[paper-kernel-postmortem-forensics](paper-kernel-postmortem-forensics/SKILL.md)** | Forensics & Autopsies | Hardware race condition diagnoses, shared memory deadlock resolution, mixed-precision overflow contracts, L2 cache eviction cascades, and negative result defenses. |
| **[paper-arxiv-submission-packager](paper-arxiv-submission-packager/SKILL.md)** | Release & Packaging | Pre-generated `.bbl` verification, zero-warning LaTeX compilation gates (0 undefined citations/references, 0 overfull \hbox > 5pt), vector font embedding, and `.tar.gz` packaging for `cs.LG`/`cs.DC`/`cs.AI`. |

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
