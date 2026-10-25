---
name: report-chapter-3-writer
description: "Write Chapter 3 (Methodology) using AUDIT_01, AUDIT_02, AUDIT_05, and docs/thesis_agents guides"
version: "1.0.0"
tags: ["neuralops", "report", "chapter-3", "methodology", "theorems", "architecture", "thesis_agents"]
---

# Skill: Chapter 3 (Methodology) Formal Formulation and System Architecture

## When to use
Use this skill when drafting, verifying, or typesetting Chapter 3 (Methodology and System Architecture) of the doctoral dissertation (`academic/thesis/chapters/chapter03_methodology_part1.tex`, `part2.tex`, `part3a.tex`, and `part3b.tex`). It dictates the methodology for specifying the 7-layer NeuralOps software architecture, the 3-level kernel fallback dispatch contract, the 4-stage surgical transplant protocol, all 10 formal mathematical theorems with complete proofs, and experimental quality gates according to Agents 07 through 11.

## Prerequisites
- Direct compliance with `docs/thesis_agents/AGENT_07_CHAPTER_3_ARCHITECTURE_SURGERY.md`.
- Direct compliance with `docs/thesis_agents/AGENT_08_CHAPTER_3_DISPATCH_CORRECTNESS.md`.
- Direct compliance with `docs/thesis_agents/AGENT_09_CHAPTER_3_EXPERIMENTAL_SETUP.md`.
- Direct compliance with `docs/thesis_agents/AGENT_10_CHAPTER_3_THEOREMS_1_TO_5.md`.
- Direct compliance with `docs/thesis_agents/AGENT_11_CHAPTER_3_THEOREMS_6_TO_10.md`.
- Full compliance with `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md` and `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.
- Target files: `academic/thesis/chapters/chapter03_methodology_part1.tex`, `chapter03_methodology_part2.tex`, `chapter03_methodology_part3a.tex`, and `chapter03_methodology_part3b.tex`.

## Canonical Audit Files Reference
Chapter 3 derives its theoretical formulations, software architecture layers, and proof derivations directly from the six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced in Section 3.3 and 3.4 for the 4-stage surgical protocol, closed-form Ridge calibration formulations, and Theorem 10 (Distillation Convergence Bounds).
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced in Section 3.4 for Theorems 1, 2, and 3 covering low-rank error bounds, Cholesky covariance whitening, and singular value truncation optimality.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced in Section 3.2 and 3.4 for 3-level kernel fallback timing and Theorems 4 and 5 covering roofline saturation and memory bandwidth limits.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced in Section 3.2 for kernel dispatch caching and state cache memory contracts.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced in Section 3.1 and 3.5 for the 7-layer architectural decomposition, testbed disclosures, and forensic Git commit hashes.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Sourced for mapping Chapter 3 writing responsibilities across Agents 07 to 11 and coordinating formal proof sections.

## Canonical Repository Invariants
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Detailed Section-by-Section Drafting Architecture

### Section 3.1 — Platform Architecture & In-Place Surgery (Agent 07)
Target: `academic/thesis/chapters/chapter03_methodology_part1.tex` (`sec:platform_architecture`, `sec:surgical_protocol`).
- **7-Layer Decoupled Architecture**: Detail L1 (Telemetry) through L7 (Application/CLI). Explain PEP 562 dynamic lazy loading (`__getattr__`) to prevent startup latency penalties across 203 modules.
- **4-Stage Surgical Protocol**:
  1. *Stage 1 (Architecture Inspection)*: Recursive traversal of module graph, tensor dimension extraction, and hook attachment.
  2. *Stage 2 (Closed-Form Ridge Calibration)*:
     $$\min_{\mathbf{W}_p} \|\mathbf{X}\mathbf{W}_{\text{orig}} - \mathbf{X}\mathbf{W}_p\|_F^2 + \lambda \|\mathbf{W}_p\|_F^2 \implies \mathbf{W}_p^* = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top (\mathbf{X}\mathbf{W}_{\text{orig}})$$
  3. *Stage 3 (Zero-Shot In-Place Transplant)*: Dynamic submodule replacement via PyTorch `setattr()`, preserving forward hooks and state dictionaries.
  4. *Stage 4 (Distillation Healing)*: 30 optimization steps using `heal_replaced_model()` with composite loss function.

### Section 3.2 — Multi-Tier Kernel Dispatch & Sanity Gates (Agent 08)
Target: `academic/thesis/chapters/chapter03_methodology_part2.tex` (`sec:dispatch_stack`, `sec:numerical_gates`).
- **3-Level Fallback Hierarchy**:
  - *Level 1 (Native CUDA C++ / HIP)*: Hand-tuned custom kernels compiled via `torch.utils.cpp_extension` for SM 8.9.
  - *Level 2 (Triton JIT)*: Autotuned Triton kernels supporting FP16, BF16, and FP32 with block tiling.
  - *Level 3 (Vectorized PyTorch Reference)*: Universal fallback for CPU, testing, and debugging.
- **Zero-Overhead Dispatch Mechanism**: Cached routing using `@functools.lru_cache(maxsize=128)` achieving dispatch latency under $1.5\mu\text{s}$.
- **Numerical Sanity Gate**: Strict numerical tolerance requirements:
  $$\max_{i,j} |y_{ij}^{\text{kernel}} - y_{ij}^{\text{ref}}| < 10^{-2}\text{ (FP16/BF16)}, \quad < 10^{-4}\text{ (FP32)}$$
  Finite difference autograd gradient checks must satisfy $\text{atol} = 10^{-3}, \epsilon = 10^{-5}$.

### Section 3.3 — Experimental Setup, Models & Quality Gate (Agent 09)
Target: `academic/thesis/chapters/chapter03_methodology_part2.tex` (`sec:experimental_setup`, `sec:quality_gates`).
- **Hardware Testbed**: NVIDIA GeForce RTX 4070 (SM 8.9 Ada Lovelace, 5888 CUDA cores, 12 GB GDDR6 @ 504 GB/s, PCIe 4.0 x16).
- **Target Architectures**: DistilBERT-base-uncased (66M), RoBERTa-base (125M), GPT-2-small (124M).
- **Evaluation Tasks**: GLUE Benchmark (SST-2, MNLI, QQP, MRPC, QNLI) and WikiText-2 / PTB language modeling.
- **Strict 56% Accuracy Gate**: A transplanted module must achieve $\ge 56.0\%$ on SST-2 to be classified as viable (PASS). Scores below 56% indicate catastrophic structural collapse.

### Section 3.4 — Theorems 1 to 5 (Agent 10)
Target: `academic/thesis/chapters/chapter03_methodology_part3a.tex` (`sec:theorems_1_to_5`).
Include full formal statements and complete LaTeX `proof` environments for:
- **Theorem 1 (Cholesky Activation Whitening Optimality)**:
  Optimal rank-$r$ weight factor recovery minimizing expected output reconstruction error $\mathbb{E}_{\mathbf{x} \sim \mathcal{D}}[\|\mathbf{W}\mathbf{x} - \widehat{\mathbf{W}}\mathbf{x}\|_2^2]$ under $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$.
- **Theorem 2 (Streaming ASVD Covariance Accumulation Error Bounds)**:
  Cumulative covariance discrepancy under chunked streaming accumulation $\boldsymbol{\Sigma}^{(k)} = \alpha \boldsymbol{\Sigma}^{(k-1)} + (1-\alpha) \mathbf{X}_k^\top \mathbf{X}_k / B$, proving bounded spectral norm perturbation $\|\boldsymbol{\Sigma} - \boldsymbol{\Sigma}^{(K)}\|_2 \le \mathcal{O}(e^{-\gamma K})$.
- **Theorem 3 (FP8 Quantization Noise Bounds Under Low-Rank Projections)**:
  Analytical signal-to-quantization-noise ratio (SQNR) preserving dynamic range under E4M3 vs E5M2 formats for low-rank factored factors $\mathbf{U}_r \boldsymbol{\Sigma}_r$ and $\mathbf{V}_r^\top \mathbf{L}^{-1}$.
- **Theorem 4 (Dynamic Sparsity Allocation (DSA) Attention Alignment)**:
  Theoretical convergence guarantee that thresholded dynamic attention masks $\mathbf{M}_{\text{sparse}} = \mathbb{I}(|\mathbf{S}| > \tau)$ retain $\ge (1 - \epsilon)$ of total attention mass under sub-exponential tail assumptions.
- **Theorem 5 (REAP Unbiased Gradient Estimator in Attention Surgery)**:
  Proof that the Relative Error Approximation Protocol (REAP) gradient estimator satisfies $\mathbb{E}[\nabla_{\mathbf{W}_p} \mathcal{L}_{\text{REAP}}] = \nabla_{\mathbf{W}_p} \mathbb{E}[\mathcal{L}]$, guaranteeing unbiased stochastic optimization during healing.

### Section 3.5 — Theorems 6 to 10 (Agent 11)
Target: `academic/thesis/chapters/chapter03_methodology_part3b.tex` (`sec:theorems_6_to_10`).
Include full formal statements and complete LaTeX `proof` environments for:
- **Theorem 6 (Rank Collapse in Extreme Attention Linearization)**:
  Proof that naive first-order Taylor expansion of softmax attention causes rank collapse where $\operatorname{rank}(\operatorname{Attn}(Q, K, V)) \le \min(d, r_Q, r_K)$, yielding degenerative sequence representations when $r < d/4$.
- **Theorem 7 (Softmax Simplex Preservation Under Taylor Expansions)**:
  Necessary and sufficient constraints on normalization factors to preserve probability simplex non-negativity and row-sum unity $\sum_j \mathbf{A}_{ij} = 1$ in kernelized linear attention.
- **Theorem 8 (HNC-SVD Optimality and Energy Concentration)**:
  Proof that Hierarchical Nullspace Calibration (HNC) concentrates $> 92\%$ of spectral residual energy into the orthogonal complement of the uncalibrated weight subspace.
- **Theorem 9 (Householder Reflector Stability for Non-Square Projections)**:
  Proof that Householder orthonormal basis transformations $\mathbf{H} = \mathbf{I} - 2\mathbf{v}\mathbf{v}^\top$ remain numerically stable with condition number $\kappa(\mathbf{H}) = 1.0$ under arbitrary low-rank dimension reductions.
- **Theorem 10 (6-Pillar Distillation Contraction Mapping and Convergence Rate)**:
  Proof that the composite 6-pillar knowledge distillation loss forms a strongly convex contraction operator with convergence rate $\|\mathbf{W}^{(t)} - \mathbf{W}^*\|_F \le \rho^t \|\mathbf{W}^{(0)} - \mathbf{W}^*\|_F$ where $\rho = \max_i |1 - \eta \lambda_i(H)| < 1$.

---

## Persian Typography and XePersian Rules
Per `docs/thesis_agents/PERSIAN_WRITING_RULES.md`:
1. **No `\lr{}` in Math**: Write `\min_{\mathbf{W}_p}`, not `\min_{\lr{W}_p}`.
2. **No `\lr{}` around `\item`**: Correct: `\item \lr{PyTorch Reference}`.
3. **Mandatory ZWNJ (نیم‌فاصله)**: Use `U+200C` in technical terms (e.g. «پشتهٔ گسیل»، «عملگرهای»، «مقداردهی اولیه»).
4. **Digit Format**: Persian digits (۱۳۱، ۷۲، ۵۶٪) in narrative Persian text, English digits (56.0%, 1.2 us) in tables and math.
5. **Data Filter**: Do not cite flawed preliminary runs (<35% accuracy, uncalibrated checkpoints).

---

## LaTeX Integration Example

```latex
\begin{theorem}[بهینگی سفیدسازی کوواریانس چولسکی \cite{audit02}]
\label{thm:cholesky_optimality}
فرض کنید $\mathbf{X} \in \mathbb{R}^{T \times d}$ ماتریس فعال‌سازی کالیبراسیون با کوواریانس $\boldsymbol{\Sigma} = \mathbf{X}^\top \mathbf{X} / T = \mathbf{L}\mathbf{L}^\top$ باشد. برای هر ماتریس تقریب کم‌مرتبه $\widehat{\mathbf{W}} = \mathbf{A}\mathbf{B}$ با رتبه $r \le k$، خطای بازسازی وزنی زیر نرم فربنیوس حداقل می‌شود اگر و تنها اگر فاکتورها از طریق تجزیه مقادیر منفرد ماتریس سفیدشده $\widetilde{\mathbf{W}} = \mathbf{W}\mathbf{L}$ حاصل شوند:
\begin{equation}
\min_{\operatorname{rank}(\widehat{\mathbf{W}}) \le r} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\left[ \|\mathbf{W}\mathbf{x} - \widehat{\mathbf{W}}\mathbf{x}\|_2^2 \right] = \sum_{i=r+1}^d \sigma_i^2(\mathbf{W}\mathbf{L})
\end{equation}
\end{theorem}
\begin{proof}
با تعریف نرم القایی، $\mathbb{E}[\|\mathbf{W}\mathbf{x} - \widehat{\mathbf{W}}\mathbf{x}\|_2^2] = \operatorname{tr}((\mathbf{W} - \widehat{\mathbf{W}})\boldsymbol{\Sigma}(\mathbf{W} - \widehat{\mathbf{W}})^\top) = \|(\mathbf{W} - \widehat{\mathbf{W}})\mathbf{L}\|_F^2$. با تغییر متغیر $\widetilde{\mathbf{W}} = \mathbf{W}\mathbf{L}$ و $\widetilde{\mathbf{M}} = \widehat{\mathbf{W}}\mathbf{L}$، مسئله به حداقل‌سازی $\|\widetilde{\mathbf{W}} - \widetilde{\mathbf{M}}\|_F^2$ با قید $\operatorname{rank}(\widetilde{\mathbf{M}}) \le r$ تبدیل می‌شود. بر اساس قضیه اکرت-یانگ-میرسکی، جواب بهینه از تجزیه مقادیر منفرد بریده‌شده $\widetilde{\mathbf{W}}$ به دست می‌آید. بنابراین $\widehat{\mathbf{W}}^* = \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top \mathbf{L}^{-1}$ بهینه‌ساز یکتا است.
\end{proof}
```

---

## Verification Protocol
1. Confirm all 4 Chapter 3 files exist and are populated:
   - `academic/thesis/chapters/chapter03_methodology_part1.tex`
   - `academic/thesis/chapters/chapter03_methodology_part2.tex`
   - `academic/thesis/chapters/chapter03_methodology_part3a.tex`
   - `academic/thesis/chapters/chapter03_methodology_part3b.tex`
2. Verify all 10 theorems contain explicit formal statements and complete proof blocks:
   ```bash
   grep -c "\\begin{theorem}" academic/thesis/chapters/chapter03_methodology_part*.tex
   grep -c "\\begin{proof}" academic/thesis/chapters/chapter03_methodology_part*.tex
   ```
3. Verify that the 3-level fallback hierarchy is documented with Level 1 (CUDA C++), Level 2 (Triton JIT), and Level 3 (Vectorized PyTorch).
4. Verify that the closed-form Ridge regression solution is accurately stated with the $\lambda \mathbf{I}$ regularizer.
5. Check that all mathematical symbols conform to AMS-LaTeX and do not contain `\lr{}` tags inside math environments.
6. Verify that all 10 theorems cite their respective empirical provenance documents (`AUDIT_01`, `AUDIT_02`, `AUDIT_03`).

---

## Common Pitfalls to Avoid
- Omitting proof blocks for any of the 10 formal theorems.
- Using `\lr{}` within mathematical formulations or theorem statements.
- Neglecting to mention the 56% accuracy threshold quality gate.
- Documenting only 2 levels of kernel fallback instead of the canonical 3-level contract.
- Ignoring PEP 562 dynamic lazy loading when discussing module import architecture.
- Using informal phrases such as "proof is omitted" or "left as an exercise to the reader".
- Overlooking the condition number preservation proof in Householder transformations.
