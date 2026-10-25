---
name: report-chapter-2-writer
description: "Write Chapter 2 (Background) using AUDIT_01, AUDIT_02, AUDIT_03, and docs/thesis_agents guides"
version: "1.0.0"
tags: ["neuralops", "report", "chapter-2", "background", "taxonomy", "thesis_agents"]
---

# Skill: Chapter 2 (Background) Architectural Taxonomy and Theoretical Foundations

## When to use
Use this skill when authoring, expanding, or verifying Chapter 2 (Theoretical Foundations and Related Literature) of the doctoral dissertation (`academic/thesis/chapters/chapter02_theory_part1.tex` and `chapter02_theory_part2.tex`). It dictates the methodology for presenting the 131-attention taxonomy, the 72-compression paradigm classification, singular value decomposition (SVD) low-rank theory, and accelerator roofline modeling according to the operational guides of Agents 03, 04, 05, and 06.

## Prerequisites
- Direct compliance with `docs/thesis_agents/AGENT_03_CHAPTER_2_SDPA_ROOFLINE.md`.
- Direct compliance with `docs/thesis_agents/AGENT_04_CHAPTER_2_SVD_CHOLESKY.md`.
- Direct compliance with `docs/thesis_agents/AGENT_05_CHAPTER_2_ATTENTION_TAXONOMY.md`.
- Direct compliance with `docs/thesis_agents/AGENT_06_CHAPTER_2_COMPRESSION_PARADIGMS.md`.
- Full compliance with `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md` and `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.
- Target files: `academic/thesis/chapters/chapter02_theory_part1.tex` and `academic/thesis/chapters/chapter02_theory_part2.tex`.
- Exclusion of early buggy runs: cite only calibrated, stable measurements (RTX 4070 SM 8.9, 504 GB/s theoretical bandwidth, 46.6 TFLOPS FP32 peak compute).

## Canonical Audit Files Reference
Chapter 2 directly references and grounds all theoretical taxonomies in the six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced in Section 2.2 for knowledge distillation background and cross-entropy teacher-student calibration theory.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced in Section 2.2 and 2.3 for the 72 compression paradigms, Eckart-Young-Mirsky theorem formulations, Cholesky activation whitening, and singular value decay spectra.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced in Section 2.1 and 2.4 for the complete 131 attention mechanism taxonomy, 7 mathematical families, and the Williams hardware roofline model (183.1 FLOP/Byte).
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced in Section 2.1 for asymptotic algorithmic complexity comparisons ($\mathcal{O}(N^2)$ vs $\mathcal{O}(N)$ vs $\mathcal{O}(N \log N)$).
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced in Section 2.1 for invariant registry counts (131/72) and testbed architectural constraints.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Sourced for mapping Chapter 2 drafting responsibilities across Agents 03 to 06 and coordinating deliverable boundaries.

## Canonical Repository Invariants
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Detailed Section-by-Section Drafting Architecture

### Section 2.1 — Standard Attention and Roofline Modeling (Agent 03)
Target: `academic/thesis/chapters/chapter02_theory_part1.tex` (`sec:standard_attention`, `sec:roofline`).
- **Mathematical Variance Control**: Prove analytically why the $1/\sqrt{d_k}$ scaling factor is mandatory for dot-product attention when inputs are i.i.d. with mean 0 and unit variance:
  $$\operatorname{Var}\left(\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}\right) = \frac{1}{d_k} \operatorname{Var}(S) = \frac{d_k}{d_k} = 1$$
- **Causal Mask Formulation**: Express strictly without `\lr{}` inside the equation:
  $$\mathbf{M}_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$
- **RoPE Geometry**: Formulate Rotary Position Embedding via block-diagonal orthogonal rotation matrices:
  $$\mathbf{R}_{\Theta, m}^d = \operatorname{diag}\left(\mathbf{R}_{\theta_1, m}, \dots, \mathbf{R}_{\theta_{d/2}, m}\right), \quad \mathbf{R}_{\theta, m} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$
- **MHA vs GQA vs MQA**: Contrast multi-head, grouped-query, and multi-query memory bandwidth demands.
- **Roofline Formulation**: Ada Lovelace RTX 4070 (SM 8.9) with $504\text{ GB/s}$ bandwidth and $46.6\text{ TFLOP/s}$ FP32 peak throughput. Standard attention is firmly memory-bound ($I \le 183.1\text{ FLOP/B} < I_{\text{knee}} = 276.0\text{ FLOP/B}$).

### Section 2.2 — Mathematical SVD and Whitening (Agent 04)
Target: `academic/thesis/chapters/chapter02_theory_part2.tex` (`sec:svd_foundations`, `sec:cholesky_whitening`).
- **Eckart-Young-Mirsky Theorem**:
  For any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ with singular value decomposition $\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$, the optimal rank-$k$ approximation under the Frobenius norm satisfies:
  $$\min_{\operatorname{rank}(\mathbf{A}_k) \le k} \|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^{\min(m, n)} \sigma_i^2}$$
- **Activation Covariance Metric**:
  Given calibration input activations $\mathbf{X} \in \mathbb{R}^{T \times d}$, the empirical sample covariance matrix $\boldsymbol{\Sigma}_{XX} \in \mathbb{R}^{d \times d}$ is computed as:
  $$\boldsymbol{\Sigma}_{XX} = \frac{1}{T} \mathbf{X}^\top \mathbf{X} + \epsilon \mathbf{I}_d$$
  where $\epsilon = 10^{-6}$ ensures strict positive definiteness.
- **Cholesky Whitening Decomposition**:
  Compute the lower triangular Cholesky factor $\mathbf{L}$ such that $\boldsymbol{\Sigma}_{XX} = \mathbf{L} \mathbf{L}^\top$. The preconditioned weight matrix is:
  $$\widetilde{\mathbf{W}} = \mathbf{W} \mathbf{L}$$
- **Singular Value Decomposition on Whitened Weights**:
  $$\widetilde{\mathbf{W}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top \implies \mathbf{W} \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top \mathbf{L}^{-1}$$
  This minimizes the expected output reconstruction error $\mathbb{E}_{\mathbf{x} \sim \mathcal{D}}[\|\mathbf{W}\mathbf{x} - \widehat{\mathbf{W}}\mathbf{x}\|_2^2]$.
- **LoRA Parameterization Compatibility**:
  $$\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}, \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$$

### Section 2.3 — Attention Mechanism Taxonomy (131 Modules) (Agent 05)
Target: `academic/thesis/chapters/chapter02_theory_part2.tex` (`sec:attention_taxonomy`).
- Classify all 131 registered attention architectures into the 7 canonical mathematical families:
  1. **Exact Softmax & Flash Tiling (18 modules)**: Standard scaled dot-product, FlashAttention-1/2/3, FlashDecoding. Formulation: $\operatorname{Attn}(Q, K, V) = \operatorname{softmax}(QK^\top/\sqrt{d_k})V$.
  2. **Recurrent & State-Space Models (SSM) (24 modules)**: Mamba, RWKV, RetNet, S4, Linear RNNs. Recurrence: $h_t = A_t h_{t-1} + B_t x_t, y_t = C_t h_t + D_t x_t$.
  3. **Kernelized & Random Feature Maps (22 modules)**: Performer, Cosformer, Random Subspace Attention. Kernel approximation: $\phi(Q)\phi(K)^\top V$.
  4. **Low-Rank & Latent KV Projections (19 modules)**: Linformer, Multi-Head Latent Attention (MLA), LoRA Attention. Latent compression: $K' = K E, V' = V F$.
  5. **Sparse & Adaptive Block Patterns (21 modules)**: BigBird, Longformer, Sparse Sinkhorn, Splash Attention. Structured sparsity mask $\mathbf{S} \odot (QK^\top)$.
  6. **Differential & Multi-Scale Attention (12 modules)**: Differential Transformer, Multi-Scale Retention. Subtraction of cancelation maps $\operatorname{softmax}(A_1) - \lambda \operatorname{softmax}(A_2)$.
  7. **Hybrid & Segmented Schemes (15 modules)**: Griffin SSD, Hymba TransX, Mamba-Attention Hybrids. Multi-stage sliding local windows and global linear recurrent branches.
- **Table 2.1**: Complete 131-module taxonomy table without truncation using `longtable`.
- **Table 2.2**: Macro averages across the 7 families (Speedup, Latency, Peak VRAM, FLOP reduction).

### Section 2.4 — Model Compression Paradigms (72 Engines) (Agent 06)
Target: `academic/thesis/chapters/chapter02_theory_part2.tex` (`sec:compression_paradigms`).
- Categorize the 72 post-training compression engines into 5 paradigms:
  1. **Low-Rank Matrix Factorization (24 engines)**: SVD, HNC-SVD, FW-SVD, Truncated SVD, Randomized SVD, CUR Matrix Decomposition.
  2. **Structured & Unstructured Pruning (16 engines)**: Magnitude Pruning, Wanda Pruning, SparseGPT, Movement Pruning, Semi-structured 2:4 sparsity.
  3. **Quantization Engines (18 engines)**: Uniform INT8/INT4, AWQ (Activation-aware Weight Quantization), GPTQ, KMeans Quantization, SmoothQuant.
  4. **Knowledge Distillation Bridges (8 engines)**: Logit KL Divergence, Intermediate Feature MSE, Spectral Attention Alignment, Contrastive Distillation.
  5. **Hybrid Low-Rank Quantization (6 engines)**: Q-SVD, SpinQuant Factorization, Low-Rank Plus Sparse (L+S) Decomposition.
- **Table 2.3**: Full 72-row compression engine taxonomy table with formal compression ratios and calibration requirements.

---

## Persian Typography and XePersian Rules
Per `docs/thesis_agents/PERSIAN_WRITING_RULES.md`:
1. **Never use `\lr{}` inside math environments**: Use `\operatorname{Var}`, `\operatorname{Softmax}`, or `\text{...}`.
2. **Never wrap `\item` in `\lr{}`**: Correct: `\item \lr{FlashAttention-2}`.
3. **Mandatory ZWNJ (نیم‌فاصله)**: Use `U+200C` in compound terms (e.g. «مدل‌های»، «لایه‌ای»، «بهینه‌سازی»).
4. **Digit Format**: Persian digits (۱۳۱، ۷۲) in running narrative, English digits (131, 72) in tables and equations.
5. **No early buggy runs**: Never cite flawed preliminary runs (<35% accuracy, uncalibrated checkpoints).

---

## LaTeX Integration Example

```latex
\begin{table}[H]
\centering
\small
\caption{خلاصه تله‌متری کلان در میان ۷ خانواده مکانیزم توجه \cite{audit03}}
\label{tab:family_macro_averages}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Attention Family} & \textbf{Count} & \textbf{Mean Latency (ms)} & \textbf{Mean Speedup} & \textbf{Mean VRAM (MB)} & \textbf{Source} \\
\midrule
Exact Softmax \& Flash & 18 & 0.482 & 2.84$\times$ & 19.4 & \texttt{AUDIT\_03} \\
Recurrent \& SSM & 24 & 0.395 & 3.42$\times$ & 14.2 & \texttt{AUDIT\_03} \\
Kernelized \& Linear & 22 & 0.412 & 3.18$\times$ & 15.8 & \texttt{AUDIT\_03} \\
Low-Rank \& Latent & 19 & 0.441 & 2.95$\times$ & 16.1 & \texttt{AUDIT\_03} \\
Sparse \& Adaptive & 21 & 0.510 & 2.65$\times$ & 18.2 & \texttt{AUDIT\_03} \\
Differential \& Multi-Scale & 12 & 0.465 & 2.88$\times$ & 17.5 & \texttt{AUDIT\_03} \\
Hybrid \& Segmented & 15 & 0.428 & 3.10$\times$ & 15.0 & \texttt{AUDIT\_03} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Complete Verification Protocol

1. Verify that `academic/thesis/chapters/chapter02_theory_part1.tex` and `chapter02_theory_part2.tex` contain all sections.
2. Check for zero table truncation: 131 attentions in Table 2.1 and 72 compressors in Table 2.3.
3. Verify absence of banned XePersian patterns:
   ```bash
   grep -n "\\\\lr{" academic/thesis/chapters/chapter02_theory_part*.tex | grep -E "begin\{equation\}|\$|item"
   ```
4. Verify that early buggy run numbers (<50%) are omitted from the main thesis text.
5. Confirm that arithmetic intensity for standard attention is correctly documented as 183.1 FLOP/Byte.
6. Verify that all 7 attention families and 5 compression paradigms have formal mathematical definitions.

---

## Common Pitfalls to Avoid
- Inserting `\lr{}` inside equations or around `\item` commands.
- Truncating 131-row or 72-row taxonomy tables with ellipses or partial subsets.
- Including early uncalibrated checkpoint numbers instead of canonical verified results.
- Forgetting ZWNJ (نیم‌فاصله) in compound Persian technical phrases.
- Confusing theoretical memory bandwidth (504 GB/s) with measured sustained throughput.
- Omitting Cholesky whitening inversion proofs when presenting low-rank factorization.
