<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction — motivation, problem, contributions, organization
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 – 2 pages (800 – 1200 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The four attention projections $W_Q, W_K, W_V, W_O$ in a standard transformer 
layer account for the majority of attention parameters. For DistilBERT 
($d = 768$), each projection contains $768 \times 768 = 589{,}824$ parameters; 
the four together contain ~2.36M parameters per layer.

**Low-rank factorization** replaces $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
with the product $U V$ where $U \in \mathbb{R}^{d_\text{out} \times r}$ and 
$V \in \mathbb{R}^{r \times d_\text{in}}$. Parameter count drops from 
$d_\text{in} d_\text{out}$ to $r (d_\text{in} + d_\text{out})$. For 
$768 \times 768$ at $r = 128$, this is a **3× reduction** in attention 
parameters.

The **rank choice** is the central design decision. Truncated SVD gives the 
best rank-$r$ approximation in Frobenius norm (Eckart–Young), but the theorem 
says nothing about what $r$ should be. Existing approaches fall into three 
categories:

1. **Fixed global rank.** Apply the same $r$ to every projection. Simple, but 
   ignores that projections have very different **spectral decay rates**. 
   **Figure 1.1** shows singular value spectra of two DistilBERT projections: 
   one decays rapidly (~20 effective rank), another decays slowly (~200).

2. **Validation-driven escalation.** Start low, increase rank if Frobenius 
   error exceeds a threshold. Requires multiple SVDs per layer.

3. **Activation-aware rank.** Weight the SVD input space by empirical 
   activation statistics (Fisher diagonal, covariance). Requires calibration 
   data.

We propose a **fourth approach** that is self-contained: use the singular 
values themselves to determine rank. Specifically, threshold **cumulative 
squared energy**:

$$
\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_\text{in}, d_\text{out})} \sigma_j^2}, \quad
r^*(\tau) = \min\{r : \mathcal{E}(r) \geq \tau\}
\tag{1.1}
$$

with default $\tau = 0.92$. Fast-decaying projections get 
small $r$, slow-decaying ones get large $r$, all determined automatically.

## 1.2 Problem Statement

Given a pretrained attention module with projections 
$W_P \in \mathbb{R}^{d_\text{out}^{(P)} \times d_\text{in}^{(P)}}$ for 
$P \in \{Q, K, V, O\}$, and an energy threshold $\tau \in (0, 1)$, we seek 
per-projection factorizations $W_P \approx U_P V_P$ such that:

1. **Energy preservation:** $\mathcal{E}_P(r_P) \geq \tau$ for each $P$.
2. **Parameter reduction:** $r_P (d_\text{in}^{(P)} + d_\text{out}^{(P)}) < d_\text{in}^{(P)} d_\text{out}^{(P)}$.
3. **No base retraining:** $U_P, V_P$ are frozen; only adapters train.
4. **No calibration data:** rank selected purely from the weight's spectrum.
5. **Multi-architecture support:** flat, nested, causal, T5, GQA, RoPE.
6. **Initial forward fidelity:** with adapters zero-initialized, the initial 
   forward pass matches the truncation approximation exactly.

## 1.3 Our Approach

**Adaptive SVD Attention** combines four design choices:

1. **Cumulative energy rank.** For each projection, compute SVD, then select 
   the smallest rank $r$ where cumulative squared energy $\mathcal{E}(r) \geq \tau$.

2. **Symmetric factor distribution.** Split $\Sigma^{1/2}$ between factors:

$$
U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{1.2}
$$

This keeps $\|U_r\|_2 = \|V_r\|_2 = \sqrt{\sigma_\text{max}}$, avoiding FP16 overflow.

3. **Residual adapters.** Add zero-initialized $B A$ with $B = 0$ at init. The 
   adapter contributes zero to the initial forward, giving a stable starting 
   point for recovery.

4. **Universal architecture support.** The injection handles flat (DistilBERT), 
   nested (RoBERTa/BERT), fused Conv1D (GPT-2), GQA (Qwen), RoPE (LLaMA), 
   and T5 relative attention bias.

## 1.4 Contributions

- **Energy-thresholded rank selection.** A self-contained, calibration-free 
  criterion for per-layer rank based on cumulative singular value energy. 
  Section 4.2 formalizes this.
- **Symmetric SVD + residual adapter.** A factorization that keeps both 
  factors balanced in magnitude and adds zero-initialized residuals for 
  recovery. Section 4.3–4.4.
- **Broad architecture support.** The method works on flat, nested, and causal 
  attention, with optional T5 bias, GQA, and RoPE. Section 5.
- **Systematic evaluation.** We benchmark on 3 models on SST-2 with ablations 
  over $\tau$, residual rank, and recovery steps. Section 7.
- **Honest reporting.** We report which models benefit most, which have 
  limitations, and what we tried that failed (Appendix D).
- **Full reproducibility.** Code, checkpoints, and evaluation scripts released 
  under MIT.

## 1.5 Paper Organization

Section 2 reviews related work in low-rank compression, activation-aware 
methods, and parameter-efficient fine-tuning. Section 3 provides background 
on SVD, energy thresholds, and LoRA. Section 4 presents the method. Section 5 
describes integration with real transformer architectures. Section 6 details 
experimental setup. Section 7 reports results. Section 8 analyzes the 
per-layer rank distribution and ablation contributions. Section 9 discusses 
practical usage. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $W_P$ | Attention projection weight for $P \in \{Q, K, V, O\}$ | §1.1 |
| $U, V$ | Low-rank factors | §1.1 |
| $r$ | Per-projection rank | §1.1 |
| $\sigma_i$ | $i$-th singular value | §1.1 |
| $\mathcal{E}(r)$ | Cumulative squared energy at rank $r$ | §1.1 |
| $\tau$ | Energy threshold | §1.1 |
| $A, B$ | Residual adapter factors | §1.3 |
| $r_L$ | Residual adapter rank | §1.3 |
| $d_h$ | Per-head dimension | §1.4 |
| $H$ | Query head count | §1.4 |
| $H_{kv}$ | KV head count (GQA) | §1.4 |
| $N_l$ | Number of layers | §1.4 |
