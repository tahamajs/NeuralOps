<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The parameter budget of a transformer language model is dominated by two 
components: the feed-forward network (FFN) and the self-attention mechanism. 
In DistilBERT, attention projections account for approximately **70%** of the 
parameters in the transformer block. For GPT-2, this fraction rises to 
**75%**. Reducing attention parameter cost is therefore a high-leverage 
compression strategy.

Standard multi-head self-attention (MHSA) stores four dense matrices per 
layer:

$$
W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}
\tag{1.1}
$$

totaling $4d^2$ parameters. For DistilBERT ($d = 768$), this is $4 \times 589{,}824 = 2{,}359{,}296$ 
parameters per layer, or 14.2M across 6 layers — a substantial share of the 
66.9M total.

**Low-rank factorization** is a natural primitive for reducing this cost. 
For each $W$, one replaces it with a product:

$$
W \approx U V, \qquad
U \in \mathbb{R}^{d \times r}, \quad V \in \mathbb{R}^{r \times d}, \quad r \ll d
\tag{1.2}
$$

The parameter count drops from $d^2$ to $r(d + d) = 2rd$. The **reduction 
factor** is $d / 2r$. For $d = 768$ and $r = 64$, this is **6×**. Classical 
SVD (Eckart–Young, 1936) gives the optimal rank-$r$ approximation in 
Frobenius norm.

**The rank problem.** Classical SVD requires an a-priori choice of $r$, which 
is either:

- **Too small** — the approximation error is high, and downstream accuracy 
  drops.
- **Too large** — the parameter reduction is minimal, defeating the purpose.

In practice, different layers have different intrinsic ranks. Attention 
projections in early layers of DistilBERT tend to have rapidly-decaying 
spectra (effective rank ~30), while projections in middle layers decay 
slowly (effective rank ~100). A single global $r$ is suboptimal for both.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_l$ layers and a target 
energy threshold $\eta \in (0, 1)$, we seek a compressed model $\mathcal{M}'$ 
such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.3}
$$

with the following constraints:

1. **Per-layer adaptive rank.** Different layers may have different ranks 
   $r_\ell$ chosen automatically.
2. **Energy preservation.** For each factorized layer, 
   $\sum_{i \leq r_\ell} \sigma_i^2 \geq \eta \sum_i \sigma_i^2$.
3. **Parameter reduction.** $r_\ell (d + d) < d^2$ for each layer.
4. **Multi-architecture support.** Must work on flat (DistilBERT), nested 
   (RoBERTa), causal (GPT-2), and GQA (Qwen) transformers.
5. **No base retraining.** Only LoRA adapters may be trained.

## 1.3 Our Approach: MinimalNOVA

**MinimalNOVA** is a drop-in replacement for the four attention projections 
of a transformer layer. It has three key components:

1. **SVD factorization.** Each $W$ is factorized as $U V$ via truncated SVD, 
   with $U \in \mathbb{R}^{d_\text{out} \times r_\ell}$ and 
   $V \in \mathbb{R}^{r_\ell \times d_\text{in}}$.

2. **Energy-adaptive rank.** Instead of fixing $r$, we choose:

   $$
   r_\ell = \min\!\left(r_\text{max}, \left|\left\{i : \sum_{j \leq i} \sigma_j^2 < \eta \sum_j \sigma_j^2\right\}\right| + 1\right)
   \tag{1.4}
   $$

   This selects the smallest rank that captures at least $\eta$ fraction of 
   the total spectral energy, capped at $r_\text{max}$.

3. **LoRA recovery.** After factorization, the base factors $U, V$ are 
   frozen and LoRA adapters $A, B$ are added:

   $$
   y = (x V^\top) U^\top + (x A^\top) B^\top \cdot s + b
   \tag{1.5}
   $$

   with $B = 0$ at initialization, ensuring the factorized output equals 
   the SVD approximation at step 0.

## 1.4 Contributions

We make the following contributions:

- **Energy-adaptive rank selection.** We propose a spectral-energy criterion 
  that automatically selects a different rank per layer, avoiding the 
  fixed-rank trade-off.
- **A multi-architecture implementation.** We handle four attention families: 
  flat (DistilBERT), nested (RoBERTa), causal with Conv1D (GPT-2), and GQA 
  with RoPE (Qwen3.5). The nested-encoder bypass and Conv1D transposition 
  are critical for correctness.
- **A production-hardened pipeline.** Strict Frobenius validation, safe 
  fallback to original layers, and detailed per-layer statistics.
- **A systematic evaluation.** Benchmark on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over rank cap $r_\text{max}$, energy threshold 
  $\eta$, and LoRA rank.
- **Full reproducibility.** Code, checkpoints, and scripts released under 
  Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, spectral 
energy, and LoRA. Section 4 presents MinimalNOVA. Section 5 describes 
integration with four architectures. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes the energy-adaptive rank behavior. 
Section 9 discusses practical guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Layer weight matrix | §1.1 |
| $U, V$ | Low-rank factors | §1.1 |
| $r$ | Target rank | §1.1 |
| $r_\ell$ | Adaptive rank for layer $\ell$ | §1.3 |
| $\sigma_i$ | $i$-th singular value | §1.3 |
| $\eta$ | Energy threshold | §1.3 |
| $r_\text{max}$ | Rank cap | §1.3 |
| $d$ | Hidden size | §1.1 |
| $H$ | Number of heads | §3.2 |
| $H_{kv}$ | Number of KV heads (GQA) | §3.2 |
| $d_h$ | Per-head dimension | §3.1 |
| $\alpha$ | LoRA scaling | §3.4 |
| $A, B$ | LoRA factors | §3.4 |

---