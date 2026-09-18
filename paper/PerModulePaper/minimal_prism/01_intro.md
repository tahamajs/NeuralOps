<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer-based language models concentrate the majority of their parameters 
in two architectural blocks: the feed-forward network and the multi-head 
self-attention (MHSA) mechanism. In DistilBERT-base, the four attention 
projections $W_Q, W_K, W_V, W_O$ account for approximately **70%** of the 
parameters in each transformer block. Reducing the parameter cost of attention 
is therefore a high-leverage compression strategy.

Standard MHSA stores four dense matrices per layer:

$$
W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}
\tag{1.1}
$$

totaling $4d^2$ parameters. For DistilBERT ($d = 768$), this is $4 \times 589{,}824 = 2{,}359{,}296$ 
parameters per layer — or 14.2M across six layers, a substantial share of the 
66.9M total.

**Low-rank factorization** is the classical approach. For each $W$, one 
replaces it with a product:

$$
W \approx U V, \qquad
U \in \mathbb{R}^{d_\text{out} \times r}, \quad
V \in \mathbb{R}^{r \times d_\text{in}}, \quad r \ll \min(d_\text{in}, d_\text{out})
\tag{1.2}
$$

The parameter count drops from $d_\text{in} d_\text{out}$ to 
$r(d_\text{in} + d_\text{out})$. The **reduction factor** is 
$\rho = d_\text{in} d_\text{out} / (r(d_\text{in} + d_\text{out}))$. For square 
matrices at $d = 768$ and $r = 128$: $\rho = 768 / 256 = 3.0$. At $r = 64$: 
$\rho = 6.0$. Classical SVD (Eckart–Young, 1936) gives the optimal rank-$r$ 
approximation in Frobenius norm.

**The rank-selection problem.** Classical SVD requires choosing $r$ 
a-priori, which is a hard trade-off:

- **Too small** — the approximation error is high, and downstream accuracy 
  drops.
- **Too large** — the parameter reduction is minimal, defeating the purpose.

In practice, different layers have different **intrinsic ranks**. Attention 
projections in early layers of DistilBERT tend to have rapidly-decaying 
spectra (effective rank ~30–48), while middle layers decay more slowly 
(effective rank ~80–100). A single global $r$ is suboptimal for both.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_l$ layers and a target 
energy threshold $\eta \in (0, 1)$, we seek a compressed model 
$\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.3}
$$

subject to:

1. **Per-layer adaptive rank.** Different layers may use different 
   $r_\ell$ chosen automatically.
2. **Spectral energy preservation.** For each factorized layer, 
   $E(r_\ell) \geq \eta$.
3. **Parameter reduction.** $r_\ell (d_\text{in} + d_\text{out}) < d_\text{in} d_\text{out}$.
4. **Multi-architecture support.** Flat (DistilBERT), nested (RoBERTa), 
   causal with Conv1D (GPT-2), GQA + RoPE (Qwen).
5. **No base retraining.** Only LoRA adapters may be trained.

## 1.3 Our Approach: MinimalPRISM

**MinimalPRISM** is a drop-in replacement for the four attention projections 
with three key components:

1. **SVD factorization.** Each $W$ is factorized as $U V$ via truncated SVD, 
   with $U \in \mathbb{R}^{d_\text{out} \times r_\ell}$ and 
   $V \in \mathbb{R}^{r_\ell \times d_\text{in}}$, where $\Sigma^{1/2}$ is 
   distributed symmetrically.

2. **Spectral energy-adaptive rank.** For each weight matrix, compute 
   $E(r) = \sum_{i \leq r} \sigma_i^2 / \sum_i \sigma_i^2$ and select:

   $$
   r_\ell = \min\!\left(r_\text{max}, \left|\{i : E(i) < \eta\}\right| + 1\right)
   \tag{1.4}
   $$

   This selects the smallest rank that captures at least $\eta$ fraction of 
   the spectral energy, capped at $r_\text{max}$.

3. **LoRA recovery.** After factorization, the base factors $U, V$ are 
   frozen and LoRA adapters $A, B$ are added:

   $$
   y = (x V^\top) U^\top + (x A^\top) B^\top \cdot s + b
   \tag{1.5}
   $$

   with $B = 0$ at initialization, so $y$ at step 0 equals the SVD 
   approximation.

**Injection safety.** MinimalPRISM detects nested encoders (RoBERTa) at 
injection time rather than only in weight-copy. It runs a **pre-test on the 
first layer** before mutating the rest of the model, aborting if weight 
extraction fails. This is critical for production correctness.

## 1.4 Contributions

- **Energy-adaptive rank selection.** A principled criterion that selects 
  a different rank per layer based on spectral decay, avoiding the 
  fixed-rank trade-off.
- **A safe injection protocol.** Nested-encoder detection at injection time, 
  pre-test on the first layer, and strict Frobenius norm validation of 
  extracted weights.
- **A multi-architecture implementation.** Four families supported 
  (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with explicit Conv1D handling, GQA 
  support, and RoPE integration.
- **A systematic evaluation.** Benchmark on three models on SST-2 with 
  ablations over $r_\text{max}$, $\eta$, and LoRA rank.
- **Full reproducibility.** Code, checkpoints, and scripts released under 
  Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background. Section 4 
presents MinimalPRISM. Section 5 describes architecture integration. Section 
6 details the setup. Section 7 reports results. Section 8 analyzes 
energy-adaptive behavior. Section 9 discusses guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Weight matrix |
| $U, V$ | Low-rank factors |
| $r$ | Target rank |
| $r_\ell$ | Adaptive rank for layer $\ell$ |
| $r_\text{max}$ | Rank cap |
| $\sigma_i$ | $i$-th singular value |
| $E(r)$ | Cumulative spectral energy |
| $\eta$ | Energy threshold |
| $d$ | Hidden size |
| $H$ | Query head count |
| $H_{kv}$ | KV head count (GQA) |
| $d_h$ | Per-head dimension |
| $\alpha, r_L$ | LoRA scaling, rank |

---