# 1. Introduction

## 1.1 Motivation

The four attention projections $W_Q, W_K, W_V, W_O$ account for the majority 
of parameters in the attention block of a transformer. For a 768-dim model 
with 12 heads, these contribute $4 d^2 = 2.36$M parameters — comparable to 
the FFN in DistilBERT and a significant fraction in GPT-2.

**Low-rank factorization** is a natural approach: replace $W \in \mathbb{R}^{d \times d}$ 
with $U V$ where $U \in \mathbb{R}^{d \times r}$ and $V \in \mathbb{R}^{r \times d}$ 
for $r \ll d$. Parameter count drops from $d^2$ to $2 r d$.

**The rank $r$ is critical.** Too small → severe approximation error. Too 
large → poor compression. Two existing approaches dominate:

1. **Fixed rank** (SOA Fused Attention, vanilla SVD): choose a global $r$ 
   (e.g., 128) and apply it to all projections.
2. **Empirical tuning** (A3, AASVD): run ablations to find $r$ per model.

Both ignore that **each projection has a different intrinsic rank**. A 
projection with rapidly decaying singular values can be compressed more 
aggressively than one with a flat spectrum.

## 1.2 Problem Statement

Given a pretrained attention with projections $\{W^{(i)} \in \mathbb{R}^{d_i \times d_i}\}$, 
we seek factorizations $\{U^{(i)} V^{(i)}\}$ such that:

1. **Parameter reduction:** $2 r^{(i)} d_i < d_i^2$ for each projection.
2. **Adaptive rank:** $r^{(i)}$ differs per projection based on its spectrum.
3. **Spectral fidelity:** the retained energy fraction satisfies:
   
   $$\frac{\sum_{j=1}^{r^{(i)}} \sigma_j^2}{\sum_{j=1}^{d_i} \sigma_j^2} \geq \tau$$
   
   for a threshold $\tau$ (e.g., 0.92).
4. **No calibration data required:** initialization is zero-shot.
5. **Learnable recovery:** adapters provide fine-tuning capacity.
6. **Full architecture support:** T5 relative bias, cross-attention, GQA, RoPE.

## 1.3 Our Approach

We propose **Adaptive SVD Attention**, which combines three ideas:

### 1.3.1 Energy-Thresholded Rank Selection

For each projection $W$, we compute the SVD:

$$
W = U \Sigma V^\top
\tag{1.1}
$$

and select the **smallest rank** $r$ such that the cumulative squared 
singular value energy reaches $\tau$:

$$
r^* = \min\left\{ r : \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{d} \sigma_j^2} \geq \tau \right\}
\tag{1.2}
$$

This is principled: it directly bounds the Frobenius reconstruction error 
relative to the total matrix energy.

### 1.3.2 Symmetric $\sqrt{\Sigma}$ Split

We split the singular values symmetrically:

$$
U_r = U_{:, :r} \Sigma_r^{1/2}, \quad V_r = \Sigma_r^{1/2} V_{:r, :}^\top
\tag{1.3}
$$

This balances magnitudes between the two factors, which is numerically 
favorable.

### 1.3.3 Residual Adapters

We freeze the SVD base and add a trainable low-rank residual:

$$
W_\text{eff} \approx U_r V_r + B A
\tag{1.4}
$$

where $A \in \mathbb{R}^{r_a \times d}$, $B \in \mathbb{R}^{d \times r_a}$, and 
$B = 0$ initially.

### 1.3.4 All Four Projections

Unlike SOA Fused Attention (Q/K/V only), we factorize **all four** including 
$W_O$. This is possible because the residual adapter absorbs the additional 
error introduced by factorizing $W_O$.

## 1.4 Contributions

- **Adaptive rank selection.** We introduce energy-thresholded rank 
  selection per projection, replacing fixed-rank heuristics.
- **All-four-projection factorization.** We compress $W_Q, W_K, W_V, W_O$ 
  using the same primitive.
- **T5 and cross-attention support.** We handle T5 relative attention bias 
  and encoder-decoder cross-attention — a capability lacking in prior 
  attention-compression methods.
- **A production-hardened implementation.** Supports DistilBERT, RoBERTa, 
  GPT-2, Qwen3.5, and T5 with robust weight extraction.
- **A comprehensive evaluation.** We benchmark on three models on SST-2 with 
  ablations over energy threshold, residual rank, and rank policy.
- **Full reproducibility.** Code, weights, and scripts released under {{LICENSE}}.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, energy 
thresholding, and attention. Section 4 presents Adaptive SVD Attention. 
Section 5 describes integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes rank selection. Section 9 discusses 
guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension |
| $r$ | Selected rank |
| $H$ | Number of heads |
| $H_{kv}$ | KV head count |
| $d_h$ | Head dimension |
| $\sigma_i$ | $i$-th singular value |
| $\tau$ | Energy threshold |
| $r_a$ | Residual adapter rank |
| $U_r, V_r$ | Factorized SVD components |
| $A, B$ | Residual adapter matrices |
```

---