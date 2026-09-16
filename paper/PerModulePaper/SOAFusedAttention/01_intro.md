# 1. Introduction

## 1.1 Motivation

In a standard transformer attention block, the four projections $W_Q, W_K, 
W_V, W_O$ dominate parameter count. For a 768-dim model with 12 heads and 
64 head dim:

| Projection | Shape | Parameters |
|---|---|---|
| $W_Q$ | $768 \times 768$ | 589,824 |
| $W_K$ | $768 \times 768$ | 589,824 |
| $W_V$ | $768 \times 768$ | 589,824 |
| $W_O$ | $768 \times 768$ | 589,824 |
| **Total** | | **2.36M** |

Compare to the FFN, which contributes $2 \times 768 \times 3072 = 4.72M$ for 
a 4× expansion. In models with smaller FFN expansion (e.g., DistilBERT's 4×), 
attention is the majority of the parameter count.

**Low-rank factorization** is a natural compression primitive. For a weight 
$W \in \mathbb{R}^{d \times d}$, one replaces it with:

$$
W \approx U V, \quad U \in \mathbb{R}^{d \times k}, \quad V \in \mathbb{R}^{k \times d}
\tag{1.1}
$$

with $k \ll d$. The parameter count drops from $d^2$ to $kd + dk = 2kd$. For 
$k = 128$ and $d = 768$, this is a 3× reduction.

**The standard approach** is to compute the truncated SVD of $W$:

$$
W \approx U_k \Sigma_k V_k^\top
\tag{1.2}
$$

and use $U = U_k$, $V = \Sigma_k V_k^\top$ (or distribute $\Sigma$ symmetrically).

This is optimal in Frobenius norm (Eckart–Young), and **requires no calibration 
data** — the matrix itself is the only input needed.

## 1.2 Problem Statement

Given a pretrained transformer with attention projections $\{W_Q, W_K, W_V, W_O\}$, 
we seek a compressed attention block with projections $\{U_Q V_Q, U_K V_K, U_V V_V, 
W_O'\}$ such that:

1. **Parameter reduction:** $\text{params}(U_Q V_Q) < \text{params}(W_Q)$ for 
   each projection, i.e., $2k < d$.
2. **Reconstruction bound:** $\|W - UV\|_F / \|W\|_F \leq \tau$ for 
   threshold $\tau$.
3. **No calibration data required:** the initialization must be zero-shot.
4. **Learnable recovery:** LoRA adapters attached to $U$ (or $V$) provide 
   recovery capacity.
5. **Architecture portable:** must work with flat, nested, and causal attention 
   and support GQA and RoPE.

## 1.3 Our Approach

We introduce **SOA Fused Attention**, which replaces the standard Q/K/V 
projections with factorized pairs:

$$
W_Q \to (U_Q^{\text{down}}, U_Q^{\text{up}}), \quad
W_K \to (U_K^{\text{down}}, U_K^{\text{up}}), \quad
W_V \to (U_V^{\text{down}}, U_V^{\text{up}})
$$

Each pair is initialized from the truncated SVD of the corresponding 
pretrained weight. We distribute the singular values symmetrically:

$$
W = U \Sigma V^\top \approx U_k \Sigma_k V_k^\top
= \underbrace{(U_k \Sigma_k^{1/2})}_{\text{up}} 
  \underbrace{(\Sigma_k^{1/2} V_k^\top)}_{\text{down}}
\tag{1.3}
$$

**Key design choices:**

1. **Zero-shot SVD init:** no calibration pass needed.
2. **Frozen down and up:** base weights are not trained.
3. **LoRA on up:** trainable adapter for recovery.
4. **Output projection preserved:** $W_O$ is not factorized (it combines 
   heads and is critical).

## 1.4 Contributions

- **A zero-shot attention compressor.** SOA Fused Attention factorizes Q, K, 
  V using SVD of the pretrained weights themselves — no calibration data.
- **Real parameter reduction.** Unlike attention-replacement methods, we 
  **reduce** parameter count by a factor of $d / (2k)$.
- **A production-hardened implementation.** We handle GQA, RoPE, causal 
  masking, nested encoders, and multiple architectures with robust weight 
  extraction and per-projection Frobenius validation.
- **A comprehensive evaluation.** We benchmark on DistilBERT, RoBERTa, and 
  GPT-2 on SST-2 with ablations over rank, LoRA rank, and recovery steps.
- **Full reproducibility.** Code, weights, and evaluation scripts are released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, LoRA, 
and attention. Section 4 presents SOA Fused Attention. Section 5 describes 
integration. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the reconstruction-accuracy relationship. Section 9 
discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension |
| $k$ | Bottleneck rank |
| $H$ | Number of heads |
| $H_{kv}$ | Number of KV heads |
| $d_h$ | Head dimension |
| $W_Q, W_K, W_V, W_O$ | Attention projections |
| $U, V$ | Low-rank factors (up, down) |
| $\Sigma$ | Singular values |
| $\tau$ | Reconstruction error threshold |
| $r_L$ | LoRA rank |
```

---