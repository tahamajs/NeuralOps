# 1. Introduction

## 1.1 Motivation

The four attention projection matrices $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ 
together contain $4 d^2$ parameters per layer — for $d = 768$, that's **2.36M 
parameters per layer**, dominating the attention block. Across 12 layers of 
RoBERTa, this is **28.3M parameters** just in projections.

Low-rank factorization reduces this cost. For a matrix $W \in \mathbb{R}^{d \times d}$, 
SVD replaces it with $U_r V_r$ where $U_r \in \mathbb{R}^{d \times r}$, 
$V_r \in \mathbb{R}^{r \times d}$, and $r \ll d$. Parameter count drops from 
$d^2$ to $2 r d$.

**Standard SVD compression** treats each projection independently:

$$
W_Q \to U_{Q,r} V_{Q,r}, \quad W_K \to U_{K,r} V_{K,r}, \quad W_V \to U_{V,r} V_{V,r}, \quad W_O \to U_{O,r} V_{O,r}
\tag{1.1}
$$

This costs $4 \cdot 2 r d = 8 r d$ parameters.

**Key observation.** In pretrained transformers, the Q, K, V projections of a 
head are **highly correlated** — they share a common input basis. This suggests 
that a **shared** input-space basis $V_{QKV}$ may capture the structure of all 
three with minimal loss.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with hidden dimension $d$ and 
head dimension $d_h$, we seek compressed projections 
$\{W_Q, W_K, W_V, W_O\}$ such that:

$$
\text{Cost}(\mathcal{A}') \ll 4 d^2, \qquad
\mathcal{A}' \approx \mathcal{A}
\tag{1.2}
$$

subject to:

1. **Parameter reduction:** Compressed projection count $< 4 d^2$.
2. **Accuracy preservation:** Near-baseline on SST-2.
3. **Multi-architecture:** Support flat, nested, causal, and GQA models.
4. **Recovery capacity:** LoRA adapters for fine-tuning.
5. **Correctness:** Handle nested encoders without double projection.

## 1.3 Our Approach: Joint QKV SVD

We stack the Q, K, V projections:

$$
W_{QKV} = \begin{pmatrix} W_Q \\ W_K \\ W_V \end{pmatrix} \in \mathbb{R}^{(d + 2 d_{kv}) \times d}
\tag{1.3}
$$

where $d_{kv} = H_{kv} \cdot d_h$ for grouped-query attention.

We then compute a single SVD:

$$
W_{QKV} = U_{QKV} \Sigma_{QKV} V_{QKV}^\top
\tag{1.4}
$$

and truncate to rank $r$:

$$
W_{QKV} \approx U_{QKV, r} \Sigma_{QKV, r}^{1/2} \cdot \Sigma_{QKV, r}^{1/2} V_{QKV, r}^\top
\tag{1.5}
$$

**Advantages:**

1. **Shared basis:** $V_{QKV, r}$ is a single input basis for Q, K, V.
2. **Smaller parameter count:** $r(d + 2d_{kv}) + r \cdot d = 2rd + 2r d_{kv}$ vs 
   independent $6rd$.
3. **Correlation exploitation:** SVD automatically captures the shared 
   structure.

We apply separate SVD to the output projection $W_O \in \mathbb{R}^{d \times d_h H}$:

$$
W_O \approx U_{O, r} \Sigma_{O, r}^{1/2} \cdot \Sigma_{O, r}^{1/2} V_{O, r}^\top
\tag{1.6}
$$

## 1.4 Contributions

- **Joint QKV SVD.** We propose stacking Q, K, V and factorizing them with a 
  single SVD, sharing the input basis across all three projections.
- **Multi-architecture support.** The implementation handles flat (DistilBERT), 
  nested (RoBERTa), Conv1D (GPT-2), and GQA+RoPE (Qwen) models.
- **Correct nested-encoder detection.** We check for `out_lin` / `c_proj` / 
  `o_proj` **before** declaring a model nested, fixing a subtle double-
  projection bug that caused 50.92% collapse in earlier implementations.
- **Safe failure handling.** Weight extraction failures trigger warnings, not 
  exceptions, allowing graceful degradation.
- **LoRA recovery on both branches.** Both the QKV and output branches have 
  independent LoRA adapters.
- **Honest parameter accounting.** We report exact parameter reductions.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, LoRA, 
and attention. Section 4 presents Compressed MHA. Section 5 describes 
integration. Section 6 details the setup. Section 7 reports results. Section 8 
analyzes trade-offs. Section 9 discusses practical guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension |
| $d_h$ | Head dimension |
| $H$ | Query heads |
| $H_{kv}$ | KV heads |
| $d_{kv}$ | $H_{kv} \cdot d_h$ |
| $r$ | SVD rank |
| $r_L$ | LoRA rank |
| $W_Q, W_K, W_V, W_O$ | Projection matrices |
| $U, \Sigma, V$ | SVD factors |
| $\text{is\_nested}$ | Nested encoder flag |

---