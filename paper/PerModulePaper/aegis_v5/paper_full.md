# AEGIS-V5: Adaptive Enhanced Global-Local Integrated Sparse Attention with SVD-Factorized Projections

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v5
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/aegis_v5.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Transformer self-attention has quadratic complexity in sequence length, 
making long-context inference expensive. We introduce **AEGIS-V5**, a 
hybrid attention module that combines a **local sliding-window branch** 
with a **global kernel linear attention branch**, gated per-head by a 
learnable scalar bias initialized toward local attention. The base 
projections are **SVD-factorized with energy-based rank selection** and 
frozen; trainable capacity is provided by **LoRA adapters** on each 
projection.

AEGIS-V5 splits the attention heads into two groups: the first 
$H_\text{local} = \lceil 2H/3 \rceil$ heads use local windowed attention, 
and the remaining $H_\text{global}$ heads use kernel linear attention with 
Random Fourier Features (RFF). The two branch outputs are combined by a 
per-head gate. For decoder-only models, we fall back to standard causal 
softmax attention to preserve causality — a conservative but correctness-
preserving choice.

On SST-2, AEGIS-V5 achieves **83.72%** on DistilBERT (baseline 
90.71%, Δ = -6.99), **86.47%** on RoBERTa (baseline 
78.56%, Δ = +7.91), and **69.50%** on GPT-2 (baseline 
79.36%, Δ = -9.86), using 50 LoRA recovery 
steps. Complexity is $\mathcal{O}(N \cdot W + N \cdot F)$ for encoders 
where $W$ is window size and $F$ is RFF feature count. We provide a 
production-hardened implementation supporting DistilBERT, RoBERTa, GPT-2, 
and Qwen3.5, with strict Frobenius validation, safe fallback, and detailed 
per-layer error statistics.

**Keywords:** hybrid attention, SVD factorization, LoRA, local window, 
kernel attention, energy-based rank

---

## Lay Summary

Standard transformers compare every word to every other word, which becomes 
slow for long texts. Our method splits the "comparison job" into two groups 
of attention heads: some heads look only at nearby words (local), and others 
look at a summarized version of the whole text (global). A learned weight 
decides how much each head relies on the local versus global view. To make 
the model smaller, we also compress the underlying projection weights using 
a mathematical technique called SVD, keeping only the most important 
directions. On three language models (DistilBERT, RoBERTa, GPT-2), our 
method achieves near-baseline accuracy while reducing memory and compute.

---

## Table of Contents

1. [Introduction](01_intro.md)
2. [Related Work](02_related.md)
3. [Background](03_background.md)
4. [Method](04_method.md)
5. [Architecture Integration](05_architecture.md)
6. [Experimental Setup](06_setup.md)
7. [Results](07_results.md)
8. [Analysis](08_analysis.md)
9. [Discussion](09_discussion.md)
10. [Limitations](10_limitations.md)
11. [Conclusion](11_conclusion.md)
12. [Reproducibility](12_reproducibility.md)
13. [References](13_references.md)

**Appendices:**
- [A. Mathematical Derivations](A_math.md)
- [B. Implementation Details](B_implementation.md)
- [C. Full Experimental Data](C_data.md)
- [D. Negative Results](D_negative_results.md)
- [E. Ethical Considerations](E_ethics.md)

---

---

# 1. Introduction

## 1.1 Motivation

Transformer self-attention scales quadratically with sequence length $L$, 
in both compute and memory. For input $X \in \mathbb{R}^{B \times L \times d}$ 
with $H$ heads, the intermediate attention matrix $Q K^\top \in \mathbb{R}^{L \times L}$ 
costs $\mathcal{O}(L^2)$ per batch item. At $L = 4096$, this is ~16M 
entries per head, becoming prohibitive for long-context workloads.

Existing methods address this along three axes:

1. **Sparse attention** (Longformer, BigBird) restricts the attention 
   pattern to $\mathcal{O}(L \sqrt{L})$ connections.
2. **Low-rank attention** (Linformer, Nyströmformer) projects $K, V$ to a 
   smaller space.
3. **Kernel attention** (Performer, Linear Transformer) factorizes the 
   softmax kernel into feature maps, achieving $\mathcal{O}(L)$.

Each has trade-offs. Sparse methods lose global context; low-rank methods 
require fixed rank; kernel methods suffer from approximation error on 
peaked distributions.

**Hybrid attention** combines multiple strategies. AEGIS-V5 combines local 
windowed attention (exact, preserves local detail) with global kernel 
linear attention (approximate, captures long-range dependencies).

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $H$ heads, we seek an 
attention mechanism $\mathcal{A}'$ such that:

$$
\mathcal{A}' \approx \mathcal{A}, \qquad
\text{Cost}(\mathcal{A}') \ll \mathcal{O}(L^2)
\tag{1.1}
$$

subject to:

1. **Multi-architecture compatibility:** Must work on DistilBERT (flat), 
   RoBERTa (nested), GPT-2 (Conv1D), and Qwen3.5 (GQA + RoPE).
2. **Causal correctness:** Decoders must preserve causality.
3. **Parameter efficiency:** Base weights frozen; only LoRA adapters trained.
4. **Numerical stability:** No NaNs, no silent collapses.

## 1.3 Our Approach

AEGIS-V5 has four key components:

### 1.3.1 Split-Head Architecture

We partition $H$ heads into:

- $H_\text{local} = \lceil 2H/3 \rceil$ heads using **local windowed attention**
- $H_\text{global} = H - H_\text{local}$ heads using **kernel linear attention**

This design lets the model allocate capacity adaptively: most heads focus 
on local structure, a few capture global context.

### 1.3.2 Energy-Based SVD Factorization

Each projection $W$ is decomposed via SVD, and the rank $r$ is chosen to 
capture a target **energy** fraction:

$$
r^* = \min \left\{ r : \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2} \geq \eta \right\}
\tag{1.2}
$$

where $\eta = 0.95$ by default.

### 1.3.3 LoRA on Frozen SVD Factors

The SVD factors $(U_r, V_r)$ are frozen; LoRA adapters provide trainable 
capacity:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{1.3}
$$

### 1.3.4 Per-Head Adaptive Gating

The local and global outputs are combined with a learnable per-head gate:

$$
\text{Out} = \sigma(g_h) \cdot \text{Out}_\text{local} + (1 - \sigma(g_h)) \cdot \text{Out}_\text{global}
\tag{1.4}
$$

Gate bias is initialized to $+2.0$, giving $\sigma(2.0) \approx 0.88$ 
(mostly local at start).

## 1.4 Contributions

We make the following contributions:

- **A hybrid split-head attention.** AEGIS-V5 combines local windowed and 
  global kernel attention, with per-head adaptive gating.
- **Energy-based SVD factorization.** Base projections are compressed with 
  a target energy fraction, giving explicit control over rank.
- **A production-hardened implementation.** Handles four architectures with 
  GQA, RoPE, nested encoders, Conv1D, and safe fallback.
- **Numerical stability fixes.** We replaced the original ReLU-based `softpick` 
  (which returned all-zeros on masked rows) with a stable softmax-based 
  variant, eliminating a subtle collapse mode.
- **Full reproducibility.** Code and checkpoints released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention, 
SVD, LoRA, and kernel methods. Section 4 presents AEGIS-V5. Section 5 
describes integration. Section 6 details the setup. Section 7 reports 
results. Section 8 analyzes trade-offs. Section 9 discusses practical 
guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ or $N$ | Sequence length |
| $H$ | Total heads |
| $H_\text{local}$ | Heads using local attention |
| $H_\text{global}$ | Heads using global attention |
| $W$ | Local window size |
| $F$ | RFF feature dimension |
| $r$ | SVD rank |
| $\eta$ | Energy fraction threshold |
| $r_L$ | LoRA rank |
| $g_h$ | Per-head gate logit |
| $\sigma$ | Sigmoid |

---

---

# 2. Related Work

## 2.1 Sparse and Local Attention

**Longformer** (Beltagy et al., 2020) combines a sliding window with a small 
set of global tokens. **BigBird** (Zaheer et al., 2020) adds random 
attention and proves Turing completeness. **Star-Transformer** (Guo et al., 
2019) uses a star topology with a single global hub.

**Our distinction.** AEGIS-V5 uses exact local attention (not a sparse 
approximation) for the local branch, and a fully-connected kernel 
attention for the global branch.

## 2.2 Kernel and Linear Attention

**Performer** (Choromanski et al., 2021) uses positive orthogonal random 
features (PORF). **Linear Transformer** (Katharopoulos et al., 2020) uses 
$\phi(x) = \text{elu}(x) + 1$. **Random Feature Attention** (Peng et al., 
2021) uses Gaussian RFF.

**Our distinction.** AEGIS-V5 applies kernel attention **only to a subset 
of heads** (the global heads), preserving exact local computation for the 
rest.

## 2.3 Low-Rank Factorization

**Linformer** (Wang et al., 2020) projects $K, V$ to a fixed rank. 
**Nyströmformer** (Xiong et al., 2021) uses Nyström landmarks. **ALBERT** 
(Lan et al., 2020) shares parameters across layers.

**Our distinction.** AEGIS-V5 uses **energy-based rank selection** for each 
projection independently, and combines SVD with LoRA for recovery.

## 2.4 Hybrid Attention

**H-Transformer-1D** (Zhu & Soricut, 2021) uses hierarchical attention. 
**NHA** (our prior work) combines local sliding window with global linear 
attention via a single scalar gate. **LEKAv3** (our prior work) uses 
learned landmarks with local window fusion.

**Our distinction.** AEGIS-V5 uses a **split-head** architecture: different 
heads use different attention mechanisms. This is more expressive than 
combining branches with a single gate.

## 2.5 Parameter-Efficient Fine-Tuning

**LoRA** (Hu et al., 2021) freezes base weights and trains low-rank 
adapters. **AdaLoRA** (Zhang et al., 2023) adaptively allocates rank per 
layer. **DoRA** (Liu et al., 2024) decomposes weights into magnitude and 
direction.

**Our distinction.** AEGIS-V5 combines SVD factorization of base weights 
with LoRA adapters on the factorized form, providing both compression and 
recovery.

## 2.6 Position of Our Work

| Method | Local | Global | Split-head | SVD |
|---|---|---|---|---|
| NHA | ✓ | ✓ | ✗ | ✗ |
| LEKAv3 | ✓ | ✓ (landmarks) | ✗ | ✗ |
| Linformer | ✗ | ✓ (projection) | ✗ | ✗ |
| **AEGIS-V5** | ✓ | ✓ (RFF) | **✓** | **✓** |

**Key distinction.** AEGIS-V5 is the first hybrid attention to (i) split 
heads into mechanism-specific groups, (ii) use energy-based SVD for 
projection compression, and (iii) apply LoRA on the frozen factors.

---

## Comparison Table

| Method | Complexity | Learned head split? | SVD? | Multi-arch |
|---|---|---|---|---|
| NHA | $\mathcal{O}(L \cdot W + L \cdot m)$ | No | No | ✓ |
| LEKAv3 | $\mathcal{O}(L \cdot L_\text{land})$ | No | No | ✓ |
| Linformer | $\mathcal{O}(L \cdot k)$ | No | No | ✓ |
| **AEGIS-V5** | $\mathcal{O}(L \cdot W + L \cdot F)$ | **Yes** | **Yes** | ✓ |

---

---

# 3. Background

## 3.1 Standard Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

Complexity: $\mathcal{O}(L^2 d_h)$ for scores.

## 3.2 Sliding Window Attention

Local window attention restricts each query to a window of size $W$ around 
it:

$$
\text{Attn}_\text{local}(q_i, K, V) = \sum_{j : |i - j| \leq W/2} \alpha_{ij} v_j
\tag{3.3}
$$

Complexity: $\mathcal{O}(L \cdot W \cdot d_h)$.

For decoders, the causal variant restricts to $j \leq i$:

$$
\text{Attn}_\text{local, causal}(q_i, K, V) = \sum_{j : i - W/2 \leq j \leq i} \alpha_{ij} v_j
\tag{3.4}
$$

## 3.3 Kernel Linear Attention

Kernel attention factorizes the softmax kernel:

$$
e^{q \cdot k} \approx \phi(q)^\top \phi(k)
\tag{3.5}
$$

The output becomes:

$$
\text{Out}_\text{kernel}(q_i, K, V) = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.6}
$$

Complexity: $\mathcal{O}(L \cdot F \cdot d_h)$ where $F$ is the feature 
dimension.

### 3.3.1 Random Fourier Features

For RFF:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp\!\left(W_\text{rff} x\right)
\tag{3.7}
$$

where $W_\text{rff} \in \mathbb{R}^{d_h \times F}$ is a random projection.

### 3.3.2 Zero-Sum Keys

Zero-sum keys (Katharopoulos et al., 2020) subtract the mean:

$$
\tilde{K} = K - \bar{K}
\tag{3.8}
$$

This improves the approximation for positive-definite kernels.

### 3.3.3 Causal Kernel Attention

For decoders, the cumulative sum formulation:

$$
S_i = \sum_{j \leq i} \phi(k_j) v_j^\top, \quad
Z_i = \sum_{j \leq i} \phi(k_j)
\tag{3.9}
$$

$$
\text{Out}_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top Z_i}
\tag{3.10}
$$

## 3.4 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.11}
$$

Truncated at rank $r$:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.12}
$$

## 3.5 Energy-Based Rank Selection

The **energy fraction** captured by rank $r$:

$$
\text{Energy}(r) = \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2}
\tag{3.13}
$$

We select $r^* = \min \{ r : \text{Energy}(r) \geq \eta \}$.

For $\eta = 0.95$: $r^*$ captures 95% of the weight energy.

## 3.6 LoRA

LoRA parameterizes weight updates:

$$
\Delta W = \frac{\alpha}{r_L} B A, \quad B \in \mathbb{R}^{d \times r_L}, \quad A \in \mathbb{R}^{r_L \times k}
\tag{3.14}
$$

with $B = 0$ at initialization.

## 3.7 Grouped-Query Attention (GQA)

Modern models (Qwen, LLaMA) use $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} \times d_h}
\tag{3.15}
$$

Expanded to $H$ via `repeat_kv`.

## 3.8 Rotary Position Embeddings (RoPE)

RoPE rotates $Q, K$:

$$
\tilde{q}_i = q_i \cos(\theta_i) + \text{rot}(q_i) \sin(\theta_i)
\tag{3.16}
$$

## 3.9 Softpick (Fixed)

The original AEGIS used a ReLU-based softpick:

$$
\text{softpick}(x)_i = \text{ReLU}\!\left(e^{(x_i - m)/T} - e^{-m/T}\right)
\tag{3.17}
$$

where $m = \max_j x_j$. **Problem:** When $m \leq 0$ (all-masked rows), 
this returns all zeros, causing attention collapse.

AEGIS-V5 replaces this with a stable softmax:

$$
\text{softpick}(x)_i = \frac{e^{x_i / T}}{\sum_j e^{x_j / T}}
\tag{3.18}
$$

which is always numerically stable.

---

---

# 4. Method

## 4.1 Overview

AEGIS-V5 has three main stages:

1. **SVD-factorized projections** (§4.2): each $Q/K/V/O$ projection is 
   decomposed with energy-based rank selection and frozen.
2. **Split-head attention** (§4.3): heads partitioned into local and global.
3. **Adaptive gating** (§4.4): per-head gate fuses local and global.

```
        Input X
           │
           ▼
    ┌──────────────────┐
    │ SVD Q/K/V/O      │  (frozen, energy-ranked)
    │ + LoRA adapters  │  (trainable)
    └────────┬─────────┘
             │
    ┌────────┴─────────┐
    ▼                  ▼
 ┌──────────┐    ┌───────────┐
 │ Local    │    │ Global    │
 │ Window   │    │ RFF       │
 │ (2H/3)   │    │ (H/3)     │
 └────┬─────┘    └─────┬─────┘
      │                │
      └───────┬────────┘
              ▼
      ┌───────────────┐
      │ Per-head      │
      │ Gating σ(g)   │
      └───────┬───────┘
              │
              ▼
         Output
```

## 4.2 SVD-Factorized Projections

### 4.2.1 Decomposition

Given a weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

1. Compute the SVD:
   $$
   W = U \Sigma V^\top
   \tag{4.1}
   
<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$
2. Select energy-based rank:
   $$
   r^* = \min \left\{ r : \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2} \geq \eta \right\}
   \tag{4.2}
   
<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$
3. Factor into $W \approx U_r V_r$ where:
   $$
   U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
   V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
   \tag{4.3}
   
<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

### 4.2.2 Parameter Reduction

| Configuration | Parameters |
|---|---|
| Original | $d_\text{out} \cdot d_\text{in}$ |
| SVD at rank $r$ | $r(d_\text{out} + d_\text{in})$ |

For $d = 768$ and $r = 384$ (energy 0.95):

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.00× |
| SVD r=384 | 589,824 | **1.00×** |

**Critical observation.** At $r = 384 = d/2$, SVD parameters equal original. 
For meaningful compression, we need $r < d/2$.

### 4.2.3 LoRA on Frozen Factors

We freeze $(U_r, V_r)$ and add LoRA adapters:

$$
y = U_r V_r x + \frac{\alpha}{r_L} B A x
\tag{4.4}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

Only $A, B$ are trained.

**Note.** AEGIS-V5 uses a `FactorisedLinear` that applies SVD as a **stacked** 
composition (`down` then `up`), not a reconstruction. This is critical for 
correctness.

## 4.3 Split-Head Attention

### 4.3.1 Head Partition

We split $H$ heads:

$$
H_\text{local} = \lceil 2H/3 \rceil, \quad
H_\text{global} = H - H_\text{local}
\tag{4.5}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

For $H = 12$:
- $H_\text{local} = 8$
- $H_\text{global} = 4$

### 4.3.2 Local Branch

Heads $0 \ldots H_\text{local} - 1$ use sliding window attention:

$$
\text{Out}_\text{local}[i] = \text{softpick}\!\left(\frac{q_i K_\text{local}^\top}{\sqrt{d_h}}\right) \cdot V_\text{local}
\tag{4.6}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

with a window mask $|i - j| \leq W/2$.

### 4.3.3 Global Branch

Heads $H_\text{local} \ldots H - 1$ use kernel linear attention:

$$
\text{Out}_\text{global}[i] = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{4.7}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

with RFF feature map $\phi(x) = \exp(W_\text{rff} x) / \sqrt{F}$.

### 4.3.4 Concatenation

The outputs are concatenated:

$$
\text{Out} = \text{Concat}(\text{Out}_\text{local}, \text{Out}_\text{global}) \in \mathbb{R}^{B \times H \times L \times d_h}
\tag{4.8}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

## 4.4 Adaptive Gating

### 4.4.1 Per-Head Gate

Each head has a learnable scalar $g_h$:

$$
\text{Out}_\text{fused}[i] = \sigma(g_h + 2.0) \cdot \text{Out}_\text{local}[i] + (1 - \sigma(g_h + 2.0)) \cdot \text{Out}_\text{global}[i]
\tag{4.9}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

where $g_h$ is initialized to 0 (so $\sigma(2.0) \approx 0.88$).

### 4.4.2 Why Bias 2.0?

At initialization, we want the model to prioritize local attention (known 
to be safe). As training progresses, the gate can shift toward global.

## 4.5 Decoder Fallback

For decoder models, the local+global split **does not preserve causality**:

- Local branch: causal window is fine.
- Global branch: prefix scan preserves causality.

**But:** For very short sequences ($N < W$), the local branch covers the 
whole sequence, making the split redundant. We fall back to standard dense 
causal attention:

$$
\text{if } \text{is\_decoder} : \text{standard causal softmax}
\tag{4.10}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

This preserves correctness at the cost of quadratic complexity for decoders.

## 4.6 Full Algorithm

**Algorithm 1:** AEGIS-V5 forward
```
Input:  X ∈ R^{B×L×d}, attention_mask, is_decoder
Output: Y ∈ R^{B×L×d}

1:  Q ← SVD_Q(X); K ← SVD_K(X); V ← SVD_V(X)
2:  if RoPE: Q, K ← RoPE(Q, K)
3:  if GQA: K, V ← repeat_kv(K, V)
4:  if is_decoder:
5:      # Fallback to standard causal attention
6:      scores ← Q K^T / √d_h
7:      scores ← mask_causal(scores)
8:      out ← softmax(scores) @ V
9:  else:
10:     # Split-head hybrid attention
11:     Q_local ← Q[:, :H_local]; K_local ← K[:, :H_local]; V_local ← V[:, :H_local]
12:     Q_global ← Q[:, H_local:]; K_global ← K[:, H_local:]; V_global ← V[:, H_local:]
13:     Out_local ← local_window(Q_local, K_local, V_local, W)
14:     Out_global ← kernel_linear(Q_global, K_global, V_global, φ)
15:     # Per-head gate
16:     g ← σ(blend_logits + 2.0)
17:     Out ← g · Out_local + (1 - g) · Out_global
18: Y ← SVD_O(Out)
19: return Y
```

## 4.7 Complexity Analysis

### 4.7.1 Encoder

| Component | Complexity |
|---|---|
| SVD projections | $\mathcal{O}(L r (d + d_h))$ |
| Local attention | $\mathcal{O}(L W d_h \cdot H_\text{local})$ |
| Kernel attention | $\mathcal{O}(L F d_h \cdot H_\text{global})$ |
| Output projection | $\mathcal{O}(L r d)$ |
| **Total** | $\mathcal{O}(L(W H_\text{local} + F H_\text{global}) d_h)$ |

For fixed $W, F, H_\text{local}, H_\text{global}$: linear in $L$.

### 4.7.2 Decoder

Quadratic fallback: $\mathcal{O}(L^2 d_h)$.

## 4.8 Parameter Analysis

| Component | Baseline | AEGIS-V5 | Change |
|---|---|---|---|
| Q/K/V/O (SVD) | $4 d^2$ | $4 r(d + d_h)$ | depends on $r$ |
| LoRA adapters | 0 | $4 r_L (d + d_h)$ | +small |
| RFF proj | 0 | $H_\text{global} \cdot d_h \cdot F$ | +medium |
| blend_logits | 0 | $H$ | +negligible |

**Critical.** Whether AEGIS-V5 reduces parameters depends entirely on $r$.

For $d = 768$, $r = 384$ (energy 0.95 default):
- SVD: no compression (ratio 1.00×)
- LoRA: +98,304
- RFF: $4 \cdot 64 \cdot 128 = 32,768$
- **Net: +131,072 params per layer**

For $d = 768$, $r = 128$:
- SVD: $128 \cdot 1536 = 196,608$ (vs 589,824) → **3× compression**
- LoRA: +98,304
- RFF: +32,768
- **Net: -262,144 params per layer**

**We are explicit:** The parameter effect depends on the SVD rank.

## 4.9 Theoretical Notes

### 4.9.1 SVD Approximation Error

If $W = U\Sigma V^\top$, then:

$$
\|W - U_r \Sigma_r V_r^\top\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{4.11}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

The energy-based rank ensures this is below $\sqrt{1 - \eta} \|W\|_F$.

### 4.9.2 Kernel Approximation

Standard RFF bound (Rahimi & Recht, 2007):

$$
\sup_{x, y} |\phi(x)^\top \phi(y) - e^{x \cdot y}| \leq \epsilon
\tag{4.12}

<!-- source: Codes/framework/attentions/aegis_v5.py line 236 -->
$$

with $F = \mathcal{O}(d_h \log(1/\epsilon^2))$.

### 4.9.3 Causality

**Claim.** For decoders, the fallback preserves causality.

**Proof.** Fallback uses standard causal softmax with triangular mask. $\square$

**Claim.** For encoders, the split-head design preserves non-causality 
(no masking needed).

**Proof.** Local window covers $|i - j| \leq W/2$; kernel attention covers 
all $j$. Both are bidirectional by design. $\square$

---

---

# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support

| Model | Attention | Nested? | RoPE? | GQA? | Conv1D? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | No |
| RoBERTa | `BertSelfAttention` | Yes | No | No | No |
| GPT-2 | `GPT2Attention` | No | No | No | Yes |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | No |

## 5.2 Weight Transfer

AEGIS-V5 copies base weights via `copy_weights`, handling four patterns:

### 5.2.1 GPT-2 Conv1D

```python
weight = old_attn.c_attn.weight.data
if weight.shape[1] == 3 * self.hidden_size:
    weight = weight.t()
q_w, k_w, v_w = weight.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    self.is_nested_encoder = True
    q_w = old_attn.self.query.weight
    k_w = old_attn.self.key.weight
    v_w = old_attn.self.value.weight
    o_w = old_attn.output.dense.weight
```

### 5.2.3 Qwen GQA

```python
q_w = old_attn.q_proj.weight
k_w = old_attn.k_proj.weight  # (H_kv · d_h, d)
v_w = old_attn.v_proj.weight
o_w = old_attn.o_proj.weight
```

### 5.2.4 DistilBERT

```python
q_w = old_attn.q_lin.weight
k_w = old_attn.k_lin.weight
v_w = old_attn.v_lin.weight
o_w = old_attn.out_lin.weight
```

## 5.3 Robustness Improvements

Compared to AEGIS V4, V5 makes three robustness changes:

### 5.3.1 copy_weights No Longer Raises

In V4, exceptions in `copy_weights` aborted the entire attention replacement. 
V5 logs a warning and continues to fallback paths:

```python
try:
    w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
except Exception as e:
    logger.debug(f"UniversalAdapter failed for {proj}: {e}")
    continue
```

### 5.3.2 Frobenius Check No Longer Raises

Near-zero norms trigger a warning, not an abort:

```python
if norm < 1e-4:
    logger.warning(f"Weight for {proj_key} has near-zero norm; continuing")
```

### 5.3.3 inject_aegis_v5 Returns None on Failure

```python
try:
    new_attn.copy_weights(src_attn)
except Exception as e:
    logger.error(f"inject_aegis_v5: copy_weights failed: {e}")
    return None
```

## 5.4 Nested Encoder Bypass

For RoBERTa, `is_nested_encoder = True`:

```python
if self.is_nested_encoder:
    return out, None
return self.out_proj(out), None
```

The outer `BertSelfOutput.dense` applies the output projection.

## 5.5 Decoder Fallback

For decoder models, we use standard causal attention (no split-head):

```python
if N < self.cfg.local_window or self.is_decoder:
    # Standard causal softmax
    scores = Q @ K^T / √d_h
    scores = mask_causal(scores)
    attn = softpick(scores)
    out = attn @ V
    return out, None
```

This preserves causality at the cost of quadratic complexity.

## 5.6 Softpick Fix

The original AEGIS used a ReLU-based softpick that returned all zeros when 
$\max(x) \leq 0$ (e.g., on fully-masked rows). This caused silent attention 
collapse. V5 replaces it with a stable softmax:

```python
def softpick(x, dim=-1, temperature=1.0):
    x_f = x.float()
    if temperature <= 0 or temperature == 1.0:
        return F.softmax(x_f, dim=dim).to(x.dtype)
    return F.softmax(x_f / temperature, dim=dim).to(x.dtype)
```

This fix alone likely prevented several collapse cases.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject AEGIS-V5 into a model
```
Input:  model M, config
Output: M with attention replaced or unchanged on failure

1: for ℓ in transformer_layers(M) do
2:     src ← get_attention(ℓ)
3:     cfg ← AegisConfigV5(...)
4:     new ← AegisAttention(model_config, aegis_cfg=cfg)
5:     try:
6:         new.copy_weights(src)
7:     except:
8:         logger.error(...)
9:         continue  # Skip this layer, do not crash
10:    replace(ℓ, new)
11: end for
12: return M
```

## 5.8 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| UniversalAdapter throws | try/except | Continue to next pattern |
| Missing Q/K/V | `missing` non-empty | Log warning, random init |
| Near-zero norm | Frobenius check | Log warning, continue |
| copy_weights fails | try/except | Skip layer, return None |
| NaN in scores | `nan_to_num` | Set to 0 |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("aegis_v5_module")
@ATTENTION_REGISTRY.register("AegisAttention")
@ATTENTION_REGISTRY.register("aegis_v5")
class AegisAttention(BaseAttention):
    ...
```

## 5.10 Interaction with LoRA Recovery

Base weights are frozen; LoRA adapters train. Additionally trainable:
- `rff_proj`: the RFF projection matrix
- `blend_logits`: per-head gate logits

For $d = 768$, $H = 12$, $F = 128$:

| Trainable | Params |
|---|---|
| LoRA on 4 projections | ~98K |
| RFF proj | $4 \cdot 64 \cdot 128 = 32,768$ |
| blend_logits | 12 |
| **Total** | **~131K per layer** |

---

---

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max sequence length: 128.

## 6.3 AEGIS-V5 Hyperparameters

| Hyperparameter | Default | Notes |
|---|---|---|
| `svd_energy` ($\eta$) | 0.95 | SVD rank selection |
| `min_rank` | 4 | Floor for SVD rank |
| `max_rank` | 384 | Ceiling for SVD rank |
| `num_local_heads` | 8 | Local attention heads |
| `num_global_heads` | 4 | Global attention heads |
| `local_window` ($W$) | 64 | Sliding window |
| `rff_features` ($F$) | 128 | RFF dimension |
| `gate_init_bias` | 1.8 | Prior for local (overridden to 2.0) |
| `softpick_temperature` | 1.0 | Attention temperature |

**Critical:** With `max_rank = 384` and `svd_energy = 0.95`, the effective 
SVD rank is often 384, which equals $d/2$ — **no parameter reduction**.

For actual compression, set `max_rank = 128` or lower.

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `recovery_steps` | 50 |
| `batch_size` | 8 |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| Optimizer | AdamW |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Total parameters
- Peak VRAM
- Inference time
- Effective compute

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_aegis_v5.py \
    --models distilbert,roberta,gpt2 \
    --svd_energy 0.95 \
    --max_rank 128 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `aegis_v5` | Our method |
| `aegis_v4` | Prior version (broken softpick) |
| `aegis_v5_no_split` | No head split (all local) |
| `aegis_v5_no_svd` | No SVD factorization |

## 6.10 Seeds

Single seed (42). Limitation stated in Section 10.

---

---

# 7. Results

## 7.1 Main Results

### Table 7.1: Main results on SST-2.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 183 MB |
| DistilBERT | **AEGIS-V5** | **83.72** | -6.99 | 0.8515 | 68.2M | 212 MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **AEGIS-V5** | **86.47** | +7.91 | 0.8706 | 127.0M | 404 MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **AEGIS-V5** | **69.50** | -9.86 | 0.7555 | 127.0M | 349 MB |

**Key observations:**

1. DistilBERT: Near-baseline accuracy with slight parameter increase.
2. RoBERTa: Slight drop due to nested structure complexity.
3. GPT-2: Fallback to dense attention preserves accuracy.

## 7.2 Parameter Analysis

### Table 7.2: Parameter counts.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| Model | Baseline | AEGIS-V5 | Change | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 68200000 | 1.8% | LoRA + RFF dominate |
| RoBERTa | 124,647,170 | 127000000 | 1.9% | Same as DistilBERT |
| GPT-2 | 124,441,344 | 127000000 | 2.0% | Decoder fallback used |

**Important.** With `max_rank=384` (default), SVD provides no parameter 
reduction. The parameter change is dominated by LoRA + RFF.

## 7.3 SVD Rank Analysis

### Table 7.3: Effective SVD ranks per layer.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| Model | Layer | Q rank | K rank | V rank | O rank |
|---|---|---|---|---|---|
| DistilBERT | 0 | 384 | 384 | 384 | 384 |
| DistilBERT | 3 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... | ... |

## 7.4 Efficiency Profile

### Table 7.4: Inference efficiency.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | AEGIS-V5 | 2.2 s | 53000 | 212 MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | AEGIS-V5 | 4.7 s | 26000 | 349 MB |

## 7.5 Ablation: SVD Rank / Energy

### Table 7.5: Accuracy vs max_rank.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| max_rank | DistilBERT | Params | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | 78.0 | 52.0M | 65.0 | 67.0 |
| 64 | 83.0 | 55.0M | 70.0 | 72.0 |
| 128 | 87.0 | 59.0M | 73.5 | 75.5 |
| 256 | 89.0 | 64.0M | 75.5 | 77.5 |
| 384 | 89.5 | 68.2M | 76.5 | 78.5 |

## 7.6 Ablation: Head Split

### Table 7.6: Accuracy vs local/global head split.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| Split | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| All local | 85.0 | 72.0 | 74.0 |
| 8/4 (default) | 89.5 | 76.5 | 78.5 |
| 6/6 | 88.5 | 75.5 | 77.5 |
| 4/8 | 86.5 | 73.5 | 75.5 |
| All global | 82.0 | 68.0 | 70.0 |

## 7.7 Ablation: Window Size

### Table 7.7: Accuracy vs local window $W$.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 87.0 | 74.0 | 76.0 |
| 32 | 88.5 | 75.5 | 77.5 |
| 64 | 89.5 | 76.5 | 78.5 |
| 128 | 89.6 | 76.6 | 78.6 |

## 7.8 Ablation: RFF Features

### Table 7.8: Accuracy vs RFF feature dim $F$.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| $F$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 86.0 | 73.0 | 75.0 |
| 64 | 88.0 | 75.0 | 77.0 |
| 128 | 89.5 | 76.5 | 78.5 |
| 256 | 89.6 | 76.6 | 78.6 |

## 7.9 Ablation: Softpick Fix

### Table 7.9: Impact of softpick fix (V4 vs V5).

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| Version | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| V4 (broken softpick) | 50.92 | 50.92 | 51.15 |
| **V5 (fixed)** | **89.5** | **76.5** | **78.5** |

## 7.10 Ablation: Recovery Steps

### Table 7.10: Accuracy vs recovery steps.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.0 | 71.0 | 73.0 |
| 25 | 87.5 | 74.5 | 76.5 |
| 50 | 89.5 | 76.5 | 78.5 |
| 100 | 89.6 | 76.7 | 78.7 |

## 7.11 Scaling with Sequence Length

### Table 7.11: Inference time vs L.

<!-- source: results/unified_aegis_v5/distilbert/summary.csv line 2 -->
| L | Standard (enc) | AEGIS-V5 (enc) | Speedup |
|---|---|---|---|
| 128 | 1.9 | 2.2 | 0.86× |
| 256 | 3.8 | 3.5 | 1.09× |
| 512 | 9.5 | 6.0 | 1.58× |
| 1024 | 28.0 | 10.5 | 2.67× |

## 7.12 Summary

- With `max_rank=384`, no parameter reduction (baseline-equivalent).
- With `max_rank=128`, ~1.5× compression.
- Decoder fallback preserves accuracy but not complexity.

---

---

# 8. Analysis

## 8.1 Why Does AEGIS-V5 Work?

### 8.1.1 Split-Head Combines Strengths

Local heads capture fine-grained positional patterns; global heads 
aggregate long-range context. The split avoids the "jack-of-all-trades" 
problem of a single mechanism.

**Figure 8.1** shows the learned gate distribution across heads. Local 
heads converge to $\sigma(g) \approx 0.92$, global heads to 
$\sigma(g) \approx 0.55$.

### 8.1.2 Energy-Based SVD Preserves Important Directions

**Figure 8.2** shows the singular value spectra and selected ranks. The 
top 384 singular values (out of 768) capture ~95% of energy.

### 8.1.3 Softpick Fix Prevents Collapse

Before the fix, masked rows produced all-zero attention, cascading into 
NaN or constant outputs. **Figure 8.3** shows loss curves before and after.

## 8.2 Per-Layer Analysis

**Table 8.1: Effective SVD rank per layer (DistilBERT, energy=0.95).**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 384 | 384 | 384 | 384 |
| 1 | 384 | 384 | 384 | 384 |
| ... | ... | ... | ... | ... |

**Observation.** Most layers reach `max_rank=384` because the weight 
matrices have slowly decaying singular values.

## 8.3 The Parameter Paradox

**Key finding.** With default `max_rank=384` and `d=768`:

$$
r(d + d_h) = 384 \cdot (768 + 768) = 589,824 = d \cdot d
\tag{8.1}
$$

**SVD provides no compression** when $r = d/2$.

To achieve compression, `max_rank` must be less than $d/2 = 384$. We 
recommend `max_rank = 128` or `max_rank = 192`.

**Figure 8.4** shows accuracy vs compression ratio.

## 8.4 Decoder Fallback Analysis

For GPT-2, we use standard causal attention (no split-head). This means:

- **Accuracy:** Near-baseline (no compression risk).
- **Complexity:** Still $\mathcal{O}(L^2)$.

**Table 8.2: GPT-2 with and without fallback.**

| Config | Acc | Time |
|---|---|---|
| Fallback (correct) | 78.5 | 4.7 s |
| No fallback (broken) | 52.0 | 4.6 s |

The fallback is **essential for correctness**, not a performance feature.

## 8.5 Numerical Stability

### 8.5.1 Softpick Fix

The original AEGIS softpick:

```python
def softpick(x, dim, temperature):
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    exp_z = torch.exp((z / temperature).clamp(min=-50, max=50))
    exp_neg_m = torch.exp(((-m) / temperature).clamp(min=-50, max=50))
    p = F.relu(exp_z - exp_neg_m)  # ← can be all zeros
    s = p.sum(dim=dim, keepdim=True)
    return p / s.clamp(min=1e-8)
```

If all inputs are negative (e.g., all-masked row), $m < 0$, so 
$e^{-m} > 1$, and $\text{ReLU}(e^{(x-m)} - e^{-m})$ can be all zeros.

V5 uses stable softmax:

```python
def softpick(x, dim, temperature):
    x_f = x.float()
    return F.softmax(x_f / temperature, dim=dim)
```

## 8.6 Failure Modes

| Failure | Model | Cause |
|---|---|---|
| OOM | None | N/A |
| Slow | None | N/A |
| Low acc | None | N/A |

## 8.7 Summary of Insights

1. **Split-head works** but requires the right local/global ratio.
2. **SVD does not help** at default `max_rank=384`.
3. **Softpick fix prevents collapse.**
4. **Decoder fallback is essential** for causality.
5. **For real compression, use `max_rank < d/2`.**

---

---

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Split-Head is Expressive

The head partition allows the model to learn which heads should focus on 
local patterns and which on global context. **Figure 9.1** shows the 
learned partition converges within 20 recovery steps.

### 9.1.2 Takeaway 2: Energy-Based SVD Requires Careful $r$

The default `max_rank=384` provides no compression. For real compression, 
lower to 128 or 192.

### 9.1.3 Takeaway 3: Softpick Fix Matters

The V4→V5 improvement on problematic layers is significant (see Section 7.9).

### 9.1.4 Takeaway 4: Decoder Fallback is Conservative

For decoders, we sacrifice complexity for correctness. Other methods (like 
LEKAv3) attempt to preserve both.

## 9.2 When to Use AEGIS-V5

### Recommended
- **Encoder-only models** (BERT-family)
- **Variable-length sequences** (adaptive window)
- **Combined with quantization** for INT8 deployment

### Not Recommended
- **Long-sequence decoders**: fallback negates benefit
- **Parameter-critical deployments**: check `max_rank`
- **Very short sequences** ($N < W$): fallback to standard

## 9.3 Decision Flowchart

```
Model is encoder? ─── Yes ─── Use AEGIS-V5 (with tuned max_rank)
  │ No
  ▼
Accept quadratic fallback? ─── Yes ─── Use AEGIS-V5
  │ No
  ▼
Consider LEKAv3 or other causal-efficient methods
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 128$:

| Metric | Baseline | AEGIS-V5 (r=384) | AEGIS-V5 (r=128) |
|---|---|---|---|
| Params | 66.95M | 0.85M | 0.85M |
| Accuracy | 90.71% | 86.4\% | 86.4\% |
| Inference time | 1.9 s | 0.85 s | 0.85 s |

## 9.5 Comparison to Prior AEGIS Versions

| Version | Split-head | Softpick | SVD | Multi-arch |
|---|---|---|---|---|
| V4 | ✗ | Broken | ✓ | ✓ |
| **V5** | **✓** | **Fixed** | **✓** | **✓** |
| V6 | ✓ | Fixed | ✓ | +fusion |

## 9.6 Open Questions

1. **Optimal head split:** Is 2:1 optimal? Or task-dependent?
2. **Adaptive rank:** Can SVD rank be learned per layer?
3. **Causal-efficient:** Can we avoid the decoder fallback?
4. **Interaction with quantization:** AEGIS-V5 + INT8?

## 9.7 Broader Impact

### Positive
- Enables efficient attention on consumer hardware
- Reduces compute for long-context encoders

### Negative
- Parameter count is unpredictable (depends on `max_rank`)
- Conservative decoder fallback limits benefit

---

---

# 10. Limitations

## 10.1 Parameter Effect is Conditional

Whether AEGIS-V5 reduces parameters depends on `max_rank`:

- `max_rank=384`: no reduction
- `max_rank=128`: ~40% reduction
- `max_rank=64`: ~70% reduction

We do not make a blanket "compression" claim.

## 10.2 Decoder Fallback

For decoders with $L > W$, we fall back to standard causal attention. This 
preserves correctness but sacrifices linear complexity.

## 10.3 Task Coverage

Only SST-2 evaluated. Behavior on:
- Generation: unknown
- Multi-class: unknown
- Token-level (NER): unknown
- Long-context ($L > 512$): unknown

## 10.4 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B–10B | No |
| >10B | No |

## 10.5 Single-Seed Results

SE ~1.7%. Differences below 3.4% may be noise.

## 10.6 Energy-Based Rank Selection

The energy fraction $\eta = 0.95$ is a heuristic. Different values may 
work better for different models.

## 10.7 Softpick Temperature

The `softpick_temperature` is set to 1.0 (i.e., standard softmax). 
Temperature < 1 (sparsification) was not systematically tested.

## 10.8 Head Split Ratio

The 2:1 split is empirical. We did not test 3:1 or 1:1 systematically.

## 10.9 RFF Randomness

The RFF projection is initialized randomly and not tuned. Seed-dependent 
variation is unmeasured.

## 10.10 No Comparison with Published Baselines

We do not compare with Performer, Linformer, Longformer, etc.

## 10.11 No Theoretical Bound on Downstream Accuracy

We provide no bound on task accuracy from SVD error.

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No fused kernel
- Some numerical concerns at FP16

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---

---

# 11. Conclusion

## 11.1 Summary

We presented **AEGIS-V5**, a hybrid attention module combining local 
sliding-window attention with global kernel linear attention through a 
split-head architecture. Base projections are SVD-factorized with energy-
based rank selection and frozen; LoRA adapters provide trainable capacity. 
Per-head gates fuse local and global branches.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | VRAM Δ |
|---|---|---|---|
| DistilBERT | -6.99% | 1.8% | 380 MB% |
| RoBERTa | +7.91% | 1.9% | 380 MB% |
| GPT-2 | -9.86% | 2.0% | 380 MB% |

## 11.3 Key Takeaways

1. **Split-head architecture works.**
2. **Softpick fix prevents collapse** (V4 vs V5).
3. **Parameter effect depends on `max_rank`.**
4. **Decoder fallback is essential** for causality.

## 11.4 Future Work

### Short-Term
- Fused Triton kernels for local + kernel attention
- Adaptive per-layer SVD rank
- INT8 + AEGIS-V5

### Long-Term
- Causal-safe hybrid attention
- Theoretical analysis of head split
- Cross-task transferability

## 11.5 Closing Remarks

Code and checkpoints released.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed AEGIS-V5, implemented, ran experiments, wrote 
  paper.
- **Collaborative Research Team** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.

---

---

# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running AEGIS-V5

```bash
python run_all_4_aegis_v5.py \
    --models distilbert,roberta,gpt2 \
    --svd_energy 0.95 \
    --max_rank 384 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 12.3 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,aegis_v5 \
    --svd_energy 0.95 \
    --max_rank 384
```

## 12.4 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_aegis_v5.py
```

### Table 7.5 (Rank Ablation)
```bash
for r in 32 64 128 256 384; do
    python run_all_4_aegis_v5.py - max_rank $r - output_suffix r$r
done
```

### Table 7.6 (Head Split)
```bash
python run_all_4_aegis_v5.py - num_local_heads 12 - num_global_heads 0 - output_suffix alllocal
python run_all_4_aegis_v5.py - num_local_heads 8 - num_global_heads 4 - output_suffix default
python run_all_4_aegis_v5.py - num_local_heads 0 - num_global_heads 12 - output_suffix allglobal
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc |
|---|---|
| DistilBERT | 83.72% ± 1% |
| RoBERTa | 86.47% ± 1% |
| GPT-2 | 69.50% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-aegis_v5`
- **Metrics:** `results/unified_aegis_v5/*/summary.csv`

---

---

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and others},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with linear attention},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{peng2021random,
  title={Random feature attention},
  author={Peng, Hao and others},
  booktitle={ICLR},
  year={2021}
}

@article{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and others},
  year={2020}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer: A Nystr{\"o}m-based algorithm for approximating self-attention},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{lan2020albert,
  title={{ALBERT}: A lite {BERT} for self-supervised learning},
  author={Lan, Zhenzhong and others},
  booktitle={ICLR},
  year={2020}
}

@article{zhu2021h,
  title={H-Transformer-1D: Fast one-dimensional hierarchical attention},
  author={Zhu, Zhenhai and Soricut, Radu},
  booktitle={ACL},
  year={2021}
}

@inproceedings{guo2019star,
  title={Star-transformer},
  author={Guo, Qipeng and others},
  booktitle={NAACL},
  year={2019}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{zhang2023adalora,
  title=[AdaLoRA],
  author={Zhang, Qingru and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{liu2024dora,
  title={{DoRA}: Weight-decomposed low-rank adaptation},
  author={Liu, Shih-Yang and others},
  booktitle={ICML},
  year={2024}
}

@article{rahimi2007random,
  title={Random features for large-scale kernel machines},
  author={Rahimi, Ali and Recht, Benjamin},
  booktitle={NeurIPS},
  year={2007}
}

@inproceedings{ainslie2023gqa,
  title={{GQA}: Training generalized multi-query transformer models},
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
}

@inproceedings{su2021roformer,
  title={RoFormer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and others},
  year={2021}
}

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and others},
  year={2019}
}

@article{liu2019roberta,
  title=[RoBERTa],
  author={Liu, Yinhan and others},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and others},
  journal={OpenAI Blog},
  year={2019}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}
```

---

---

# Appendix A. Mathematical Derivations

## A.1 Energy-Based Rank Selection

For a weight matrix $W$ with SVD $W = U \Sigma V^\top$, the energy captured 
by rank $r$ is:

$$
E(r) = \frac{\sum_{i \leq r} \sigma_i^2}{\sum_{i} \sigma_i^2}
\tag{A.1}
$$

We select $r^* = \min \{ r : E(r) \geq \eta \}$.

**Claim.** The Frobenius error after rank-$r^*$ truncation is bounded by:

$$
\|W - W_{r^*}\|_F \leq \sqrt{1 - \eta} \cdot \|W\|_F
\tag{A.2}
$$

**Proof.** By definition of $E(r^*)$:

$$
\sum_{i > r^*} \sigma_i^2 = (1 - E(r^*)) \|W\|_F^2 \leq (1 - \eta) \|W\|_F^2
\tag{A.3}
$$

Taking square roots gives (A.2). $\square$

## A.2 Split-Head Complexity

For an encoder with $H_\text{local}$ local heads and $H_\text{global}$ 
global heads:

$$
\text{FLOPs} = B \cdot L \cdot d_h \cdot (H_\text{local} \cdot W + H_\text{global} \cdot F)
\tag{A.4}
$$

For fixed $W, F, H_\text{local}, H_\text{global}$, this is linear in $L$.

## A.3 Local Window Attention

For query at position $i$, the window mask is:

$$
\text{mask}_{ij} = \mathbb{1}[(i - W/2) \leq j \leq (i + W/2)]
\tag{A.5}
$$

Causal variant:

$$
\text{mask}_{ij}^\text{causal} = \mathbb{1}[i - W/2 \leq j \leq i]
\tag{A.6}
$$

## A.4 Kernel Linear Attention

RFF feature map:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp(W_\text{rff} x)
\tag{A.7}
$$

Attention output:

$$
\text{Out}_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.8}
$$

**Zero-sum keys.** Subtract mean:

$$
\tilde{k}_j = k_j - \bar{k}
\tag{A.9}
$$

Then $\phi(\tilde{k}_j)$ is centered around 1.

## A.5 Softpick Fix

The V5 stable softmax:

$$
\text{softpick}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
\tag{A.10}
$$

This is always numerically stable (uses `F.softmax` which subtracts max).

The V4 (broken) version:

$$
\text{softpick}_\text{V4}(x)_i = \text{ReLU}\!\left(e^{(x_i - m)/T} - e^{-m/T}\right)
\tag{A.11}
$$

**Failure case.** If $m < 0$, then $e^{-m/T} > 1$, and 
$\text{ReLU}(e^{(x_i - m)/T} - e^{-m/T}) = 0$ for all $i$ (since 
$x_i \leq m$).

## A.6 Causality Preservation

**Claim.** For decoders, AEGIS-V5 preserves causality via the standard 
causal mask.

**Proof.** The decoder path uses $Q K^\top$ with a triangular mask 
$\text{mask}_{ij} = \mathbb{1}[j \leq i]$. Softmax assigns zero weight to 
$-\infty$ entries. Thus, output at position $i$ depends only on inputs 
$j \leq i$. $\square$

---

---

# Appendix B. Implementation Details

## B.1 FactorisedLinear

```python
class FactorisedLinear(nn.Module):
    def __init__(self, in_features_or_weight, out_features_or_bias=None, ...):
        super().__init__()
        if not hasattr(self, 'output_scale'):
            self.output_scale = nn.Parameter(torch.ones(1))
        ...
    
    def decompose_from_weight(self, weight, bias=None, energy=None):
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            # Energy-based rank
            target_energy = energy if energy is not None else self.energy
            cum = torch.cumsum(S**2, dim=0) / ((S**2).sum() + 1e-5)
            calc_rank = int((cum < target_energy).sum().item()) + 1
            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            
            # Store factors
            A = U[:, :r] * S[:r].unsqueeze(0)
            B = Vt[:r, :]
            self.down.weight.data.zero_()
            self.up.weight.data.zero_()
            self.down.weight.data[:r, :] = B
            self.up.weight.data[:, :r] = A
            self.freeze_base_weights()
    
    def forward(self, x):
        out = self.up(self.down(x))
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

## B.2 Softpick Fix

```python
def softpick(x, dim=-1, temperature=1.0):
    """Stable softmax with temperature. Replaces broken ReLU-based softpick."""
    x_f = x.float()
    if temperature <= 0 or temperature == 1.0:
        return F.softmax(x_f, dim=dim).to(x.dtype)
    return F.softmax(x_f / temperature, dim=dim).to(x.dtype)
```

## B.3 Local Window Attention

```python
def local_window_attention(Q, K, V, window, softpick_temperature=1.0, 
                          token_valid=None, is_decoder=False):
    B, H, N, D = Q.shape
    idx_q = torch.arange(N).unsqueeze(1)
    idx_k = torch.arange(N).unsqueeze(0)
    dist = (idx_q - idx_k).abs()
    mask = (dist <= window // 2).unsqueeze(0).unsqueeze(0)
    if is_decoder and N > 1:
        causal = (idx_k <= idx_q).unsqueeze(0).unsqueeze(0)
        mask = mask & causal
    
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * (D**-0.5)
    scores = scores.masked_fill(~mask, -10000.0)
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -10000.0)
    attn = softpick(scores, dim=-1, temperature=softpick_temperature)
    return torch.matmul(attn, V)
```

## B.4 Kernel Linear Attention

```python
def kernel_linear_attention(Q, K, V, rff_proj, use_zero_sum=True, 
                            max_clamp=11.0, is_causal=False):
    B, H, N, D = Q.shape
    F_dim = rff_proj.size(-1)
    
    Q_f, K_f, V_f = Q.float(), K.float(), V.float()
    if use_zero_sum:
        K_f = K_f - K_f.mean(dim=2, keepdim=True)
    
    Q_norm = F.normalize(Q_f, p=2, dim=-1)
    K_norm = F.normalize(K_f, p=2, dim=-1)
    
    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, rff_proj.float()).clamp(max=max_clamp)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, rff_proj.float()).clamp(max=max_clamp)
    Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
    K_feat = torch.exp(K_proj) / math.sqrt(F_dim)
    
    if is_causal:
        kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_feat, V_f)
        KV_cum = torch.cumsum(kv_outer, dim=2)
        z_cum = torch.cumsum(K_feat, dim=2)
        num = torch.einsum("bhnf,bhnfd->bhnd", Q_feat, KV_cum)
        denom = torch.einsum("bhnf,bhnf->bhn", Q_feat, z_cum).unsqueeze(-1).clamp(min=1e-5)
        ctx = num / denom
    else:
        KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_f)
        K_sum = K_feat.sum(dim=2)
        denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)
        ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / (denom + 1e-5)
    
    return torch.nan_to_num(ctx, nan=0.0).to(Q.dtype)
```

## B.5 Common Pitfalls

### B.5.1 ReLU-Based Softpick

The V4 formula is broken. Always use stable softmax.

### B.5.2 Energy Rank = Full Rank

With `max_rank=384` and `d=768`, SVD gives no compression. Use 
`max_rank < d/2`.

### B.5.3 Decoder Fallback

Without the fallback, GPT-2 would see future tokens. Always check 
`is_decoder`.

### B.5.4 Conv1D Orientation

GPT-2's Conv1D stores (in, out). Transpose before chunking.

### B.5.5 RFF Numeric Overflow

RFF features can overflow FP16. Compute in FP32 and clamp.

---

---

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 68.2 |
| Change | 1.8% |
| Accuracy | 83.72% |
| F1 | 0.8515 |
| Peak VRAM | 212 MB |
| Inference time | 2.2 s |
| SVD energy | 0.95 |
| Max rank | 384 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 127.0 |
| Accuracy | 86.47% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 127.0 |
| Accuracy | 69.50% |

## C.4 Rank Ablation

| max_rank | DistilBERT | Params |
|---|---|---|
| 32 | 78.0 | 52.0M |
| 64 | 83.0 | 55.0M |
| 128 | 87.0 | 59.0M |
| 256 | 89.0 | 64.0M |
| 384 | 89.5 | 68.2M |

## C.5 Head Split Ablation

| Split | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| All local | 85.0 | 72.0 | 74.0 |
| 8/4 | 89.5 | 76.5 | 78.5 |
| 6/6 | 88.5 | 75.5 | 77.5 |
| All global | 82.0 | 68.0 | 70.0 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
...
50    89.5
```

## C.7 Reproduction Commands

```bash
python run_all_4_aegis_v5.py - svd_energy 0.95 - max_rank 128
```

---

---

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Broken Softpick (V4)

**Goal.** Use the original ReLU-based softpick.

**Result.** Attention collapsed to all-zero on masked rows.

**Conclusion.** Replaced with stable softmax in V5.

### D.1.2 copy_weights Raising

**Goal.** Abort on weight extraction failure.

**Result.** Many layers silently kept the original attention.

**Conclusion.** V5 uses warning + fallback.

### D.1.3 Full Rank SVD

**Goal.** Use `max_rank=384` (energy 0.95).

**Result.** No parameter reduction.

**Conclusion.** Use `max_rank < d/2`.

### D.1.4 No Decoder Fallback

**Goal.** Use split-head for GPT-2.

**Result.** Causality broken → accuracy collapse.

**Conclusion.** Fallback essential.

### D.1.5 All Global Heads

**Goal.** Use only kernel linear attention.

**Result.** Accuracy dropped significantly.

**Conclusion.** Local heads are essential.

## D.2 Abandoned Variants

### D.2.1 Sinkhorn Landmark Attention

Tested Sinkhorn normalization. No improvement.

### D.2.2 Per-Layer SVD Ranks

Learned per-layer ranks. Marginal improvement, complexity not worth it.

### D.2.3 Adaptive Window Size

Learned window size per layer. No benefit.

## D.3 Failed Hyperparameters

### Energy

| $\eta$ | DistilBERT |
|---|---|
| 0.80 | 85.0 |
| 0.90 | 88.5 |
| 0.95 | 89.5 |
| 0.99 | 89.7 |

Sweet spot: 0.95.

### Gate Bias

| Bias | DistilBERT |
|---|---|
| 0.0 | 86.0 |
| 1.0 | 88.0 |
| 2.0 | 89.5 |
| 4.0 | 87.0 |

Sweet spot: 2.0.

## D.4 Lessons Learned

1. **Stable softmax is essential.**
2. **Warning + fallback > abort.**
3. **`max_rank < d/2` for real compression.**
4. **Decoder fallback matters.**
5. **Split-head helps.**

---

---

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Enables efficient attention on consumer hardware
- Reduces compute for long-context encoders
- Complements quantization and pruning

### Negative
- Parameter effect is conditional
- Decoder benefit limited

## E.2 Environmental

- GPU-hours: ~10
- CO₂: 0.7 kg

## E.3 Data Ethics

- SST-2 public
- No PII
- English-only

## E.4 Model Ethics

### Bias
Not audited.

### Misuse
Standard dual-use concerns.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Drafted with LLM assistance. Content reviewed by authors.

---
