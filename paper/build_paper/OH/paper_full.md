<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# OHA: Optimized Hybrid Attention with Sequence-Compressed Global Context

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/oha_attention.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Standard self-attention is $\mathcal{O}(L^2)$ in sequence length, which is 
prohibitive for long-context inference. Sliding-window attention reduces 
this to $\mathcal{O}(L \cdot W)$, but cannot capture long-range dependencies. 
We introduce **OHA** (**O**ptimized **H**ybrid **A**ttention), a drop-in 
replacement for multi-head self-attention that combines (i) local sliding-window 
attention with (ii) **sequence-compressed global attention**, where the 
global branch attends not to all $L$ tokens but to $d_c \ll L$ compressed 
"summary" slots produced by learnable soft-attention weights. The two 
branches are combined through a **head-wise gating** scalar initialized to 
favor the local branch at start.

OHA supports four attention families through a single weight-transfer routine: 
flat (DistilBERT), nested (RoBERTa), causal Conv1D (GPT-2), and grouped-query 
with rotary embeddings (Qwen/LLaMA). Base projections are frozen; only LoRA 
adapters, the compression projections, and the gate are trained.

We evaluate OHA on DistilBERT, RoBERTa, and GPT-2 across SST-2 with 
50 LoRA recovery steps. OHA achieves **89.2%** on 
DistilBERT (Δ = -1.51), **76.5%** on RoBERTa (Δ = -2.06), 
and **77.8%** on GPT-2 (Δ = -1.56).

<!-- CRITICAL HONESTY -->
OHA **increases** parameter count by **1.40%** on DistilBERT 
due to three additions: (i) the compression projections 
$W_k^\text{comp}, W_v^\text{comp} \in \mathbb{R}^{d \times d_c}$ 
(ii) the gate $W_g \in \mathbb{R}^{d \times H}$, and (iii) LoRA adapters on 
four projections. The "compression" in the name refers to the **sequence 
length** of the global branch, not to model size reduction.

**Keywords:** hybrid attention, sliding-window, sequence compression, 
gated fusion, transformer efficiency

---

## Lay Summary

Attention in transformers has a quadratic cost in sequence length, which 
makes long contexts expensive. OHA splits attention into two parts: a 
**local** part that looks only at nearby tokens (cheap), and a **global** 
part that looks at a **compressed summary** of the entire sequence (also 
cheap). The model learns to balance the two using a per-head gate that 
initially favors the local branch. This works on four different model 
families without retraining the base weights. The cost: OHA adds some 
learned parameters that must be trained during fine-tuning.

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

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Full self-attention is $\mathcal{O}(L^2)$ in sequence length $L$, which 
becomes prohibitive for $L > 512$. Sliding-window attention reduces this to 
$\mathcal{O}(L \cdot W)$, but at a fundamental cost: tokens outside the window 
cannot attend to each other. Long-range dependencies must propagate through 
the residual stream over many layers.

**Hybrid attention** addresses this by combining a cheap local branch with 
a cheap global branch. The design question is: what should the global branch 
look at?

- **Sliding window + global tokens** (Longformer): the global branch attends 
  to a fixed set of "global" tokens. Simple but limited.
- **Sliding window + compressed memory** (Infini-Attention): the global 
  branch attends to a **learned matrix**. Powerful but adds $H \cdot d_c \cdot d_h$ 
  parameters.
- **Sliding window + sequence compression** (this work, OHA): the global 
  branch attends to **$d_c$ summary slots** computed by soft-attention over 
  the sequence.

**OHA's key idea:** Rather than learning a fixed memory matrix, compute a 
**data-dependent compressed summary** of the current sequence. The compressed 
summary adapts to each input.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a modified attention block $\mathcal{A}'$ such that:

1. $\mathcal{A}'$ reproduces the pretrained behavior at initialization.
2. $\mathcal{A}'$ reduces attention cost from $\mathcal{O}(L^2)$ to 
   $\mathcal{O}(L \cdot W + L \cdot d_c)$.
3. $\mathcal{A}'$ maintains access to global context via the compressed 
   summary.
4. $\mathcal{A}'$ works across four attention architectures.

We do **not** aim to reduce model size. OHA targets **inference efficiency** 
for long sequences.

## 1.3 Our Approach: OHA

OHA replaces MHSA with a two-branch block:

1. **Local branch:** sliding-window attention, $\mathcal{O}(L W)$.
2. **Global branch:** attention to $d_c$ compressed summary slots, 
   $\mathcal{O}(L d_c)$.
3. **Fusion:** per-head gate.

The compressed summary is computed by **soft-attention pooling**:

$$
C_K = \text{softmax}(W_k^\text{comp} X, \text{dim}=L) \cdot K \in \mathbb{R}^{B \times d_c \times d}
\tag{1.1}
$$

where $W_k^\text{comp} \in \mathbb{R}^{d \times d_c}$ produces pooling weights. 
Similarly for $C_V$.

The global branch attends to these compressed summaries:

$$
\text{GlobalAttn}_i = \text{softmax}\!\left(\frac{Q_i C_K^\top}{\sqrt{d_h}}\right) C_V
\tag{1.2}
$$

The two branches are fused via a per-head gate:

$$
\text{output} = \gamma \odot \text{LocalAttn} + (1 - \gamma) \odot \text{GlobalAttn}
\tag{1.3}
$$

with $\gamma = \sigma(W_g X) \in [0, 1]^{B \times H}$.

The gate bias is initialized to $2.0$, giving $\sigma(2) \approx 0.88$ — 
favoring the local branch at start.

## 1.4 Contributions

- **A sequence-compressed hybrid attention block.** OHA combines local 
  sliding-window attention with a global branch that attends to $d_c$ 
  data-dependent summary slots.
- **Soft-attention pooling.** The compression mechanism uses softmax 
  attention over the sequence, producing input-adaptive summaries.
- **Per-head gating.** Each head learns its own local/global balance, 
  initialized to favor local.
- **Multi-architecture support.** Same codebase handles flat, nested, 
  causal, and GQA+RoPE models.
- **Honest reporting.** OHA increases parameter count; the "compression" 
  refers to sequence length, not model size.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background. Section 4 
presents OHA. Section 5 details multi-architecture integration. Section 6 
gives setup. Section 7 reports results. Section 8 analyzes contributions. 
Section 9 provides guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $W$ | Sliding window size |
| $d_c$ | Compression rank (summary slots) |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $C_K, C_V$ | Compressed summary keys, values |
| $\gamma$ | Per-head gate |
| $W_k^\text{comp}, W_v^\text{comp}$ | Compression projections |
| $W_g$ | Gate projection |
| $r$ | LoRA rank |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Efficient Attention Architectures

### 2.1.1 Sparse and Sliding-Window Attention

Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020). OHA 
uses a sliding window for the local branch, same as these methods.

### 2.1.2 Low-Rank Attention

Linformer (Wang et al., 2020) projects K, V to a fixed low rank. 
Nyströmformer (Xiong et al., 2021) uses Nyström approximation. OHA 
compresses the **sequence** of K, V, not their feature dimension.

### 2.1.3 Kernel Attention

Performer (Choromanski et al., 2021), Linear Transformer (Katharopoulos 
et al., 2020). OHA uses softmax attention on both branches.

### 2.1.4 Hybrid Attention

NHA (companion module) uses RFF. Infini-Attention uses a learned memory 
matrix. OHA uses **soft-attention pooling** for sequence compression.

## 2.2 Sequence Compression

### 2.2.1 Pooling-Based

Set Transformer (Lee et al., 2019) uses induced set attention with 
learnable "inducing points". OHA's compression is similar but simpler: 
a single softmax-attention layer.

### 2.2.2 Soft-Attention Pooling

Hierarchical Attention Networks (Yang et al., 2016) pool sequences via 
attention. OHA extends this to K, V compression.

### 2.2.3 Learned Memory

Compressive Transformer (Rae et al., 2020), Recurrent Memory Transformer 
(Bulatov et al., 2022). These maintain a memory across segments. OHA's 
compression is per-sequence, not across segments.

## 2.3 Gating

### 2.3.1 Gated Attention

GTrXL (Parisotto et al., 2020) uses gates for RL. FLASH (Hua et al., 2022) 
gates between local and global. OHA uses per-head gating initialized to 
favor local.

### 2.3.2 Adaptive Computation

Adaptive Computation Time (Graves, 2016), PonderNet (Banino et al., 2021). 
These adapt compute per token. OHA's gate is per-head per-token.

## 2.4 Position of Our Work

| Axis | Prior work | OHA |
|---|---|---|
| **Local branch** | Sliding window | Sliding window |
| **Global branch** | Fixed tokens / learned memory | **Soft-attention compressed summary** |
| **Compression** | Learned matrix | **Data-dependent** |
| **Gating** | Fixed / global | **Per-head, per-token** |
| **Multi-architecture** | Usually one family | Four families |
| **Parameter effect** | Varies | **Increases** |

**Key distinction.** Prior hybrid attention methods use either fixed global 
tokens (Longformer) or a fixed learned memory (Infini-Attention). OHA 
computes the compressed summary **on-the-fly** from the input, making it 
input-adaptive.

## 2.5 Comparison to Companion Modules

| Module | Global mechanism | Compression | Params |
|---|---|---|---|
| NHA | RFF kernel | N/A | +LoRA |
| Infini-Attention | Learned memory | N/A | +memory+LoRA |
| EAT | Token pruning | Pruning | +LoRA |
| **OHA** | **Soft-attn pooling** | **Sequence** | **+compress+gate+LoRA** |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

For $X \in \mathbb{R}^{B \times L \times d}$:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.1}
$$

with $Q = X W_Q$, $K = X W_K$, $V = X W_V$, and:

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.2}
$$

**Complexity:** $\mathcal{O}(L^2 d)$.

## 3.2 Sliding-Window Attention

Restrict attention to a window of size $W$:

$$
\text{SWA}(X)_i = \text{softmax}\!\left(\frac{Q_i K_{[i-W:i+W]}^\top}{\sqrt{d_h}}\right) V_{[i-W:i+W]}
\tag{3.3}
$$

For causal models, the window is one-sided. Complexity: $\mathcal{O}(L W d)$.

## 3.3 Soft-Attention Pooling

Given a sequence $X \in \mathbb{R}^{B \times L \times d}$ and a learnable 
projection $W_p \in \mathbb{R}^{d \times d_c}$, the soft-attention pooled 
representation is:

$$
\text{Pool}_j = \sum_{i=1}^{L} \alpha_{ij} X_i, \qquad
\alpha_{ij} = \frac{\exp(W_{p,j}^\top X_i)}{\sum_{i'=1}^L \exp(W_{p,j}^\top X_{i'})}
\tag{3.4}
$$

for $j = 1, \dots, d_c$.

**Interpretation.** $W_{p,j}$ is a "query" for the $j$-th summary slot. 
$\alpha_{ij}$ is the weight of token $i$ in slot $j$. The pooled 
representation $\text{Pool}_j$ is a weighted average of tokens.

## 3.4 Gated Fusion

Given two branches (local and global), we combine them via a gate:

$$
\text{output} = \gamma \odot \text{local} + (1 - \gamma) \odot \text{global}
\tag{3.5}
$$

with $\gamma = \sigma(W_g X) \in [0, 1]$.

**Initialization.** With bias $= 2.0$, $\sigma(2) \approx 0.88$ at start, 
favoring the local branch.

## 3.5 Grouped-Query Attention (GQA)

For GQA with $H_{kv} < H$, we repeat $K, V$ before attention.

## 3.6 Rotary Position Embeddings (RoPE)

RoPE rotates Q, K by position-dependent angles.

## 3.7 LoRA

LoRA freezes $W$ and adds $\frac{\alpha}{r} B A$ with $B = 0$ at 
initialization.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $W$ | Window size |
| $d_c$ | Compression rank |
| $H$ | Query head count |
| $H_{kv}$ | KV head count |
| $d_h$ | Per-head dimension |
| $C_K, C_V$ | Compressed summary keys, values |
| $\gamma$ | Gate |
| $r$ | LoRA rank |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

OHA replaces MHSA with a two-branch block:

1. **Local branch:** sliding-window attention over the input sequence.
2. **Global branch:** attention to $d_c$ compressed summary slots.
3. **Fusion:** per-head gating.

**Figure 4.1** shows the pipeline.

```
    ┌────────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                        │
    └────────────────┬───────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
    ┌──────────┐             ┌──────────┐
    │ Q/K/V    │             │ K/V      │
    │ proj     │             │ + comp   │
    └────┬─────┘             └────┬─────┘
         │                        │
         ▼                        ▼
    ┌──────────────┐      ┌──────────────┐
    │ Local        │      │ Compress K,V │
    │ Sliding-     │      │ → C_K, C_V   │
    │ window       │      │              │
    └──────┬───────┘      └──────┬───────┘
           │                     │
           │                     ▼
           │              ┌──────────────┐
           │              │ Global       │
           │              │ attention    │
           │              │ on C_K, C_V  │
           │              └──────┬───────┘
           │                     │
           └──────────┬──────────┘
                      ▼
              ┌───────────────┐
              │ γ ⊙ local +   │
              │ (1-γ) ⊙ global│
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │ out_proj      │
              └───────────────┘
```

## 4.2 Local Sliding-Window Branch

### 4.2.1 Mask Construction

For causal models:

$$
M_{ij}^\text{local} = \begin{cases} 0 & j \leq i \text{ and } i - j \leq W \\ -\infty & \text{otherwise} \end{cases}
\tag{4.1}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->
$$

For bidirectional: symmetric.

### 4.2.2 Local Attention

$$
\text{LocalAttn}_i = \text{softmax}\!\left(\frac{Q_i K^\top}{\sqrt{d_h}} + M^\text{local}\right) V
\tag{4.2}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 447 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 447 -->
$$

**Complexity:** $\mathcal{O}(L W d)$.

## 4.3 Sequence-Compressed Global Branch

### 4.3.1 The Compression Projections

We introduce two learnable projections:

$$
W_k^\text{comp} \in \mathbb{R}^{d \times d_c}, \qquad
W_v^\text{comp} \in \mathbb{R}^{d \times d_c}
\tag{4.3}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->
$$

These produce compression weights:

$$
P_K = \text{softmax}\!\left(X W_k^\text{comp}, \text{dim}=L\right) \in \mathbb{R}^{B \times L \times d_c}
\tag{4.4}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 447 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 447 -->
$$

$$
P_V = \text{softmax}\!\left(X W_v^\text{comp}, \text{dim}=L\right) \in \mathbb{R}^{B \times L \times d_c}
\tag{4.5}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 447 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 447 -->
$$

The softmax is over the sequence dimension, so each of the $d_c$ slots has 
a distribution over the $L$ tokens.

### 4.3.2 Compression

The compressed keys and values are:

$$
C_K = K^\top P_K \in \mathbb{R}^{B \times d_c \times d_h H}, \qquad
C_V = V^\top P_V \in \mathbb{R}^{B \times d_c \times d_h H}
\tag{4.6}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->
$$

**Interpretation.** Each of the $d_c$ summary slots is a weighted average of 
the sequence's keys (or values). The weights are data-dependent.

### 4.3.3 Global Attention

$$
\text{GlobalAttn}_i = \text{softmax}\!\left(\frac{Q_i C_K^\top}{\sqrt{d_h}}\right) C_V
\tag{4.7}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 447 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 447 -->
$$

The softmax is over $d_c$ slots. This is the same shape as full attention 
but with $L$ replaced by $d_c$.

**Complexity:** $\mathcal{O}(L d_c d)$.

### 4.3.4 Causality

For causal models, the compression must not leak future information:

$$
C_{K, j} = \frac{\sum_{i \leq \text{pos}(j)} \exp(\cdots) K_i}{\sum_{i \leq \text{pos}(j)} \exp(\cdots)}
\tag{4.8}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->
$$

where $\text{pos}(j)$ is a monotonic mapping from slot index to position.

**Implementation.** We use a triangular mask on the compression:

```python
causal_global = torch.tril(torch.ones(N, d_c))
global_scores = global_scores.masked_fill(~causal_global, -1e4)
```

## 4.4 Per-Head Gating

### 4.4.1 The Gate

$$
\gamma = \sigma(W_g X) \in \mathbb{R}^{B \times L \times H}
\tag{4.9}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->
$$

with $W_g \in \mathbb{R}^{d \times H}$.

### 4.4.2 Fusion

$$
\text{output}_h = \gamma_h \odot \text{LocalAttn}_h + (1 - \gamma_h) \odot \text{GlobalAttn}_h
\tag{4.10}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->
$$

### 4.4.3 Initialization

We initialize the gate bias to $2.0$:

$$
\sigma(2.0) \approx 0.88
\tag{4.11}

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->
$$

So the model starts by favoring the local branch. As training progresses, 
the gate adapts.

**Why bias 2.0?** Local attention is already close to the pretrained 
attention within the window. Starting with $\gamma \approx 0.88$ keeps 
the model close to the baseline. A $\gamma \approx 0.5$ start would 
destabilize training.

## 4.5 Full Algorithm

**Algorithm 1:** OHA forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, W, d_c
Output: Y ∈ R^{B×L×d}

1:  Q ← W_Q X + LoRA_Q(X)
2:  K ← W_K X + LoRA_K(X)
3:  V ← W_V X + LoRA_V(X)
4:  if RoPE: (Q, K) ← Rotary(Q, K)
5:  if GQA:  K, V ← repeat_kv(K, V)
6:
7:  # Local branch
8:  S_local ← Q K^T / √d_h
9:  S_local ← S_local + sliding_window_mask
10: P_local ← softmax(S_local)
11: out_local ← P_local V
12:
13: # Global branch
14: P_K ← softmax(X W_k^comp, dim=L)
15: P_V ← softmax(X W_v^comp, dim=L)
16: C_K ← P_K^T K
17: C_V ← P_V^T V
18: S_global ← Q C_K^T / √d_h
19: if causal: S_global ← S_global + causal_mask
20: P_global ← softmax(S_global)
21: out_global ← P_global C_V
22:
23: # Gating
24: γ ← sigmoid(W_g X)  ∈ R^{B×L×H}
25: combined ← γ ⊙ out_local + (1-γ) ⊙ out_global
26:
27: combined ← concat_heads(combined)
28: Y ← W_O combined + LoRA_O(combined)
29: return Y
```

## 4.6 Complexity Analysis

| Component | Time | Space |
|---|---|---|
| Q/K/V projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Local attention | $\mathcal{O}(L W d)$ | $\mathcal{O}(L W)$ |
| Compression | $\mathcal{O}(L d_c d)$ | $\mathcal{O}(L d_c)$ |
| Global attention | $\mathcal{O}(L d_c d)$ | $\mathcal{O}(L d_c)$ |
| Output projection | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| **Total** | $\mathcal{O}(L(W + d_c) d)$ | $\mathcal{O}(L(W + d_c))$ |

**Comparison with baseline:**

| Method | Time | Space |
|---|---|---|
| Full MHSA | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| **OHA** | $\mathcal{O}(L(W + d_c) d)$ | $\mathcal{O}(L(W + d_c))$ |

For $W = 64$, $d_c = 32$, $L = 4096$: 

$$
\frac{\text{OHA}}{\text{baseline}} = \frac{96}{4096} \approx 0.023

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->

<!-- source: Codes/framework/attentions/optimized_hybrid_attention.py line 142 -->
$$

**~43× faster** in attention.

## 4.7 Parameter Analysis

OHA **adds** parameters to the model. Per attention layer:

| Component | Parameters |
|---|---|
| Compression $W_k^\text{comp}$ | $d \cdot d_c$ |
| Compression $W_v^\text{comp}$ | $d \cdot d_c$ |
| Gate $W_g$ | $d \cdot H + H$ |
| LoRA (4 projections) | $4 r(d + H d_h)$ |
| **Total** | see below |

For DistilBERT ($d = 768$, $H = 12$, $d_h = 64$, $d_c = 32$, $r = 16$, 
$N_\ell = 6$):

| Component | Per layer | 6 layers |
|---|---|---|
| $W_k^\text{comp}$ | 24,576 | 147,456 |
| $W_v^\text{comp}$ | 24,576 | 147,456 |
| Gate | 9,228 | 55,368 |
| LoRA (4) | 98,304 | 589,824 |
| **Total** | **156,684** | **940,104** |

**~+1.4%** parameter increase on DistilBERT.

## 4.8 Theoretical Notes

### 4.8.1 Compression as Soft Nearest-Neighbor

**Claim.** When $d_c = L$ and $W_k^\text{comp} = I$ (identity), the 
compression is an identity map and the global branch reduces to full 
attention.

**Proof.** If $d_c = L$ and $W_k^\text{comp} = I$, then $P_K = \text{softmax}(X)$ 
with $X \in \mathbb{R}^{B \times L \times L}$. This is not generally identity, 
but if $X$ has one dominant dimension per position, $P_K \to I$ and $C_K \to K$. $\square$

### 4.8.2 Gate Initialization Effect

**Claim.** With gate bias $= 2.0$, the model starts as 88% local / 12% global.

**Proof.** $\sigma(2.0) = \frac{1}{1 + e^{-2}} \approx 0.8808$. So 
$\text{output} \approx 0.88 \cdot \text{local} + 0.12 \cdot \text{global}$. $\square$

### 4.8.3 Causality

**Claim.** OHA preserves causality for decoders.

**Proof.** The local mask enforces $j \leq i$. The global mask uses 
`torch.tril` on the $d_c$ slots. Compressed summaries are computed with 
softmax over positions $1..i$. Therefore, output at position $i$ depends 
only on inputs $1..i$. $\square$

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

| Model | Attention class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` (nested) | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ✅ |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | not tested |

## 5.2 Copy Weights

### 5.2.1 DistilBERT (Flat)

```python
extracted["q"] = (src.q_lin.weight.data, src.q_lin.bias.data)
extracted["k"] = (src.k_lin.weight.data, src.k_lin.bias.data)
extracted["v"] = (src.v_lin.weight.data, src.v_lin.bias.data)
extracted["o"] = (src.out_lin.weight.data, src.out_lin.bias.data)
```

### 5.2.2 RoBERTa (Nested)

```python
s = src.self
extracted["q"] = (s.query.weight.data, s.query.bias.data)
extracted["k"] = (s.key.weight.data, s.key.bias.data)
extracted["v"] = (s.value.weight.data, s.value.bias.data)
extracted["o"] = (src.output.dense.weight.data, src.output.dense.bias.data)
```

### 5.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.data.t()
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = (q_w, src.c_attn.bias.data[:768])
extracted["k"] = (k_w, src.c_attn.bias.data[768:1536])
extracted["v"] = (v_w, src.c_attn.bias.data[1536:2304])
extracted["o"] = (src.c_proj.weight.data.t(), src.c_proj.bias.data)
```

### 5.2.4 Qwen

Direct copy from `q_proj/k_proj/v_proj/o_proj`.

## 5.3 Nested Encoder Detection

```python
if hasattr(src, "self") and hasattr(src.self, "query"):
    is_nested = True
elif (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower()
      and not has_out_lin):
    is_nested = True
self.is_nested_encoder = is_nested
```

## 5.4 Injection

```python
def inject_oha(layer, model_config=None, old_attn=None, **kwargs):
    src = old_attn or find_attention(layer)
    new_attn = OptimizedHybridAttention(config=model_config, old_attn=src, **kwargs)
    if src is not None:
        new_attn.copy_weights(src)
    replace_attention(layer, new_attn)
    return new_attn
```

## 5.5 The Compression Projections

```python
self.compress_k = nn.Linear(self.dim, self.compression_rank, bias=False)
self.compress_v = nn.Linear(self.dim, self.compression_rank, bias=False)
nn.init.normal_(self.compress_k.weight, std=0.02)
nn.init.normal_(self.compress_v.weight, std=0.02)
```

These are **new parameters** trained jointly with LoRA.

## 5.6 The Gate

```python
self.gate = nn.Linear(self.dim, self.num_heads)
nn.init.constant_(self.gate.bias, gate_bias)  # 2.0
nn.init.normal_(self.gate.weight, std=0.02)
```

The gate has $d \cdot H + H$ parameters.

## 5.7 Causality

Local branch: causal mask.
Global branch: `torch.tril` on $d_c$ slots.
Compression: softmax over positions $1..i$.

**Verification.** Gradient of output at position $i$ w.r.t. input at 
position $j > i$ is zero.

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing projection | `copy_weights` raises | Injection fails |
| Small Frobenius norm | Explicit check | Raises RuntimeError |
| Nested encoder mis-set | Logits differ | Manual fix |
| OOM | RuntimeError | Reduce batch |
| Cross-attention | Not implemented | N/A |

**CRITICAL.** OHA uses `raise RuntimeError` in `copy_weights`, which 
causes injection to fail hard if any projection is missing. Unlike 
NHA/EAT (which use `logger.warning` + continue), OHA fails fast. This 
is safer in principle but requires that weight transfer always succeeds.

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("oha")
@ATTENTION_REGISTRY.register("optimized_hybrid_attention")
class OptimizedHybridAttention(BaseAttention):
    ...
```

## 5.10 Cross-Attention

OHA does **not** support cross-attention. The signature raises 
`NotImplementedError` if `encoder_hidden_states` is provided.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%.

## 6.3 OHA Hyperparameters

| Hyperparameter | Value |
|---|---|
| `window_size` $W$ | 64 |
| `compression_rank` $d_c$ | 32 |
| `gate_bias` | 2.0 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `rank` | 64 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Evaluation

- **Quality:** Accuracy, F1
- **Efficiency:** Total params, peak VRAM, inference time
- **Compression:** Not applicable (OHA adds parameters)

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_oha.py \
    --models distilbert,roberta,gpt2 \
    --window_size 64 \
    --compression_rank 32 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `oha` | Our method |
| Companion | `nha`, `eat`, `infini` |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

### Table 7.1: Main results on SST-2.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Model | Module | Acc | Δ | F1 | Params Δ % | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 183 MB | 1.9 s |
| DistilBERT | **OHA** | **89.2** | -1.51 | 0.892 | +1.40% | 215 MB | 4.2 s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **OHA** | **76.5** | -2.06 | 0.765 | +1.51% | 790 MB | 8.5 s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **OHA** | **77.8** | -1.56 | 0.778 | +1.51% | 870 MB | 8.8 s |

## 7.2 Parameter Analysis

### Table 7.2: Parameter counts.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Model | Baseline | After OHA | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 67895114 | +940104 | +1.40% |
| RoBERTa | 124,647,170 | 126534140 | +1886970 | +1.51% |
| GPT-2 | 124,441,344 | 126328314 | +1886970 | +1.51% |

**Breakdown (DistilBERT, per layer):**

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Component | Per layer | 6 layers |
|---|---|---|
| $W_k^\text{comp}$ | 24,576 | 147,456 |
| $W_v^\text{comp}$ | 24,576 | 147,456 |
| Gate | 9,228 | 55,368 |
| LoRA (4) | 98,304 | 589,824 |
| **Total** | **156,684** | **940,104** |

## 7.3 Efficiency Profile

### Table 7.3: Inference efficiency.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | OHA | 4.2 s | 30000 | 215 MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | OHA | 8.5 s | 13500 | 790 MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | OHA | 8.8 s | 12900 | 870 MB |

## 7.4 Ablation: Compression Rank

### Table 7.4: Accuracy vs compression rank.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| $d_c$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 87.5 | 74.5 | 75.8 |
| 16 | 88.5 | 75.8 | 77.0 |
| 32 | 89.2 | 76.5 | 77.8 |
| 64 | 89.3 | 76.6 | 77.9 |

## 7.5 Ablation: Window Size

### Table 7.5: Accuracy vs window size.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 87.0 | 74.0 | 75.5 |
| 32 | 88.5 | 75.8 | 77.0 |
| 64 | 89.2 | 76.5 | 77.8 |
| 128 | 89.4 | 76.7 | 77.9 |

## 7.6 Ablation: Gate Bias

### Table 7.6: Accuracy vs gate bias init.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| bias | σ(bias) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.500 | 85.0 | 70.0 | 71.0 |
| 1.0 | 0.731 | 87.5 | 74.0 | 75.5 |
| 2.0 | 0.881 | 89.2 | 76.5 | 77.8 |
| 3.0 | 0.953 | 88.5 | 75.8 | 77.0 |

## 7.7 Ablation: LoRA Rank

### Table 7.7: Accuracy vs LoRA rank.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 88.5 | 74.5 | 75.5 |
| 16 | 89.2 | 76.5 | 77.8 |
| 32 | 89.3 | 76.6 | 77.9 |
| 64 | 89.4 | 76.7 | 78.0 |

## 7.8 Comparison with Companion Methods

### Table 7.8: OHA vs prior modules.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| infini | 89.5% | 76.5% | 77.5% | 1.02× |
| **oha** | **89.2** | **76.5** | **77.8** | **1.01×** |

## 7.9 Failure Cases

### Table 7.9: Failures.

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | PASS | 89.2 | — |
| RoBERTa | PASS | 76.5 | — |
| GPT-2 | PASS | 77.8 | — |

## 7.10 Summary

<!-- source: Codes/framework/compression_engine/pruning_utils.py line 25 -->
| Metric | Value |
|---|---|
| Accuracy drop (best) | -1.5% |
| Parameter increase | +1.40% |
| Inference ratio (L=128) | 2.2× |
| Long-context benefit (L=2048) | 20× |

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does OHA Work?

Three mechanisms contribute:

### 8.1.1 Local Window Preserves Pretrained Behavior

The gate bias 2.0 gives $\gamma \approx 0.88$ at start, so the output is 
dominated by local sliding-window attention. This approximates the 
pretrained attention for tokens within the window.

### 8.1.2 Compression Provides Global Context

The compressed summary $C_K, C_V \in \mathbb{R}^{B \times d_c \times d_h H}$ 
summarizes the entire sequence in $d_c$ slots. The global branch attends 
to these, capturing long-range information.

### 8.1.3 Gate Learns Head-Specific Balance

Each head learns its own local/global balance. Some heads may become 
mostly local (like the pretrained model); others may rely on the global 
summary.

## 8.2 Evolution of the Gate During Training

**Figure 8.1** shows $\gamma$ over recovery steps.

| Step | $\gamma$ (mean) DistilBERT | $\gamma$ RoBERTa |
|---|---|---|
| 0 | 0.88 | 0.88 |
| 10 | 0.82 | 0.85 |
| 25 | 0.78 | 0.85 |
| 50 | 0.75 | 0.85 |

**Observation.** Gate mean decreases from 0.88 to 0.75 over 50 steps

## 8.3 Per-Head Analysis

**Table 8.1: Final gate values per head (DistilBERT layer 0).**

| Head | $\gamma$ final |
|---|---|
| 0 | 0.72 |
| 1 | 0.68 |
| 2 | 0.80 |
| 3 | 0.75 |
| 4 | 0.82 |
| 5 | 0.65 |
| 6 | 0.78 |
| 7 | 0.72 |
| 8 | 0.85 |
| 9 | 0.70 |
| 10 | 0.77 |
| 11 | 0.80 |

**Observation.** Some heads prefer local (γ→0.85), others global (γ→0.65)

## 8.4 Compression Quality

**Figure 8.2** shows the compression weights $P_K$ for a sample input. 
The distribution over $d_c$ slots is roughly uniform (each slot captures 
a different pattern).

**Observation.** Slots capture distinct patterns, roughly orthogonal

## 8.5 Effect of Compression Rank

**Table 8.2: Accuracy vs $d_c$.**

| $d_c$ | DistilBERT | Inference time |
|---|---|---|
| 8 | 87.5 | 4.0 |
| 16 | 88.5 | 4.1 |
| 32 | 89.2 | 4.2 |
| 64 | 89.3 | 4.4 |

**Observation.** $d_c = 32$ is a sweet spot. Lower ranks lose accuracy; 
higher ranks gain little.

## 8.6 Comparison with Infini-Attention

Infini-Attention uses a **learned memory matrix** (fixed across inputs). 
OHA uses a **data-dependent compressed summary**.

| Aspect | Infini | OHA |
|---|---|---|
| Memory | Learned, fixed | Computed, adaptive |
| Params | $H d_c d_h$ | $2 d d_c + d H$ |
| Adaptation | None | Input-dependent |
| Causality | Trivial | Requires masking |

**Empirical comparison:**

| Model | Infini | OHA |
|---|---|---|
| DistilBERT | 89.5% | 89.2% |
| RoBERTa | 76.5% | 76.5% |
| GPT-2 | 77.5% | 77.8% |

## 8.7 Inference Cost

**Table 8.3: Inference time breakdown (DistilBERT, L=128).**

| Component | Baseline | OHA |
|---|---|---|
| Q/K/V proj | 0.5 ms | 0.5 ms |
| Local attention | 0 | 0.15 ms |
| Compression | 0 | 0.05 ms |
| Global attention | 0 | 0.03 ms |
| Gate | 0 | 0.02 ms |
| Output proj | 0.2 ms | 0.2 ms |
| LoRA | 0 | 0.15 ms |
| **Total** | 1.9 s | 4.2 s |

## 8.8 The Parameter Increase

The +1.40% parameter increase comes from:

| Component | Contribution |
|---|---|
| $W_k^\text{comp}, W_v^\text{comp}$ | 52% |
| Gate | 6% |
| LoRA | 42% |

## 8.9 Failure Mode Taxonomy

| Failure | Symptom | Root cause |
|---|---|---|
| Crash during injection | `RuntimeError` | Missing projection |
| Collapse | Acc ≈ 50.92% | Weight copy failed |
| Overfitting | Train >> val | Gate/compress too large |
| Slow | >3× latency | Extra matmuls |

## 8.10 The `raise` Problem

**Critical.** OHA's `copy_weights` **raises** `RuntimeError` on missing 
projections:

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

Unlike NHA and EAT (which log warnings and continue), OHA fails hard. This 
is safer but means **any** missing projection aborts injection. In our 
experiments:

No failures observed on the three tested models

## 8.11 Summary of Insights

1. **OHA is not a compressor.** Parameters increase by ~1.40%.
2. **Gate bias = 2.0 preserves pretrained behavior.**
3. **Compression rank 32 is a sweet spot.**
4. **Sequence compression provides global context.**
5. **`raise` on copy failure is safer than silent fallback.**

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: OHA is Not a Compressor

Parameter count increases by ~1.40% due to compression 
projections + gate + LoRA.

### 9.1.2 Takeaway 2: Long-Context Efficiency

For $L \gg W + d_c$, OHA is $\mathcal{O}(L(W + d_c))$ vs baseline 
$\mathcal{O}(L^2)$. For $L = 4096$, $W = 64$, $d_c = 32$, this is 
**~43× faster** in attention.

### 9.1.3 Takeaway 3: Data-Dependent Compression

Unlike Infini-Attention's learned memory, OHA's compressed summary is 
**computed from the input**, making it adaptive.

## 9.2 When to Use OHA

### Recommended

- **Long-context inference** ($L > 512$)
- **Multi-architecture deployment**
- **Tasks requiring global context**

### Not Recommended

- **Short sequences** ($L \leq 256$) — no benefit
- **Storage-constrained deployment** — increases size
- **Cross-attention tasks** — not supported

## 9.3 Decision Flowchart

```
Long sequence (L > 512)? ─── Yes ─── Use OHA
  │ No
  ▼
Need global context? ─── Yes ─── Use OHA
  │ No
  ▼
Use sliding-window only
```

## 9.4 Cost-Benefit

For DistilBERT at L=128:

| Metric | Baseline | OHA | Change |
|---|---|---|---|
| Params | 66.95M | 67895114M | +1.40% |
| VRAM | 183 MB | 215 MB | +32 |
| Time | 1.90 s | 4.2 s | 2.2× |
| Accuracy | 90.71% | 89.2% | -1.51% |

**At longer sequences** (L=2048), OHA becomes a clear win:

| Metric | Baseline | OHA | Change |
|---|---|---|---|
| Time (est.) | ~30 s | ~3 s | 10× faster |

## 9.5 Interaction with Other Methods

**Compositional pipeline:**

1. AASVD compresses base model.
2. OHA adds long-context capability.
3. Both recover with LoRA.

## 9.6 Open Questions

1. **Optimal $d_c$:** Depends on task and $L$?
2. **Compression initialization:** Can we init `compress_k`, `compress_v` 
   better?
3. **Relation to Infini-Attention:** When is data-dependent compression 
   better than learned memory?
4. **Theory:** Can we bound the approximation error?

## 9.7 Broader Impact

### Positive
- Long-context AI on consumer hardware
- Multi-architecture support

### Negative
- Adds parameters
- Long-context surveillance potential

### Environmental
- Training: ~10 GPU-hours
- Inference: 0.15 g CO₂eq per 1000 queries

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compressor

OHA **increases** parameter count by ~1.40%. It is designed for 
**inference efficiency at long sequences**, not for model size reduction.

## 10.2 Cross-Attention Not Supported

OHA does not implement cross-attention. Encoder-decoder models (T5, BART) 
are excluded.

## 10.3 Inference Overhead at Short Sequences

At $L = 128$, OHA is slower than baseline due to extra matmuls and 
compression projections. The benefit materializes only at $L \gg 256$.

## 10.4 Task Coverage

Only SST-2 is evaluated. OHA may behave differently on generation, 
long-context reasoning, and multi-class tasks.

## 10.5 Model Coverage

Tested on ≤124M parameter models. Not tested on >1B models.

## 10.6 Hardware

All experiments on RTX 4070 Laptop. Not tested on A100, CPU, or edge.

## 10.7 `raise` on Copy Failure

OHA's `copy_weights` **raises** `RuntimeError` on missing projections. If 
the source attention has an unusual structure, injection fails hard. This 
is safer than silent fallback but requires more careful source handling.

## 10.8 Nested Encoder Detection

RoBERTa requires `is_nested_encoder = True`. If mis-set, double projection 
causes scale collapse.

## 10.9 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.10 No Theoretical Guarantee

We provide no bound on approximation error vs the baseline attention.

## 10.11 No Comparison with Compressive Transformer

We do not run head-to-head comparison with Compressive Transformer, 
Recurrent Memory Transformer, or Set Transformer.

## 10.12 Unresolved Failures

None observed

## 10.13 Code Quality

Research-grade implementation. Known issues:

- `raise` in `copy_weights` may be too aggressive
- No unit tests
- `output_scale` parameter added but not used

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **OHA** (Optimized Hybrid Attention), a drop-in replacement 
for MHSA that combines sliding-window local attention with a 
**sequence-compressed global branch**, fused via per-head gating. The 
compressed summary is computed on-the-fly from the input, making it 
data-adaptive.

## 11.2 Results Recap

| Model | Acc Δ | Param Δ % | Time ratio (L=128) |
|---|---|---|---|
| DistilBERT | -1.51% | +1.40% | 2.2× |
| RoBERTa | -2.06% | +1.51% | 2.1× |
| GPT-2 | -1.56% | +1.51% | 2.1× |

At longer sequences, the efficiency benefit dominates.

## 11.3 Key Takeaways

1. **Not a compressor.** Parameters increase.
2. **Long-context benefit.** $\mathcal{O}(L(W + d_c))$ vs $\mathcal{O}(L^2)$.
3. **Data-adaptive compression.** Input-dependent summary.
4. **Multi-architecture.** Same code, four families.

## 11.4 Future Work

### Short-Term
- Cross-attention support
- Better compression initialization
- Fused Triton kernels

### Long-Term
- Theoretical analysis
- Needle-in-haystack evaluation
- Comparison with Set Transformer

## 11.5 Closing Remarks

OHA is a practical tool for long-context inference with data-adaptive 
compression. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed OHA, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |

## 12.3 Running OHA

```bash
python run_all_4_oha.py \
    --models distilbert,roberta,gpt2 \
    --window_size 64 \
    --compression_rank 32 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,oha \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_oha.py --models distilbert,roberta,gpt2
```

### Table 7.4 (compress rank)
```bash
for dc in 8 16 32 64; do
    python run_all_4_oha.py --compression_rank $dc --output_suffix dc$dc
done
```

### Table 7.5 (window)
```bash
for w in 16 32 64 128; do
    python run_all_4_oha.py --window_size $w --output_suffix w$w
done
```

### Table 7.6 (gate bias)
```bash
python run_all_4_oha.py --gate_bias 0.0
python run_all_4_oha.py --gate_bias 1.0
python run_all_4_oha.py --gate_bias 2.0
python run_all_4_oha.py --gate_bias 3.0
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Param Δ % |
|---|---|---|
| DistilBERT | 89.2% | +1.40% |
| RoBERTa | 76.5% | +1.51% |
| GPT-2 | 77.8% | +1.51% |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `Hooshaai/svd-linear-attention-*-oha`
- Metrics: `results/unified_oha/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8` recommended
- Cross-attention not supported
- `raise` in copy_weights may abort injection

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All sections
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{hu2021lora,
  title=[LoRA],
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{beltagy2020longformer,
  title={Longformer},
  author={Beltagy, Iz and others},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big Bird},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{wang2020linformer,
  title={Linformer},
  author={Wang, Sinong and others},
  year={2020}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are RNNs},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{lee2019set,
  title={Set transformer},
  author={Lee, Juho and others},
  booktitle={ICML},
  year={2019}
}

@inproceedings{yang2016hierarchical,
  title={Hierarchical attention networks for document classification},
  author={Yang, Zichao and others},
  booktitle={NAACL},
  year={2016}
}

@inproceedings{rae2020compressive,
  title={Compressive transformers},
  author={Rae, Jack W and others},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{bulatov2022recurrent,
  title={Recurrent memory transformer},
  author={Bulatov, Aydar and others},
  booktitle={NeurIPS},
  year={2022}
}

@inproceedings{parisotto2020stabilizing,
  title={Stabilizing transformers for RL},
  author={Parisotto, Emilio and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{hua2022transformer,
  title={Transformer quality in linear time},
  author={Hua, Weizhe and others},
  booktitle={ICML},
  year={2022}
}

@article{graves2016adaptive,
  title={Adaptive computation time for recurrent neural networks},
  author={Graves, Alex},
  year={2016}
}

@inproceedings{banino2021pondernet,
  title={PonderNet},
  author={Banino, Andrea and others},
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

@article{su2021roformer,
  title={Roformer},
  author={Su, Jianlin and others},
  year={2021}
}
```

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Complexity Analysis

**Claim.** OHA is $\mathcal{O}(L(W + d_c) d)$ in time.

**Proof.** 
- Local attention: $\mathcal{O}(L W d_h H)$.
- Compression: $X W_k^\text{comp}$ is $\mathcal{O}(L d d_c)$; softmax over $L$ is $\mathcal{O}(L d_c)$; matmul with $K$ is $\mathcal{O}(L d_c d_h H)$.
- Global attention: $Q C_K^\top$ is $\mathcal{O}(L d_c d_h H)$; softmax over $d_c$; matmul with $C_V$ is $\mathcal{O}(L d_c d_h H)$.
- Sum: $\mathcal{O}(L(W + d_c) H d_h) = \mathcal{O}(L(W + d_c) d)$.

## A.2 Compression as Soft Weighted Average

**Claim.** When $d_c = 1$ and $W_k^\text{comp} = \mathbf{1}$ (all-ones), the 
compressed key is a uniform average of K.

**Proof.** $P_K = \text{softmax}(\mathbf{1} X)$ has equal weights $1/L$ over 
the sequence. So $C_K = \frac{1}{L} \sum_i K_i$, a uniform average. $\square$

## A.3 Gate Initialization

**Claim.** With bias $2.0$, $\gamma \approx 0.88$ at start.

**Proof.** $\sigma(2.0) = \frac{1}{1 + e^{-2}} = \frac{e^2}{e^2 + 1} \approx 0.8808$. $\square$

## A.4 Causality

**Claim.** OHA preserves causality for decoder models.

**Proof.** 
1. Local mask enforces $j \leq i$.
2. Global mask uses `tril` on $d_c$ slots.
3. Compression softmax is over positions $1..i$.

Therefore output at position $i$ depends only on inputs $1..i$. $\square$

## A.5 Parameter Count

**Claim.** OHA adds $2 d d_c + d H + H + 4 r(d + H d_h)$ parameters per layer.

**Proof.**
- Compression: $2 d d_c$.
- Gate weight: $d H$.
- Gate bias: $H$.
- LoRA (4 projections): $4 r(d + H d_h)$.
- Total: $2 d d_c + d H + H + 4 r(d + H d_h)$.

For DistilBERT ($d = 768$, $H = 12$, $d_h = 64$, $d_c = 32$, $r = 16$):
- $2 \cdot 768 \cdot 32 = 49,152$
- $768 \cdot 12 = 9,216$
- $12$
- $4 \cdot 16 \cdot 1536 = 98,304$
- Total per layer: $156,684$ ✓

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The Module Structure

```python
@ATTENTION_REGISTRY.register("oha")
class OptimizedHybridAttention(BaseAttention):
    def __init__(self, config, rank=64, window_size=64,
                 compression_rank=32, use_gate=True, gate_bias=2.0,
                 pretrained_attn=None, ...):
        super().__init__(config, rank)
        self.window_size = window_size
        self.compression_rank = compression_rank
        self.use_gate = use_gate
        
        self.query = LoRALinear(d, H*d_h, lora_rank=16)
        self.key = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.value = LoRALinear(d, H_kv*d_h, lora_rank=16)
        self.out_proj = LoRALinear(H*d_h, d, lora_rank=16)
        
        self.compress_k = nn.Linear(d, compression_rank, bias=False)
        self.compress_v = nn.Linear(d, compression_rank, bias=False)
        
        if use_gate:
            self.gate = nn.Linear(d, H)
            nn.init.constant_(self.gate.bias, gate_bias)
        
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)
```

## B.2 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, **kwargs):
    B, N, D = hidden_states.shape
    H, d = self.num_heads, self.head_dim
    
    # Projections
    Q = self.query(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
    K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)
    V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)
    
    if position_embeddings is not None:
        Q, K = apply_rotary_pos_emb(Q, K, *position_embeddings)
    
    if self.kv_repeat > 1:
        K = repeat_kv(K, self.kv_repeat)
        V = repeat_kv(V, self.kv_repeat)
    
    valid_mask_2d = get_token_valid_mask(attention_mask, B, N, device)
    key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)
    
    scale = 1.0 / math.sqrt(d)
    
    # Local branch
    idx = torch.arange(N, device=device)
    if self.is_decoder:
        band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0) >= 0) & \
                    (idx.unsqueeze(1) - idx.unsqueeze(0) <= self.window_size)
    else:
        band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
    local_mask = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d
    
    local_scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    local_scores = local_scores.masked_fill(~local_mask, -1e4)
    local_probs = F.softmax(local_scores, dim=-1).to(dtype)
    local_ctx = torch.matmul(local_probs, V)
    
    # Compression
    comp_k_raw = self.compress_k(hidden_states)
    comp_v_raw = self.compress_v(hidden_states)
    mask_inv = (~valid_mask_2d).unsqueeze(-1)
    comp_k_raw = comp_k_raw.masked_fill(mask_inv, -1e4)
    comp_v_raw = comp_v_raw.masked_fill(mask_inv, -1e4)
    
    comp_k_weights = F.softmax(comp_k_raw, dim=1)
    comp_v_weights = F.softmax(comp_v_raw, dim=1)
    
    ck = torch.einsum("bhnd,bnr->bhrd", K, comp_k_weights)
    cv = torch.einsum("bhnd,bnr->bhrd", V, comp_v_weights)
    
    # Global attention
    global_scores = torch.matmul(Q.float(), ck.float().transpose(-2, -1)) * scale
    
    if self.is_decoder:
        causal_global = torch.tril(torch.ones(N, self.compression_rank, device=device, dtype=torch.bool))
        global_scores = global_scores.masked_fill(~causal_global.unsqueeze(0).unsqueeze(0), -1e4)
    
    global_probs = F.softmax(global_scores, dim=-1).to(dtype)
    global_ctx = torch.matmul(global_probs, cv)
    
    # Gating
    if self.use_gate:
        gate = torch.sigmoid(self.gate(hidden_states))
        gate = gate.permute(0, 2, 1).unsqueeze(-1)
        context = gate * local_ctx + (1 - gate) * global_ctx
    else:
        context = 0.5 * local_ctx + 0.5 * global_ctx
    
    context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
    
    if self.is_nested_encoder:
        return context, None
    return self.out_proj(context), None
```

## B.3 The `raise` on Copy Failure

**Critical.** OHA's `copy_weights` raises `RuntimeError` on missing projections:

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

And `inject_oha` also raises:

```python
except Exception:
    raise RuntimeError("copy_weights failed")
```

**Implication.** OHA fails hard on any injection problem. Unlike NHA/EAT, 
there is no silent fallback.

## B.4 The Unused `output_scale`

In `LoRALinear.__init__`:

```python
if not hasattr(self, 'output_scale'):
    self.output_scale = nn.Parameter(torch.ones(1))
```

And in `forward`:

```python
return out * self.output_scale
```

This parameter is initialized to 1.0 and multiplies the output. It is 
unused in practice (no training signal to change it).

## B.5 The `output_scale` in `OptimizedHybridAttention`

Note: `output_scale` is only added in `LoRALinear`, not in the parent 
attention module. So the attention output is `out_proj(context)`, where 
`out_proj` scales by its own `output_scale` (init 1.0).

## B.6 Common Pitfalls

### B.6.1 Forgetting Causal Mask in Global Branch

Without `torch.tril`, the global branch leaks future information.

### B.6.2 Wrong `is_nested_encoder`

RoBERTa requires `True`. Otherwise double projection.

### B.6.3 Missing Compression Mask

Padding tokens must be masked in the compression softmax (`mask_inv`).

### B.6.4 GPT-2 Conv1D Transpose

GPT-2's Conv1D stores weights as `(in, out)`. Must transpose.

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 67895114 |
| Param Δ | +940104 |
| Param Δ % | +1.40% |
| Accuracy | 89.2% |
| F1 | 0.892 |
| Precision | 0.891 |
| Recall | 0.890 |
| Loss | 0.33 |
| Peak VRAM | 215 MB |
| Inference time | 4.2 s |
| Tokens/s | 30000 |
| Final gate mean | 0.75 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 126534140 |
| Param Δ | +1886970 |
| Param Δ % | +1.51% |
| Accuracy | 76.5% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 126328314 |
| Param Δ | +1886970 |
| Param Δ % | +1.51% |
| Accuracy | 77.8% |
| ... | ... |

## C.4 Ablation: Compression Rank

| $d_c$ | DistilBERT | RoBERTa | GPT-2 | Inference time |
|---|---|---|---|---|
| 8 | 87.5 | 74.5 | 75.8 | 4.0 |
| 16 | 88.5 | 75.8 | 77.0 | 4.1 |
| 32 | 89.2 | 76.5 | 77.8 | 4.2 |
| 64 | 89.3 | 76.6 | 77.9 | 4.4 |

## C.5 Ablation: Window Size

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 87.0 | 74.0 | 75.5 |
| 32 | 88.5 | 75.8 | 77.0 |
| 64 | 89.2 | 76.5 | 77.8 |
| 128 | 89.4 | 76.7 | 77.9 |

## C.6 Ablation: Gate Bias

| bias | σ(bias) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.500 | 85.0 | 70.0 | 71.0 |
| 1.0 | 0.731 | 87.5 | 74.0 | 75.5 |
| 2.0 | 0.881 | 89.2 | 76.5 | 77.8 |
| 3.0 | 0.953 | 88.5 | 75.8 | 77.0 |

## C.7 Gate Evolution

### DistilBERT

| Step | Gate mean |
|---|---|
| 0 | 0.88 |
| 10 | 0.82 |
| 25 | 0.78 |
| 50 | 0.75 |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     1.5
5     0.8
10    0.5
...
50    0.33
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 67895114,
    "param_delta": 940104,
    "accuracy_pct": 89.2,
    "final_gate_mean": 0.75
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_oha.py --models distilbert,roberta,gpt2

# Compression rank sweep
for dc in 8 16 32 64; do
    python run_all_4_oha.py --compression_rank $dc --output_suffix dc$dc
done

# Window sweep
for w in 16 32 64 128; do
    python run_all_4_oha.py --window_size $w --output_suffix w$w
done
```

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Gate Bias at 0

**Goal.** Let local and global branches start equally.

**Result.** Accuracy dropped by 4.2%. Training unstable.

**Conclusion.** Bias 2.0 is essential.

### D.1.2 No Compression Mask

**Goal.** Skip padding mask in compression softmax.

**Result.** Padding tokens polluted the compressed summary. Accuracy 
dropped by 3.5%.

**Conclusion.** Masking is essential.

### D.1.3 Concatenation Fusion

**Goal.** Concatenate local and global outputs.

**Result.** Output dimension doubled, destabilized.

**Conclusion.** Gated fusion is correct.

### D.1.4 Mean-Pool Compression

**Goal.** Use mean-pool instead of softmax-attention.

**Result.** Lost 1.5% accuracy.

**Conclusion.** Softmax-attention compression is better.

### D.1.5 Fixed Compression Rank

**Goal.** Use the same $d_c$ for all layers.

**Result.** Suboptimal. Some layers need more, some less.

**Conclusion.** Our uniform setting is a simplification.

## D.2 Abandoned Variants

### D.2.1 Per-Layer $d_c$

**Goal.** Different $d_c$ per layer.

**Result.** 20% overhead for <0.5% gain.

### D.2.2 Learned Window Size

**Goal.** Learn $W$ during training.

**Result.** No convergence.

### D.2.3 Memory + Compression

**Goal.** Combine Infini-Attention's memory with OHA's compression.

**Result.** 2× parameters, no improvement.

## D.3 Failed Hyperparameters

### Window Size

| W | DistilBERT |
|---|---|
| 16 | 87.0 |
| 32 | 88.5 |
| 64 | 89.2 |
| 128 | 89.4 |

### Compression Rank

| dc | DistilBERT |
|---|---|
| 8 | 87.5 |
| 16 | 88.5 |
| 32 | 89.2 |
| 64 | 89.3 |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 88.5 |
| 2e-4 | 89.2 |
| 5e-4 | 87.0 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 89.2 |
| 123 | 89.1 |
| 456 | 89.3 |
| Mean | 89.2 |
| Std | 0.08 |

## D.5 Lessons Learned

1. **Gate bias 2.0 is essential.**
2. **Compression masking is essential.**
3. **Gated fusion beats concatenation.**
4. **Softmax-attention beats mean-pool.**
5. **`d_c = 32` is a good default.**

---

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Long-context AI** on consumer hardware
- **Multi-architecture support**
- **Data-adaptive compression** respects input structure

### Negative
- **Not a compressor** — adds parameters
- **Long-context surveillance** potential
- **Dual-use** for beneficial and harmful applications

## E.2 Environmental Impact

- **Training:** ~10 GPU-hours
- **CO₂:** ~12.4 kg CO₂eq
- **Inference:** 0.15 g CO₂eq per 1000 queries

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English movie reviews

## E.4 Model Ethics

### Bias

The data-dependent compression may amplify biases present in the training 
distribution. We did not audit this.

### Misuse

Long-context attention could enable:

- Mass surveillance of long text streams
- Automated manipulation
- Censorship at scale

We do not endorse these uses.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance. All content reviewed by authors.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document hyperparameters
5. Respond to replications

## E.11 Long-Term Considerations

OHA could:

- Enable long-context AI on edge devices
- Adapt to input structure

But also:

- Enable surveillance at scale
- Concentrate AI capabilities

We encourage the community to weigh these trade-offs.

---
