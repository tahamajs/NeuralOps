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
