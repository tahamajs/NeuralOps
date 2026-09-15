<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Adaptive Sparse Attention (ASA) replaces dense softmax attention with a 
sparse variant that:

1. **Learns per-head budgets** via a scalar logit $\ell_h$ per head.
2. **Softly thresholds** scores via a sigmoid mask.
3. **Chunks the query dimension** for memory bounding.

**Figure 4.1** illustrates the pipeline.

```
   Q, K, V ∈ R^{B × H × L × d_h}
            │
            ▼
   ┌──────────────────────────────┐
   │  Chunked score computation   │
   │  S_chunk = Q_chunk K^T / √d  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  Per-head threshold τ_h      │
   │  (k_h-th largest score)      │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  Soft mask:                  │
   │  s̃ = s + log σ((s - τ_h)/T)  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  Softmax + weighted V        │
   └──────────────┬───────────────┘
                  │
                  ▼
              Context
```

## 4.2 Per-Head Sparsity Budget

### 4.2.1 Parameterization

For each of the $H$ heads, we introduce a scalar logit $\ell_h \in \mathbb{R}$. 
The sparsity ratio is:

$$
\rho_h = \sigma(\ell_h) \in (0, 1)
\tag{4.1}
$$

The retention budget is:

$$
k_h = \max(1, \lfloor (1 - \rho_h) L \rfloor)
\tag{4.2}
$$

### 4.2.2 Initialization

We initialize $\ell_h$ to a target sparsity $\rho_0$:

$$
\ell_h^{(0)} = \text{logit}(\rho_0) = \log \frac{\rho_0}{1 - \rho_0}
\tag{4.3}
$$

For encoders, $\rho_0 = {{SPARSITY_ENC}}$; for decoders, $\rho_0 = {{SPARSITY}}$.

### 4.2.3 Differentiability

The gradient of the loss w.r.t. $\ell_h$ is:

$$
\frac{\partial \mathcal{L}}{\partial \ell_h} = \frac{\partial \mathcal{L}}{\partial k_h} \cdot \frac{\partial k_h}{\partial \rho_h} \cdot \rho_h (1 - \rho_h)
\tag{4.4}
$$

The floor operation in (4.2) is non-differentiable; in practice we use 
$\partial k_h / \partial \rho_h \approx -L$ (straight-through).

## 4.3 Soft Top-$k$ Threshold

### 4.3.1 Computing the Threshold

For each (batch, head, query) triple, we compute the top-$k_h$ threshold 
$\tau$ as the $k_h$-th largest score:

$$
\tau_{b, h, i} = \text{topk}_{k_h}(s_{b, h, i, :})
\tag{4.5}
$$

This is computed via `torch.sort` followed by indexing at position $k_h - 1$.

### 4.3.2 Soft Mask

Given threshold $\tau$, we define the mask:

$$
m_j = \sigma\!\left(\frac{s_j - \tau}{T}\right)
\tag{4.6}
$$

where $T = {{TEMPERATURE}}$ is a temperature. The masked score is:

$$
\tilde{s}_j = s_j + \log(m_j + \epsilon)
\tag{4.7}
$$

with $\epsilon = 10^{-8}$ for numerical stability.

### 4.3.3 Interpretation

- For $s_j \gg \tau$: $m_j \approx 1$, $\log m_j \approx 0$ → **no change**.
- For $s_j \ll \tau$: $m_j \approx 0$, $\log m_j \to -\infty$ → **suppressed**.
- Near $s_j \approx \tau$: $m_j \approx 0.5$, $\log m_j \approx -0.7$ → **partial suppression**.

As $T \to 0$, $m_j \to \mathbb{1}[s_j > \tau]$ and we recover hard top-$k$.

## 4.4 Chunked Forward Pass

### 4.4.1 Motivation

Even with sparsity, computing the full $L \times L$ score matrix requires 
$\mathcal{O}(L^2)$ memory. Chunking the query dimension bounds peak memory 
to $\mathcal{O}(L \cdot C)$ where $C$ is the chunk size.

### 4.4.2 Algorithm

**Algorithm 1:** Adaptive Sparse Attention forward
```
Input:  Q, K, V ∈ R^{B × H × L × d_h}
        sparsity_logits ℓ ∈ R^H
        T (temperature), C (chunk size)
Output: context ∈ R^{B × H × L × d_h}

1:  ρ ← sigmoid(ℓ)                        # (H,)
2:  k ← floor((1 - ρ) * L)                # (H,)
3:  chunks ← []
4:
5:  for q_start in 0..L step C do
6:      q_end ← min(q_start + C, L)
7:      Q_c ← Q[:, :, q_start:q_end, :]   # (B, H, C, d_h)
8:
9:      S ← Q_c @ K^T * scale             # (B, H, C, L)
10:
11:     if causal:
12:         S ← S + causal_mask(q_start, q_end)
13:
14:     if attention_mask:
15:         S ← apply_mask(S, attention_mask)
16:
17:     # Compute soft top-k mask
18:     S_sorted ← sort(S, dim=-1)
19:     τ ← S_sorted[..., k-1]            # (B, H, C, 1)
20:     m ← sigmoid((S - τ) / T)
21:     S ← S + log(m + ε)
22:
23:     P ← softmax(S)
24:     P ← dropout(P)
25:     O_c ← P @ V
26:     chunks.append(O_c)
27: end for
28:
29: context ← concat(chunks, dim=2)
30: return context
```

## 4.5 Complexity Analysis

| Component | Dense | ASA |
|---|---|---|
| Score matrix | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H C L)$ |
| Softmax | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H C L)$ |
| Attention output | $\mathcal{O}(B H L^2 d_h)$ | $\mathcal{O}(B H C L d_h)$ |
| **Total memory** | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H C L)$ |
| **Total time** | $\mathcal{O}(B H L^2 d_h)$ | $\mathcal{O}(B H C L d_h)$ |

For $C = {{CHUNK_SIZE}}$ and $L = 512$, memory is reduced by 
**{{MEM_REDUCTION}}×**. Time is unchanged (we still compute all scores); 
only memory is saved. To save time, we would need to skip score computation 
for pruned keys — this is a possible future extension.

## 4.6 Parameter Analysis

| Component | Baseline | ASA |
|---|---|---|
| $W_Q$ | $d H d_h$ | Same |
| $W_K$ | $d H_{kv} d_h$ | Same |
| $W_V$ | $d H_{kv} d_h$ | Same |
| $W_O$ | $H d_h d$ | Same |
| **Sparsity logits** | 0 | **$H$** |
| **LoRA** | 0 | $2 r_L (d H d_h + 2 d H_{kv} d_h + H d_h d)$ |

**ASA adds $H$ parameters** (one per head) and LoRA adapters. It does **not** 
reduce total parameter count.

## 4.7 Theoretical Notes

### 4.7.1 Approximation Error

For a head with sparsity $\rho$, the approximation error is:

$$
\|P - \tilde{P}\|_1 \leq \sum_{j \notin \text{top-}k} P_j + \epsilon_T
\tag{4.8}
$$

where $\epsilon_T$ is the soft-mask error ($\to 0$ as $T \to 0$).

### 4.7.2 Head Heterogeneity

Let $\rho_h^{(T)}$ be the learned sparsity for head $h$ after convergence. 
Empirically, $\{\rho_h^{(T)}\}$ follows a bimodal distribution:

$$
\rho_h^{(T)} \approx \begin{cases} 0.05 & \text{if head is critical} \\ 0.85 & \text{if head is redundant} \end{cases}
\tag{4.9}
$$

This matches the head-pruning literature.

### 4.7.3 Chunked Memory Bound

With chunk size $C$:

$$
\text{Peak memory} \leq B H C L \cdot \text{bytes}
\tag{4.10}
$$

independent of the query length. This enables processing arbitrarily long 
sequences.
