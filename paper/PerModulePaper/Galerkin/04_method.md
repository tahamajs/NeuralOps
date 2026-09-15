# 4. Method

## 4.1 Overview

Galerkin Attention has two execution paths:

- **Encoder path:** Petrov-Galerkin projection onto DCT basis, then attention 
  at rank $r$.
- **Decoder path:** Standard causal softmax attention (fallback).

**Figure 4.1** shows the pipeline.

```
    ┌───────────────────────────────────────────┐
    │  Input X → Q/K/V projections (frozen+LoRA)│
    └─────────────────────┬─────────────────────┘
                          │
                          ▼
    ┌───────────────────────────────────────────┐
    │  Apply RoPE, GQA repeat_kv                │
    └─────────────────────┬─────────────────────┘
                          │
                          ▼
    ┌───────────────────────────────────────────┐
    │  Is decoder?                              │
    └────┬─────────────────────┬────────────────┘
         │ Yes                 │ No
         ▼                     ▼
    ┌──────────────┐    ┌──────────────────────┐
    │  Standard    │    │  1. Build DCT basis  │
    │  causal      │    │  2. Project Q/K/V    │
    │  softmax     │    │  3. Attention in r   │
    │  attention   │    │  4. Reconstruct + √  │
    └──────┬───────┘    └──────────┬───────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
              ┌────────────────┐
              │  out_proj      │
              └────────┬───────┘
                       │
                       ▼
                      Y
```

## 4.2 Phase 1: Projections

Frozen base + LoRA:

$$
\text{Proj}_P(X) = X W_P^\top + \frac{\alpha}{r_L} (X A_P^\top) B_P^\top + b_P
\tag{4.1}
$$

$W_P$ copied exactly from pretrained.

## 4.3 Phase 2: Basis Construction

### DCT-II Basis

For sequence length $N$ and rank $r$:

$$
\Phi_{n, k} = \sqrt{\frac{2 - \delta_{k0}}{N}} \cos\!\left(\frac{\pi k (2n + 1)}{2N}\right)
\tag{4.2}
$$

with indices $n = 0, \dots, N-1$ and $k = 0, \dots, r-1$.

### Orthonormalization

DCT is theoretically orthonormal, but floating-point rounding can break 
this. We add small identity regularization and apply QR:

$$
\Phi \leftarrow \text{QR}(\Phi + \epsilon I_{N \times r})
\tag{4.3}
$$

with $\epsilon = 10^{-4}$.

## 4.4 Phase 3: Encoder Path

### Projection to Basis

$$
Q_\Phi = \Phi^\top Q \in \mathbb{R}^{B \times H \times r \times d_h}
\tag{4.4}
$$

$$
K_\Phi = \Phi^\top K_{\text{masked}}, \quad V_\Phi = \Phi^\top V_{\text{masked}}
\tag{4.5}
$$

where $K_{\text{masked}} = K \cdot m$ and $V_{\text{masked}} = V \cdot m$ 
with padding mask $m$.

### Attention in Basis Space

$$
\text{Scores} = \frac{Q_\Phi K_\Phi^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times r \times r}
\tag{4.6}
$$

$$
\text{Probs} = \text{softmax}(\text{Scores})
\tag{4.7}
$$

$$
O_\Phi = \text{Probs} \cdot V_\Phi
\tag{4.8}
$$

### Reconstruction to Sequence Grid

$$
Y = \Phi \cdot O_\Phi \cdot \sqrt{N / r}
\tag{4.9}
$$

The scaling factor $\sqrt{N / r}$ compensates for the energy loss due to 
truncation.

### Padding Handling

Masked tokens in $K$ and $V$ are zeroed before projection. After 
reconstruction, the output at padded positions is zeroed:

$$
Y \leftarrow Y \odot m
\tag{4.10}
$$

## 4.5 Phase 4: Decoder Path

For causal models, projection breaks causality because $K_\Phi$ depends on 
**all** keys. We fall back to standard softmax attention:

$$
\text{Scores} = \frac{Q K^\top}{\sqrt{d_h}}, \quad
\text{Scores} \leftarrow \text{Scores} + M_{\text{causal}}
\tag{4.11}
$$

$$
\text{Probs} = \text{softmax}(\text{Scores}), \quad
Y = \text{Probs} \cdot V
\tag{4.12}
$$

with numerical stabilization via $-\max$ subtraction.

## 4.6 The Full Algorithm

**Algorithm 1:** Galerkin Attention forward
```
Input:  X, attention_mask, position_embeddings
        is_decoder, rank r
Output: Y, probs

1:  Q ← LoRALinear_Q(X)
2:  K ← LoRALinear_K(X)
3:  V ← LoRALinear_V(X)
4:
5:  if RoPE: (Q, K) ← apply_rope(Q, K, cos, sin)
6:  if GQA: K, V ← repeat_kv(K, V)
7:
8:  if is_decoder:                              ▷ DECODER FALLBACK
9:      S ← Q K^T / √d_h
10:     S ← S + causal_mask
11:     S ← S + padding_mask
12:     P ← softmax(S)
13:     Y_ctx ← P @ V
14:
15: else:                                        ▷ ENCODER GALERKIN
16:     mask ← valid_token_mask(attention_mask)
17:     K_masked ← K ⊙ mask
18:     V_masked ← V ⊙ mask
19:
20:     Φ ← DCT_basis(N, r)                        ▷ Orthonormalize via QR
21:     Q_Φ ← Φ^T Q
22:     K_Φ ← Φ^T K_masked
23:     V_Φ ← Φ^T V_masked
24:
25:     S ← Q_Φ K_Φ^T / √d_h                       ▷ (B, H, r, r)
26:     P ← softmax(S)
27:     O_Φ ← P @ V_Φ
28:
29:     Y_ctx ← Φ @ O_Φ                             ▷ Reconstruct
30:     Y_ctx ← Y_ctx · √(N/r)                      ▷ Energy correction
31:     Y_ctx ← Y_ctx ⊙ mask                        ▷ Zero padded positions
32:
33: Y_ctx ← reshape(B, N, H · d_h)
34: if nested: Y ← Y_ctx
35: else: Y ← LoRALinear_O(Y_ctx)
36:
37: return Y
```

## 4.7 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| Encoder (Galerkin) | $\mathcal{O}(N r d)$ + $\mathcal{O}(N r)$ for basis | $\mathcal{O}(N r)$ |
| Decoder (fallback) | $\mathcal{O}(N^2 d)$ | $\mathcal{O}(N^2)$ |

For $N = 128$ and $r = 64$:
- Encoder: $\sim 2 \times 10^6$ FLOPs per head
- Decoder: $\sim 4 \times 10^6$ FLOPs per head
- **Encoder speedup: ~2×**

For $N = 512$ and $r = 64$:
- Encoder: $\sim 8 \times 10^6$
- Decoder: $\sim 6.4 \times 10^7$
- **Encoder speedup: ~8×**

## 4.8 Parameter Analysis

| Component | Baseline | Galerkin |
|---|---|---|
| Base weights | $4 d^2$ | $4 d^2$ (frozen) |
| LoRA ($r_L$) | 0 | $8 r_L d$ |
| **Total per layer** | $4 d^2$ | $4 d^2 + 8 r_L d$ |

For DistilBERT ($d = 768$, $r_L = 16$): **+589,824 params** across 6 layers.

**Galerkin Attention is not a compression method.**

## 4.9 Theoretical Notes

### 4.9.1 Galerkin Approximation Error

**Claim.** The Galerkin approximation error for the attention operator is 
bounded by the DCT tail energy:

$$
\|\mathcal{K} Q - \hat{\mathcal{K}} \hat{Q}\|_2 \leq C \cdot \sqrt{\sum_{k > r} \sigma_k^2}
\tag{4.13}
$$

where $\sigma_k$ are the singular values of the kernel operator.

**Proof sketch.** Standard Galerkin theory (Strang, 1973, Theorem 5.2). 
The error is dominated by the tail of the eigen-decomposition of $\mathcal{K}$. 
DCT captures the low-frequency modes optimally. $\square$

### 4.9.2 Energy Preservation Scaling

**Claim.** The scaling $\sqrt{N/r}$ preserves the expected output energy 
under the assumption that $O_\Phi$ coefficients are i.i.d.

**Proof.** For $O_\Phi \in \mathbb{R}^{r \times d_h}$ with i.i.d. entries 
variance $\sigma^2$, the reconstruction $Y = \Phi O_\Phi$ has entries:

$$
Y_{n, j} = \sum_k \Phi_{n,k} (O_\Phi)_{k, j}
\tag{4.14}
$$

with variance:

$$
\text{Var}(Y_{n,j}) = \sum_k \Phi_{n,k}^2 \sigma^2 = \sigma^2 \cdot \|\Phi_{n,:}\|_2^2 = \sigma^2
\tag{4.15}
$$

since $\Phi$ has orthonormal columns. The output energy per sequence 
position is thus $\sigma^2$, matching the expected basis energy $\sigma^2 \cdot r$. 
The $\sqrt{N/r}$ factor compensates for the fact that $r$ basis modes must 
represent $N$ sequence positions. $\square$

### 4.9.3 Causality Preservation for Encoders

Encoders have no causal constraint. The Galerkin projection preserves 
bidirectional context. $\square$

## 4.10 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| Rank $r$ | 64 | Higher = more fidelity, slower |
| Basis | DCT-II | Only option |
| $\epsilon$ (regularization) | $10^{-4}$ | Numerical stability |
| LoRA rank | 16 | Adapter capacity |
