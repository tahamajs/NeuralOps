# 4. Method

## 4.1 Overview

LEKAv1 replaces standard attention with a kernel approximation pipeline:

1. **Project** Q, K, V (frozen base + LoRA).
2. **Normalize** each head with LayerNorm.
3. **Map** to features via QR-orthogonalized RFF.
4. **Aggregate** causally (decoder) or globally (encoder).
5. **Output** via projection + preservation gate.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
              ▼
      Q/K/V Projection
              │
              ▼
        Head Norm
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
      φ(Q)  φ(K)   V
       │      │     │
       │      ▼     │
       │  KV = φ(K)^T V
       │  Z  = φ(K)^T 1
       │      │     │
       └──────┼─────┘
              ▼
         A = φ(Q) KV / (φ(Q) Z)
              │
              ▼
        out_proj (LoRA)
              │
              ▼
      α·A + (1-α)·X
              │
              ▼
           Output Y
```

## 4.2 The Kernel Feature Map

**Step 1: Head-wise LayerNorm.**

$$
\tilde{x} = \text{LN}(x) \in \mathbb{R}^{d_h}
\tag{4.1}
$$

**Step 2: Random projection.**

$$
f = \tilde{x} W, \quad W \in \mathbb{R}^{d_h \times F}
\tag{4.2}
$$

**Step 3: cos/sin feature.**

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(f) \| \sin(f)] \in \mathbb{R}^{2F}
\tag{4.3}
$$

**Kernel interpretation.** For $x, y \in \mathbb{R}^{d_h}$:

$$
\phi(x)^\top \phi(y) = \frac{1}{F} \sum_{i=1}^F \cos(w_i^\top (x - y))
\tag{4.4}
$$

which is the unbiased RFF estimator of $\exp(-\|x - y\|^2 / 2)$.

## 4.3 Projection Matrix Initialization

We initialize $W \in \mathbb{R}^{d_h \times F}$ via QR orthogonalization:

**Case 1:** $F \leq d_h$ (typical):

$$
G \sim \mathcal{N}(0, I_{d_h \times F}), \quad G = Q R, \quad W = Q
\tag{4.5}
$$

This ensures $W^\top W = I_F$.

**Case 2:** $F > d_h$:

We concatenate orthogonal blocks of size $d_h$:

$$
W = [Q_1 \| Q_2 \| \dots], \quad Q_k \in \mathbb{R}^{d_h \times d_h}
\tag{4.6}
$$

Both cases give approximately orthonormal columns.

## 4.4 Linear Attention Computation

### 4.4.1 Causal (Decoder)

For decoders, we compute cumulative sums:

$$
\text{KV}_i = \sum_{j \leq i} \phi(k_j) v_j^\top \in \mathbb{R}^{2F \times d_h}
\tag{4.7}
$$

$$
\text{Z}_i = \sum_{j \leq i} \phi(k_j) \in \mathbb{R}^{2F}
\tag{4.8}
$$

Implemented as `torch.cumsum` on the outer product $\phi(k) \otimes v$.

**Attention output:**

$$
A_i = \frac{\phi(q_i)^\top \text{KV}_i}{\phi(q_i)^\top \text{Z}_i + \epsilon}
\tag{4.9}
$$

### 4.4.2 Bidirectional (Encoder)

For encoders:

$$
\text{KV} = \sum_{j=1}^L \phi(k_j) v_j^\top, \quad
\text{Z} = \sum_{j=1}^L \phi(k_j)
\tag{4.10}
$$

$$
A_i = \frac{\phi(q_i)^\top \text{KV}}{\phi(q_i)^\top \text{Z} + \epsilon}
\tag{4.11}
$$

## 4.5 Masking

**Padding.** We apply the mask before aggregation:

$$
\phi(k_j) \leftarrow \phi(k_j) \cdot m_j, \quad
v_j \leftarrow v_j \cdot m_j
\tag{4.12}
$$

Then $\text{KV}$ and $\text{Z}$ automatically account for valid tokens.

**Causality.** Handled by cumsum.

## 4.6 Preservation Gate

$$
\alpha = \sigma(w), \quad w_0 = 2
\tag{4.13}
$$

$$
\text{out} = \alpha \odot A + (1 - \alpha) \odot X
\tag{4.14}
$$

**Why this gate?**

1. **Residual safety:** If the kernel approximation is poor, the model 
   can lean toward identity.
2. **Numerical stability:** At initialization, $\alpha \approx 0.88$, 
   giving strong attention but retaining residual structure.
3. **Graceful degradation:** Under distribution shift, the gate can 
   shift toward identity.

## 4.7 Output Scaling

We also learn an `output_scale` parameter:

$$
\text{out} = s \cdot \text{out}, \quad s_0 = 1
\tag{4.15}
$$

This provides a per-layer gain that can be adjusted during recovery.

## 4.8 Algorithm

**Algorithm 1:** LEKAv1 forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×L×d}

 1: Q ← query(X);  K ← key(X);  V ← value(X)
 2: reshape to (B, H, L, d_h)
 3: if RoPE: Q, K ← apply_rotary(Q, K, pos)
 4: if GQA:  K, V ← repeat_kv
 5: φ_Q ← kernel_map(Q)             ▷ (B, H, L, 2F)
 6: φ_K ← kernel_map(K)
 7: if mask: φ_K ← φ_K · m; V ← V · m
 8: if decoder:
 9:     KV ← cumsum(φ_K ⊗ V)          ▷ (B, H, L, 2F, d_h)
10:     Z  ← cumsum(φ_K)              ▷ (B, H, L, 2F)
11: else:
12:     KV ← sum(φ_K ⊗ V)
13:     Z  ← sum(φ_K)
14: A ← (φ_Q · KV) / (φ_Q · Z + ε)
15: A ← reshape to (B, L, d)
16: A ← out_proj(A)
17: α ← sigmoid(pres_gate)
18: Y ← α · A + (1 - α) · X
19: Y ← Y · output_scale
20: return Y
```

## 4.9 Complexity Analysis

For a single block:

| Component | Time | Space |
|---|---|---|
| Q/K/V projections | $O(L d^2)$ | $O(L d)$ |
| Feature map | $O(L d F)$ | $O(L F)$ |
| Outer product | $O(L F d_h)$ | $O(L F d_h)$ or $O(F d_h)$ (causal) |
| Aggregation | $O(L F d_h)$ | $O(F d_h)$ |
| Output | $O(L d^2)$ | $O(L d)$ |
| **Total** | $O(L d (d + F))$ | $O(L d)$ |

Compare to standard attention ($O(L^2 d)$). For $L \gg F$, LEKAv1 is 
**much faster**.

**Causal memory.** The causal version materializes the cumulative KV 
of size $(B, H, L, 2F, d_h)$, which is $O(L F d_h)$ — linear in L. 
This is still much smaller than $O(L^2)$.

## 4.10 Parameter Analysis

For $d = 768$, $F = 256$, $H = 12$, $d_h = 64$, LoRA $r_L = 16$:

| Component | Baseline | LEKAv1 |
|---|---|---|
| Q/K/V/O projections | $4 d^2 = 2.36M$ | $2.36M$ (frozen) |
| LoRA on Q/K/V/O | 0 | $4 \cdot 2 r_L d = 98K$ |
| Random projection | 0 | $H d_h F = 196K$ |
| LayerNorm | 0 | $2 d_h = 128$ |
| Preservation gate | 0 | 1 |
| Output scale | 0 | 1 |
| **Total per layer** | 2.36M | 2.65M |
| **Increase** | — | **+12%** |

The parameter increase is modest: **~12%**.

---