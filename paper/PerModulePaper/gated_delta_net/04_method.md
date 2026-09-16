<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Gated DeltaNet has four components:

1. **Projections:** $Q, K, V, Z$ from hidden states.
2. **Gating:** per-head scalars $\alpha_t, \beta_t$ from hidden states.
3. **Recurrence:** state update via the Delta Rule with gating.
4. **Bidirectional scan** (encoders only) or **causal scan** (decoders).

```
                Input X
                   │
        ┌──────────┼──────────┬──────────┐
        │          │          │          │
        ▼          ▼          ▼          ▼
    ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
    │  Q   │  │  K   │  │  V   │  │  Z   │
    │ proj │  │ proj │  │ proj │  │ proj │
    └───┬──┘  └───┬──┘  └───┬──┘  └───┬──┘
        │         │         │         │
        │         │         │         └───────┐
        │         │         │                 │
        │  ┌──────┴──────┐  │                 │
        │  │  β, α gates │  │                 │
        │  └──────┬──────┘  │                 │
        │         │         │                 │
        └─────────┴─────────┴─────┐           │
                                 │           │
                                 ▼           │
                        ┌────────────────┐   │
                        │ Delta Rule     │   │
                        │ Recurrence     │   │
                        └────────┬───────┘   │
                                 │           │
                                 ▼           ▼
                        ┌──────────────────────┐
                        │  SiLU gating         │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  out_proj            │
                        └──────────────────────┘
```

## 4.2 Projections

Given hidden states $X \in \mathbb{R}^{B \times L \times d}$:

$$
\begin{aligned}
Q &= X W_Q \in \mathbb{R}^{B \times L \times H d_h} \\
K &= X W_K \in \mathbb{R}^{B \times L \times H_{kv} d_h} \\
V &= X W_V \in \mathbb{R}^{B \times L \times H_{kv} d_h} \\
Z &= X W_Z \in \mathbb{R}^{B \times L \times H d_h}
\end{aligned}
\tag{4.1}
$$

**Note.** $W_Z$ is a **new parameter** not present in the pretrained model. 
We initialize it from $W_Q$ (see Section 5.3).

## 4.3 Gating

Per-head scalars are computed from the input:

$$
\alpha_t = \sigma(W_\alpha x_t + b_\alpha), \quad
\beta_t = \sigma(W_\beta x_t + b_\beta)
\tag{4.2}
$$

where $\sigma$ is the sigmoid function and $W_\alpha, W_\beta \in \mathbb{R}^{H \times d}$.

**Initialization.** We set $b_\beta = 2.0$ so that $\beta_0 \approx 0.88$, 
encouraging active state updates early in training. We set $b_\alpha = 0$ 
so that $\alpha_0 = 0.5$.

## 4.4 The Delta Rule Recurrence

The state $S \in \mathbb{R}^{d_h \times d_h}$ is updated as:

$$
\boxed{
S_t = \beta_t S_{t-1} + \alpha_t \cdot \frac{k_t^\top (v_t - k_t S_{t-1})}{1 + \|k_t\|^2}
}
\tag{4.3}
$$

where:
- $k_t \in \mathbb{R}^{d_h}$ is the normalized key.
- $v_t \in \mathbb{R}^{d_h}$ is the value.
- $k_t S_{t-1}$ is the **prediction** of $v_t$ from the current state.
- $v_t - k_t S_{t-1}$ is the **prediction error**.
- The $1 / (1 + \|k_t\|^2)$ normalization stabilizes the update.

**Output:**

$$
y_t = q_t S_t
\tag{4.4}
$$

where $q_t$ is the normalized query.

## 4.5 Bidirectional Scan (Encoders)

For encoder models, we compute two independent scans:

$$
y_\text{fwd} = \text{scan}(Q, K, V, \alpha, \beta)
\tag{4.5}
$$

$$
y_\text{bwd} = \text{flip}(\text{scan}(\text{flip}(Q), \text{flip}(K), \text{flip}(V), 
\text{flip}(\alpha), \text{flip}(\beta)))
\tag{4.6}
$$

The final output is the average:

$$
y = \frac{1}{2}(y_\text{fwd} + y_\text{bwd})
\tag{4.7}
$$

## 4.6 Causal Scan (Decoders)

For decoder models, we use only the forward scan:

$$
y = y_\text{fwd}
\tag{4.8}
$$

This preserves causality: $y_t$ depends only on $S_t$, which depends only 
on $X_1, \dots, X_t$.

## 4.7 Chunked Implementation

The reference implementation uses **chunked recurrence** to amortize Python 
overhead:

1. Pad sequence to multiple of `chunk_size = 8`.
2. Split into `num_chunks = L / 8` chunks.
3. **Outer loop** over chunks: for each chunk, call `_delta_chunk_forward`.
4. **Inner loop** over tokens within chunk (8 iterations).

**Complexity.** $\mathcal{O}(L / C \cdot C) = \mathcal{O}(L)$ Python 
iterations, where $C$ is the chunk size.

**This is the primary bottleneck** (see Section 8).

## 4.8 SiLU Gating

After the recurrence, we apply gated linear unit with SiLU:

$$
y_\text{final} = y \odot \text{SiLU}(Z)
\tag{4.9}
$$

This adds a trainable nonlinearity without introducing additional parameters 
beyond $W_Z$.

## 4.9 Algorithm

**Algorithm 1:** Gated DeltaNet forward pass
```
Input:  X ∈ R^{B×L×d}, is_decoder, chunk_size C
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V;  Z ← X W_Z
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA:  (K, V) ← repeat_kv(K, V)
 4: Q ← normalize(Q);  K ← normalize(K) / sqrt(d)
 5: β ← sigmoid(W_β X);  α ← sigmoid(W_α X)   ▷ [B, H, L]
 6:
 7: # Forward scan
 8: y_fwd ← chunked_scan(Q, K, V, α, β, C)
 9:
10: if not is_decoder:
11:     # Backward scan
12:     Q' ← flip(Q); K' ← flip(K); V' ← flip(V)
13:     α' ← flip(α); β' ← flip(β)
14:     y_bwd ← flip(chunked_scan(Q', K', V', α', β', C))
15:     y ← 0.5 * (y_fwd + y_bwd)
16: else:
17:     y ← y_fwd
18:
19: y ← y ⊙ SiLU(Z)
20: return out_proj(y)
```

## 4.10 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Projections | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(Ld)$ |
| Gating | $\mathcal{O}(LdH)$ | $\mathcal{O}(LH)$ |
| Recurrence (FLOPs) | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(d^2)$ |
| Recurrence (wall-clock) | $\mathcal{O}(L)$ sequential | — |
| SiLU gating | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(Ld)$ |
| **Baseline MHSA** | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |

**For $L = 128$, $d = 64$:**
- FLOPs: Gated DeltaNet is ~2× fewer.
- Wall-clock: Gated DeltaNet is **~45× slower** due to Python loops.

## 4.11 Parameter Analysis

| Component | Params per layer |
|---|---|
| Q projection + LoRA | $d \cdot H d_h + r_L(d + H d_h)$ |
| K projection + LoRA | $d \cdot H_{kv} d_h + r_L(d + H_{kv} d_h)$ |
| V projection + LoRA | same as K |
| **Z projection + LoRA** | $d \cdot H d_h + r_L(d + H d_h)$ ← **new** |
| O projection + LoRA | $H d_h \cdot d + r_L(H d_h + d)$ |
| $\alpha$ projection | $d \cdot H$ |
| $\beta$ projection | $d \cdot H$ |

**Example (DistilBERT, $r_L = 16$, $d = 768$):**
- Z projection: +589,824 params per layer
- $\alpha, \beta$: +18,432 per layer
- LoRA: +589,824 per layer
- **Total: +1.2M per layer × 6 layers = +7.2M (+10.7%)**

The **Z projection is the largest overhead**. It is not present in baseline 
attention.

---