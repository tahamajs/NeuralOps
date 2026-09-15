<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Galerkin Attention has two paths:

1. **Encoder path** (bidirectional models): project attention onto DCT basis.
2. **Decoder path** (causal models): fall back to standard causal attention.

```
                    Input X
                       │
                       ▼
                 ┌───────────┐
                 │  Q/K/V    │
                 │ projection│
                 └─────┬─────┘
                       │
                       ▼
                 ┌───────────┐
                 │ is_decoder│
                 └──┬────┬───┘
                    │ No │ Yes
                    │    │
                    │    ▼
                    │  ┌──────────────┐
                    │  │ Causal       │
                    │  │ attention    │
                    │  │ (standard)   │
                    │  └──────┬───────┘
                    │         │
                    ▼         │
            ┌───────────────┐ │
            │ Galerkin proj │ │
            │ + softmax     │ │
            └───────┬───────┘ │
                    │         │
                    └────┬────┘
                         │
                         ▼
                    Output Y
```

## 4.2 The Galerkin Basis

We construct the DCT-II basis $\Phi \in \mathbb{R}^{N \times r}$:

$$
\Phi_{n, k} = \cos\!\left(\frac{\pi k (2n + 1)}{2N}\right) \cdot
c_k, \quad c_k = \begin{cases} 1/\sqrt{N} & k = 0 \\ \sqrt{2/N} & k \geq 1 \end{cases}
\tag{4.1}
$$

We then orthonormalize via QR to correct for numerical error:

$$
\Phi \leftarrow Q \text{ where } \Phi + 10^{-4} I = Q R
\tag{4.2}
$$

## 4.3 Projection and Attention

For each head:

1. **Project** Q, K, V onto the basis:
   $$
   Q_\Phi = \Phi^\top Q \in \mathbb{R}^{B \times H \times r \times d}
   \tag{4.3}
   $$
   
2. **Compute attention** in the reduced space:
   $$
   A_\Phi = \mathrm{softmax}\!\left(\frac{Q_\Phi K_\Phi^\top}{\sqrt{d}}\right) \in \mathbb{R}^{B \times H \times r \times r}
   \tag{4.4}
   $$
   
3. **Apply to values:**
   $$
   O_\Phi = A_\Phi V_\Phi \in \mathbb{R}^{B \times H \times r \times d}
   \tag{4.5}
   $$
   
4. **Reconstruct** to original space:
   $$
   O = \Phi O_\Phi \in \mathbb{R}^{B \times H \times N \times d}
   \tag{4.6}
   $$

5. **Energy correction:**
   $$
   O \leftarrow O \cdot \sqrt{N / r}
   \tag{4.7}
   $$

## 4.4 Why the Energy Correction?

The projection $\Phi^\top$ is contractive: $\|\Phi^\top Q\|_F \leq \|Q\|_F$. 
For an orthonormal basis of $r$ columns out of $N$, the average energy 
preserved is $r/N$. Multiplying by $\sqrt{N/r}$ approximately restores 
the original norm.

**Caveat.** This correction is a heuristic and does not account for the 
actual distribution of energy across the basis. If energy is concentrated 
in the first $r$ components, the correction **overshoots**; if it is 
uniform, the correction is exact.

## 4.5 Decoder Fallback

For causal models, we use standard attention with explicit causal masking:

$$
\text{scores}_{ij} = \begin{cases}
Q_i K_j^\top / \sqrt{d} & j \leq i \\
-\infty & j > i
\end{cases}
\tag{4.8}
$$

This preserves the autoregressive property. The Galerkin path is **not** 
used for decoders.

## 4.6 Parameter Analysis

Galerkin Attention **adds parameters** via LoRA:

| Component | Params |
|---|---|
| Original attention | $d_\text{in} d_\text{out}$ |
| Q projection + LoRA | $d_\text{in} d_\text{out} + r_L(d_\text{in} + d_\text{out})$ |
| K projection + LoRA | same |
| V projection + LoRA | same |
| O projection + LoRA | same |
| **Total per layer** | $4 d_\text{in} d_\text{out} + 4 r_L(d_\text{in} + d_\text{out})$ |

**Example (DistilBERT, $r_L = 16$, $d = 768$):** +589,824 params per layer.

**Note.** Galerkin Attention is **not a compression method**. It aims to 
reduce **attention-matrix memory**, not parameter count.

## 4.7 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Q/K/V proj | $\mathcal{O}(Nd^2)$ | $\mathcal{O}(Nd)$ |
| Basis projection | $\mathcal{O}(Nrd)$ | $\mathcal{O}(Nrd)$ |
| Attention | $\mathcal{O}(r^2 d)$ | $\mathcal{O}(r^2)$ |
| Reconstruction | $\mathcal{O}(Nrd)$ | $\mathcal{O}(Nrd)$ |
| O proj | $\mathcal{O}(Nd^2)$ | $\mathcal{O}(Nd)$ |
| **Baseline MHSA** | $\mathcal{O}(N^2 d)$ | $\mathcal{O}(N^2)$ |
| **Galerkin** | $\mathcal{O}(Nrd + r^2 d)$ | $\mathcal{O}(Nr)$ |

For $r \ll N$, Galerkin attention is asymptotically faster in the 
attention matrix, but the projection/reconstruction overhead dominates 
for short sequences.

## 4.8 Algorithm

**Algorithm 1:** Galerkin Attention forward
```
Input:  X ∈ R^{B×N×d}, is_decoder
Output: Y ∈ R^{B×N×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA: (K, V) ← repeat_kv(K, V)
 4:
 5: if is_decoder:
 6:     A ← causal_softmax(Q K^T / sqrt(d))
 7:     O ← A V
 8: else:
 9:     Φ ← DCT_basis(N, r)               ▷ Orthonormal
10:     Q_Φ ← Φ^T Q
11:     K_Φ ← Φ^T K
12:     V_Φ ← Φ^T V
13:     A_Φ ← softmax(Q_Φ K_Φ^T / sqrt(d))
14:     O_Φ ← A_Φ V_Φ
15:     O ← Φ O_Φ
16:     O ← O * sqrt(N / r)               ▷ Energy correction
17:
18: return out_proj(O)
```

## 4.9 Theoretical Basis

The method is motivated by the following observation: for smooth attention 
kernels $k(x, y)$, projection onto the first $r$ DCT basis functions 
preserves energy at rate:

$$
\frac{\|\Phi_r^\top k\|_2}{\|k\|_2} \approx 1 - \mathcal{O}(r^{-2\alpha})
\tag{4.9}
$$

where $\alpha$ is the smoothness parameter of $k$. For $C^\infty$ kernels, 
the decay is super-polynomial.

**Empirical caveat.** Trained transformer attention kernels are **not** 
smooth. They are sharply peaked (attention sinks, heavy hitters) and highly 
content-dependent. This is why our method fails in practice (Section 8).

---
