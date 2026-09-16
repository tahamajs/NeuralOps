# 4. Method

## 4.1 Overview

LEKAv2 replaces standard attention with a hybrid pipeline:

1. **Project** Q, K, V (frozen base + LoRA).
2. **Map** to dual-scale RFF features.
3. **Select** landmarks via a learned scorer.
4. **Compute** global landmark attention with **softpick**.
5. **Compute** local sliding-window attention.
6. **Fuse** via per-head gates.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
         Q/K/V Projection (frozen + LoRA)
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
      Q      K      V
       │      │      │
       ▼      ▼      │
     φ(Q)   φ(K)     │  (Dual-scale RFF)
       │      │      │
       │      ▼      │
       │  Landmark  │
       │  Scorer    │
       │      │      │
       │      ▼      │
       │  K_land, V_land
       │      │      │
       └──────┼──────┘
              ▼
      ┌───────────────┐
      │  Softpick     │
      │  Attention    │
      │  (Global)     │
      └───────┬───────┘
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
    Local   Global   Gate
    Attn    Attn    (σ(w_h))
      │       │       │
      └───────┼───────┘
              ▼
      (1-g)Local + g·Global
              │
              ▼
          out_proj
              │
              ▼
           Output Y
```

## 4.2 Dual-Scale Random Features

Two projections:

$$
W_1 \in \mathbb{R}^{d_h \times F}, \quad W_2 \in \mathbb{R}^{d_h \times F/2}
\tag{4.1}
$$

Both initialized as $\mathcal{N}(0, 1/\sqrt{d_h})$ (scaled Gaussian).

**Feature map:**

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(x W_1) \| \sin(x W_1)] \| \frac{1}{\sqrt{F/2}} [\cos(x W_2) \| \sin(x W_2)]
\tag{4.2}
$$

Output dimension: $2F + F = 3F$.

**Why two scales?** The two projections capture different frequency 
bands, similar to multi-resolution analysis.

## 4.3 Landmark Distillation

**Scorer.** A small MLP:

$$
s_\theta(\phi(k)) = \text{MLP}(\text{LN}(\phi(k))) \in \mathbb{R}
\tag{4.3}
$$

with architecture: LayerNorm($3F$) → Linear($3F$, 64) → GELU → Linear(64, 1).

**Selection.**

$$
L_\text{max} = \min\!\left(\max(L_\text{min}, \lceil \rho L \rceil), L\right)
\tag{4.4}
$$

$$
\mathcal{S}_\text{land} = \text{TopK}(s_\theta, L_\text{max})
\tag{4.5}
$$

**Position 0 guarantee.** We set $s_\theta[:, 0] \leftarrow 10^4$ to force 
retention.

**Gather.** $K_\text{land} = K[\mathcal{S}_\text{land}]$, $V_\text{land} = V[\mathcal{S}_\text{land}]$.

## 4.4 Correction Matrix

Per-head learnable correction applied to landmark values:

$$
V_\text{land}^\text{adj} = V_\text{land} \cdot W_\text{corr}^h
\tag{4.6}
$$

where $W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$ is 
initialized to the identity per head.

**Purpose.** Allows the model to learn head-specific value transformations 
during recovery.

## 4.5 Softpick Attention

**Scores.**

$$
\text{scores}_{ij} = \phi(q_i)^\top \phi(K_\text{land})_j^\top
\tag{4.7}
$$

**Masking.** Padding and causal masks applied.

**Softpick.**

$$
m_i = \max_j \text{scores}_{ij}, \quad
p_{ij} = \text{ReLU}\!\left(e^{(\text{scores}_{ij} - m_i)/T} - e^{-m_i/T}\right)
\tag{4.8}
$$

with $T = 1.0$ and fallback to uniform when $\sum_j p_{ij} < 10^{-6}$.

**Global context.**

$$
A_\text{global} = \sum_j p_{ij} V_{\text{land}, j}^\text{adj}
\tag{4.9}
$$

## 4.6 Local Sliding Window

**Window mask.**

$$
\text{mask}_{ij} = \begin{cases} 1 & |i - j| \leq W/2 \\ 0 & \text{otherwise} \end{cases}
\tag{4.10}
$$

**Local scores.** Standard scaled dot-product:

$$
\text{local}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}} \cdot \text{mask}_{ij}
\tag{4.11}
$$

Then causal + padding masks.

**Local attention.**

$$
A_\text{local} = \text{softmax}(\text{local}) \cdot V
\tag{4.12}
$$

## 4.7 Local-Global Fusion

Per-head gate:

$$
g_h = \sigma(w_h), \quad w_h^{(0)} = -2
\tag{4.13}
$$

**Fusion.**

$$
\text{out}_h = (1 - g_h) \cdot A_\text{local}^h + g_h \cdot A_\text{global}^h
\tag{4.14}
$$

With $g_h^{(0)} \approx 0.12$, the local branch dominates initially.

## 4.8 Algorithm

**Algorithm 1:** LEKAv2 forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×L×d}

 1: Q ← query(X);  K ← key(X);  V ← value(X)
 2: reshape to (B, H, L, d_h)
 3: if RoPE: Q, K ← rotary(Q, K)
 4: if GQA:  K, V ← repeat_kv
 5: φ_Q ← dual_rff(Q)                 ▷ (B, H, L, 3F)
 6: φ_K ← dual_rff(K)
 7: L_max ← max(min_landmarks, ρ·L)
 8: scores ← landmark_scorer(φ_K)     ▷ (B, H, L)
 9: scores[:, :, 0] ← 1e4
10: idx ← topk(scores, L_max)
11: K_land ← gather(φ_K, idx); V_land ← gather(V, idx)
12: V_land_adj ← V_land · W_corr
13: A_glob ← softpick(φ_Q @ K_land^T) @ V_land_adj
14: A_loc ← softmax(masked(QK^T)) @ V  ▷ window + causal
15: g ← sigmoid(local_gate)            ▷ (H,)
16: out ← (1-g)·A_loc + g·A_glob
17: reshape to (B, L, d); out_proj
18: return Y
```

## 4.9 Complexity Analysis

Per block:

| Component | Time | Params |
|---|---|---|
| Q/K/V projections | $O(L d^2)$ | $3 d^2$ |
| Dual RFF | $O(L d (3F))$ | 0 (buffers) |
| Landmark scorer | $O(L \cdot 3F)$ | $O(F)$ |
| Landmark attention | $O(L \cdot L_\text{max} \cdot 3F)$ | 0 |
| Local window attention | $O(L \cdot W \cdot d_h)$ | 0 |
| Correction matrix | $O(L_\text{max} \cdot d_h^2)$ | $H d_h^2$ |
| Fusion | $O(L d)$ | $H$ |
| Output | $O(L d^2)$ | $d^2$ |
| **Total** | $O(L d^2 + L \cdot L_\text{max} \cdot F)$ | $\sim 4 d^2 + H d_h^2$ |

For $\rho = 0.5$ and $L \gg F$, the landmark term is 
$O(L^2 \cdot F)$ — **quadratic** in $L$! But with $L_\text{max} = \rho L$ 
and $\rho = 0.5$, we get $O(L^2)$ again.

**Note.** LEKAv2 is **not truly linear** unless we set $L_\text{max} = $ 
constant. The current implementation uses $L_\text{max} = \rho L$ — this 
is effectively **linear-quadratic** (still $O(L^2)$ but with a smaller 
constant).

## 4.10 Parameter Analysis

For $d = 768$, $H = 12$, $d_h = 64$, $F = 128$, LoRA $r_L = 16$:

| Component | Baseline | LEKAv2 |
|---|---|---|
| Q/K/V/O | $4 d^2 = 2.36M$ | $2.36M$ (frozen) |
| LoRA | 0 | $98K$ |
| RFF buffers | 0 | 0 (not params) |
| Landmark scorer | 0 | ~$9K$ |
| Correction matrix | 0 | $H d_h^2 = 49K$ |
| Local gate | 0 | 12 |
| **Total per layer** | 2.36M | 2.51M |
| **Increase** | — | **+6%** |

Modest parameter increase.

## 4.11 Training

Trainable:
- LoRA adapters (frozen base).
- Landmark scorer.
- Correction matrix.
- Local gate.

Frozen:
- Q/K/V/O base weights.
- Random projection buffers (not parameters).

---