# 4. Method

## 4.1 Overview

LEKAv3 replaces standard attention with a two-branch mechanism:

```
    Q, K, V
        │
        ▼
    ┌───────────────────────────┐
    │ RoPE (if applicable)      │
    └───────────┬───────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
    ┌───────────┐  ┌───────────────┐
    │ Global    │  │ Local         │
    │ Landmark  │  │ Sliding       │
    │ Branch    │  │ Window        │
    └─────┬─────┘  └───────┬───────┘
          │                │
          └────────┬───────┘
                   ▼
          ┌─────────────────┐
          │ Gated Fusion    │
          │ σ(g) per head   │
          └────────┬────────┘
                   ▼
             Output
```

## 4.2 Kernel Feature Map

We map each head vector $x \in \mathbb{R}^{d_h}$ to feature space:

$$
\phi(x) = \frac{1}{\sqrt{d_\text{feat}}} \begin{pmatrix} \cos(W_1 x) \\ \sin(W_1 x) \end{pmatrix}
\tag{4.1}
$$

where $W_1 \in \mathbb{R}^{d_h \times d_\text{feat}/2}$ is a random 
projection matrix. Feature dimension is $d_\text{feat}$ (default 128).

**Multi-scale.** We use two projections $W_1, W_2$ with different scales 
$\sqrt{d_h/2}$ and $\sqrt{d_h/4}$. The concatenated feature dimension is 
$d_\text{feat} + d_\text{feat}/2 = 1.5 d_\text{feat}$.

**Why cos/sin?** These provide a bounded, well-conditioned approximation 
of the softmax kernel without the exploding variance of pure RFF.

## 4.3 Landmark Selection

### 4.3.1 Scorer

A small MLP scores each key-feature:

$$
s_i = f_\theta(K_\text{feat}[i]) \in \mathbb{R}
\tag{4.2}
$$

where $f_\theta$ is: Linear($2 d_\text{feat}$, 64) → GELU → Linear(64, 1).

### 4.3.2 Selection

We retain the top-$L_\text{land}$ indices:

$$
\mathcal{L} = \text{TopK}(s, L_\text{land})
\tag{4.3}
$$

with the constraint that index 0 is always included:

$$
s_0 \leftarrow 10^4
\tag{4.4}
$$

Padding tokens are masked:

$$
s_i \leftarrow -10^4 \quad \text{for padding}
\tag{4.5}
$$

### 4.3.3 Effective Landmark Count

To balance compute and accuracy across sequence lengths:

$$
L_\text{land} = \min(N, \max(L_0, \lceil N/2 \rceil))
\tag{4.6}
$$

where $L_0$ is the base budget (default 64). This means:
- Short sequences ($N \leq L_0$): $L_\text{land} = L_0$
- Medium ($L_0 < N \leq 2 L_0$): $L_\text{land} = N/2$
- Long ($N > 2 L_0$): $L_\text{land} = N/2$

## 4.4 Global Landmark Attention

### 4.4.1 Landmark Features

Gather landmark features and values:

$$
K_\mathcal{L} = K_\text{feat}[\mathcal{L}], \qquad V_\mathcal{L} = V[\mathcal{L}]
\tag{4.7}
$$

### 4.4.2 Correlation Matrix

LEKAv3 applies a per-head learnable correlation:

$$
V_\mathcal{L}^\text{adj} = V_\mathcal{L} \cdot W_\text{corr}
\tag{4.8}
$$

where $W_\text{corr} \in \mathbb{R}^{d_h \times d_h}$ is initialized to 
identity. This lets each head adjust the landmark values.

### 4.4.3 Attention

$$
\text{attn}_{ij} = \text{softpick}\!\left(\phi(q_i) \cdot \phi(k_{\mathcal{L}_j})\right)
\tag{4.9}
$$

The output:

$$
\text{Out}_\text{land}[i] = \sum_j \text{attn}_{ij} \cdot V_\mathcal{L}^\text{adj}[j]
\tag{4.10}
$$

### 4.4.4 Causal Masking

For decoders:

$$
\text{attn}_{ij} \leftarrow -\infty \quad \text{if } \mathcal{L}_j > i
\tag{4.11}
$$

The mask uses original positions from `land_idx`.

## 4.5 Local Sliding Window

### 4.5.1 Window Definition

$$
\text{window}(i, j) = |i - j| \leq W/2
\tag{4.12}
$$

### 4.5.2 Local Attention

Standard scaled dot-product:

$$
\text{scores}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}} \cdot \mathbb{1}[\text{window}(i,j)]
\tag{4.13}
$$

For decoders, additionally mask $j > i$:

$$
\text{scores}_{ij} \leftarrow -\infty \quad \text{if } j > i
\tag{4.14}
$$

### 4.5.3 Output

$$
\text{Out}_\text{local} = \text{softmax}(\text{scores}) \cdot V
\tag{4.15}
$$

## 4.6 Gated Fusion

The final output combines the two branches:

$$
\text{Out} = \sigma(g) \cdot \text{Out}_\text{land} + (1 - \sigma(g)) \cdot \text{Out}_\text{local}
\tag{4.16}
$$

where $g \in \mathbb{R}^H$ is a per-head learnable bias. We initialize 
$g = 2.0$ so that $\sigma(g) \approx 0.88$, biasing the model toward the 
global branch initially.

**Why this initialization?** The landmark branch provides global context 
which is more important at the start. As training progresses, the gate can 
shift toward local.

## 4.7 Full Algorithm

**Algorithm 1:** LEKAv3 forward
```
Input:  X ∈ R^{B×L×d}, attention_mask
Output: Y ∈ R^{B×L×d}

1:  Q ← X W_Q; K ← X W_K; V ← X W_V
2:  if RoPE: Q, K ← RoPE(Q, K)
3:  if GQA: K, V ← repeat_kv(K, V)
4:  Q_feat ← φ(Q); K_feat ← φ(K)
5:  s ← landmark_scorer(K_feat)
6:  s[0] ← 1e4; s[padding] ← -1e4
7:  L ← TopK(s, L_land)
8:  K_L ← K_feat[L]; V_L ← V[L]
9:  attn_land ← softpick(Q_feat @ K_L^T)
10: if causal: attn_land ← mask(attn_land, L > i)
11: Out_land ← attn_land @ (V_L @ W_corr)
12: scores_local ← Q @ K^T / √d_h
13: scores_local ← mask_window(scores_local)
14: if causal: scores_local ← mask_causal(scores_local)
15: Out_local ← softmax(scores_local) @ V
16: g ← σ(gate)
17: Out ← g · Out_land + (1-g) · Out_local
18: Y ← Out_proj(Out)
19: return Y
```

## 4.8 Complexity Analysis

| Component | Time | Space |
|---|---|---|
| Q/K/V projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Kernel features | $\mathcal{O}(L d_h d_\text{feat})$ | $\mathcal{O}(L d_\text{feat})$ |
| Landmark scoring | $\mathcal{O}(L d_\text{feat})$ | $\mathcal{O}(L)$ |
| TopK | $\mathcal{O}(L \log L_\text{land})$ | $\mathcal{O}(L)$ |
| Landmark attention | $\mathcal{O}(L L_\text{land} d_h)$ | $\mathcal{O}(L L_\text{land})$ |
| Local attention | $\mathcal{O}(L W d_h)$ | $\mathcal{O}(L W)$ |
| Output projection | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| **Total** | $\mathcal{O}(L L_\text{land} d_h + L W d_h)$ | $\mathcal{O}(L(L_\text{land} + W))$ |

**Comparison to standard attention:** $\mathcal{O}(L^2 d_h)$. LEKAv3 is linear 
when $L_\text{land}, W \sim \mathcal{O}(1)$.

For $L = 128$:
- Standard: $L^2 = 16{,}384$
- LEKAv3: $L \cdot L_\text{land} + L \cdot W = 128 \cdot 64 + 128 \cdot 16 = 10{,}240$

For $L = 512$:
- Standard: $262{,}144$
- LEKAv3: $512 \cdot 256 + 512 \cdot 16 = 139{,}264$

Speedup grows with $L$.

## 4.9 Parameter Analysis

| Component | Baseline | LEKAv3 | Change |
|---|---|---|---|
| Q/K/V/O projections | $4 d^2$ | $4 d^2$ (frozen) | 0 |
| LoRA adapters | 0 | $4 r_L (d + d)$ | +small |
| Kernel projections | 0 | $d_h d_\text{feat} \cdot 1.5$ | +small |
| Landmark scorer | 0 | $2 d_\text{feat} \cdot 64 + 64 + 1$ | +small |
| $W_\text{corr}$ | 0 | $H d_h^2$ | +medium |
| Gate | 0 | $H$ | negligible |

For $d = 768$, $H = 12$, $d_h = 64$, $d_\text{feat} = 128$:

| Component | Params |
|---|---|
| Baseline (per layer) | 2,359,296 |
| LoRA (rank 16) | 98,304 |
| Kernel projections | ~12,288 |
| Landmark scorer | 16,513 |
| W_corr | 49,152 |
| **Total added** | ~176,257 |
| **Change** | **+7.5%** |

LEKAv3 **increases** parameter count by ~7.5% per layer.

## 4.10 Theoretical Notes

### 4.10.1 Kernel Approximation

Under the RFF guarantee (Rahimi & Recht, 2007):

$$
\sup_{x, y} |\phi(x)^\top \phi(y) - e^{x \cdot y}| \leq \epsilon
\tag{4.17}
$$

with $d_\text{feat} = \mathcal{O}(d_h \log(1/\epsilon^2))$.

### 4.10.2 Landmark Selection Guarantee

**Claim.** If landmarks are chosen to maximize coverage of the key 
distribution, then landmark attention approximates full attention.

**Sketch.** Given the query's attention concentrates on a small set of 
keys (Zhang et al., 2023), selecting those keys as landmarks preserves 
the output. $\square$

### 4.10.3 Causality

**Claim.** For decoders, output at position $i$ depends only on inputs 
$j \leq i$.

**Proof.** Both the landmark attention (via causal mask on original 
positions) and the local attention (via triangular mask) enforce $j \leq i$. 
The gate is a scalar independent of positions. $\square$

---