<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Adaptive Span Attention (ASA-Span) replaces the standard softmax attention 
with a **span-masked** variant:

1. Compute standard scores $S = QK^\top / \sqrt{d_h}$.
2. Apply causal, padding, and position-bias masks.
3. Apply the **per-head soft ramp** $g_{z_h}(d)$.
4. Renormalize via softmax.
5. Multiply with $V$.

**Figure 4.1** illustrates the pipeline.

```
    Q, K, V ∈ R^{B × H × L × d_h}
             │
             ▼
    ┌──────────────────────────────┐
    │  S = Q K^T / √d_h            │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Apply masks (causal, pad)   │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Per-head ramp:              │
    │  g_z(d) = clamp((R+z-d)/R)   │
    │  S̃ = S + log(g_z + ε)        │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  P = softmax(S̃)              │
    └──────────────┬───────────────┘
                   │
                   ▼
              O = P · V
```

## 4.2 Learnable Span

### 4.2.1 Parameterization

For each of the $H$ heads, we introduce a scalar logit $\ell_h \in \mathbb{R}$. 
The span is:

$$
z_h = \sigma(\ell_h) \cdot S_\text{max}
\tag{4.1}
$$

where $\sigma$ is the sigmoid and $S_\text{max}$ is the maximum span 
(hyperparameter).

### 4.2.2 Initialization

We initialize $\ell_h$ to a value that makes $z_h \approx S_\text{max}$:

$$
\ell_h^{(0)} = \log \frac{0.982}{1 - 0.982} \approx 4.0
\tag{4.2}
$$

This gives $\sigma(4.0) \approx 0.982$, so $z_h \approx 0.982 \cdot S_\text{max}$. 
The span starts **nearly fully open** and shrinks during training via the 
L1 penalty.

### 4.2.3 Why Start Open?

Starting with a small span would immediately hurt accuracy. Starting open 
lets the model decide which spans to shrink, based on task loss.

## 4.3 Soft Ramp Mask

### 4.3.1 Definition

Given query position $i$ and key position $j$, define the distance:

$$
d_{ij} = \begin{cases} |i - j| & \text{if bidirectional} \\ \max(0, i - j) & \text{if causal} \end{cases}
\tag{4.3}
$$

For causal models, keys after the query ($j > i$) are masked separately; 
the ramp only applies to **past** keys.

The ramp mask is:

$$
g_{z_h}(d) = \text{clamp}\!\left(\frac{R + z_h - d}{R}, 0, 1\right)
\tag{4.4}
$$

### 4.3.2 Properties

- **Continuous:** $g_z$ is continuous in $z$ and $d$.
- **Piecewise-linear:** Three regimes (full, linear decay, zero).
- **Differentiable in $z$:**
  $$
  \frac{\partial g_z(d)}{\partial z} = \begin{cases} 0 & d < z \\ 1/R & z \le d \le z + R \\ 0 & d > z + R \end{cases}
  \tag{4.5}
  $$
- **Bounded:** $g_z \in [0, 1]$.

### 4.3.3 Numerical Stability

We compute the ramp in FP32 and apply in log space:

$$
\tilde{S}_{ij} = S_{ij} + \log(g_{z_h}(d_{ij}) + \epsilon)
\tag{4.6}
$$

with $\epsilon = 10^{-8}$. Then standard softmax.

## 4.4 Renormalization

After masking, the attention row does not sum to 1. We renormalize:

$$
P_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_j \exp(\tilde{S}_{ij})}
\tag{4.7}
$$

This is equivalent to applying the ramp as a **multiplicative factor** before 
softmax:

$$
P_{ij} = \frac{g_{z_h}(d_{ij}) \cdot \exp(S_{ij})}{\sum_j g_{z_h}(d_{ij}) \cdot \exp(S_{ij})}
\tag{4.8}
$$

## 4.5 Fast Path Optimization

If the ramp has no effect (all heads have $z_h + R \geq L$), we skip ramp 
computation:

```python
if not ((span + self.ramp_size) >= N_k).all():
    # compute ramp
    ...
```

This avoids the $\mathcal{O}(H L^2)$ ramp allocation when unnecessary.

## 4.6 L1 Span Penalty

### 4.6.1 Formulation

We add to the loss:

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{task} + \lambda \sum_{h=1}^H z_h
\tag{4.9}
$$

### 4.6.2 Effect

The gradient w.r.t. $\ell_h$:

$$
\frac{\partial \mathcal{L}_\text{span}}{\partial \ell_h} = \lambda \cdot S_\text{max} \cdot \sigma(\ell_h)(1 - \sigma(\ell_h))
\tag{4.10}
$$

always negative (pushes $z_h$ toward 0). The task loss pushes $z_h$ up 
when the model needs long-range context. The equilibrium depends on 
$\lambda$.

### 4.6.3 Recommended Values

| $\lambda$ | Effect |
|---|---|
| 0.0 | No compression; spans stay open |
| 0.001 | Mild compression |
| 0.01 | Balanced (recommended) |
| 0.1 | Aggressive; likely accuracy drop |

We use $\lambda = 0.01$ by default.

## 4.7 Complete Forward Pass

**Algorithm 1:** Adaptive Span Attention forward
```
Input:  Q, K, V ∈ R^{B × H × L × d_h}
        span_logits ℓ ∈ R^H
        max_span S_max, ramp_size R
        attention_mask, is_causal
Output: context ∈ R^{B × H × L × d_h}

1:  span ← sigmoid(ℓ) * S_max                # (H,)
2:
3:  S ← Q @ K^T * scale                      # (B, H, L, L)
4:
5:  S ← apply_attention_mask(S, attention_mask)
6:  if is_causal:
7:      S ← S + causal_mask
8:
9:  # Apply ramp mask
10: if (span + R).min() < L:
11:     d_q ← arange(L)[:, None]
12:     d_k ← arange(L)[None, :]
13:     if is_causal:
14:         dist ← clamp(d_q - d_k, min=0)
15:     else:
16:         dist ← abs(d_q - d_k)
17:     span_b ← span.view(1, H, 1, 1)
18:     ramp ← clamp((R + span_b - dist) / R, 0, 1)
19:     S ← S + log(ramp + ε)
20: end if
21:
22: P ← softmax(S)
23: P ← dropout(P)
24: O ← P @ V
25: return O
```

## 4.8 Complexity Analysis

| Component | Dense | ASA-Span |
|---|---|---|
| Score matrix | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H L^2)$ |
| Ramp mask | — | $\mathcal{O}(B H L^2)$ |
| Softmax | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H L^2)$ |

**Note.** ASA-Span does **not reduce complexity**. The full score matrix is 
still computed. Its benefit is **interpretability** and **pruning** — the 
model learns which spans matter. To realize speedup, one would apply a hard 
mask after training and use sparse kernels.

## 4.9 Parameter Analysis

| Component | Baseline | ASA-Span |
|---|---|---|
| $W_Q$ | $d H d_h$ | Same |
| $W_K$ | $d H_{kv} d_h$ | Same |
| $W_V$ | $d H_{kv} d_h$ | Same |
| $W_O$ | $H d_h d$ | Same |
| **Span logits** | 0 | **$H$** |
| **LoRA** | 0 | $2 r_L (\cdots)$ |

ASA-Span adds **$H$ parameters** (one logit per head). It does not reduce 
parameter count.

## 4.10 Theoretical Notes

### 4.10.1 Approximation Quality

For a head with span $z$ and ramp $R$, the approximation error is bounded by:

$$
\|P - \tilde{P}\|_1 \leq \sum_{(i,j) : |i-j| > z} P_{ij} + \epsilon_R
\tag{4.11}
$$

where $\epsilon_R$ is the ramp-induced error:

$$
\epsilon_R \leq \frac{R}{\sum_j g_z(d_{ij}) \exp(S_{ij})} \cdot \sum_j \exp(S_{ij})
\tag{4.12}
$$

### 4.10.2 Head Span Heterogeneity

Empirically, spans converge to a bimodal distribution:

$$
z_h^{(T)} \approx \begin{cases} 16 & \text{if head is local} \\ S_\text{max} & \text{if head is global} \end{cases}
\tag{4.13}
$$

### 4.10.3 Equilibrium of L1 Penalty

The equilibrium span satisfies:

$$
\frac{\partial \mathcal{L}_\text{task}}{\partial z_h} + \lambda = 0
\tag{4.14}
$$

The task loss derivative depends on the model, but in practice heads with 
small gradients converge to small spans.

---