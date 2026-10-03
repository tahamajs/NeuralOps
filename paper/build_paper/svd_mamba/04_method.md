<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

SVD-Mamba replaces MHSA with a three-path block:

1. **Attention path** (SVD-factorized projections).
2. **SSM path** (selective state space with chunked scan).
3. **Global mixing path** (low-rank global aggregation).

Fused by a gated combiner.

**Figure 4.1** shows the pipeline.

```
    ┌───────────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                          │
    └────────────────┬──────────────────────────────┘
                     │
        ┌────────────┼────────────────┐
        ▼            ▼                ▼
    ┌────────┐  ┌──────────┐    ┌──────────┐
    │ Attn   │  │ SSM      │    │ Global   │
    │ path   │  │ branch   │    │ mixing   │
    │ (SVD)  │  │ (Mamba)  │    │ (SVD)    │
    └────┬───┘  └─────┬────┘    └─────┬────┘
         │            │               │
         │            ▼               │
         │      ┌──────────┐          │
         │      │ + Global │          │
         │      └─────┬────┘          │
         │            │               │
         │            ▼               │
         │      ┌──────────┐          │
         │      │ out_proj │          │
         │      └─────┬────┘          │
         │            │               │
         └────────────┼───────────────┘
                      ▼
              ┌───────────────┐
              │ γ ⊙ SSM +     │
              │ (1-γ) ⊙ Attn  │
              └───────┬───────┘
                      ▼
                    Output
```

## 4.2 SVD-Factorized Linear

Each projection is stored as:

$$
W = U \Sigma V^\top, \qquad
U \in \mathbb{R}^{d_\text{out} \times r}, \; \Sigma \in \mathbb{R}^r, \;
V \in \mathbb{R}^{d_\text{in} \times r}
\tag{4.1}
$$

The forward pass is:

$$
y = ((x V) \odot \Sigma) U^\top + b
\tag{4.2}
$$

**Parameter count:** $r(d_\text{in} + d_\text{out} + 1)$ vs $d_\text{in} d_\text{out}$.

**LoRA recovery.** A residual LoRA path is added:

$$
y = ((x V) \odot \Sigma) U^\top + (x A^\top) B^\top \cdot s + b
\tag{4.3}
$$

where $s = \alpha / r_L$ and only $A, B$ are trained.

## 4.3 Attention Path

Standard MHSA with SVD-factorized projections:

$$
Q = \text{SVDLinear}_Q(X), \quad
K = \text{SVDLinear}_K(X), \quad
V_a = \text{SVDLinear}_V(X)
\tag{4.4}
$$

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_{a,h}
\tag{4.5}
$$

$$
Y_\text{attn} = \text{SVDLinear}_O(\text{Concat}(\text{head}_1, \dots, \text{head}_H))
\tag{4.6}
$$

**Causality.** For decoders, a triangular mask is applied to the score 
matrix.

## 4.4 SSM Path

### 4.4.1 Input Projection

The hidden states are projected to a gated 2×inner dimension:

$$
[x; z] = \text{SVDLinear}_\text{in}(X) \in \mathbb{R}^{B \times L \times 2 d_\text{inner}}
\tag{4.7}
$$

where $d_\text{inner} = \text{expand} \cdot d$.

### 4.4.2 Causal Convolution

A depthwise 1D convolution is applied:

$$
x_\text{conv} = \text{SiLU}(\text{Conv1d}(x))
\tag{4.8}
$$

with kernel size $d_\text{conv}$ and no padding (causal).

### 4.4.3 Selective Parameters

The SSM parameters are computed:

$$
[\Delta; B; C] = \text{SVDLinear}_{x}(x_\text{conv})
\tag{4.9}
$$

$$
\Delta = \text{softplus}(\text{SVDLinear}_{\Delta}([\Delta; B; C]) + b_\Delta)
\tag{4.10}
$$

### 4.4.4 Discretization

With diagonal $A = -\exp(A_\text{log})$:

$$
\bar{A}_t = \exp(\Delta_t A), \qquad
\bar{B}_t = \Delta_t B_t
\tag{4.11}
$$

### 4.4.5 Chunked Scan

For chunk size $C = 64$:

Within a chunk, compute:

$$
\lambda_{c, i} = \sum_{j=1}^i \log \bar{A}_{c, j}
\tag{4.12}
$$

Then:

$$
h_{c, i}^\text{intra} = e^{\lambda_{c, i}} \sum_{j=1}^i e^{-\lambda_{c, j}} \bar{B}_{c, j} x_{c, j}
\tag{4.13}
$$

Plus inter-chunk state:

$$
h_{c, i} = h_{c, i}^\text{intra} + e^{\lambda_{c, i}} h_{c-1, C}
\tag{4.14}
$$

### 4.4.6 Output with Denominator

The output is:

$$
y_t = \frac{C_t \cdot h_t}{\max(|C_t| \cdot |h_t^\text{denom}|, 10^{-4})} + D x_t
\tag{4.15}
$$

where $h_t^\text{denom}$ tracks $\sum |\bar{B}_j| x_j$.

### 4.4.7 Gating

$$
Y_\text{SSM} = y \odot \text{SiLU}(z)
\tag{4.16}
$$

## 4.5 Global SVD Mixing

A low-rank global aggregation:

$$
\text{global}_\text{scores} = X V_g \in \mathbb{R}^{B \times L \times r_g}
\tag{4.17}
$$

$$
Y_\text{global} = \text{SVDLinear}_\text{global}(\text{global}_\text{scores})
\tag{4.18}
$$

Added to $Y_\text{SSM}$:

$$
Y_\text{SSM}^\text{final} = Y_\text{SSM} + Y_\text{global}
\tag{4.19}
$$

## 4.6 Gated Fusion

The two main paths are fused:

$$
\gamma = \sigma(W_g X)
\tag{4.20}
$$

$$
Y = \gamma \odot Y_\text{SSM}^\text{final} + (1 - \gamma) \odot Y_\text{attn}
\tag{4.21}
$$

**Initialization.** The gating bias is $b_g = -7$, so:

$$
\sigma(-7) \approx 9 \times 10^{-4}
\tag{4.22}
$$

The SSM is silent at start; the model behaves like the pretrained attention.

## 4.7 Full Algorithm

**Algorithm 1:** SVD-Mamba forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
Output: Y ∈ R^{B×L×d}

 1: # Attention path
 2: Q ← SVDLinear_Q(X); K ← SVDLinear_K(X); V_a ← SVDLinear_V(X)
 3: if RoPE: (Q, K) ← Rotary(Q, K)
 4: if GQA:  K, V_a ← repeat_kv(K, V_a)
 5: scores ← Q K^T / √d_h
 6: scores ← scores + causal_mask + pad_mask
 7: Y_attn ← SVDLinear_O(softmax(scores) V_a)
 8:
 9: # SSM path
10: [x; z] ← SVDLinear_in(X)
11: x_conv ← SiLU(Conv1d(x))
12: [Δ; B; C] ← SVDLinear_x(x_conv)
13: Δ ← softplus(SVDLinear_Δ([Δ; B; C]) + b_Δ)
14: Ā ← exp(Δ A)
15: for c in chunks:
16:     h_c ← scan(Ā, B̄, x_conv) within chunk
17:     h_c ← h_c + carry_over
18:     y_c ← (C h_c) / (|C| |h_denom|)_clamp
19: Y_ssm ← y ⊙ SiLU(z)
20: Y_ssm ← Y_ssm + global_mix(X)
21: Y_ssm ← SVDLinear_out(Y_ssm)
22:
23: # Fusion
24: γ ← σ(W_g X)
25: Y ← γ ⊙ Y_ssm + (1 - γ) ⊙ Y_attn
26: return Y
```

## 4.8 Complexity Analysis

| Path | Time | Space |
|---|---|---|
| Attention | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| SSM (scan) | $\mathcal{O}(L d_\text{inner} d_\text{state})$ | $\mathcal{O}(L d_\text{inner} d_\text{state})$ |
| Global mixing | $\mathcal{O}(L d r_g)$ | $\mathcal{O}(L r_g)$ |
| Fusion | $\mathcal{O}(L d)$ | $\mathcal{O}(L d)$ |
| **Total** | $\mathcal{O}(L^2 d + L d_\text{inner} d_\text{state})$ | $\mathcal{O}(L^2)$ |

For $L \gg d$, the SSM cost dominates the linear part, and attention 
dominates the quadratic part.

## 4.9 Parameter Analysis

| Component | Formula |
|---|---|
| SVD projections | $r(d_\text{in} + d_\text{out} + 1)$ each |
| SSM in_proj | $r_\text{in}(d + 2 d_\text{inner} + 1)$ |
| SSM x_proj | $r_x(d_\text{inner} + d_\text{dt} + 2 d_\text{state} + 1)$ |
| SSM dt_proj | $r_\text{dt}(d_\text{dt} + d_\text{inner} + 1)$ |
| SSM out_proj | $r_\text{out}(d_\text{inner} + d + 1)$ |
| Conv1d | $d_\text{inner} d_\text{conv}$ |
| A_log, D, dt_bias | $d_\text{state} + 2 d_\text{inner}$ |
| Global V | $d r_g$ |
| Global out | $r_g(d_\text{inner} + r_g + 1)$ |
| Gating | $d^2 + d$ |
| LoRA (per projection) | $r_L(d_\text{in} + d_\text{out})$ |

For DistilBERT ($d = 768$, $d_\text{state} = 16$, $d_\text{conv} = 4$, 
$d_\text{inner} = 768$, $r = 192$, $r_L = 16$):

| Component | Per layer | 6 layers |
|---|---|---|
| Q/K/V/O projections | ~600K | ~3.6M |
| SSM branch | ~4M | ~24M |
| Global mixing | ~50K | ~300K |
| Gating | ~590K | ~3.5M |
| LoRA | ~100K | ~600K |
| **Total** | **~5.3M** | **~32M** |

**Net change:** baseline has ~1.77M per layer (Q/K/V/O). SVD-Mamba has 
~5.3M. **~3× parameter increase.**

## 4.10 Theoretical Notes

### 4.10.1 Gate Silence at Init

**Claim.** With $b_g = -7$, the SSM contributes < 0.1% to the output at 
initialization.

**Proof.** $\sigma(-7) = \frac{1}{1 + e^7} \approx 9.1 \times 10^{-4}$. $\square$

### 4.10.2 Chunked Scan Equivalence

**Claim.** The chunked scan is mathematically equivalent to the sequential 
recurrence.

**Proof.** Within a chunk, the intra-chunk scan computes $h_{c,i}$ assuming 
$h_{c,0} = 0$. The inter-chunk term adds $e^{\lambda_{c,i}} h_{c-1, C}$. 
Summing gives the full recurrence. $\square$

### 4.10.3 Causality

**Claim.** SVD-Mamba preserves causality.

**Proof.** The SSM is causal by construction (recurrence). The convolution 
is causal (no padding on the right). The attention mask is causal. Therefore 
the output at position $i$ depends only on inputs $1..i$. $\square$

---