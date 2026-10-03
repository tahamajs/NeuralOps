<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method — most important section
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 – 5 pages (1500 – 2500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Adaptive SVD compresses each attention projection in four steps:

1. **Full SVD** (§4.2): compute $U \Sigma V^\top$ in FP32.
2. **Energy-thresholded rank** (§4.3): find smallest $r$ with $\mathcal{E}(r) \geq \tau$.
3. **Symmetric factorization** (§4.4): set $U_r = U_{:, :r} \Sigma^{1/2}$, 
   $V_r = \Sigma^{1/2} V_{:r, :}^\top$.
4. **Residual adapter** (§4.5): attach zero-initialized $B A$ for recovery.

**Figure 4.1** shows the pipeline. There is no calibration phase.

```
    ┌──────────────────────────────────────────┐
    │  Pretrained attention module A           │
    │  with projections W_Q, W_K, W_V, W_O     │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │  For each projection P:                  │
    │    Phase 1: SVD(W_P) → U_P Σ_P V_P^T     │
    │    Phase 2: Find r_P via τ               │
    │    Phase 3: Truncate + distribute Σ^{1/2}│
    │    Phase 4: Add zero-init BA adapter     │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │  Compressed attention A' with:           │
    │    - Frozen U_P, V_P for each P          │
    │    - Trainable A_P, B_P for each P       │
    └──────────────────────────────────────────┘
```

## 4.2 Phase 1: Full SVD

For each projection weight $W_P \in \mathbb{R}^{d_\text{out}^{(P)} \times d_\text{in}^{(P)}}$:

$$
W_P = U_P \Sigma_P V_P^\top
\tag{4.1}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

We compute this in **FP32** regardless of the model dtype. FP16 SVD can 
produce NaN for ill-conditioned matrices.

**Complexity.** $\mathcal{O}(d_\text{in} d_\text{out} \min(d_\text{in}, d_\text{out}))$. 
For $768 \times 768$, this is ~450M FLOPs, ~50 ms on CPU, ~5 ms on GPU.

## 4.3 Phase 2: Energy-Thresholded Rank

### 4.3.1 Definition

Given singular values $\sigma_1 \geq \sigma_2 \geq \dots$, define:

$$
s_i = \sigma_i^2, \quad \mathcal{E}(r) = \frac{\sum_{i=1}^{r} s_i}{\sum_{j=1}^{\min(d_\text{in}, d_\text{out})} s_j}
\tag{4.2}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

The rank is:

$$
r_P = \min\{r : \mathcal{E}(r) \geq \tau\}
\tag{4.3}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

### 4.3.2 Implementation

```python
def _compute_energy_rank(self, S, max_rank, min_rank=16):
    s_squared = S**2
    total_energy = s_squared.sum()
    if total_energy <= 1e-12:
        return max(1, min(min_rank, max_rank))

    cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
    idx = torch.where(cum_energy >= self.energy_threshold)[0]
    r = int(idx[0].item()) + 1 if len(idx) > 0 else max_rank
    return max(1, min(max(r, min_rank), max_rank))
```

**Complexity.** $\mathcal{O}(\min(d_\text{in}, d_\text{out}))$ via `cumsum`.

### 4.3.3 Minimum Rank Constraint

We impose $r_P \geq r_\text{min}$ (default 16) to prevent numerical 
instability in very low-rank layers. This bounds $r_P$ from below, but for 
$\tau = 0.92$ and typical attention weights, the natural rank is far above 
$r_\text{min}$.

### 4.3.4 Maximum Rank Cap

We cap $r_P \leq \min(d_\text{in}, d_\text{out})$. In practice, $\tau < 1.0$ 
ensures $r_P < \min(d_\text{in}, d_\text{out})$, but the cap protects against 
edge cases.

## 4.4 Phase 3: Symmetric Factorization

### 4.4.1 Factor Construction

Given $r_P$, set:

$$
U_{P, r} = U_{P, :, :r_P} \Sigma_{P, :r_P, :r_P}^{1/2}, \quad
V_{P, r} = \Sigma_{P, :r_P, :r_P}^{1/2} V_{P, :r_P, :}^\top
\tag{4.4}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

Then $U_{P, r} V_{P, r} = W_{P, r_P}$ (the truncated SVD).

### 4.4.2 Why Symmetric?

Three reasons:

1. **Numerical stability.** Both factors have magnitude $\sqrt{\sigma_\text{max}}$, 
   avoiding FP16 overflow.

2. **Balanced gradient flow.** During recovery, adapters interact symmetrically 
   with both factors.

3. **Orthogonality.** When singular values are distinct, columns of $U_{P, r}$ 
   and rows of $V_{P, r}$ are orthogonal.

### 4.4.3 The Factorized Forward Pass

The compressed projection computes:

$$
\text{Proj}_P(x) = (x V_{P, r}^\top) U_{P, r}^\top + b_P
\tag{4.5}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

**Parameter count:**

| Configuration | Parameters |
|---|---|
| Original | $d_\text{in}^{(P)} d_\text{out}^{(P)}$ |
| Adaptive SVD | $r_P (d_\text{in}^{(P)} + d_\text{out}^{(P)})$ |

**Reduction factor:** $\frac{d_\text{in} d_\text{out}}{r_P (d_\text{in} + d_\text{out})}$.

## 4.5 Phase 4: Residual Adapter

### 4.5.1 The Residual

The truncation residual is:

$$
\Delta W_P = W_P - U_{P, r} V_{P, r}
\tag{4.6}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

Its Frobenius norm is $\sqrt{(1 - \mathcal{E}(r_P))} \cdot \|W_P\|_F \leq \sqrt{1 - \tau} \|W_P\|_F$. 
At $\tau = 0.92$, this is ~28% of $\|W_P\|_F$.

### 4.5.2 Adapter Parameterization

We approximate the residual with a rank-$r_L$ factor:

$$
\Delta W_P \approx B_P A_P, \quad
A_P \in \mathbb{R}^{r_L \times d_\text{in}}, \quad
B_P \in \mathbb{R}^{d_\text{out} \times r_L}
\tag{4.7}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

### 4.5.3 Initialization

$$
B_P = 0, \quad A_P \sim \mathcal{N}(0, 1/\sqrt{r_L})
\tag{4.8}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

The zero initialization of $B_P$ ensures $\Delta W_P = B_P A_P = 0$ at step 0.

### 4.5.4 Full Forward Pass

$$
\text{Proj}_P(x) = (x V_{P, r}^\top) U_{P, r}^\top + (x A_P^\top) B_P^\top + b_P
\tag{4.9}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

**Parameter overhead:** $r_L (d_\text{in} + d_\text{out})$ per projection.

## 4.6 The Full Algorithm

**Algorithm 1:** Adaptive SVD compression for one attention module
```
Input:  pretrained attention A with projections P ∈ {Q, K, V, O}
        energy threshold τ ∈ (0, 1)
        residual rank r_L
        minimum rank r_min
Output: compressed attention A'

1:  for P in {Q, K, V, O} do
2:      W_P ← weight(A.P)                            ▷ Extract
3:      U_P, Σ_P, V_P^T ← svd(W_P)                   ▷ Full SVD in FP32
4:      s ← Σ_P^2
5:      E ← cumsum(s) / sum(s)                      ▷ Cumulative energy
6:      idx ← first index where E ≥ τ
7:      r_P ← max(idx + 1, r_min)                   ▷ Enforce min rank
8:
9:      Σ_r^{1/2} ← sqrt(Σ_P[:r_P])
10:     U_r ← U_P[:, :r_P] @ diag(Σ_r^{1/2})        ▷ Symmetric split
11:     V_r ← diag(Σ_r^{1/2}) @ V_P^T[:r_P, :]
12:
13:     B_P ← zeros(d_out, r_L)                     ▷ Zero init
14:     A_P ← kaiming_uniform(r_L, d_in)
15:
16:     replace A.P with AdaptiveSVDLinear(U_r, V_r, A_P, B_P)
17: end for
18:
19: freeze(U_r, V_r) for all P                      ▷ Base frozen
20: unfreeze(A_P, B_P) for all P                    ▷ Adapters trainable
21:
22: return A'
```

## 4.7 Complexity Analysis

### 4.7.1 Time Complexity

| Phase | Time |
|---|---|
| SVD per projection | $\mathcal{O}(d_\text{in} d_\text{out} \min(d_\text{in}, d_\text{out}))$ |
| Rank selection | $\mathcal{O}(\min(d_\text{in}, d_\text{out}))$ |
| Factor truncation | $\mathcal{O}(r_P (d_\text{in} + d_\text{out}))$ |
| Adapter init | $\mathcal{O}(r_L (d_\text{in} + d_\text{out}))$ |
| **Per projection** | $\mathcal{O}(d_\text{in} d_\text{out} \min(d_\text{in}, d_\text{out}))$ |
| **Per attention module** | $\mathcal{O}(4 \cdot d^3)$ for square $d \times d$ |
| **Total ($N_l$ layers)** | $\mathcal{O}(4 N_l d^3)$ |

For DistilBERT ($N_l = 6$, $d = 768$): ~30 seconds total on GPU.

### 4.7.2 Space Complexity

Peak memory per projection: $O(d^2)$ for the SVD. All covariances 
and calibration states are avoided.

### 4.7.3 Compression Time Comparison

| Method | Calibration | SVD | Total |
|---|---|---|---|
| A3 | ~10s | ~20s | ~30s |
| AASVD | ~15s | ~20s | ~35s |
| **Adaptive SVD** | **0s** | **~15s** | **~15s** |

## 4.8 Parameter Analysis

For a square $d \times d$ projection:

| Configuration | Parameters |
|---|---|
| Original | $d^2$ |
| Adaptive SVD | $2 r_P d$ |
| Adaptive SVD + LoRA($r_L$) | $2 d (r_P + r_L)$ |

**Reduction factor (with adapter):**

$$
\frac{d^2}{2 d (r_P + r_L)} = \frac{d}{2(r_P + r_L)}
\tag{4.10}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

**Example.** $d = 768$, $r_P = 128$, $r_L = 16$:

| Configuration | Parameters | Ratio |
|---|---|---|
| Original | 589,824 | 1.00× |
| Adaptive SVD (base) | 196,608 | 3.00× |
| + Adapter | 221,184 | 2.67× |

## 4.9 Theoretical Notes

### 4.9.1 Reconstruction Bound

**Proposition.** With energy threshold $\tau$:

$$
\frac{\|W_P - U_{P, r} V_{P, r}\|_F}{\|W_P\|_F} = \sqrt{1 - \mathcal{E}(r_P)} \leq \sqrt{1 - \tau}
\tag{4.11}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

**Proof.** $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = (1 - \mathcal{E}(r)) \|W\|_F^2$ by 
SVD truncation formula. Since $\mathcal{E}(r_P) \geq \tau$ by rank selection, 
the bound follows. $\square$

At $\tau = 0.92$, the maximum relative error is $\sqrt{0.08} \approx 0.283$.

### 4.9.2 Exactness at Full Rank

**Proposition.** If $r_P = \mathrm{rank}(W_P)$, then $U_{P, r} V_{P, r} = W_P$ exactly.

**Proof.** Direct from SVD. $\square$

### 4.9.3 Initial Forward Pass Fidelity

**Proposition.** At initialization ($B_P = 0$):

$$
\text{Proj}_P(x) = (x V_{P, r}^\top) U_{P, r}^\top + b_P = W_{P, r_P} x + b_P
\tag{4.12}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

**Proof.** Substitute $B_P = 0$ into (4.9), and note $U_{P, r} V_{P, r} = W_{P, r_P}$. 
$\square$

**Corollary.** The initial forward pass equals the truncation approximation, 
not the pretrained model (unless $r_P = \text{rank}(W_P)$).

### 4.9.4 Rank Reduction Factor

The parameter ratio per projection is:

$$
\rho_P = \frac{2 r_P d}{d^2} = \frac{2 r_P}{d} \quad \text{(for square } d \times d\text{)}
\tag{4.13}

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->

<!-- source: neuralops/attentions/adaptive_svd_attention.py line 137 -->
$$

At $r_P = d/2$, $\rho_P = 1$ (no reduction). At $r_P = d/4$, $\rho_P = 0.5$ (2× reduction).

## 4.10 Hyperparameter Guidance

| Hyperparameter | Default | Range tested |
|---|---|---|
| $\tau$ | 0.92 | 0.85 – 0.99 |
| $r_L$ | 16 | 4 – 64 |
| $r_\text{min}$ | 16 | 8 – 32 |
| Recovery steps | 50 | 10 – 200 |

**Choosing $\tau$:**

- $\tau = 0.85$: aggressive, may lose accuracy
- $\tau = 0.92$: balanced (default)
- $\tau = 0.95$: conservative, less compression
- $\tau = 0.99$: near-lossless, minimal compression

**Choosing $r_L$:**

- $r_L = 8$: minimal adapter overhead
- $r_L = 16$: balanced (default)
- $r_L = 32$: more recovery capacity
- $r_L = 64$: rarely needed
