<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method — most important section
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages (1500 - 2500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

A3 compresses a pretrained transformer in **five phases**:

1. **Calibration** (§4.2): collect activation statistics via streaming hooks.
2. **Weighted SVD** (§4.3): factorize each weight matrix in weighted space.
3. **Validation** (§4.4): compute Frobenius reconstruction error.
4. **Adaptive escalation** (§4.4.2): increase rank if error exceeds $\tau$.
5. **LayerNorm recalibration** (§4.5): absorb distributional shift.

**Figure 4.1** shows the pipeline. Phase 1 runs once on the whole model; 
phases 2–4 run per layer; phase 5 runs once at the end.

```
    ┌──────────────────────────────────────────────────┐
    │  Phase 1: Calibration (streaming)                │
    │  Input: unlabeled text                           │
    │  Output: {F^(ℓ), C_X^(ℓ)} per layer              │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  For each layer ℓ:                               │
    │    Phase 2: Weighted SVD at rank r               │
    │    Phase 3: Frobenius validation                 │
    │    Phase 4: If err > τ: escalate rank, retry     │
    │    If err > τ after max retries: fallback        │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  Phase 5: LayerNorm recalibration (global)       │
    │  Rescale γ to match target σ                     │
    └──────────────────────────────────────────────────┘
```

## 4.2 Phase 1: Streaming Calibration

### 4.2.1 Problem

We want to compute, for each linear layer $\ell$:

$$
F^{(\ell)} = \frac{1}{N} \sum_{t=1}^{N} \left(x_t^{(\ell)}\right)^2, \qquad
C_X^{(\ell)} = \frac{1}{N} \sum_{t=1}^{N} x_t^{(\ell)} \left(x_t^{(\ell)}\right)^\top
\tag{4.1}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

where $x_t^{(\ell)} \in \mathbb{R}^{d_\text{in}^{(\ell)}}$ is the input to 
layer $\ell$ at token $t$.

Naively, storing all $N$ activations per layer costs $\mathcal{O}(N \sum_\ell d_\text{in}^{(\ell)})$, 
which for $N = 10^4$ and $d_\text{in} = 768$ over 12 layers is $\sim 9 \times 10^7$ 
values, or 180 MB at FP16. Streaming avoids this.

### 4.2.2 Streaming Accumulation

We register forward hooks on every linear-like module. For each forward pass 
with input $x \in \mathbb{R}^{B \times L \times d}$, the hook:

1. Reshapes $x$ to $(BL) \times d$.
2. Filters padding tokens: keeps only rows where $\|x_i\|_1 > 10^{-5}$.
3. Accumulates $C_\text{batch} = x_\text{valid}^\top x_\text{valid} \in \mathbb{R}^{d \times d}$ and $f_\text{batch} = \sum_i x_{\text{valid}, i}^2 \in \mathbb{R}^d$.
4. Adds these to CPU-side running buffers.

After all calibration passes:

$$
F^{(\ell)} = \frac{\sum_k f^{(k)}}{\sum_k N^{(k)}}, \qquad
C_X^{(\ell)} = \frac{\sum_k C^{(k)}}{\sum_k N^{(k)}}
\tag{4.2}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

**Memory.** $\mathcal{O}(d^2)$ per layer, or $\mathcal{O}(N_l d^2)$ total. 
For $d = 768$ and $N_l = 12$, this is $\sim 8.5$M values, or 17 MB at FP32.

### 4.2.3 Calibration Sources

We support three sources, in priority order:

1. **Explicit `calibration_loader`.** A `DataLoader` of input batches.
2. **`calibration_texts`.** A list of strings; tokenized on-the-fly.
3. **Default.** Random inputs of the appropriate shape (weak calibration).

If none are provided, A3 falls back to unweighted SVD (see §4.3.4).

## 4.3 Phase 2: Weighted SVD

### 4.3.1 The Weighting Idea

Given weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, standard 
SVD minimizes $\|W - W_r\|_F^2$. We instead minimize an **activation-weighted** 
error:

$$
\mathcal{L}_\text{weighted}(W_r) = \mathbb{E}_x\!\left[\|W x - W_r x\|_2^2\right]
\tag{4.3}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

If we assume $x$ has approximately diagonal covariance 
$\mathrm{diag}(F_1, \dots, F_{d_\text{in}})$, then:

$$
\mathcal{L}_\text{weighted}(W_r) = \sum_{i=1}^{d_\text{in}} F_i \|(W - W_r)_{:, i}\|_2^2 = \|(W - W_r) \cdot \mathrm{diag}(\sqrt{F})\|_F^2
\tag{4.4}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

So the activation-weighted problem is equivalent to an **unweighted** problem 
on the transformed matrix:

$$
W_\text{weighted} = W \cdot \mathrm{diag}(\sqrt{F + \epsilon})
\tag{4.5}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

where $\epsilon = 10^{-6}$ prevents division by zero.

### 4.3.2 The Algorithm

1. Compute $W_\text{weighted}$ using (4.5).
2. Compute the truncated SVD:
   $$
   W_\text{weighted} \approx U_r \Sigma_r V_r^\top
   \tag{4.6}
   
<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$
3. Unweight the right factor to recover the original space:
   $$
   V_r^\text{unweighted} = V_r^\top \cdot \mathrm{diag}\!\left(\frac{1}{\sqrt{F + \epsilon}}\right)
   \tag{4.7}
   
<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$
4. Set $U = U_r \Sigma_r^{1/2}$ and $V = \Sigma_r^{1/2} V_r^\text{unweighted}$.

The factorized layer computes:

$$
y = (x V^\top) U^\top + b
\tag{4.8}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

### 4.3.3 Scale Normalization

To prevent extreme weighting values, we optionally normalize $F$:

$$
F_\text{norm} = \frac{F}{\|F\|_2 / \sqrt{d_\text{in}}}
\tag{4.9}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

and clamp the square root to $[10^{-3}, 10^3]$. This is controlled by the 
`normalize_scale` flag.

### 4.3.4 Fallback to Unweighted SVD

If no calibration data is available, we set $F = \mathbf{1}$, giving 
standard SVD. This is a safe default but sacrifices the activation-aware 
benefit.

## 4.4 Phase 3 & 4: Validation and Adaptive Escalation

### 4.4.1 Frobenius Error Validation

After factorization, we compute:

$$
\text{err}(W, U, V) = \frac{\|W - U V\|_F}{\|W\|_F + 10^{-8}}
\tag{4.10}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

If $\text{err} \leq \tau$, the layer is replaced. Otherwise:

### 4.4.2 Adaptive Rank Escalation

We retry with an increased rank:

$$
r_{k+1} = r_k + \max\!\left(8, \frac{\min(d_\text{in}, d_\text{out}) - r_k}{K - k}\right)
\tag{4.11}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

where $K$ is the maximum number of retries (default 3). The step size is 
chosen so the rank reaches $\min(d_\text{in}, d_\text{out})$ in exactly $K$ 
steps if no improvement is seen.

If even at maximum rank the error exceeds $\tau$, the layer **falls back to 
the original** (no compression), and we log the failure.

### 4.4.3 Why Adaptive?

Choosing rank upfront is a trade-off:

- Too high → poor compression
- Too low → excessive error

Our approach defers the decision to per-layer validation, which handles the 
natural variance in layer importance.

## 4.5 Phase 5: LayerNorm Recalibration

### 4.5.1 Motivation

After factorizing a layer, its output distribution changes. The LayerNorm 
that follows now sees a different input distribution, which may have 
non-unit variance even after normalization.

The LayerNorm computes:

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma(x) + \epsilon} + \beta
\tag{4.12}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

If the input variance changed from $\sigma_\text{old}$ to $\sigma_\text{new}$, 
the output scaled by $\gamma / \sigma_\text{new}$ instead of $\gamma / \sigma_\text{old}$. 
We can compensate by adjusting $\gamma$:

$$
\gamma \leftarrow \gamma \cdot \frac{\sigma_\text{target}}{\sigma_\text{current}}
\tag{4.13}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

where $\sigma_\text{target}$ is the pre-compression standard deviation and 
$\sigma_\text{current}$ is the post-compression one.

### 4.5.2 Procedure

1. Before compression, record $\sigma_\text{target}^{(\ell)}$ for every 
   LayerNorm via a calibration pass.
2. After compression, run another calibration pass to compute $\sigma_\text{current}^{(\ell)}$.
3. For each LayerNorm, update $\gamma$ via (4.13).

This is applied only if `recalibrate_ln = True` and calibration sources are 
available.

## 4.6 Combined Algorithm

**Algorithm 1:** A3 compression
```
Input:  model M, rank r₀, threshold τ, calibration data D
Output: compressed model M', statistics

 1: for ℓ in linear_layers(M) do
 2:     F^(ℓ), C_X^(ℓ) ← running_accumulate(ℓ, D)
 3: end for
 4: σ_target ← record_layernorm_scales(M, D)
 5:
 6: for ℓ in linear_layers(M) do
 7:     W ← weight(ℓ)
 8:     r ← r₀
 9:     for attempt in 0..K do
10:         W_weighted ← W · diag(√(F^(ℓ) + ε))
11:         U_r, Σ_r, V_r ← svd(W_weighted, r)
12:         V_r ← unweight(V_r, F^(ℓ))
13:         err ← ‖W - U_r V_r‖_F / ‖W‖_F
14:         if err ≤ τ: break
15:         r ← escalate(r, K, attempt)
16:     end for
17:     if err > τ: fallback(ℓ) else: replace(ℓ, U_r, V_r)
18: end for
19:
20: if recalibrate_ln:
21:     σ_current ← record_layernorm_scales(M', D)
22:     for ln in layernorms(M') do
23:         ln.γ ← ln.γ · σ_target[ln] / σ_current[ln]
24:     end for
25: end if
26: return M'
```

## 4.7 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Calibration | $\mathcal{O}(N \sum_\ell d_\ell^2)$ | $\mathcal{O}(N_l d^2)$ |
| Weighted SVD | $\mathcal{O}(\sum_\ell d_\text{out}^{(\ell)} d_\text{in}^{(\ell)} \min(d_\text{out}, d_\text{in}))$ | $\mathcal{O}(d_\text{max}^2)$ |
| Validation | $\mathcal{O}(\sum_\ell r d_\text{out} d_\text{in})$ | $\mathcal{O}(d_\text{out} d_\text{in})$ |
| LN recalibration | $\mathcal{O}(N)$ | $\mathcal{O}(N_l d)$ |

**Full compression time** on DistilBERT (66.9M params): ~45 
seconds on RTX 4070.

## 4.8 Parameter Analysis

For a layer with weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

| Configuration | Parameters |
|---|---|
| Original | $d_\text{out} \cdot d_\text{in}$ |
| A3 at rank $r$ | $r(d_\text{out} + d_\text{in})$ |
| A3 at rank $r$ + LoRA($r_L$) | $r(d_\text{out} + d_\text{in}) + r_L(d_\text{out} + d_\text{in})$ |
| Reduction factor | $\frac{d_\text{out} d_\text{in}}{r(d_\text{out} + d_\text{in})}$ |

**Example.** For $d = 768$ and $r = 64$:

| Configuration | Params |
|---|---|
| Original | 589,824 |
| A3 rank 64 | 98,304 |
| A3 rank 64 + LoRA rank 16 | 122,880 |
| Reduction | **4.8×** |

## 4.9 Theoretical Notes

### 4.9.1 Optimality of Weighted SVD

**Proposition.** The truncated SVD of $W_\text{weighted} = W \cdot \mathrm{diag}(\sqrt{F})$ 
minimizes $\|(W - W_r) \mathrm{diag}(\sqrt{F})\|_F$ over all rank-$r$ matrices 
$W_r$.

**Proof.** This follows directly from Eckart–Young applied to the matrix 
$W_\text{weighted}$. $\square$

### 4.9.2 Relation to Standard SVD

If $F = \mathbf{1}$ (uniform), then $W_\text{weighted} = W$ and the weighted 
SVD reduces to standard SVD.

### 4.9.3 Error Bound

For any $W$, $U_r$, $V_r$ from weighted SVD:

$$
\|W - U_r V_r\|_F \leq \left\|W - W^*_r\right\|_F + \|W^*_r - U_r V_r\|_F
\tag{4.14}

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->

<!-- source: Codes/framework/compression_engine/a3_compressor.py line 152 -->
$$

where $W^*_r$ is the unweighted optimal rank-$r$ approximation. In practice, 
the weighted SVD often achieves lower $\|W - U_r V_r\|_F$ than unweighted 
because the weighting suppresses reconstruction error in low-variance 
directions.
