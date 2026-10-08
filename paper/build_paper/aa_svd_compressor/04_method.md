<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

AASVD compresses each targeted `nn.Linear` (or `Conv1D`) layer in four phases:

1. **Calibration** (§4.2): collect $C_X$ per layer via streaming EMA.
2. **Weighted SVD** (§4.3): factorize $W C_X^{1/2}$, then unweight.
3. **Validation** (§4.4): compute Frobenius reconstruction error.
4. **Replacement** (§4.5): substitute `SVDLinear` if error is within threshold.

**Figure 4.1** shows the pipeline.

```
   ┌───────────────────────────────────────────┐
   │  Phase 1: Streaming EMA Calibration       │
   │  Input: bundled or user texts             │
   │  Output: C_X^(ℓ) for each target layer    │
   └────────────────┬──────────────────────────┘
                    │
                    ▼
   ┌───────────────────────────────────────────┐
   │  For each layer ℓ:                        │
   │    Phase 2: Weighted SVD at rank r        │
   │    Phase 3: Frobenius validation          │
   │    Phase 4: Replace or fallback           │
   └────────────────┬──────────────────────────┘
                    │
                    ▼
   ┌───────────────────────────────────────────┐
   │  Compressed model with SVDLinear layers   │
   │  Ready for LoRA recovery fine-tuning      │
   └───────────────────────────────────────────┘
```

## 4.2 Phase 1: Streaming Covariance Calibration

### 4.2.1 Target Statistic

For each target linear layer $\ell$, we want:

$$
C_X^{(\ell)} = \mathbb{E}_{x^{(\ell)} \sim \mathcal{D}}\!\left[x^{(\ell)} (x^{(\ell)})^\top\right]
\tag{4.1}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

where $x^{(\ell)}$ is the layer input during calibration.

### 4.2.2 EMA Update

Given a batch of $B_\text{calib}$ calibration inputs $X \in \mathbb{R}^{B_\text{calib} \times d}$, 
the batch covariance is:

$$
C_\text{batch} = \frac{1}{N_\text{valid}} \sum_{i=1}^{N_\text{valid}} x_i x_i^\top
\tag{4.2}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

where $N_\text{valid} \leq B_\text{calib}$ counts non-padding tokens. We 
then update the running estimate:

$$
\hat{C}_X \leftarrow \beta \hat{C}_X + (1 - \beta) C_\text{batch}
\tag{4.3}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

with $\beta = 0.9$ by default. The running estimate starts as the first 
batch covariance $\hat{C}_X^{(0)} = C_\text{batch}^{(0)}$.

### 4.2.3 Padding Token Filtering

To avoid bias from padding tokens (which are typically zero):

$$
\text{valid}(x_i) = \begin{cases} 1 & \text{if } \|x_i\|_1 > 10^{-5} \\ 0 & \text{otherwise} \end{cases}
\tag{4.4}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

Only valid tokens contribute to $C_\text{batch}$. This is critical for 
batched calibration, where padding dominates the raw token count.

### 4.2.4 Memory Layout

We compute $C_\text{batch}$ on GPU (fast matmul), then **move it to CPU 
immediately**:

```python
batch_cov = (torch.matmul(x_active.T, x_active) / max(1, x_active.size(0))).cpu()
```

This prevents GPU memory accumulation over calibration iterations. The total 
GPU footprint of calibration is one batch's activation, plus the running 
covariance estimate (kept on CPU).

### 4.2.5 Calibration Corpus

We ship a **default multi-domain corpus** of 5 documents:

| # | Domain | Approximate length |
|---|---|---|
| 1 | Technical prose (attention mechanisms) | 45 tokens |
| 2 | Mathematical prose (SVD, activations) | 38 tokens |
| 3 | Source code (Python attention) | 42 tokens |
| 4 | Technical prose (distributed training) | 42 tokens |
| 5 | Mathematical prose (chain-of-thought) | 48 tokens |

Total: ~215 tokens. The corpus is designed to activate diverse subspaces 
of the residual stream, providing a broad calibration signal even without 
access to the target task.

If user-supplied `calibration_texts` are provided, they **replace** the 
default corpus.

## 4.3 Phase 2: Weighted SVD

### 4.3.1 The Weighted Objective

Standard SVD minimizes $\|W - W_r\|_F^2$, treating all input dimensions 
equally. We instead minimize the activation-weighted error:

$$
\mathcal{L}_\text{weighted}(W_r) = \mathbb{E}_{x}\!\left[\|W x - W_r x\|_2^2\right]
\tag{4.5}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->
$$

Under the second-moment assumption $\mathbb{E}[x x^\top] = C_X$:

$$
\mathcal{L}_\text{weighted}(W_r) = \mathrm{Tr}\!\left[(W - W_r) C_X (W - W_r)^\top\right] = \|(W - W_r) C_X^{1/2}\|_F^2
\tag{4.6}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->
$$

which is exactly an unweighted problem on the transformed matrix:

$$
W_\text{weighted} = W C_X^{1/2}
\tag{4.7}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->
$$

### 4.3.2 The Algorithm

1. Compute $W_\text{weighted} = W C_X^{1/2}$.
2. Compute the truncated SVD:
   $$
   W_\text{weighted} \approx U_r \Sigma_r V_r^\top
   \tag{4.8}
   
<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->
$$
3. Unweight the right factor:
   $$
   V_r^\text{unweighted} = V_r^\top C_X^{-1/2}
   \tag{4.9}
   
<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->
$$
4. Distribute the singular values symmetrically:
   $$
   U = U_r \Sigma_r^{1/2}, \qquad
   V = \Sigma_r^{1/2} V_r^\text{unweighted}
   \tag{4.10}
   
<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 527 -->
$$
5. The factorized layer computes:
   $$
   y = (x V^\top) U^\top + b
   \tag{4.11}
   
<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

### 4.3.3 Eigendecomposition with Regularization

We compute $C_X^{1/2}$ and $C_X^{-1/2}$ via eigendecomposition:

$$
C_X = Q \Lambda Q^\top, \quad
C_X^{1/2} = Q \Lambda^{1/2} Q^\top, \quad
C_X^{-1/2} = Q \Lambda^{-1/2} Q^\top
\tag{4.12}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

Eigenvalues are regularized:

$$
\lambda_i \leftarrow \max(\lambda_i, \epsilon_\text{reg}), \qquad
\epsilon_\text{reg} = 10^{-6} \cdot \max_j \lambda_j + 10^{-8}
\tag{4.13}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

This ensures $C_X^{-1/2}$ is finite for near-zero eigenvalues.

### 4.3.4 Rank Selection

We use a rank constrained by **both** the target rank $r$ and a parameter 
budget constraint:

$$
r_\text{eff} = \min\!\left(r, \left\lfloor 0.75 \cdot \frac{d_\text{in} d_\text{out}}{d_\text{in} + d_\text{out}} \right\rfloor \right)
\tag{4.14}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

The $0.75$ factor enforces that the factorized layer uses at most 75% of the 
original parameters, guaranteeing meaningful compression.

If $r_\text{eff} < r$, the requested rank is reduced; this is common for 
narrow layers (e.g., $d_\text{in} = d_\text{out} = 128$ at $r = 128$).

## 4.4 Phase 3: Frobenius Validation

After factorization, we compute:

$$
\text{err}(W, U, V) = \frac{\|W - UV\|_F}{\|W\|_F + 10^{-12}}
\tag{4.15}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

**Two possible outcomes:**

1. **err $\leq \tau$.** The factorization is accepted. `SVDLinear` replaces 
   the original layer.
2. **err $> \tau$.** If `fallback_to_original = True`, the original layer 
   is preserved and logged. Otherwise, the factorization is applied anyway 
   with a warning.

We default to `fallback_to_original = True` and `max_error = 0.50`.

## 4.5 Phase 4: Structural Replacement

### 4.5.1 The `SVDLinear` Module

The replacement layer implements:

$$
y = (x V^\top) U^\top + b + \text{LoRA}(x) \cdot s
\tag{4.16}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

where:

- $V \in \mathbb{R}^{r \times d_\text{in}}$: down-projection
- $U \in \mathbb{R}^{d_\text{out} \times r}$: up-projection
- $b \in \mathbb{R}^{d_\text{out}}$: bias (if original had one)
- $\text{LoRA}(x) = (x A^\top) B^\top$: optional recovery adapter

### 4.5.2 Parameter Count

| Configuration | Parameters |
|---|---|
| Original | $d_\text{in} d_\text{out}$ |
| AASVD at rank $r$ | $r(d_\text{in} + d_\text{out})$ |
| AASVD + LoRA($r_L$) | $r(d_\text{in} + d_\text{out}) + r_L(d_\text{in} + d_\text{out})$ |

For a $768 \times 768$ matrix at $r = 64$:

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.00× |
| AASVD r=64 | 98,304 | **6.0×** |
| AASVD r=64 + LoRA r=16 | 122,880 | **4.8×** |

### 4.5.3 LoRA-Freeze Semantics

When LoRA is enabled, the base factors $U, V$ are **frozen** and only LoRA 
adapters $A, B$ are trainable. This ensures that:

1. The compression is not undone during recovery.
2. The parameter reduction persists after fine-tuning.
3. Gradient flow through $U, V$ is blocked to avoid drift.

The freeze is implemented via `requires_grad = False` on $U, V$.

## 4.6 The Full Algorithm

**Algorithm 1:** AASVD compression
```
Input:  model M, rank r, threshold τ, β=0.9
        calibration texts D_calib
Output: compressed model M', statistics

1:  for ℓ in target_layers(M) do
2:      register_forward_hook(ℓ, accumulate_C_X)
3:  end for
4:
5:  # Phase 1: Calibration
6:  inputs ← tokenize(D_calib)
7:  forward(M, inputs)
8:  remove_hooks()
9:
10: # Cache eigendecomposition per layer
11: for ℓ in target_layers(M) do
12:     Q, Λ ← eigh(C_X^(ℓ))
13:     C_sqrt[ℓ] ← Q @ diag(√Λ) @ Q.T
14:     C_inv_sqrt[ℓ] ← Q @ diag(1/√Λ) @ Q.T
15: end for
16:
17: # Phase 2-4: Weighted SVD + validation
18: for ℓ in target_layers(M) do
19:     W ← weight(ℓ)
20:     W_weighted ← W @ C_sqrt[ℓ]
21:     U, Σ, V ← svd(W_weighted)
22:     U_r, V_r ← truncate(U, Σ, V, r)
23:     V_r ← V_r @ C_inv_sqrt[ℓ]
24:     W_rec ← U_r @ V_r
25:     err ← ‖W - W_rec‖_F / (‖W‖_F + ε)
26:
27:     if err ≤ τ:
28:         replace(ℓ, SVDLinear(U_r, V_r))
29:     elif fallback_to_original:
30:         log_warning(ℓ, err)
31:         mark_fallback(ℓ)
32:     else:
33:         log_warning(ℓ, err)
34:         replace(ℓ, SVDLinear(U_r, V_r))
35: end for
36:
37: return M'
```

## 4.7 Complexity Analysis

| Phase | Time | GPU Memory |
|---|---|---|
| Calibration | $\mathcal{O}(N B d^2)$ | $\mathcal{O}(B d)$ |
| Eigendecomposition | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |
| Weighted SVD | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |
| Validation | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| **Total** | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(N_l d^2)$ |

CPU memory for storing all covariances: $\mathcal{O}(N_l d^2)$. For 
DistilBERT ($N_l = 6$, $d = 768$): ~14M values, ~56 MB at FP32.

## 4.8 Theoretical Notes

### 4.8.1 Optimality of Weighted SVD

**Proposition.** $W C_X^{1/2}$ truncated at rank $r$ minimizes $\|(W - W_r) C_X^{1/2}\|_F$ 
over all rank-$r$ matrices $W_r$.

**Proof.** This is a direct application of Eckart–Young to the matrix 
$W C_X^{1/2}$. $\square$

### 4.8.2 Relation to Standard SVD

If $C_X = I$ (identity, uncorrelated unit-variance inputs), then 
$W C_X^{1/2} = W$ and AASVD reduces to standard SVD.

If $C_X = \mathrm{diag}(F)$ (independent but non-uniform variance), then 
$W C_X^{1/2} = W \mathrm{diag}(\sqrt{F})$ and AASVD reduces to A3's 
diagonal weighting.

**AASVD is strictly more general** than both standard SVD and diagonal 
weighting.

### 4.8.3 Error Bound

**Proposition.** For AASVD with rank $r$:

$$
\|W - U_r V_r\|_F \leq \frac{\sqrt{\sum_{i > r} \sigma_i^2}}{\sqrt{\lambda_\text{min}(C_X)}}
\tag{4.17}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 530 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 530 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 530 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 530 -->
$$

where $\sigma_i$ are the singular values of $W C_X^{1/2}$ and $\lambda_\text{min}$ 
is the smallest eigenvalue of $C_X$ (before regularization).

**Proof.** We have:

$$
\|W - U_r V_r\|_F \cdot \sqrt{\lambda_\text{min}(C_X)}
\leq \|(W - U_r V_r) C_X^{1/2}\|_F
\leq \sqrt{\sum_{i > r} \sigma_i^2}
\tag{4.18}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 530 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 530 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 530 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 530 -->
$$

Dividing gives (4.17). $\square$

### 4.8.4 EMA Convergence

**Proposition.** The EMA estimate $\hat{C}_X^{(k)}$ converges to the true 
$\mathbb{E}[C_\text{batch}]$ as $k \to \infty$ with rate $\mathcal{O}(\beta^k)$.

**Proof sketch.** The EMA recursion $\hat{C}^{(k)} = \beta \hat{C}^{(k-1)} + (1-\beta) C_k$ 
has the closed form:

$$
\hat{C}^{(k)} = \beta^{k-1} C_1 + (1 - \beta) \sum_{j=2}^{k} \beta^{k-j} C_j
\tag{4.19}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

Taking expectations (assuming $\mathbb{E}[C_k] = \bar{C}$):

$$
\mathbb{E}[\hat{C}^{(k)}] = \bar{C} + \beta^{k-1} (C_1 - \bar{C})
\tag{4.20}

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: neuralops/compression_engine/aa_svd_compressor.py line 183 -->
$$

so the bias decays as $\beta^{k-1}$. $\square$
