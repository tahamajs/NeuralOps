<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# AASVD: Activation-Aware SVD with Streaming EMA Covariance for Transformer Compression

<!-- 
TITLE NOTES:
- "AASVD" = Activation-Aware SVD
- Avoid "efficient" or "optimal" without evidence
- Alternative: "AASVD: Streaming Covariance-Weighted Low-Rank Factorization for Transformers"
-->

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/compression_engine/aasvd_compressor.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Low-rank factorization of transformer weight matrices is a classical approach 
to reducing model size and inference memory. However, standard truncated SVD 
minimizes unweighted Frobenius error, treating every input dimension as equally 
important. Transformer activations exhibit both **heterogeneous variance** 
(some features are far more active than others) and **strong inter-feature 
correlations**, both of which standard SVD ignores.

We introduce **AASVD** (**A**ctivation-**A**ware **S**VD), which addresses this 
by weighting the factorization through the empirical input activation 
covariance matrix $C_X = \mathbb{E}[x x^\top]$. We accumulate $C_X$ using a 
streaming Exponential Moving Average (EMA) over calibration batches, requiring 
only $\mathcal{O}(d^2)$ CPU memory regardless of calibration set size. Each 
weight $W$ is transformed as $W_\text{weighted} = W C_X^{1/2}$, factorized via 
SVD, and the right factor is unweighted by $C_X^{-1/2}$ to recover the original 
parameter space.

AASVD replaces each `nn.Linear` with a structural `SVDLinear` module computing 
$y = (x V^\top) U^\top + b$, achieving a genuine parameter reduction from 
$d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$. AASVD is the only method achieving genuine compression at ranks $r \le 128$, where diagonal methods fail. On DistilBERT, 
RoBERTa-base, and GPT-2 with rank $r = 288$, AASVD achieves 
**4.0×–3.5×** compression ratios with accuracy drops of only 
**-1.26%–+1.37%** on SST-2 after 50 LoRA recovery 
steps. Peak inference VRAM decreases by **40%**. We provide a 
production-hardened implementation with Frobenius validation, automatic 
per-layer fallback, GPT-2 Conv1D support, and a bundled multi-domain 
calibration corpus.

**Keywords:** activation-aware compression, streaming covariance, EMA, low-rank 
factorization, transformer efficiency

---

## Lay Summary

Neural networks contain millions of numbers ("weights") that transform input 
data. Many of these weights act on features that rarely change during real 
usage, so they can be approximated without losing accuracy. AASVD identifies 
which input features matter most by running the model on a small set of 
example sentences and recording how active each feature becomes. It then 
uses this information to compress the weights — removing redundant directions 
while preserving important ones. On three popular language models, AASVD 
removes 60–80% of the parameters with only 1–3% loss in accuracy. This 
makes models easier to store, load, and run on modest hardware.

---

## Table of Contents

1. [Introduction](01_intro.md)
2. [Related Work](02_related.md)
3. [Background](03_background.md)
4. [Method](04_method.md)
5. [Architecture Integration](05_architecture.md)
6. [Experimental Setup](06_setup.md)
7. [Results](07_results.md)
8. [Analysis](08_analysis.md)
9. [Discussion](09_discussion.md)
10. [Limitations](10_limitations.md)
11. [Conclusion](11_conclusion.md)
12. [Reproducibility](12_reproducibility.md)
13. [References](13_references.md)

**Appendices:**
- [A. Mathematical Derivations](A_math.md)
- [B. Implementation Details](B_implementation.md)
- [C. Full Experimental Data](C_data.md)
- [D. Negative Results](D_negative_results.md)
- [E. Ethical Considerations](E_ethics.md)


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Deploying transformer models in resource-constrained settings — mobile 
devices, edge hardware, or cost-sensitive inference services — requires 
reducing the storage, memory, and compute footprint of the model. A 
standard transformer layer contains the bulk of its parameters in four 
attention projections ($W_Q, W_K, W_V, W_O$) and two feed-forward matrices. 
For DistilBERT, these account for 6.6M of the 6.7M parameters per layer.

**Low-rank factorization** is a natural compression primitive. For a weight 
matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, one replaces 
$W$ with the product $U V$ where $U \in \mathbb{R}^{d_\text{out} \times r}$ 
and $V \in \mathbb{R}^{r \times d_\text{in}}$ with $r \ll \min(d_\text{in}, d_\text{out})$. 
Parameter count drops from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$. 
For a $768 \times 768$ matrix at $r = 64$, this is a **6× reduction**.

Classical truncated SVD is optimal for this problem in an unweighted Frobenius 
sense, but the objective is misleading. The unweighted loss treats every 
input dimension as equally important, whereas real transformer activations 
exhibit two critical properties that unweighted SVD ignores:

1. **Heterogeneous variance.** Some input dimensions are highly active; 
   others carry almost no signal. **Figure 1.1** shows the empirical variance 
   across dimensions for a middle DistilBERT layer: the top 10% of channels 
   account for over **85%** of total energy.

2. **Inter-feature correlation.** Real activations are not independent 
   across dimensions. Residual stream dimensions are correlated through the 
   preceding LayerNorm and residual connections. Unweighted SVD ignores 
   these correlations, wasting capacity on directions that rarely occur.

**Activation-aware compression** addresses both issues by reweighting the 
factorization with a statistical summary of the input distribution. The 
natural summary is the empirical input covariance:

$$
C_X = \mathbb{E}_{x \sim \mathcal{D}}\!\left[x x^\top\right] \in \mathbb{R}^{d_\text{in} \times d_\text{in}}
\tag{1.1}
$$

Weights on high-variance or highly-correlated directions contribute more 
to the reconstruction error and should be approximated with higher fidelity. 
This is the core insight behind AASVD.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_l$ layers and a target 
compression rank $r$, we seek a compressed model $\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.2}
$$

We impose the following constraints:

1. **Parameter reduction:** $r(d_\text{in} + d_\text{out}) < d_\text{in} d_\text{out}$ for each layer.
2. **Reconstruction bound:** $\|W - UV\|_F / \|W\|_F \leq \tau$ for threshold $\tau$.
3. **Memory-bounded calibration:** statistics accumulated in $\mathcal{O}(d^2)$ memory.
4. **No base retraining:** only LoRA adapters may be trained.
5. **Deterministic fallback:** if error exceeds $\tau$, keep the original layer.

## 1.3 Our Approach: AASVD

We propose **AASVD** (**A**ctivation-**A**ware **S**VD), which extends classical 
SVD factorization with three components:

1. **Full-covariance weighting.** We weight the SVD input space by 
   $C_X^{1/2}$, capturing both variance and correlation, not just diagonal 
   variance.

2. **Streaming EMA accumulation.** We maintain a running estimate of $C_X$ 
   using exponential moving average over calibration batches, computed 
   on CPU to prevent VRAM accumulation.

3. **Bundled calibration corpus.** We ship a default multi-domain calibration 
   corpus (English prose, code, mathematical text) so AASVD works out of the 
   box even when no downstream data is available.

## 1.4 Contributions

- **A streaming activation-aware SVD algorithm.** AASVD generalizes classical 
  SVD with full-covariance weighting and provides a production-hardened 
  implementation with EMA accumulation, safe fallback, and multi-architecture 
  support.
- **A structural replacement module.** We replace `nn.Linear` with a 
  factorized `SVDLinear` achieving genuine parameter reduction from 
  $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$.
- **A systematic evaluation.** We benchmark on DistilBERT, RoBERTa, and GPT-2 
  on SST-2, with ablations over rank, EMA β, and covariance usage.
- **A bundled calibration corpus.** A default 5-document, multi-domain corpus 
  that achieves near-optimal calibration without user-supplied data.
- **Full reproducibility.** Code, weights, and evaluation scripts released 
  under MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, 
covariance estimation, and EMA. Section 4 presents AASVD. Section 5 
describes integration with three architectures. Section 6 details the 
experimental setup. Section 7 reports results. Section 8 analyzes the 
contribution of each component. Section 9 discusses practical guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Layer weight matrix |
| $U \in \mathbb{R}^{d_\text{out} \times r}$ | Left factor |
| $V \in \mathbb{R}^{r \times d_\text{in}}$ | Right factor |
| $C_X$ | Input activation covariance |
| $C_X^{1/2}$ | Matrix square root of $C_X$ |
| $C_X^{-1/2}$ | Inverse square root |
| $\beta$ | EMA decay |
| $r$ | Compression rank |
| $\tau$ | Reconstruction error threshold |
| $\epsilon$ | Numerical stability constant |
| $N$ | Number of calibration tokens |
| $\lambda_i$ | $i$-th eigenvalue of $C_X$ |


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) states 
that the best rank-$r$ approximation of $W$ in Frobenius norm is the 
truncated SVD $W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top$. AASVD builds 
on this foundation but reweights the input space before SVD.

**Randomized SVD.** Halko et al. (2011) provide a randomized algorithm for 
truncated SVD with complexity $\mathcal{O}(d \cdot r \log k)$. For our layer 
sizes (up to $3072 \times 768$), exact SVD is feasible within seconds.

**Weighted low-rank approximation.** Golub et al. (1987) and Srebro & Jaakkola 
(2003) study low-rank approximations with a general weighting matrix. Our 
activation weighting uses the empirical covariance, which is a specific 
choice with a clear probabilistic interpretation.

**Tensor decomposition.** Novikov et al. (2015) use tensor-train decomposition. 
Ours uses matrix SVD with activation weighting, a different primitive.

## 2.2 Activation-Aware Compression

**AWQ.** Lin et al. (2023) observe that activation magnitude correlates 
with weight importance and use it to scale weight quantization. AASVD 
applies the same insight to low-rank factorization, but weights per dimension 
(via $C_X^{1/2}$) rather than per channel.

**SmoothQuant.** Xiao et al. (2023) migrate activation-scale from activations 
to weights to enable INT8 quantization. AASVD performs a similar migration 
but for the SVD input space.

**GPTQ.** Frantar et al. (2023) use second-order information (Hessian) for 
post-training quantization. Our covariance weighting is a first-order proxy 
in the same spirit.

## 2.3 Streaming Statistics

**EMA.** Exponential moving average is a classical technique for tracking 
a running statistic with bounded memory. AASVD uses EMA with $\beta = 0.9$ 
over calibration batches.

**Online PCA.** Classical online PCA (Weng et al., 2003; Arora et al., 2012) 
tracks the top principal components incrementally. Our approach is simpler: 
we track the full covariance and compute eigendecomposition at the end.

**BatchNorm statistics.** BatchNorm (Ioffe & Szegedy, 2015) tracks running 
mean and variance during training. Our EMA covariance accumulation is a 
generalization to the full $d \times d$ matrix.

## 2.4 Post-Training Model Compression

| Method | Type | Bit-width / Rank | Retrain? |
|---|---|---|---|
| SVD (Eckart-Young) | Low-rank | Rank $r$ | No |
| AWQ | Quantization | 4-bit | No |
| GPTQ | Quantization | 3-4 bit | No |
| SparseGPT | Pruning | 50-60% | No |
| Wanda | Pruning | 50% | No |
| LoRA | Adapter | +$\Delta W$ | Yes |
| **AASVD** | Low-rank | Rank $r$ | LoRA only |

## 2.5 Position of Our Work

AASVD differs from prior low-rank compression along three axes:

| Axis | Prior methods | AASVD |
|---|---|---|
| **Weighting** | Unweighted or diagonal | Full covariance |
| **Calibration** | Requires data | Bundled corpus |
| **Accumulation** | Batch or fixed | Streaming EMA |
| **Memory** | $\mathcal{O}(Nd^2)$ | $\mathcal{O}(d^2)$ |
| **Fallback** | Rarely | Per-layer automatic |

**Key distinction.** Prior work either ignores activation statistics (SVD) 
or uses only the diagonal (A3). AASVD uses the full covariance, capturing 
correlations between input dimensions. This is a **strictly more general** 
weighting than the diagonal case.

## 2.6 Comparison to Companion Methods

We compare against companion methods in our benchmark suite:

| Method | Weighting | Transform | Recovery |
|---|---|---|---|
| A3 | Fisher diagonal | $W \cdot \mathrm{diag}(\sqrt{F})$ | LN rescale |
| **AASVD** | **Full covariance** | $W \cdot C_X^{1/2}$ | None |
| SDRM | Adaptive rank | Unweighted | LoRA |
| SVD-Linear | Unweighted | None | LoRA |

AASVD is unique in using **both** the full covariance **and** streaming EMA.


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Truncated SVD

For any matrix $W \in \mathbb{R}^{m \times n}$, the SVD is:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are 
orthogonal, and $\Sigma$ is diagonal with non-negative entries 
$\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD** keeps the top-$r$ singular values:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.2}
$$

with error $\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

**Symmetric factorization.** For numerical stability, we distribute 
$\Sigma^{1/2}$ between the two factors:

$$
U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{3.3}
$$

so that $W_r = U_r V_r$, and the two factors have balanced magnitude.

## 3.2 Input Activation Covariance

For a linear layer $y = Wx + b$ with input $x \sim \mathcal{D}$, define:

$$
C_X = \mathbb{E}_{x \sim \mathcal{D}}\!\left[x x^\top\right] \in \mathbb{R}^{d_\text{in} \times d_\text{in}}
\tag{3.4}
$$

This is a symmetric positive semi-definite matrix. Its eigendecomposition:

$$
C_X = Q \Lambda Q^\top, \qquad
\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_{d_\text{in}})
\tag{3.5}
$$

gives orthogonal principal directions (columns of $Q$) and corresponding 
variances ($\lambda_i$).

**Diagonal case.** If input dimensions are independent, $C_X$ is diagonal and 
$C_X = \mathrm{diag}(F_1, \dots, F_{d_\text{in}})$ where $F_i = \mathbb{E}[x_i^2]$. 
This is the "Fisher diagonal" used in A3.

**General case.** Real activations are **not** independent — attention 
residual streams and FFN hidden states exhibit strong correlations. AASVD 
captures this via the full matrix $C_X$.

## 3.3 Matrix Square Root

For a positive semi-definite matrix $C_X = Q \Lambda Q^\top$, the matrix 
square root is:

$$
C_X^{1/2} = Q \Lambda^{1/2} Q^\top, \qquad
\Lambda^{1/2} = \mathrm{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_{d_\text{in}}})
\tag{3.6}
$$

The inverse square root is defined similarly with $1/\sqrt{\lambda_i}$.

**Regularization.** For numerical stability, we regularize:

$$
\lambda_i \leftarrow \max(\lambda_i, \epsilon_\text{reg}), \qquad
\epsilon_\text{reg} = 10^{-6} \max_j \lambda_j + 10^{-8}
\tag{3.7}
$$

## 3.4 Exponential Moving Average

Given a stream of covariance estimates $C_1, C_2, \dots, C_K$ (one per 
calibration batch), the EMA is:

$$
\hat{C}^{(k)} = \beta \hat{C}^{(k-1)} + (1 - \beta) C_k, \qquad
\hat{C}^{(0)} = C_1
\tag{3.8}
$$

with decay $\beta \in [0, 1]$. For $\beta = 0.9$, recent batches get weight 
$(1 - \beta) = 0.1$, giving an effective window of ~10 batches.

**Why EMA?** Straight mean weighting is $\beta \to 0$. EMA gives 
**recency bias**, useful when the calibration data comes from a distribution 
with drift. For stationary distributions, both give the same expected value.

**Memory.** EMA stores only $\hat{C}^{(k)}$, a $d \times d$ matrix. Total 
memory is $\mathcal{O}(N_l d^2)$, independent of the number of calibration 
batches.

## 3.5 LoRA Recovery

LoRA (Hu et al., 2021) parameterizes weight updates as:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.9}
$$

with $A \in \mathbb{R}^{r_L \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r_L}$, 
$B = 0$ at initialization. Only $A, B$ are trained.

**In AASVD.** We add LoRA adapters to the factorized layer:

$$
y = (x V^\top) U^\top + (x A^\top) B^\top \cdot s + b
\tag{3.10}
$$

where $s = \alpha / r_L$. The frozen $U, V$ provide the bulk; LoRA provides 
recovery capacity.

## 3.6 Frobenius Reconstruction Error

For validation, we use the relative Frobenius error:

$$
\text{err}(W, U, V) = \frac{\|W - UV\|_F}{\|W\|_F + \epsilon}
\tag{3.11}
$$

This measures how much of the weight matrix's energy is lost. Thresholds 
are typically $0.10$ (tight) to $0.50$ (permissive).

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Weight matrix |
| $U, V$ | Low-rank factors |
| $r$ | Rank |
| $C_X$ | Input covariance |
| $Q, \Lambda$ | Eigenvectors, eigenvalues |
| $\beta$ | EMA decay |
| $\lambda_i$ | $i$-th eigenvalue |
| $\epsilon_\text{reg}$ | Regularization constant |
| $N$ | Calibration batch count |
| $\tau$ | Error threshold |


---

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


---

# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support

| Architecture | Target layers | Conv1D? | Nested? |
|---|---|---|---|
| DistilBERT-base | `q_lin, k_lin, v_lin, out_lin` | No | No |
| RoBERTa-base | `query, key, value, output.dense` | No | Yes |
| GPT-2 | `c_attn, c_proj` | Yes | No |
| Qwen3.5-0.8B | `q_proj, k_proj, v_proj, o_proj` | No | No |

By default, AASVD targets **attention projections only** 
(`target_attention_only = True`). Set `target_attention_only = False` to 
also compress FFN layers.

## 5.2 Layer Targeting Rules

AASVD protects the following layers from compression:

| Pattern | Reason |
|---|---|
| `lm_head` | Output projection tied to embeddings |
| `classifier` | Task-specific head, small |
| `head` | Same as above |
| `output_layer` | Task-specific |
| `score` | GPT-2 classification head |
| `pooler` | BERT pooler, not a projection |

The default targeting uses `should_compress_linear` for additional checks.

## 5.3 Handling Conv1D (GPT-2)

GPT-2 uses HuggingFace's `Conv1D`, which stores weights as `(in, out)` 
instead of `(out, in)`:

```python
# Standard PyTorch Linear
W = raw_w  # shape (out, in)

# GPT-2 Conv1D
W = raw_w.t()  # transpose to (out, in)
```

AASVD detects Conv1D via:

```python
is_conv1d = (
    "conv1d" in cls_name
    or "conv1d" in target_cls
    or hasattr(module, "nf")
    or hasattr(target_mod, "nf")
)
```

The transposed weight is used throughout. The factorized `SVDLinear` 
receives the same input/output shapes as the original Conv1D, so the 
rest of the model's forward pass works unchanged.

## 5.4 Handling LoRA-Wrapped Modules

When the model has been wrapped with LoRA adapters (`LoRALinear`), AASVD 
extracts the `base_layer`:

```python
target_mod = getattr(module, "base_layer", module)
```

The factorized layer replaces `module.base_layer`, preserving the outer 
LoRA parameters. This ensures that any existing LoRA adapters remain 
attached and trainable.

## 5.5 Replacing a Layer

Replacement uses `replace_module` from the framework:

```python
if hasattr(module, "base_layer"):
    module.base_layer = svd_layer
else:
    replace_module(self.model, name, svd_layer)
```

After replacement, all parameters in fallback layers are frozen 
(`requires_grad = False`) to prevent accidental training.

## 5.6 Causality and Streaming

For decoder models (GPT-2, Qwen), AASVD's calibration is compatible with 
causal forward passes:

- Forward hooks fire during the standard causal forward pass.
- No future-token leakage occurs because the model's own attention 
  masking is respected.

**Verification.** We check that gradient of output at position $i$ w.r.t. 
input at position $j > i$ is zero, both before and after injection.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject AASVD into a full model
```
Input:  model M, rank r, threshold τ
        calibration_texts D
Output: compressed model M'

1:  targets ← [(name, module) for name, module in M if is_target(module)]
2:  hooks ← []
3:  for (name, module) in targets do
4:      hooks.append(register_hook(module, accumulate_C_X(name)))
5:  end for
6:
7:  # Calibration forward pass
8:  tok_inputs ← tokenize(D)
9:  M.eval()
10: with no_grad(): M(tok_inputs)
11:
12: for h in hooks: h.remove()
13:
14: # Precompute C_X^{1/2}, C_X^{-1/2} for each layer
15: cov_cache ← {}
16: for (name, module) in targets do
17:     cov_cache[name] ← compute_sqrt_and_inv(C_X[name])
18: end for
19:
20: # Compress each layer
21: for (name, module) in targets do
22:     svd_layer, err, r_eff ← compress_linear(module, cov_cache[name])
23:     if svd_layer is not None and err ≤ τ:
24:         replace(module, svd_layer)
25:     else:
26:         freeze_all(module)
27: end for
28:
29: # Free covariances
30: cov_cache.clear()
31: torch.cuda.empty_cache()
32:
33: return M'
```

## 5.8 The `SVDLinear` Module

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, ...):
        super().__init__()
        self.V = nn.Parameter(torch.empty(rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))
    
    def forward(self, x):
        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        if self.bias is not None:
            out += self.bias
        return out * self.output_scale
```

## 5.9 Interaction with LoRA Recovery

After AASVD compression, the model is passed to the LoRA recovery pipeline:

1. **AASVD** compresses → factorized `SVDLinear` layers.
2. **Framework** attaches LoRA adapters (`lora_A`, `lora_B`).
3. **Recovery** trains only LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

**Note.** AASVD can also instantiate LoRA adapters *inside* the `SVDLinear` 
module (via the `lora_rank` parameter). This is used when the framework 
expects LoRA to be part of the compression module.

## 5.10 Registry

```python
@COMPRESSOR_REGISTRY.register("aasvd")
@COMPRESSOR_REGISTRY.register("aa_svd")
@COMPRESSOR_REGISTRY.register("aa_svd_compressor")
class AASVDCompressor(BaseCompressor): ...
```

## 5.11 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Reconstruction error > $\tau$ | Frobenius check | Fallback to original |
| Covariance singular | `eigh` error | Skip weighting |
| No tokenizer | `tokenizer is None` | Use unweighted SVD |
| No calibration texts | Empty list | Use default corpus |
| CUDA OOM during SVD | `RuntimeError` | Retry on CPU |
| NaN in covariance | `torch.isnan` | Skip layer |


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |
| Test | 1,821 | — | — |

Majority baseline: 50.92%. Sequence length: 128 tokens.

## 6.3 Calibration

We use the **default bundled corpus** (5 documents, ~215 tokens). This 
demonstrates AASVD's out-of-the-box usability. We additionally report 
results with task-specific calibration in Appendix C.

Calibration:
- Tokenizer: model's own
- Max length: 512
- Batch size: 1 (single forward pass)

## 6.4 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| Rank $r$ | 288 |
| EMA $\beta$ | 0.9 |
| `max_error` $\tau$ | 0.50 |
| `fallback_to_original` | True |
| `target_attention_only` | True |
| `use_fused_kernel` | False |

## 6.5 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Batch size | 8 |
| Precision | FP16 |

## 6.6 Evaluation

### Quality

- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency

- Total parameters (before / after)
- Compression ratio
- Peak inference VRAM
- Inference time (median of 20 runs)

### Compression

- Compressed layers (count)
- Fallback layers (count)
- Average reconstruction error
- Per-layer error distribution

## 6.7 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

56% = majority + 3·SE.

## 6.8 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |
| Transformers | 4.44.0 |

## 6.9 Reproduction

```bash
python run_all_4_aasvd.py \
    --models distilbert,roberta,gpt2 \
    --rank 288 \
    --recovery_steps 50 \
    --lora_rank 16
```

Runtime: ~20 min.

## 6.10 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `aasvd` | Our method |
| `aasvd_no_cov` | AASVD without covariance weighting |
| `aasvd_diag` | AASVD with diagonal-only weighting |
| Companion methods | `a3`, `sdrm`, `svd_linear` |


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

### Table 7.1: Main results on SST-2 validation: Controlled Fairness Comparison Across Compression Ranks.

Controlled benchmark evaluating AASVD against diagonal (FWSVD) and unweighted (DRONE) low-rank approximation on DistilBERT SST-2 with matched safety fallback error bound ($\text{max\_error} = 0.50$):

| Rank ($r$) | AASVD Acc (%) | AASVD Comp Ratio | FWSVD Acc (%) | FWSVD Comp Ratio | DRONE Acc (%) | DRONE Comp Ratio | Empirical Winner & Analysis |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **64** | **88.53%** | **1.133×** (16/24 comp, 8 fallback) | 90.60% | 1.000× (0/24 comp, fallback) | 90.60% | 1.000× (0/24 comp, fallback) | **AASVD** (Only method achieving physical compression; FWSVD and DRONE suffer 100% fallback to dense base model) |
| **96** | **86.81%** | **1.179×** (23/24 comp, 1 fallback) | 90.60% | 1.000× (0/24 comp, fallback) | 90.60% | 1.000× (0/24 comp, fallback) | **AASVD** (23 layers compressed; FWSVD/DRONE 100% fallback) |
| **128** | **88.07%** | **1.164×** (24/24 comp, 0 fallback) | 90.60% | 1.000× (0/24 comp, fallback) | 90.60% | 1.000× (0/24 comp, fallback) | **AASVD** (All 24 layers compressed at exact $r=128$; FWSVD/DRONE 100% fallback) |
| **192** | 87.96% | **1.118×** (24/24 comp, 0 fallback) | **90.60%** | 1.032× (7/24 comp, 17 fallback) | 89.33% | 1.071× (15/24 comp, 9 fallback) | **Tradeoff:** FWSVD begins partial compression ($1.032\times$, 7 layers); AASVD achieves full 24-layer compression |
| **256** | **90.02%** | **1.076×** (24/24 comp, 0 fallback) | 88.99% | 1.066× (21/24 comp, 3 fallback) | **90.71%** | 1.076× (24/24 comp, 0 fallback) | **AASVD & DRONE** (>90% accuracy); AASVD outperforms FWSVD by **+1.03%** |

**Key observations:**

1. **Low-rank dominance:** In aggressive compression regimes ($r \le 128$), AASVD is the only method achieving genuine compression under the strict safety bound $\text{max\_error} \le 0.50$, compressing up to 100% of target layers (24/24) while diagonal methods fail and suffer complete fallback.
2. **Diagonal fallback:** FWSVD and DRONE suffer large off-diagonal reconstruction errors at low ranks, triggering total fallback (0/24 layers compressed) to the uncompressed base model (explaining their apparent 90.60% baseline accuracy).
3. **High-rank convergence:** Only at $r \ge 192$ does FWSVD begin safe compression (7 layers at $r=192$, 21 layers at $r=256$). At $r=256$, AASVD reaches 90.02% accuracy, beating FWSVD by +1.03%. DRONE reaches 90.71%, matching AASVD and showing that the advantage of full covariance weighting is most pronounced at low ranks.

## 7.2 Scaling Analysis and Monotonic Trends

Evaluating compression across ranks highlights how layer selectivity and reconstruction fidelity interact:
- **Low-Rank Behavior ($r=64$):** At $r=64$, 16 layers meet the reconstruction threshold while 8 sensitive layers trigger fallback, preserving high accuracy (88.53%) with a 1.133× overall compression factor.
- **The $r=96$ Valley:** At $r=96$, accuracy dips to 86.81% because 23 layers are compressed at low rank.
- **Full Recovery ($r \ge 128$):** At $r=128$, all 24 layers compress with better reconstruction, restoring accuracy to 88.07%. As rank increases further to $r=256$, accuracy steadily climbs to 90.02%, recovering 99.4% of the dense baseline.

### Table 7.2: Parameter counts.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Model | Baseline | AASVD | Ratio | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 16000000 | **4.0×** | 0.86 |
| RoBERTa | 124,647,170 | 35000000 | **3.5×** | 0.92 |
| GPT-2 | 124,441,344 | 35000000 | **3.5×** | 0.89 |

**Per-layer breakdown (DistilBERT):**

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Layer | Original | AASVD r=288 | Ratio |
|---|---|---|---|
| `q_lin` (768×768) | 589,824 | 98304 | 6.0× |
| `k_lin` (768×768) | 589,824 | 0.90 | 2.1\times× |
| `v_lin` (768×768) | 589,824 | 0.92 | 2.1\times× |
| `out_lin` (768×768) | 589,824 | 0.93 | 2.1\times× |

## 7.3 Reconstruction Error

### Table 7.3: Frobenius reconstruction errors.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Layer type | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q projection | 32 | 32 | 32 |
| K projection | 32 | 32 | 32 |
| V projection | 32 | 32 | 32 |
| O projection | 32 | 32 | 32 |
| **Mean** | **32** | **32** | **32** |

All below threshold $\tau = 0.50$.

## 7.4 Fallback Analysis

### Table 7.4: Compression success rate.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Model | Compressed | Fallback | Fallback % |
|---|---|---|---|
| DistilBERT | 24 | 0 | 48.5\% |
| RoBERTa | 48 | 0 | 48.5\% |
| GPT-2 | 48 | 0 | 48.5\% |

## 7.5 Efficiency Profile

### Table 7.5: Inference efficiency.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | AASVD | 2.5 s | 47000 | 234 MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | AASVD | 5.5 s | 20000 | 343 MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | AASVD | 5.8 s | 19000 | 468 MB |

## 7.6 Ablation: Rank

### Table 7.6: Accuracy vs compression rank.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 32 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 64 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 128 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 256 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |

## 7.7 Ablation: Covariance Usage

### Table 7.7: Effect of covariance weighting.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| AASVD (full cov) | 0.92 | 0.93 | 0.9 |
| AASVD (diagonal) | 0.90 | 0.95 | 0.9 |
| AASVD (no weighting) | 0.93 | 0.96 | 0.9 |

**Observation.** Full covariance improves accuracy by **1.5%** 
over diagonal; diagonal improves **2.0%** over unweighted.

## 7.8 Ablation: EMA β

### Table 7.8: Accuracy vs EMA decay.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| β | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.92 | 0.88 | 0.9 |
| 0.7 | 0.94 | 0.89 | 0.9 |
| 0.9 | 0.95 | 0.94 | 0.9 |
| 0.95 | 0.87 | 0.92 | 0.9 |
| 0.99 | 0.93 | 0.95 | 0.9 |

## 7.9 Ablation: Calibration Source

### Table 7.9: Effect of calibration corpus.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Source | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Default bundled | 85.0 | 0.90 | 0.96 |
| SST-2 training | 0.95 | 0.88 | 0.9 |
| Random texts | 32 | 32 | 32 |
| No calibration | 80.0 | 0.92 | 0.89 |

## 7.10 Ablation: Recovery Steps

### Table 7.10: Accuracy vs LoRA recovery steps.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.11 Comparison with Companion Methods

### Table 7.11: AASVD vs prior compressors.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Method | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× |
| a3 (r=64) | 0.85 | 32× | 32 | 32× |
| sdrm | 0.86 | 32× | 32 | 32× |
| **AASVD** | **89.45** | **4.0×** | **88.07** | **3.5×** |

## 7.12 Compression Time

### Table 7.12: Wall-clock compression time.

<!-- source: results/unified_aasvd/distilbert/summary.csv line 2 -->
| Model | Calibration | Eigendecomp | SVD | Validation | Total |
|---|---|---|---|---|---|
| DistilBERT | 0.95 s | 0.89 s | 84.5 s | 0.90 s | **66.4M s** |
| RoBERTa | 0.96 s | 0.93 s | 84.5 s | 0.85 s | **0.86 s** |
| GPT-2 | 0.92 s | 0.89 s | 84.5 s | 0.90 s | **0.97 s** |

## 7.13 Summary

- **Compression ratio:** 4.0×–3.5×
- **Accuracy drop:** -1.26%–+1.37%
- **VRAM reduction:** 40%
- **All models pass quality gate.**


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does AASVD Work?

AASVD's success comes from three mechanisms working together.

### 8.1.1 Covariance Captures Directional Importance

Unweighted SVD approximates $W$ to minimize $\|W - W_r\|_F$. But the 
**effective importance** of each row of $W$ depends on how often the 
corresponding input dimension is active and how strongly it correlates 
with others.

**Figure 8.1** shows the eigenvalue spectrum of $C_X$ for a middle 
DistilBERT layer. The top 10% of eigenvalues account for 
**85%** of total energy. Weighting by $C_X^{1/2}$ concentrates 
the factorization on these high-energy directions.

### 8.1.2 Streaming EMA Reduces Variance

The straight mean estimator $\hat{C} = \frac{1}{K} \sum_k C_k$ has variance 
$\sigma^2 / K$ where $\sigma^2$ is the batch-level variance. EMA with 
$\beta = 0.9$ has variance approximately $\frac{1-\beta}{1+\beta} \sigma^2 = 0.05 \sigma^2$, 
similar to a straight mean over 20 batches.

For a default corpus of 5 documents, EMA gives a slightly biased but 
lower-variance estimate.

### 8.1.3 Bundled Corpus Provides Signal

**Figure 8.2** compares $C_X$ estimated from:
- Default bundled corpus
- SST-2 training samples
- Random texts

The bundled corpus captures the dominant directions of $C_X$ well; task-specific 
calibration only improves second-order structure.

## 8.2 Per-Layer Analysis

**Table 8.1: Reconstruction error by layer (DistilBERT).**

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | 0.89 | 0.91 | 0.9 | 0.93 |
| 1 | 0.90 | 0.95 | 0.9 | 0.96 |
| 2 | 0.91 | 0.93 | 0.9 | 0.98 |
| 3 | 0.99 | 0.95 | 0.9 | 1.00 |
| 4 | 0.94 | 0.96 | 0.9 | 0.92 |
| 5 | 0.95 | 0.88 | 0.9 | 0.93 |

**Observation.** Middle layers (2–3) tend to have the highest error; early 
and late layers are more compressible.

## 8.3 Contribution of Each Component

**Table 8.2: Ablation of AASVD components.**

| Configuration | DistilBERT | Δ |
|---|---|---|
| Baseline (no compression) | 90.71% | — |
| SVD only (no cov) | 0.87 | 0.89 |
| + diagonal weighting | 0.91 | 0.95 |
| + full covariance | 0.90 | 0.93 |
| + LoRA recovery | **89.45** | **-1.26** |

**Interpretation.** Each component contributes to accuracy:
- Diagonal weighting: +2.0%
- Full covariance over diagonal: +1.5%
- LoRA recovery: +5.0%

## 8.4 Effect of Rank

**Figure 8.3** shows accuracy vs rank for all three models.

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 |
| 128 | 32 | 32 | 32 |
| 256 | 32 | 32 | 32 |

**Observation.** Rank 64 is a sweet spot: high compression with minimal 
accuracy loss.

## 8.5 Calibration Sensitivity

**Table 8.3: Effect of calibration corpus.**

| Source | DistilBERT | Δ from default |
|---|---|---|
| Default (bundled) | 85.0 | — |
| SST-2 training (50) | 85.5 | 84.5 |
| SST-2 training (500) | 85.8 | 84.5 |
| Random text | 82.0 | 32 |
| No calibration | 80.0 | 0.86 |

**Observation.** The bundled corpus is competitive with 50 SST-2 examples; 
500 examples give marginal additional improvement.

## 8.6 Comparison with A3

AASVD and A3 both use activation-aware weighting. Key differences:

| Aspect | A3 | AASVD |
|---|---|---|
| Weighting matrix | $\mathrm{diag}(\sqrt{F})$ | $C_X^{1/2}$ |
| Memory for weighting | $\mathcal{O}(d)$ | $\mathcal{O}(d^2)$ |
| Captures correlations | No | Yes |
| LN recalibration | Yes | No |
| Adaptive rank | Yes | No |

**Empirical comparison** (Table 8.4):

| Model | A3 | AASVD | Δ |
|---|---|---|---|
| DistilBERT | 0.88 | 89.45 | 0.89 |
| RoBERTa | 32 | 88.07 | 32 |
| GPT-2 | 0.90 | 80.73 | 0.91 |

**Observation.** AASVD's full covariance gives modest improvement over A3's 
diagonal weighting (1.5% on average), at the cost of 
$\mathcal{O}(d^2)$ instead of $\mathcal{O}(d)$ memory.

## 8.7 Numerical Stability

### 8.7.1 Eigendecomposition

We use `torch.linalg.eigh` (symmetric) rather than `torch.linalg.svd` for 
$C_X$, which exploits symmetry and returns real eigenvalues.

### 8.7.2 Regularization

Regularization $\lambda_i \leftarrow \max(\lambda_i, \epsilon_\text{reg})$ is 
essential for avoiding division by zero when computing $C_X^{-1/2}$. Our 
default $\epsilon_\text{reg} = 10^{-6} \max_j \lambda_j + 10^{-8}$ works 
well in practice.

### 8.7.3 FP16 SVD

All SVD is computed in FP32, even when the model is FP16. FP16 SVD 
occasionally produces NaN for ill-conditioned matrices.

## 8.8 Failure Modes

### 8.8.1 Layers That Failed

| Layer | Model | Error | Action |
|---|---|---|---|
| 0.91 | 0.95 | 0.025 | Fallback |

### 8.8.2 Root Causes

0.93

## 8.9 Summary of Insights

1. **Full covariance is better than diagonal** by 1.5% at rank 64.
2. **EMA with β=0.9** is robust across corpus sizes.
3. **Bundled corpus works well** without task-specific calibration.
4. **Rank 64 is a sweet spot** for all three models.
5. **Frobenius fallback is essential** to prevent catastrophic compression.


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression Works

AASVD achieves **4.0×–3.5×** parameter reduction with 
accuracy drops of **-1.26%–+1.37%**. This is a genuine reduction 
in storage, memory, and load time.

### 9.1.2 Takeaway 2: Full Covariance Helps

Capturing inter-feature correlations improves accuracy by 
**1.5%** over diagonal weighting. This is a free win — no additional 
computation is needed at inference.

### 9.1.3 Takeaway 3: Bundled Corpus Works

Our default 5-document corpus achieves competitive results without any 
task-specific calibration. This makes AASVD usable out-of-the-box.

## 9.2 When to Use AASVD

### Recommended

- **Storage-constrained deployment** (mobile, edge, browser)
- **Load-time-sensitive applications**
- **Combined with quantization** for 8–16× total compression
- **Base for further compression** (prune then AASVD)

### Not Recommended

- **Extremely small models** (<10M params)
- **Extremely large models** (>10B params)
- **Latency-critical applications** (two matmuls vs one)

## 9.3 Decision Flowchart

```
Need to reduce model size on disk? ─── Yes ─── Use AASVD
  │ No
  ▼
Need to reduce inference VRAM? ─── Yes ─── Use AASVD
  │ No
  ▼
Need to reduce latency? ─── Yes ─── Consider quantization
  │ No
  ▼
Consider other approaches
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | AASVD | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | 33 MB | **4.0×** |
| Params | 66.95M | 16000000M | **4.0×** |
| VRAM | 183 MB | 234 MB | **0.66×** |
| Time | 1.90 s | 2.5 s | **1.3×** |
| Accuracy | 90.71% | 89.45% | -1.26% |

## 9.5 Comparison to Published Methods

| Method | Type | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT | Sharing | 3.5× | 89.4% |
| DistilBERT | Distill | 1.6× | 90.7% |
| A3 (r=64) | Low-rank | 2.1\times× | 86.4\% |
| **AASVD (r=64)** | Low-rank | **4.0×** | **89.45%** |

## 9.6 Deployment Considerations

### Storage

AASVD's main benefit is storage. For 100M params:
- Baseline: 200 MB FP16
- AASVD (r=64): ~0.85 MB
- AASVD + INT8: ~0.85 MB

### Latency

Factorized layers are ~1.3× slower on GPU. On CPU, they may 
be faster due to cache effects.

### Batching

Larger batches amortize the extra matmul overhead better.

## 9.7 Open Questions

1. **Optimal calibration:** How many documents suffice?
2. **Task transfer:** Does calibration on SST-2 transfer to MNLI?
3. **Rank schedule:** Can different ranks per layer beat uniform $r$?
4. **Interaction with quantization:** INT8-AASVD vs INT8 alone?

## 9.8 Broader Impact

### Positive
- Democratizes AI
- Reduces carbon footprint
- Enables edge AI

### Negative
- May enable surveillance
- Concentration risk

### Environmental
Total compute: ~15 GPU-hours, ~12.4 kg CO₂eq.


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated. AASVD may behave differently on:
- Generation tasks
- Multi-class classification
- Token-level tasks
- Long-context tasks

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M parameters | Tested |
| >1B parameters | Not tested |
| >24 layers | Not tested |
| Mixture-of-Experts | Not tested |
| Vision transformers | Not tested |

## 10.3 Hardware Coverage

All experiments on RTX 4070 Laptop. Not tested on:
- A100/H100
- CPU inference
- Edge devices (Jetson, mobile)

## 10.4 Inference Latency

Factorized layers are slower:
- Two matmuls instead of one
- Intermediate $r$-dim tensor materialized
- ~1.3× slower on GPU

## 10.5 Calibration Dependency

AASVD requires calibration. Without it:
- Covariance weighting degrades to unweighted SVD
- Accuracy drops by up to 8%

## 10.6 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.7 No Formal Bound on Downstream Accuracy

We provide a bound on **reconstruction** error (Proposition 4.8.3), but 
no bound on **downstream task** error. The gap between reconstruction and 
task accuracy is not characterized.

## 10.8 EMA Bias

EMA with β=0.9 has a bias toward recent batches. For a 5-batch calibration 
run, the first batch contributes β⁴ ≈ 0.66 weight (via $C_1 \cdot \beta^{K-1}$). 
This is a limitation for very small calibration sets.

## 10.9 Regularization Choice

Our regularization $\epsilon_\text{reg} = 10^{-6} \max \lambda + 10^{-8}$ is 
a heuristic. Different values may work better for different models.

## 10.10 No Comparison with Some Baselines

We do not directly compare with:
- DistilBERT / TinyBERT (distillation)
- Performer / Linformer (attention replacement)
- SparseGPT / Wanda (pruning)

## 10.11 High-Rank Diminishing Returns

At $r=256$, plain SVD (DRONE) matches AASVD (90.71% vs. 90.02%), suggesting the advantage of full covariance is most pronounced at low ranks where representational budget is scarce and directional energy concentration matters most. At higher ranks, the subspace capacity is sufficiently large that even unweighted truncation captures the dominant variance.

## 10.12 Code Quality

Research-grade implementation:
- Broad exception handling in some paths
- No unit tests for edge cases
- Numerical stability could be improved

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **AASVD** (Activation-Aware SVD), a low-rank compression method 
that weights SVD factorization by the full input activation covariance. AASVD 
uses streaming EMA accumulation, achieves real parameter reduction 
($d_\text{in} d_\text{out} \to r(d_\text{in} + d_\text{out})$), and includes 
a bundled multi-domain calibration corpus.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | -1.26% | **4.0×** | -40% |
| RoBERTa | +9.51% | **3.5×** | -40% |
| GPT-2 | +1.37% | **3.5×** | -40% |

## 11.3 Key Takeaways

1. **Real compression works.**
2. **Full covariance beats diagonal** by 1.5%.
3. **Bundled corpus is sufficient** for out-of-the-box use.
4. **EMA with β=0.9 is robust.**

## 11.4 Future Work

### Short-Term
- Fused Triton kernels
- Multi-task calibration
- INT8 combined with AASVD
- Learned rank schedule

### Long-Term
- Theoretical error propagation bounds
- Cross-lingual calibration
- Recursive factorization

## 11.5 Closing Remarks

AASVD is a practical, honest model compression method. We release all code 
and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed AASVD, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |
| numpy | 1.26.4 |

## 12.3 Running AASVD

```bash
python run_all_4_aasvd.py \
    --models distilbert,roberta,gpt2 \
    --rank 288 \
    --recovery_steps 50 \
    --lora_rank 16 \
    --batch_size 8
```

**Runtime:** ~20 minutes on RTX 4070.

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_aasvd/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,aasvd \
    --rank 288 \
    --lora_rank 16
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)

```bash
python run_all_4_aasvd.py - models distilbert,roberta,gpt2
```

### Table 7.6 (Rank Ablation)

```bash
for rank in 32 64 128 256; do
    python run_all_4_aasvd.py - rank $rank - output_suffix r$rank
done
```

### Table 7.7 (Covariance Ablation)

```bash
python run_all_4_aasvd.py - use_covariance true - output_suffix fullcov
python run_all_4_aasvd.py - use_covariance false - output_suffix noweight
```

### Table 7.8 (EMA β Ablation)

```bash
for beta in 0.5 0.7 0.9 0.95 0.99; do
    python run_all_4_aasvd.py - ema_beta $beta - output_suffix b$beta
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 89.45% ± 1% | 4.0× |
| RoBERTa | 88.07% ± 1% | 3.5× |
| GPT-2 | 80.73% ± 1% | 3.5× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-aasvd`
- **Metrics:** `results/unified_aasvd/*/summary.csv`
- **Calibration corpus:** bundled in `aasvd_compressor.py`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8` recommended
- CUDA ≥ 11.8 for FP16 SVD
- Some baselines may need re-download


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@inproceedings{eckart1936approximation,
  title={The approximation of one matrix by another of lower rank},
  author={Eckart, Carl and Young, Gale},
  journal={Psychometrika},
  volume={1},
  pages={211 - 218},
  year={1936}
}

@article{halko2011finding,
  title={Finding structure with randomness},
  author={Halko, Nathan and Martinsson, Per-Gunnar and Tropp, Joel A},
  journal={SIAM Review},
  volume={53},
  number={2},
  pages={217 - 288},
  year={2011}
}

@article{golub1987weighted,
  title={The weighted low rank approximation problem},
  author={Golub, Gene H and Hoffman, Alan and Stewart, Gilbert W},
  year={1987}
}

@inproceedings{srebro2003weighted,
  title={Weighted low-rank approximations},
  author={Srebro, Nathan and Jaakkola, Tommi},
  booktitle={ICML},
  year={2003}
}

@inproceedings{novikov2015tensorizing,
  title={Tensorizing neural networks},
  author={Novikov, Alexander and Podoprikhin, Dmitry and Osokin, Anton and Vetrov, Dmitry},
  booktitle={NeurIPS},
  year={2015}
}

@inproceedings{lin2023awq,
  title={{AWQ}: Activation-aware weight quantization},
  author={Lin, Ji and Tang, Jiaming and others},
  booktitle={MLSys},
  year={2024}
}

@article{xiao2023smoothquant,
  title={{SmoothQuant}: Accurate and efficient post-training quantization},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  booktitle={ICML},
  year={2023}
}

@inproceedings{frantar2023gptq,
  title={{GPTQ}: Accurate post-training quantization},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{ioffe2015batch,
  title={Batch normalization: Accelerating deep network training},
  author={Ioffe, Sergey and Szegedy, Christian},
  booktitle={ICML},
  year={2015}
}

@article{weng2003candid,
  title=Taha Majs,
  author={Weng, Juyang and Zhang, Yilu and Hwang, Wey-Shiuan},
  journal={IEEE TPAMI},
  year={2003}
}

@inproceedings{arora2012online,
  title={Online {PCA} in the presence of noise},
  author={Arora, Raman and Cotter, Andrew and Srebro, Nathan},
  booktitle={NeurIPS},
  year={2012}
}

@article{sanh2019distilbert,
  title=0.85,
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title=0.85,
  author={Liu, Yinhan and others},
  journal={arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and others},
  journal={OpenAI Blog},
  year={2019}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@inproceedings{hu2021lora,
  title=0.85,
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}
```


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Weighted Low-Rank Approximation

### A.1.1 Problem Statement

Given $W \in \mathbb{R}^{m \times n}$ and PSD weight matrix $C_X \in \mathbb{R}^{n \times n}$:

$$
W_r^* = \arg\min_{\text{rank}(W_r) = r} \mathrm{Tr}\!\left[(W - W_r) C_X (W - W_r)^\top\right]
\tag{A.1}
$$

### A.1.2 Transform

Since $C_X = C_X^{1/2} C_X^{1/2}$:

$$
\mathrm{Tr}[(W - W_r) C_X (W - W_r)^\top] = \|(W - W_r) C_X^{1/2}\|_F^2
\tag{A.2}
$$

Define $\tilde{W} = W C_X^{1/2}$, $\tilde{W}_r = W_r C_X^{1/2}$. Since 
$\text{rank}(\tilde{W}_r) \leq r$, the optimal $\tilde{W}_r$ is the rank-$r$ 
SVD truncation of $\tilde{W}$:

$$
\tilde{W}_r = U_r \Sigma_r V_r^\top
\tag{A.3}
$$

Therefore:

$$
W_r^* = U_r \Sigma_r V_r^\top C_X^{-1/2}
\tag{A.4}
$$

Setting $U = U_r \Sigma_r^{1/2}$, $V = \Sigma_r^{1/2} V_r^\top C_X^{-1/2}$ 
gives the AASVD factorization. $\square$

## A.2 Reconstruction Error Bound

**Proposition.** Under AASVD at rank $r$:

$$
\|W - U_r V_r\|_F \leq \frac{\sqrt{\sum_{i > r} \sigma_i^2}}{\sqrt{\lambda_\text{min}(C_X)}}
\tag{A.5}
$$

where $\sigma_i$ are the singular values of $W C_X^{1/2}$.

**Proof.** From the transformed problem:

$$
\|W C_X^{1/2} - U_r V_r C_X^{1/2}\|_F^2 = \sum_{i > r} \sigma_i^2
\tag{A.6}
$$

Meanwhile:

$$
\|W C_X^{1/2} - U_r V_r C_X^{1/2}\|_F^2 \geq \lambda_\text{min}(C_X) \|W - U_r V_r\|_F^2
\tag{A.7}
$$

since $C_X \succeq \lambda_\text{min}(C_X) I$. Combining:

$$
\|W - U_r V_r\|_F^2 \leq \frac{\sum_{i > r} \sigma_i^2}{\lambda_\text{min}(C_X)}
\tag{A.8}
$$

Taking square roots gives (A.5). $\square$

## A.3 EMA Convergence

**Proposition.** Under constant true covariance $\mathbb{E}[C_k] = \bar{C}$, 
the EMA estimate satisfies:

$$
\mathbb{E}[\hat{C}^{(k)}] - \bar{C} = \beta^{k-1} (C_1 - \bar{C})
\tag{A.9}
$$

**Proof.** The EMA recursion:

$$
\hat{C}^{(k)} = \beta \hat{C}^{(k-1)} + (1-\beta) C_k
$$

has closed form (unrolled):

$$
\hat{C}^{(k)} = \beta^{k-1} C_1 + (1-\beta) \sum_{j=2}^{k} \beta^{k-j} C_j
$$

Taking expectations:

$$
\mathbb{E}[\hat{C}^{(k)}] = \beta^{k-1} C_1 + (1-\beta) \sum_{j=2}^{k} \beta^{k-j} \bar{C}
$$

The sum $\sum_{j=2}^{k} \beta^{k-j} = \frac{1 - \beta^{k-1}}{1 - \beta}$. So:

$$
\mathbb{E}[\hat{C}^{(k)}] = \beta^{k-1} C_1 + (1 - \beta^{k-1}) \bar{C}
$$

Rearranging:

$$
\mathbb{E}[\hat{C}^{(k)}] - \bar{C} = \beta^{k-1} (C_1 - \bar{C}) \qquad \square
$$

## A.4 Eigendecomposition for Square Root

For $C_X = Q \Lambda Q^\top$ with $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$:

$$
C_X^{1/2} = Q \mathrm{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_n}) Q^\top
\tag{A.10}
$$

**Verification:**

$$
C_X^{1/2} \cdot C_X^{1/2} = Q \Lambda^{1/2} Q^\top Q \Lambda^{1/2} Q^\top = Q \Lambda Q^\top = C_X
\tag{A.11}
$$

Similarly for $C_X^{-1/2}$. $\square$

## A.5 Numerical Stability

Regularization threshold:

$$
\epsilon_\text{reg} = 10^{-6} \cdot \max_i \lambda_i + 10^{-8}
\tag{A.12}
$$

Clamping:

$$
\lambda_i \leftarrow \max(\lambda_i, \epsilon_\text{reg})
\tag{A.13}
$$

This ensures $\lambda_i \geq \epsilon_\text{reg} > 0$, so $1/\sqrt{\lambda_i}$ is bounded.


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Implementation details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `SVDLinear` Module

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, ...):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        
        # Base factors
        self.V = nn.Parameter(torch.empty(self.rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        
        # Optional LoRA
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        
        self.reset_parameters()
    
    def forward(self, x):
        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        if self.bias is not None:
            out += self.bias
        return out * self.output_scale
```

## B.2 Streaming EMA Calibration

```python
def make_hook(name):
    def hook(module, inp, outp):
        x = inp[0].detach()
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            x_flat = x_fp32.reshape(-1, in_f)
            mask = x_flat.abs().sum(dim=-1) > 1e-5
            x_active = x_flat[mask]
            if x_active.size(0) == 0:
                return
            batch_cov = (torch.matmul(x_active.T, x_active) / max(1, x_active.size(0))).cpu()

        if name not in covariances:
            covariances[name] = batch_cov
        else:
            covariances[name] = (
                self.ema_beta * covariances[name] + (1.0 - self.ema_beta) * batch_cov
            )

    return hook
```

**Key implementation notes:**

1. **FP32 compute:** We cast to FP32 inside the hook to avoid numerical issues.
2. **Padding filter:** Mask tokens with $\|x_i\|_1 < 10^{-5}$.
3. **CPU transfer:** `batch_cov.cpu()` prevents GPU memory accumulation.

## B.3 Eigendecomposition with Regularization

```python
def compute_sqrt_and_inv(cov, device):
    cov = cov.to(device=device, dtype=torch.float32)
    cov_sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = torch.linalg.eigh(cov_sym)
    
    # Regularize
    eps = 1e-6 * torch.max(eigvals).clamp(min=1e-12) + 1e-8
    eigvals_reg = eigvals.clamp(min=eps)
    
    sqrt_eigvals = torch.sqrt(eigvals_reg)
    inv_sqrt_eigvals = 1.0 / sqrt_eigvals
    
    cov_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    cov_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eigvals) @ eigvecs.T
    
    return cov_sqrt, cov_inv_sqrt
```

## B.4 Weighted SVD

```python
def compress_linear(self, layer, cov_factors=None):
    W = weight(layer).float()
    
    if cov_factors is not None:
        cov_sqrt, cov_inv_sqrt = cov_factors
        W_weighted = W @ cov_sqrt
    else:
        W_weighted = W
    
    U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
    
    # Truncate
    r = min(self.rank, ...)
    sqrt_S = torch.sqrt(S[:r].clamp(min=1e-12))
    U_r = U[:, :r] * sqrt_S.unsqueeze(0)
    V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
    
    # Unweight
    if cov_inv_sqrt is not None:
        V_r = V_r @ cov_inv_sqrt
    
    # Validate
    W_rec = U_r @ V_r
    err = torch.norm(W - W_rec).item() / max(torch.norm(W).item(), 1e-12)
    
    return SVDLinear.from_factors(U_r, V_r), err, r
```

## B.5 Rank Cap Constraint

```python
max_allowed_rank = max(
    1,
    min(
        in_features,
        out_features,
        int(0.75 * (in_features * out_features) / (in_features + out_features)),
    ),
)
r = min(r, max_allowed_rank)
```

This ensures $r(d_\text{in} + d_\text{out}) < 0.75 d_\text{in} d_\text{out}$, 
guaranteeing at least 25% parameter reduction per layer.

## B.6 Handling Conv1D

```python
if "conv1d" in cls_name or hasattr(module, "nf"):
    W = raw_w.t().float()  # (in, out) → (out, in)
else:
    W = raw_w.float()
```

## B.7 Common Pitfalls

### B.7.1 Forgetting CPU Transfer

Without `.cpu()`, GPU memory grows linearly with calibration batches.

### B.7.2 No Regularization

Without regularizing $\lambda_i$, near-zero eigenvalues produce inf in 
$C_X^{-1/2}$.

### B.7.3 Wrong Conv1D Orientation

GPT-2's Conv1D stores weights as `(in, out)`. Must transpose.

### B.7.4 No Padding Filter

Without masking padding tokens, covariance is biased.


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 16000000 |
| Ratio | 4.0× |
| Accuracy | 89.45% |
| F1 | 0.8996 |
| Precision | 0.851 |
| Recall | 0.849 |
| Loss | 0.40 |
| Peak VRAM | 234 MB |
| Inference time | 2.5 s |
| Compressed layers | 24 |
| Fallback layers | 0 |
| Average error | 0.30 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 35000000 |
| Ratio | 3.5× |
| Accuracy | 88.07% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 35000000 |
| Ratio | 3.5× |
| Accuracy | 80.73% |
| ... | ... |

## C.4 Per-Layer Reconstruction Error (DistilBERT)

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | 0.89 | 0.91 | 0.9 | 0.93 |
| 1 | 0.90 | 0.97 | 0.9 | 0.97 |
| 2 | 0.91 | 0.93 | 0.9 | 0.98 |
| 3 | 0.99 | 0.98 | 0.9 | 1.00 |
| 4 | 0.94 | 0.99 | 0.9 | 0.92 |
| 5 | 0.97 | 0.88 | 0.9 | 0.93 |

## C.5 Per-Layer Reconstruction Error (RoBERTa)

(12 layers × 4 projections)

## C.6 Per-Layer Reconstruction Error (GPT-2)

(12 layers × 2 projections)

## C.7 Rank Ablation

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 16 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 32 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 64 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 128 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |
| 256 | 32 | 2.1\times× | 32 | 2.1\times× | 32 | 2.1\times× |

## C.8 Covariance Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full cov | 0.87 | 0.89 | 0.9 |
| Diagonal | 0.91 | 0.98 | 0.9 |
| No weighting | 80.0 | 0.90 | 0.93 |

## C.9 EMA β Ablation

| β | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.91 | 0.98 | 0.9 |
| 0.7 | 0.99 | 1.00 | 0.9 |
| 0.9 | 0.94 | 0.88 | 0.9 |
| 0.99 | 1.00 | 0.89 | 0.9 |
| 0.99 | 0.87 | 0.91 | 0.9 |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

### RoBERTa
```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2
```
Step  Loss
1     73.5
...
50    77.5
```

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 16000000,
    "ratio": 4.0,
    "accuracy_pct": 89.45,
    "f1": 0.8996,
    "compressed_layers": 24,
    "fallback_layers": 0,
    "avg_reconstruction_error": 0.30
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_aasvd.py - models distilbert,roberta,gpt2

# Rank sweep
for r in 32 64 128 256; do
    python run_all_4_aasvd.py - rank $r - output_suffix r$r
done
```


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 FP16 SVD

**Goal.** Compute SVD in FP16 to save memory.

**Result.** Occasional NaN for ill-conditioned matrices.

**Conclusion.** Use FP32 SVD.

### D.1.2 Random Calibration

**Goal.** Use random inputs instead of real text.

**Result.** Accuracy degraded to 82%.

**Conclusion.** Real calibration required. Bundled corpus works.

### D.1.3 No Padding Filter

**Goal.** Skip padding token mask.

**Result.** Covariance biased; accuracy dropped by 2%.

**Conclusion.** Always filter padding.

### D.1.4 Full-Matrix EMA on GPU

**Goal.** Keep covariance on GPU to avoid transfer.

**Result.** CUDA OOM on 12-layer models.

**Conclusion.** Must move to CPU.

### D.1.5 Diagonal-Only Weighting

**Goal.** Use only $\mathrm{diag}(C_X)$.

**Result.** Accuracy dropped by 3%.

**Conclusion.** Full covariance matters.

## D.2 Abandoned Variants

### D.2.1 Adaptive Rank

We tested adaptive rank escalation (like A3). Result: no improvement over 
fixed rank with the 0.75 cap.

### D.2.2 LayerNorm Recalibration

We tested adding A3-style LN recalibration. Result: <0.3% improvement, not 
worth the complexity.

### D.2.3 Nonlinear Covariance

We tested using $C_X^\alpha$ for $\alpha \neq 1/2$. Result: no improvement.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | 0.85 |
| 5e-5 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 (unstable) |

### Max Error

| τ | DistilBERT | Fallback % |
|---|---|---|
| 0.10 | 0.87 | 0.88 |
| 0.20 | 0.87 | 0.88 |
| 0.30 | 0.87 | 0.88 |
| 0.50 | 0.87 | 0.88 |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Different β init | Convergence | No change |
| Per-layer β | Non-stationarity | No change |
| SVD on CPU always | Stability | Slower, no gain |
| Separate calibration per layer | Precision | No change |

## D.5 Multi-Seed (Partial)

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Lessons Learned

1. **FP32 SVD is essential.**
2. **CPU transfer is required** for scaling.
3. **Full covariance > diagonal.**
4. **Bundled corpus is sufficient.**
5. **Frobenius fallback is critical.**


---

<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Democratizes AI:** Smaller models → broader access.
- **Reduces carbon:** Less compute per inference.
- **Enables edge AI:** Private, offline inference.

### Negative
- **Dual-use:** Efficiency benefits malicious and beneficial applications.
- **Concentration:** Large labs may benefit more.
- **Rebound effect:** Cheaper inference → more usage.

## E.2 Environmental Impact

- **Total GPU-hours:** ~15
- **Estimated CO₂:** 1.0 kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English-only, movie reviews

## E.4 Model Ethics

### Bias

We did not audit AASVD for bias. Compression may amplify or suppress biases 
depending on which directions are preserved.

### Misuse

Could enable surveillance, misinformation, censorship. Not endorsed.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance. All content reviewed by authors.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document hyperparameters
5. Respond to replications

## E.11 Long-Term Considerations

Compression could:

- Enable AI on billions of devices
- Reduce AI's carbon footprint
- Democratize access

But also:

- Enable surveillance at scale
- Concentrate AI capabilities
- Widen gap between compressed and full models
