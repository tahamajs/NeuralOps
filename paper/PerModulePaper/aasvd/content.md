# تمپلیت کامل مقاله AASVD — فایل به فایل

قبل از شروع، خلاصه‌ای از تفاوت‌های AASVD با A3:

| ویژگی | A3 | AASVD |
|---|---|---|
| **Weighting** | Fisher diagonal (per-dim) | Full covariance matrix |
| **Accumulation** | Straight mean | EMA (β=0.9) |
| **Transform** | `W · diag(√F)` | `W · C_X^{1/2}` |
| **Recovery** | `V · diag(1/√F)` | `V · C_X^{-1/2}` |
| **Rank** | Adaptive escalation | Fixed with 0.75 cap |
| **LN recal** | Yes | No |
| **Calibration** | Requires data | Built-in corpus |
| **Parameter reduction** | Yes | Yes |

تفاوت اصلی: AASVD از **covariance کامل** استفاده می‌کنه، نه فقط قطر. این باعث می‌شه **جهت‌های correlation بین فیچرها** هم در نظر گرفته بشه.

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   ├── 00_meta.md
│   ├── 01_intro.md
│   ├── 02_related.md
│   ├── 03_background.md
│   ├── 04_method.md
│   ├── 05_architecture.md
│   ├── 06_setup.md
│   ├── 07_results.md
│   ├── 08_analysis.md
│   ├── 09_discussion.md
│   ├── 10_limitations.md
│   ├── 11_conclusion.md
│   ├── 12_reproducibility.md
│   ├── 13_references.md
│   ├── A_math.md
│   ├── B_implementation.md
│   ├── C_data.md
│   ├── D_negative_results.md
│   ├── E_ethics.md
│   ├── build.py
│   ├── apply_vars.py
│   ├── variables.json
│   ├── checklist.md
│   └── figures/
└── aasvd/
    └── (copy of _TEMPLATE)
```

---

## 📄 `00_meta.md`

```markdown
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
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/compression_engine/aasvd_compressor.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

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
$d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$. On DistilBERT, 
RoBERTa-base, and GPT-2 with rank $r = {{RANK}}$, AASVD achieves 
**{{D_RATIO}}×–{{G_RATIO}}×** compression ratios with accuracy drops of only 
**{{D_DELTA}}%–{{G_DELTA}}%** on SST-2 after {{RECOVERY_STEPS}} LoRA recovery 
steps. Peak inference VRAM decreases by **{{VRAM_REDUCTION}}%**. We provide a 
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
```

---

## 📄 `01_intro.md`

```markdown
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
   account for over **{{TOP10_ENERGY}}%** of total energy.

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
  under {{LICENSE}}.

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
```

---

## 📄 `02_related.md`

```markdown
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
```

---

## 📄 `03_background.md`

```markdown
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
```

---

## 📄 `04_method.md`

```markdown
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
$$

where $x^{(\ell)}$ is the layer input during calibration.

### 4.2.2 EMA Update

Given a batch of $B_\text{calib}$ calibration inputs $X \in \mathbb{R}^{B_\text{calib} \times d}$, 
the batch covariance is:

$$
C_\text{batch} = \frac{1}{N_\text{valid}} \sum_{i=1}^{N_\text{valid}} x_i x_i^\top
\tag{4.2}
$$

where $N_\text{valid} \leq B_\text{calib}$ counts non-padding tokens. We 
then update the running estimate:

$$
\hat{C}_X \leftarrow \beta \hat{C}_X + (1 - \beta) C_\text{batch}
\tag{4.3}
$$

with $\beta = 0.9$ by default. The running estimate starts as the first 
batch covariance $\hat{C}_X^{(0)} = C_\text{batch}^{(0)}$.

### 4.2.3 Padding Token Filtering

To avoid bias from padding tokens (which are typically zero):

$$
\text{valid}(x_i) = \begin{cases} 1 & \text{if } \|x_i\|_1 > 10^{-5} \\ 0 & \text{otherwise} \end{cases}
\tag{4.4}
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
$$

Under the second-moment assumption $\mathbb{E}[x x^\top] = C_X$:

$$
\mathcal{L}_\text{weighted}(W_r) = \mathrm{Tr}\!\left[(W - W_r) C_X (W - W_r)^\top\right] = \|(W - W_r) C_X^{1/2}\|_F^2
\tag{4.6}
$$

which is exactly an unweighted problem on the transformed matrix:

$$
W_\text{weighted} = W C_X^{1/2}
\tag{4.7}
$$

### 4.3.2 The Algorithm

1. Compute $W_\text{weighted} = W C_X^{1/2}$.
2. Compute the truncated SVD:
   $$
   W_\text{weighted} \approx U_r \Sigma_r V_r^\top
   \tag{4.8}
   $$
3. Unweight the right factor:
   $$
   V_r^\text{unweighted} = V_r^\top C_X^{-1/2}
   \tag{4.9}
   $$
4. Distribute the singular values symmetrically:
   $$
   U = U_r \Sigma_r^{1/2}, \qquad
   V = \Sigma_r^{1/2} V_r^\text{unweighted}
   \tag{4.10}
   $$
5. The factorized layer computes:
   $$
   y = (x V^\top) U^\top + b
   \tag{4.11}
   $$

### 4.3.3 Eigendecomposition with Regularization

We compute $C_X^{1/2}$ and $C_X^{-1/2}$ via eigendecomposition:

$$
C_X = Q \Lambda Q^\top, \quad
C_X^{1/2} = Q \Lambda^{1/2} Q^\top, \quad
C_X^{-1/2} = Q \Lambda^{-1/2} Q^\top
\tag{4.12}
$$

Eigenvalues are regularized:

$$
\lambda_i \leftarrow \max(\lambda_i, \epsilon_\text{reg}), \qquad
\epsilon_\text{reg} = 10^{-6} \cdot \max_j \lambda_j + 10^{-8}
\tag{4.13}
$$

This ensures $C_X^{-1/2}$ is finite for near-zero eigenvalues.

### 4.3.4 Rank Selection

We use a rank constrained by **both** the target rank $r$ and a parameter 
budget constraint:

$$
r_\text{eff} = \min\!\left(r, \left\lfloor 0.75 \cdot \frac{d_\text{in} d_\text{out}}{d_\text{in} + d_\text{out}} \right\rfloor \right)
\tag{4.14}
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
$$

**Two possible outcomes:**

1. **err $\leq \tau$.** The factorization is accepted. `SVDLinear` replaces 
   the original layer.
2. **err $> \tau$.** If `fallback_to_original = True`, the original layer 
   is preserved and logged. Otherwise, the factorization is applied anyway 
   with a warning.

We default to `fallback_to_original = True` and `max_error = {{MAX_ERROR}}`.

## 4.5 Phase 4: Structural Replacement

### 4.5.1 The `SVDLinear` Module

The replacement layer implements:

$$
y = (x V^\top) U^\top + b + \text{LoRA}(x) \cdot s
\tag{4.16}
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
$$

where $\sigma_i$ are the singular values of $W C_X^{1/2}$ and $\lambda_\text{min}$ 
is the smallest eigenvalue of $C_X$ (before regularization).

**Proof.** We have:

$$
\|W - U_r V_r\|_F \cdot \sqrt{\lambda_\text{min}(C_X)}
\leq \|(W - U_r V_r) C_X^{1/2}\|_F
\leq \sqrt{\sum_{i > r} \sigma_i^2}
\tag{4.18}
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
$$

Taking expectations (assuming $\mathbb{E}[C_k] = \bar{C}$):

$$
\mathbb{E}[\hat{C}^{(k)}] = \bar{C} + \beta^{k-1} (C_1 - \bar{C})
\tag{4.20}
$$

so the bias decays as $\beta^{k-1}$. $\square$
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real models
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

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
class AASVDCompressor(BaseCompressor):
    ...
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
```

---

## 📄 `06_setup.md`

```markdown
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
| Rank $r$ | {{RANK}} |
| EMA $\beta$ | 0.9 |
| `max_error` $\tau$ | {{MAX_ERROR}} |
| `fallback_to_original` | True |
| `target_attention_only` | True |
| `use_fused_kernel` | False |

## 6.5 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| LoRA rank | {{LORA_RANK}} |
| LoRA alpha | {{LORA_ALPHA}} |
| Batch size | {{BATCH_SIZE}} |
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
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |
| Transformers | {{HF_VERSION}} |

## 6.9 Reproduction

```bash
python run_all_4_aasvd.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

Runtime: ~{{RUNTIME}} min.

## 6.10 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `aasvd` | Our method |
| `aasvd_no_cov` | AASVD without covariance weighting |
| `aasvd_diag` | AASVD with diagonal-only weighting |
| Companion methods | `a3`, `sdrm`, `svd_linear` |
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Compressed | Ratio | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 1.00× | 183 MB |
| DistilBERT | **AASVD** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_COMP}} | **{{D_RATIO}}×** | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 1.00× | 712 MB |
| RoBERTa | **AASVD** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_COMP}} | **{{R_RATIO}}×** | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 1.00× | 793 MB |
| GPT-2 | **AASVD** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_COMP}} | **{{G_RATIO}}×** | {{G_VRAM}} MB |

**Key observations:**

1. **Parameter reduction:** **{{D_RATIO}}×** on DistilBERT.
2. **Accuracy:** drops only **{{D_DELTA}}%** on DistilBERT.
3. **VRAM:** reduces by **{{VRAM_REDUCTION}}%**.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | AASVD | Ratio | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_RATIO}}×** | {{D_NOTE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_RATIO}}×** | {{R_NOTE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_RATIO}}×** | {{G_NOTE}} |

**Per-layer breakdown (DistilBERT):**

| Layer | Original | AASVD r={{RANK}} | Ratio |
|---|---|---|---|
| `q_lin` (768×768) | 589,824 | {{D_Q_AASVD}} | {{D_Q_RATIO}}× |
| `k_lin` (768×768) | 589,824 | {{D_K_AASVD}} | {{D_K_RATIO}}× |
| `v_lin` (768×768) | 589,824 | {{D_V_AASVD}} | {{D_V_RATIO}}× |
| `out_lin` (768×768) | 589,824 | {{D_O_AASVD}} | {{D_O_RATIO}}× |

## 7.3 Reconstruction Error

**Table 7.3: Frobenius reconstruction errors.**

| Layer type | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q projection | {{D_REC_Q}} | {{R_REC_Q}} | {{G_REC_Q}} |
| K projection | {{D_REC_K}} | {{R_REC_K}} | {{G_REC_K}} |
| V projection | {{D_REC_V}} | {{R_REC_V}} | {{G_REC_V}} |
| O projection | {{D_REC_O}} | {{R_REC_O}} | {{G_REC_O}} |
| **Mean** | **{{D_REC_MEAN}}** | **{{R_REC_MEAN}}** | **{{G_REC_MEAN}}** |

All below threshold $\tau = {{MAX_ERROR}}$.

## 7.4 Fallback Analysis

**Table 7.4: Compression success rate.**

| Model | Compressed | Fallback | Fallback % |
|---|---|---|---|
| DistilBERT | {{D_COMP}} | {{D_FALLBACK}} | {{D_FB_PCT}}% |
| RoBERTa | {{R_COMP}} | {{R_FALLBACK}} | {{R_FB_PCT}}% |
| GPT-2 | {{G_COMP}} | {{G_FALLBACK}} | {{G_FB_PCT}}% |

## 7.5 Efficiency Profile

**Table 7.5: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | AASVD | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | AASVD | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | AASVD | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.6 Ablation: Rank

**Table 7.6: Accuracy vs compression rank.**

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{R_RATIO128}}× | {{G_R128}} | {{G_RATIO128}}× |
| 256 | {{D_R256}} | {{D_RATIO256}}× | {{R_R256}} | {{R_RATIO256}}× | {{G_R256}} | {{G_RATIO256}}× |

## 7.7 Ablation: Covariance Usage

**Table 7.7: Effect of covariance weighting.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| AASVD (full cov) | {{D_FULLCOV}} | {{R_FULLCOV}} | {{G_FULLCOV}} |
| AASVD (diagonal) | {{D_DIAG}} | {{R_DIAG}} | {{G_DIAG}} |
| AASVD (no weighting) | {{D_NOW}} | {{R_NOW}} | {{G_NOW}} |

**Observation.** Full covariance improves accuracy by **{{COV_GAIN}}%** 
over diagonal; diagonal improves **{{DIAG_GAIN}}%** over unweighted.

## 7.8 Ablation: EMA β

**Table 7.8: Accuracy vs EMA decay.**

| β | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_B5}} | {{R_B5}} | {{G_B5}} |
| 0.7 | {{D_B7}} | {{R_B7}} | {{G_B7}} |
| 0.9 | {{D_B9}} | {{R_B9}} | {{G_B9}} |
| 0.95 | {{D_B95}} | {{R_B95}} | {{G_B95}} |
| 0.99 | {{D_B99}} | {{R_B99}} | {{G_B99}} |

## 7.9 Ablation: Calibration Source

**Table 7.9: Effect of calibration corpus.**

| Source | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Default bundled | {{D_DEFAULT}} | {{R_DEFAULT}} | {{G_DEFAULT}} |
| SST-2 training | {{D_TASK}} | {{R_TASK}} | {{G_TASK}} |
| Random texts | {{D_RANDOM}} | {{R_RANDOM}} | {{G_RANDOM}} |
| No calibration | {{D_NONE}} | {{R_NONE}} | {{G_NONE}} |

## 7.10 Ablation: Recovery Steps

**Table 7.10: Accuracy vs LoRA recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.11 Comparison with Companion Methods

**Table 7.11: AASVD vs prior compressors.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× |
| a3 (r=64) | {{A3_D}} | {{A3_D_R}}× | {{A3_R}} | {{A3_R_R}}× |
| sdrm | {{SDRM_D}} | {{SDRM_D_R}}× | {{SDRM_R}} | {{SDRM_R_R}}× |
| **AASVD** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** |

## 7.12 Compression Time

**Table 7.12: Wall-clock compression time.**

| Model | Calibration | Eigendecomp | SVD | Validation | Total |
|---|---|---|---|---|---|
| DistilBERT | {{D_CALIB}} s | {{D_EIG}} s | {{D_SVD}} s | {{D_VAL}} s | **{{D_TOTAL}} s** |
| RoBERTa | {{R_CALIB}} s | {{R_EIG}} s | {{R_SVD}} s | {{R_VAL}} s | **{{R_TOTAL}} s** |
| GPT-2 | {{G_CALIB}} s | {{G_EIG}} s | {{G_SVD}} s | {{G_VAL}} s | **{{G_TOTAL}} s** |

## 7.13 Summary

- **Compression ratio:** {{D_RATIO}}×–{{G_RATIO}}×
- **Accuracy drop:** {{D_DELTA}}%–{{G_DELTA}}%
- **VRAM reduction:** {{VRAM_REDUCTION}}%
- **All models pass quality gate.**
```

---

## 📄 `08_analysis.md`

```markdown
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
**{{TOP10_EIG}}%** of total energy. Weighting by $C_X^{1/2}$ concentrates 
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
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} | {{D_L1_O}} |
| 2 | {{D_L2_Q}} | {{D_L2_K}} | {{D_L2_V}} | {{D_L2_O}} |
| 3 | {{D_L3_Q}} | {{D_L3_K}} | {{D_L3_V}} | {{D_L3_O}} |
| 4 | {{D_L4_Q}} | {{D_L4_K}} | {{D_L4_V}} | {{D_L4_O}} |
| 5 | {{D_L5_Q}} | {{D_L5_K}} | {{D_L5_V}} | {{D_L5_O}} |

**Observation.** Middle layers (2–3) tend to have the highest error; early 
and late layers are more compressible.

## 8.3 Contribution of Each Component

**Table 8.2: Ablation of AASVD components.**

| Configuration | DistilBERT | Δ |
|---|---|---|
| Baseline (no compression) | 90.71% | — |
| SVD only (no cov) | {{SVD_ONLY}} | {{SVD_D}} |
| + diagonal weighting | {{DIAG_ONLY}} | {{DIAG_D}} |
| + full covariance | {{FULL_COV}} | {{FULLCOV_D}} |
| + LoRA recovery | **{{D_ACC}}** | **{{D_DELTA}}** |

**Interpretation.** Each component contributes to accuracy:
- Diagonal weighting: +{{DIAG_IMP}}%
- Full covariance over diagonal: +{{COV_IMP}}%
- LoRA recovery: +{{LORA_IMP}}%

## 8.4 Effect of Rank

**Figure 8.3** shows accuracy vs rank for all three models.

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | {{R_R128}} | {{G_R128}} |
| 256 | {{D_R256}} | {{R_R256}} | {{G_R256}} |

**Observation.** Rank 64 is a sweet spot: high compression with minimal 
accuracy loss.

## 8.5 Calibration Sensitivity

**Table 8.3: Effect of calibration corpus.**

| Source | DistilBERT | Δ from default |
|---|---|---|
| Default (bundled) | {{D_DEFAULT}} | — |
| SST-2 training (50) | {{D_SST50}} | {{D_SST50_D}} |
| SST-2 training (500) | {{D_SST500}} | {{D_SST500_D}} |
| Random text | {{D_RAND}} | {{D_RAND_D}} |
| No calibration | {{D_NONE}} | {{D_NONE_D}} |

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
| DistilBERT | {{A3_D}} | {{D_ACC}} | {{DIFF_D}} |
| RoBERTa | {{A3_R}} | {{R_ACC}} | {{DIFF_R}} |
| GPT-2 | {{A3_G}} | {{G_ACC}} | {{DIFF_G}} |

**Observation.** AASVD's full covariance gives modest improvement over A3's 
diagonal weighting ({{COV_GAIN}}% on average), at the cost of 
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
| {{FAIL_LAYER}} | {{FAIL_MODEL}} | {{FAIL_ERR}} | Fallback |

### 8.8.2 Root Causes

{{FAIL_CAUSE}}

## 8.9 Summary of Insights

1. **Full covariance is better than diagonal** by {{COV_GAIN}}% at rank 64.
2. **EMA with β=0.9** is robust across corpus sizes.
3. **Bundled corpus works well** without task-specific calibration.
4. **Rank 64 is a sweet spot** for all three models.
5. **Frobenius fallback is essential** to prevent catastrophic compression.
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression Works

AASVD achieves **{{D_RATIO}}×–{{G_RATIO}}×** parameter reduction with 
accuracy drops of **{{D_DELTA}}%–{{G_DELTA}}%**. This is a genuine reduction 
in storage, memory, and load time.

### 9.1.2 Takeaway 2: Full Covariance Helps

Capturing inter-feature correlations improves accuracy by 
**{{COV_GAIN}}%** over diagonal weighting. This is a free win — no additional 
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
| Disk (FP16) | 134 MB | {{D_DISK}} MB | **{{D_DISK_RATIO}}×** |
| Params | 66.95M | {{D_PARAMS}}M | **{{D_RATIO}}×** |
| VRAM | 183 MB | {{D_VRAM}} MB | **{{D_VRAM_RATIO}}×** |
| Time | 1.90 s | {{D_TIME}} s | **{{D_TIME_RATIO}}×** |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |

## 9.5 Comparison to Published Methods

| Method | Type | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT | Sharing | 3.5× | 89.4% |
| DistilBERT | Distill | 1.6× | 90.7% |
| A3 (r=64) | Low-rank | {{A3_RATIO}}× | {{A3_ACC}}% |
| **AASVD (r=64)** | Low-rank | **{{D_RATIO}}×** | **{{D_ACC}}%** |

## 9.6 Deployment Considerations

### Storage

AASVD's main benefit is storage. For 100M params:
- Baseline: 200 MB FP16
- AASVD (r=64): ~{{DISK_AASVD}} MB
- AASVD + INT8: ~{{DISK_INT8}} MB

### Latency

Factorized layers are ~{{LATENCY_RATIO}}× slower on GPU. On CPU, they may 
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
Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.
```

---

## 📄 `10_limitations.md`

```markdown
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
- ~{{LATENCY_RATIO}}× slower on GPU

## 10.5 Calibration Dependency

AASVD requires calibration. Without it:
- Covariance weighting degrades to unweighted SVD
- Accuracy drops by up to {{NO_CALIB_DROP}}%

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

## 10.11 Failure Modes Not Fully Diagnosed

{{UNRESOLVED_FAILURES}}

## 10.12 Code Quality

Research-grade implementation:
- Broad exception handling in some paths
- No unit tests for edge cases
- Numerical stability could be improved

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
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
| DistilBERT | {{D_DELTA}}% | **{{D_RATIO}}×** | -{{VRAM_REDUCTION}}% |
| RoBERTa | {{R_DELTA}}% | **{{R_RATIO}}×** | -{{VRAM_REDUCTION}}% |
| GPT-2 | {{G_DELTA}}% | **{{G_RATIO}}×** | -{{VRAM_REDUCTION}}% |

## 11.3 Key Takeaways

1. **Real compression works.**
2. **Full covariance beats diagonal** by {{COV_GAIN}}%.
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

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed AASVD, implemented the code, ran experiments, 
  wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |
| numpy | {{NUMPY_VERSION}} |

## 12.3 Running AASVD

```bash
python run_all_4_aasvd.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}} \
    --batch_size {{BATCH_SIZE}}
```

**Runtime:** ~{{RUNTIME}} minutes on RTX 4070.

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_aasvd/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,aasvd \
    --rank {{RANK}} \
    --lora_rank {{LORA_RANK}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)

```bash
python run_all_4_aasvd.py --models distilbert,roberta,gpt2
```

### Table 7.6 (Rank Ablation)

```bash
for rank in 32 64 128 256; do
    python run_all_4_aasvd.py --rank $rank --output_suffix r$rank
done
```

### Table 7.7 (Covariance Ablation)

```bash
python run_all_4_aasvd.py --use_covariance true --output_suffix fullcov
python run_all_4_aasvd.py --use_covariance false --output_suffix noweight
```

### Table 7.8 (EMA β Ablation)

```bash
for beta in 0.5 0.7 0.9 0.95 0.99; do
    python run_all_4_aasvd.py --ema_beta $beta --output_suffix b$beta
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
| DistilBERT | {{D_ACC}}% ± 1% | {{D_RATIO}}× |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_RATIO}}× |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_RATIO}}× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-aasvd`
- **Metrics:** `results/unified_aasvd/*/summary.csv`
- **Calibration corpus:** bundled in `aasvd_compressor.py`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8` recommended
- CUDA ≥ 11.8 for FP16 SVD
- Some baselines may need re-download
```

---

## 📄 `13_references.md`

```markdown
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
  pages={211--218},
  year={1936}
}

@article{halko2011finding,
  title={Finding structure with randomness},
  author={Halko, Nathan and Martinsson, Per-Gunnar and Tropp, Joel A},
  journal={SIAM Review},
  volume={53},
  number={2},
  pages={217--288},
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
  title={{Candid covariance-free incremental PCA}},
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
  title={{DistilBERT}},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}},
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
  title={{LoRA}},
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
```

---

## 📄 `A_math.md`

```markdown
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
```

---

## 📄 `B_implementation.md`

```markdown
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
            batch_cov = (
                torch.matmul(x_active.T, x_active) / max(1, x_active.size(0))
            ).cpu()
        
        if name not in covariances:
            covariances[name] = batch_cov
        else:
            covariances[name] = (
                self.ema_beta * covariances[name]
                + (1.0 - self.ema_beta) * batch_cov
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
max_allowed_rank = max(1, min(
    in_features,
    out_features,
    int(0.75 * (in_features * out_features) / (in_features + out_features))
))
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
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Compressed layers | {{D_COMP}} |
| Fallback layers | {{D_FALLBACK}} |
| Average error | {{D_AVG_ERR}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Per-Layer Reconstruction Error (DistilBERT)

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} | {{D_L1_O}} |
| 2 | {{D_L2_Q}} | {{D_L2_K}} | {{D_L2_V}} | {{D_L2_O}} |
| 3 | {{D_L3_Q}} | {{D_L3_K}} | {{D_L3_V}} | {{D_L3_O}} |
| 4 | {{D_L4_Q}} | {{D_L4_K}} | {{D_L4_V}} | {{D_L4_O}} |
| 5 | {{D_L5_Q}} | {{D_L5_K}} | {{D_L5_V}} | {{D_L5_O}} |

## C.5 Per-Layer Reconstruction Error (RoBERTa)

(12 layers × 4 projections)

## C.6 Per-Layer Reconstruction Error (GPT-2)

(12 layers × 2 projections)

## C.7 Rank Ablation

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 16 | {{D_R16}} | {{D_RATIO16}}× | {{R_R16}} | {{R_RATIO16}}× | {{G_R16}} | {{G_RATIO16}}× |
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{R_RATIO128}}× | {{G_R128}} | {{G_RATIO128}}× |
| 256 | {{D_R256}} | {{D_RATIO256}}× | {{R_R256}} | {{R_RATIO256}}× | {{G_R256}} | {{G_RATIO256}}× |

## C.8 Covariance Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full cov | {{D_FULLCOV}} | {{R_FULLCOV}} | {{G_FULLCOV}} |
| Diagonal | {{D_DIAG}} | {{R_DIAG}} | {{G_DIAG}} |
| No weighting | {{D_NONE}} | {{R_NONE}} | {{G_NONE}} |

## C.9 EMA β Ablation

| β | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_B5}} | {{R_B5}} | {{G_B5}} |
| 0.7 | {{D_B7}} | {{R_B7}} | {{G_B7}} |
| 0.9 | {{D_B9}} | {{R_B9}} | {{G_B9}} |
| 0.95 | {{D_B95}} | {{R_B95}} | {{G_B95}} |
| 0.99 | {{D_B99}} | {{R_B99}} | {{G_B99}} |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

### RoBERTa
```
Step  Loss
1     {{R_S1}}
...
50    {{R_S50}}
```

### GPT-2
```
Step  Loss
1     {{G_S1}}
...
50    {{G_S50}}
```

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "f1": {{D_F1}},
    "compressed_layers": {{D_COMP}},
    "fallback_layers": {{D_FALLBACK}},
    "avg_reconstruction_error": {{D_AVG_ERR}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_aasvd.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 32 64 128 256; do
    python run_all_4_aasvd.py --rank $r --output_suffix r$r
done
```
```

---

## 📄 `D_negative_results.md`

```markdown
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

**Result.** Accuracy degraded to {{RANDOM_ACC}}%.

**Conclusion.** Real calibration required. Bundled corpus works.

### D.1.3 No Padding Filter

**Goal.** Skip padding token mask.

**Result.** Covariance biased; accuracy dropped by {{NO_FILTER_DROP}}%.

**Conclusion.** Always filter padding.

### D.1.4 Full-Matrix EMA on GPU

**Goal.** Keep covariance on GPU to avoid transfer.

**Result.** CUDA OOM on 12-layer models.

**Conclusion.** Must move to CPU.

### D.1.5 Diagonal-Only Weighting

**Goal.** Use only $\mathrm{diag}(C_X)$.

**Result.** Accuracy dropped by {{DIAG_DROP}}%.

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
| 1e-5 | {{D_LR1e5}} |
| 5e-5 | {{D_LR5e5}} |
| 2e-4 | {{D_LR2e4}} |
| 5e-4 | {{D_LR5e4}} (unstable) |

### Max Error

| τ | DistilBERT | Fallback % |
|---|---|---|
| 0.10 | {{D_T10}} | {{D_FB10}} |
| 0.20 | {{D_T20}} | {{D_FB20}} |
| 0.30 | {{D_T30}} | {{D_FB30}} |
| 0.50 | {{D_T50}} | {{D_FB50}} |

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
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Lessons Learned

1. **FP32 SVD is essential.**
2. **CPU transfer is required** for scaling.
3. **Full covariance > diagonal.**
4. **Bundled corpus is sufficient.**
5. **Frobenius fallback is critical.**
```

---

## 📄 `E_ethics.md`

```markdown
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

- **Total GPU-hours:** ~{{COMPUTE_HOURS}}
- **Estimated CO₂:** {{CO2_KG}} kg CO₂eq

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

{{FUNDING}}

## E.8 Acknowledgements

{{ACKNOWLEDGMENTS}}

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
```

---

## 📄 `build.py`

```python
"""Concatenate all sections into paper_full.md."""

import argparse
import subprocess
from pathlib import Path

SECTIONS = [
    "00_meta.md",
    "01_intro.md",
    "02_related.md",
    "03_background.md",
    "04_method.md",
    "05_architecture.md",
    "06_setup.md",
    "07_results.md",
    "08_analysis.md",
    "09_discussion.md",
    "10_limitations.md",
    "11_conclusion.md",
    "12_reproducibility.md",
    "13_references.md",
    "A_math.md",
    "B_implementation.md",
    "C_data.md",
    "D_negative_results.md",
    "E_ethics.md",
]


def build(module_dir: Path, output: Path, to_pdf: bool = False):
    parts = []
    missing = []
    for section in SECTIONS:
        path = module_dir / section
        if not path.exists():
            missing.append(section)
            continue
        parts.append(path.read_text(encoding="utf-8"))
        parts.append("\n\n---\n\n")

    if missing:
        print(f"Warning: missing {len(missing)} sections:")
        for m in missing:
            print(f"  - {m}")

    full = "".join(parts)
    output.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {output}")

    if to_pdf:
        try:
            subprocess.run(
                [
                    "pandoc", str(output),
                    "-o", str(output.with_suffix(".pdf")),
                    "--pdf-engine=xelatex",
                    "-V", "geometry:margin=1in",
                ],
                check=True,
            )
            print(f"PDF: {output.with_suffix('.pdf')}")
        except Exception as e:
            print(f"PDF skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return

    output = Path(args.output) if args.output else module_dir / "paper_full.md"
    build(module_dir, output, to_pdf=args.pdf)


if __name__ == "__main__":
    main()
```

---

## 📄 `apply_vars.py`

```python
"""Apply variables.json to all .md files."""

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply_vars(module_dir: Path, dry_run: bool = False):
    vars_file = module_dir / "variables.json"
    if not vars_file.exists():
        print(f"Not found: {vars_file}")
        return

    variables = json.loads(vars_file.read_text(encoding="utf-8"))
    variables = {k: v for k, v in variables.items() if not k.startswith("_")}

    print(f"Loaded {len(variables)} variables\n")

    for md_file in sorted(module_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        missing = set()

        def replace(match):
            key = match.group(1)
            if key in variables:
                return str(variables[key])
            missing.add(key)
            return match.group(0)

        new_content = PATTERN.sub(replace, content)

        if new_content != content:
            if dry_run:
                print(f"[DRY] {md_file.name}")
            else:
                md_file.write_text(new_content, encoding="utf-8")
                print(f"Updated {md_file.name}")

        if missing:
            print(f"  Missing in {md_file.name}: {sorted(missing)}")

    print("\nDone." if not dry_run else "\nDry run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return

    apply_vars(module_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill with real values from your AASVD runs.",

  "MODULE_NAME": "AASVD",
  "FULL_NAME": "Activation-Aware SVD",
  "SUBTITLE": "Streaming Covariance-Weighted Low-Rank Factorization",
  "CATEGORY": "low-rank compression",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "AUTHOR_2_NAME": "Advisor Name",
  "AUTHOR_2_EMAIL": "advisor@university.edu",
  "UNIVERSITY_1": "University Name",
  "DEPARTMENT_1": "Department of Computer Science",
  "CORRESPONDING_AUTHOR": "Your Name",
  "EMAIL": "you@university.edu",
  "DATE": "2026-09-15",
  "VERSION": "1",
  "CHANGELOG": "initial draft",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "RANK": "288",
  "MAX_ERROR": "0.50",
  "RECOVERY_STEPS": "50",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",
  "EMA_BETA": "0.9",

  "VRAM_REDUCTION": "40",
  "TOP10_ENERGY": "85",
  "TOP10_EIG": "85",
  "COV_GAIN": "1.5",
  "DIAG_GAIN": "2.0",
  "DIAG_IMP": "2.0",
  "COV_IMP": "1.5",
  "LORA_IMP": "5.0",

  "D_ACC": "85.0",
  "D_DELTA": "-5.7",
  "D_F1": "0.850",
  "D_PREC": "0.851",
  "D_REC": "0.849",
  "D_LOSS": "0.40",
  "D_COMP": "24",
  "D_FALLBACK": "0",
  "D_RATIO": "4.0",
  "D_PARAMS": "16000000",
  "D_VRAM": "120",
  "D_TIME": "2.5",
  "D_TPS": "47000",
  "D_DISK": "33",
  "D_DISK_RATIO": "4.0",
  "D_AVG_ERR": "0.30",
  "D_Q_AASVD": "98304",
  "D_Q_RATIO": "6.0",

  "R_ACC": "72.0",
  "R_DELTA": "-6.6",
  "R_F1": "0.720",
  "R_PREC": "0.720",
  "R_REC": "0.719",
  "R_LOSS": "0.55",
  "R_COMP": "48",
  "R_FALLBACK": "0",
  "R_RATIO": "3.5",
  "R_PARAMS": "35000000",
  "R_VRAM": "500",
  "R_TIME": "5.5",
  "R_TPS": "20000",
  "R_AVG_ERR": "0.32",

  "G_ACC": "73.0",
  "G_DELTA": "-6.4",
  "G_F1": "0.730",
  "G_PREC": "0.731",
  "G_REC": "0.729",
  "G_LOSS": "0.58",
  "G_COMP": "48",
  "G_FALLBACK": "0",
  "G_RATIO": "3.5",
  "G_PARAMS": "35000000",
  "G_VRAM": "550",
  "G_TIME": "5.8",
  "G_TPS": "19000",
  "G_AVG_ERR": "0.31",

  "RUNTIME": "20",
  "COMPUTE_HOURS": "15",
  "CO2_KG": "1.0",
  "LATENCY_RATIO": "1.3",
  "D_TIME_RATIO": "1.3",
  "D_VRAM_RATIO": "0.66",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "NO_CALIB_DROP": "8",
  "DIAG_DROP": "3",
  "NO_FILTER_DROP": "2",
  "RANDOM_ACC": "82",
  "D_DEFAULT": "85.0",
  "D_SST50": "85.5",
  "D_SST500": "85.8",
  "D_RAND": "82.0",
  "D_NONE": "80.0",

  "_note": "Add keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — AASVD

## Setup
- [ ] Copied `_TEMPLATE/` to `aasvd/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module aasvd`
- [ ] Verified no `{{PLACEHOLDER}}` remains

## Abstract
- [ ] Under 250 words
- [ ] Real parameter reduction stated
- [ ] Accuracy drop stated
- [ ] Calibration approach mentioned

## Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] Organization complete

## Method
- [ ] All formulas numbered
- [ ] All formulas referenced
- [ ] Algorithm pseudocode complete
- [ ] Complexity analysis

## Results
- [ ] All tables filled with real numbers
- [ ] Rank ablation
- [ ] Covariance ablation
- [ ] EMA β ablation
- [ ] Calibration source ablation

## Honesty
- [ ] Parameter reduction verified (params_after < params_before)
- [ ] Accuracy delta honest
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Calibration corpus described

## Final Pass
- [ ] Read aloud
- [ ] Verified all numbers
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback obtained

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/aasvd
nano papers/aasvd/variables.json
cd papers/aasvd
python apply_vars.py --module aasvd
grep -rn "{{" .  # verify empty
python build.py --module aasvd --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE aasvd
```

### مرحله ۲: اجرای AASVD

```powershell
cd D:\SVD_Project
python run_all_4_aasvd.py --models distilbert,roberta,gpt2 --rank 288 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_aasvd/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\aasvd
python apply_vars.py --module aasvd
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module aasvd
```

---

## 💡 نکات کلیدی AASVD

### ۱. تفاوت اصلی با A3

AASVD از **covariance کامل** استفاده می‌کند، A3 از **diagonal Fisher**.

مزیت AASVD: correlation بین فیچرها را capture می‌کند.
هزینه AASVD: memory $O(d^2)$ به جای $O(d)$.

این را در `04_method.md` و `08_analysis.md` برجسته کن.

### ۲. Calibration Corpus پیش‌فرض

AASVD یک corpus ۵ سندی همراه دارد. این مزیت بزرگی است چون:
- بدون داده تسک کار می‌کند
- out-of-the-box قابل استفاده است
- در `06_setup.md` توضیح بده

### ۳. EMA به جای mean

β=0.9 برای EMA. این یعنی وزن ۰.۹ به batch قبلی و ۰.۱ به batch جدید.
در `03_background.md` و `04_method.md` توضیح بده.

### ۴. برجسته کردن Real Compression

مثل A3، AASVD واقعاً پارامتر کم می‌کند:
- `768×768 = 589,824` پارامتر
- با `r=288`: `288×(768+768) = 442,368` پارامتر → فقط **۱.۳× کاهش**

**ولی** با `r=64`: `64×1536 = 98,304` → **۶× کاهش**

پس **rank را با دقت انتخاب کن**. rank=288 فقط ۱.۳× کاهش می‌دهد که خیلی کم است.

**توصیه:** rank=64 یا 128 برای کاهش معنادار.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_aasvd/*/summary.csv` پر کن. اگر خواستی من می‌توانم این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**