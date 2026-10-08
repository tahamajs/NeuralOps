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
