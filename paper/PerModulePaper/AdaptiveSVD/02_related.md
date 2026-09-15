<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) states 
that the best rank-$r$ approximation of $W$ in Frobenius norm is the truncated 
SVD $W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top$. Adaptive SVD builds 
on this but selects $r$ via an energy threshold rather than a validation 
loop or fixed value.

**Rank selection heuristics.** Classical rank selection includes AIC 
(Akaike, 1974), BIC (Schwarz, 1978), and cross-validation. For SVD-based 
compression specifically, prior work uses reconstruction error thresholds 
(A3), fixed budget ratios (SDRM), or optimal hard thresholding on 
singular values under noise (Gavish & Donoho, 2014). Our criterion is a 
**cumulative energy threshold**, distinct from per-singular-value thresholds.

**Randomized SVD.** Halko et al. (2011) provide a randomized algorithm with 
$\mathcal{O}(d r \log k)$ complexity. For our layer sizes (≤3072×768), exact 
SVD is faster and simpler.

## 2.2 Activation-Aware Compression

**AWQ.** Lin et al. (2023) observe that activation magnitude correlates with 
weight importance and use it to scale quantization. Applied to low-rank 
factorization, this becomes the A3 method (Fisher diagonal weighting).

**SmoothQuant.** Xiao et al. (2023) migrate activation scale to weights for 
quantization. Analogous to AASVD's covariance weighting for low-rank.

**A3.** Uses Fisher diagonal $F_i = \mathbb{E}[x_i^2]$ to weight the SVD input 
space. Requires calibration. Adaptive SVD is calibration-free.

**AASVD.** Uses full covariance $C_X = \mathbb{E}[x x^\top]$. Requires 
streaming EMA accumulation. Adaptive SVD is calibration-free.

**Key distinction.** Adaptive SVD does **not** use activation statistics. Its 
rank criterion is intrinsic to the weight matrix.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) add trainable rank-$r_L$ updates $B A$ with $B = 0$. 
Adaptive SVD uses the same residual structure, integrated with a frozen SVD base.

**AdaLoRA.** Zhang et al. (2023) adaptively allocate LoRA rank per layer based 
on singular value importance. Our adaptation is on the **base factorization**, 
not the adapter.

**PiSSA.** Meng et al. (2024) initialize LoRA adapters with principal singular 
components. Adaptive SVD uses the top-$r$ components as the **frozen base** 
and zero-initialized LoRA as the residual. These are complementary strategies.

**OFT.** Qiu et al. (2023) use orthogonal fine-tuning. Different primitive.

## 2.4 Attention-Specific Compression

**Head pruning.** Voita et al. (2019) prune entire heads. Adaptive SVD preserves 
head count and compresses per-head projections via low-rank factorization.

**GQA.** Ainslie et al. (2023) reduce KV head count during pretraining. Adaptive 
SVD preserves head count but compresses per-head projection matrices.

**KV cache compression.** H2O (Zhang et al., 2023), StreamingLLM (Xiao et al., 
2024) reduce cache size at inference. Different target.

## 2.5 Position of Our Work

| Method | Calib | Weighted | Rank | Per-layer rank | T5 bias |
|---|---|---|---|---|---|
| SVD fixed rank | No | No | Fixed | No | No |
| SDRM | No | No | Budget | No | No |
| A3 | Yes | Fisher diag | Escalation | Partial | No |
| AASVD | Yes | Full cov | Fixed | No | No |
| **Adaptive SVD** | **No** | **No** | **Energy** | **Yes** | **Yes** |

**Key distinction.** Adaptive SVD is the only method that:
1. Requires no calibration data.
2. Selects rank per-projection via an energy criterion.
3. Preserves T5 relative attention bias.

## 2.6 Comparison Table

| Feature | Value |
|---|---|
| Calibration data required | No |
| Forward passes during compression | No |
| Per-layer adaptive rank | Yes |
| Model-agnostic criterion | Yes |
| T5 bias support | Yes |
| GQA support | Yes |
| RoPE support | Yes |
