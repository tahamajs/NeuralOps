<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages (600 - 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) guarantees 
that the best rank-$r$ approximation of $W$ in Frobenius norm is the truncated 
SVD:

$$
W_r^* = U_{:, :r} \Sigma_{:, :r} V_{:r, :}^\top
\tag{2.1}
$$

This is optimal in an unweighted sense and forms the basis of most 
SVD-based compression methods.

**Randomized SVD.** Halko et al. (2011) provide a randomized algorithm for 
truncated SVD with complexity $\mathcal{O}(d_\text{in} d_\text{out} \log r)$, 
which we do not use because exact SVD is feasible for the layer sizes in our 
study.

**Weighted low-rank approximation.** Golub et al. (1987) and Srebro & Jaakkola 
(2003) study weighted variants of low-rank approximation. Our Fisher weighting 
is a diagonal special case with input-dimension-dependent weights.

## 2.2 Transformer Compression

### 2.2.1 Post-Training Pruning

Magnitude pruning (Han et al., 2015) removes individual weights. Structured 
pruning (Li et al., 2017) removes entire channels. SparseGPT (Frantar & 
Alistarh, 2023) and Wanda (Sun et al., 2024) prune large language models 
without retraining. A3 differs by factorizing rather than zeroing.

### 2.2.2 Knowledge Distillation

DistilBERT (Sanh et al., 2019), TinyBERT (Jiao et al., 2020), and MiniLM 
(Wang et al., 2020) train smaller models to mimic larger ones. A3 is 
orthogonal — it can compress an already-distilled model.

### 2.2.3 Quantization

8-bit (Dettmers et al., 2022) and 4-bit (Dettmers et al., 2023) quantization 
reduces bit-width. GPTQ (Frantar et al., 2023) uses second-order 
information. A3 is orthogonal to quantization — the two can be combined.

### 2.2.4 Low-Rank Factorization

ALBERT (Lan et al., 2020) uses cross-layer parameter sharing, a form of 
low-rank structure. Tensor train decomposition (Novikov et al., 2015) uses 
higher-order factorization. **A3 differs by using activation-aware weighting 
during factorization**, not after.

## 2.3 Activation-Aware Compression

**AWQ.** Lin et al. (2023) find that activation magnitude correlates with 
weight importance. They use it for quantization. Our Fisher weighting uses the 
same insight for low-rank factorization.

**Activation-aware pruning.** Molchanov et al. (2019) use the Taylor expansion 
of the loss with respect to activations. Our approach is simpler: we use 
second moments.

**Fisher information.** The Fisher diagonal is the classical measure of 
parameter importance (Pascanu & Bengio, 2014; Kirkpatrick et al., 2017). We 
use a cheap empirical proxy $\mathbb{E}[x^2]$ instead of the full Fisher.

## 2.4 LayerNorm Recalibration

**Post-training recalibration.** Nagel et al. (2019) recalibrate activation 
distributions for quantization. We adapt this idea to post-factorization 
distributional shift.

**Gamma rescaling.** Gholami et al. (2021) rescale LayerNorm scales after 
quantization. Our $W^* = U \Sigma V^T$ is the analogous operation 
for low-rank factorization.

## 2.5 Position of Our Work

| Axis | Prior work | A3 |
|---|---|---|
| **Factorization weighting** | Uniform (unweighted SVD) | Fisher diagonal |
| **Rank selection** | Fixed | Adaptive escalation |
| **Calibration memory** | Full activation buffer | Streaming O($d^2$) |
| **Post-factorization fix** | None | LayerNorm recalibration |
| **Error validation** | Silent | Strict Frobenius bound |
| **Fallback** | Never | Automatic per-layer |

**Key distinction.** Prior low-rank compression methods either (a) select 
rank globally before factorizing, or (b) factorize unweighted. A3 combines 
both improvements: it **selects rank per-layer based on validation**, and it 
**weights the factorization by activation statistics**. The result is 
strictly smaller reconstruction error at the same rank, or equivalently, 
smaller rank at the same error.

---

## Comparison Table

| Method | Type | Retrain? | Calib memory | Weighted? | Error bound? |
|---|---|---|---|---|---|
| SVD (Eckart-Young) | Low-rank | No | — | No | No |
| AWQ | Quant | No | O(d) | Yes | No |
| SparseGPT | Prune | No | O(d) | Yes | No |
| LoRA | Adapter | Yes | — | No | No |
| **A3** | Low-rank + adapters | LoRA only | O($d^2$) | **Yes** | **Yes** |
