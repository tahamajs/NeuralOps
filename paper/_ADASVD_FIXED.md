# AdaSVD Activation-Aware 3-Step Fix Report

## 1. Problem Diagnosis
Prior to this fix, AdaSVD collapsed on DistilBERT SST-2 (50.92% accuracy, random-guess performance) because:
1. It computed standard parameter-space truncated SVD:
   $$W \approx U_r \Sigma_r V_r^T$$
   without taking activation distribution into account. Transformer activations feature extreme outlier dimensions with variances $100\times$ larger than normal channels; standard SVD treats all dimensions equally, introducing immense distortion along outlier channels.
2. It measured Frobenius parameter error $\|W - A B\|_F / \|W\|_F$ which exceeded the fallback threshold, causing the layers to either fall back or destroy representation fidelity.
3. It allocated rank proportional to nuclear norm rather than information-theoretic spectral entropy.

---

## 2. The 3-Step Mathematical Fix

### Step 1: Cholesky Whitening
We collect empirical input activations $X \in \mathbb{R}^{N \times d_{in}}$ using the standard multi-domain calibration corpus (shared with AASVD).
The input activation covariance is:
$$C_X = \frac{1}{N} X^T X \in \mathbb{R}^{d_{in} \times d_{in}}$$
We regularize and compute the Cholesky factor:
$$\epsilon = 10^{-6} \cdot \frac{\text{Tr}(C_X)}{d_{in}}, \quad S_{white} = \text{cholesky}(C_X + \epsilon I)$$
The weight matrix is whitened prior to SVD:
$$W_{white} = W S_{white}$$
Truncated SVD is computed on the whitened matrix:
$$U, \Sigma, V^T = \text{svd}(W_{white}, \text{full\_matrices}=\text{False})$$
The right singular factor is unwhitened by post-multiplying by $S_{white}^{-1}$:
$$V^T_{unw} = V^T S_{white}^{-1}$$
Low-rank factors are constructed symmetrically:
$$A = U[:, :r] \cdot \sqrt{\Sigma[:r]}, \quad B = \sqrt{\Sigma[:r]} \cdot V^T_{unw}[:r, :]$$

### Step 2: Shannon Entropy Effective Rank Allocation
Instead of nuclear norm $\sum_i \sigma_i$, layer rank importance is determined by the Roy & Vetterli (2007) effective rank:
$$p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2 + 10^{-12}}, \quad \text{erank}(W) = \exp\left( - \sum_i p_i \ln(p_i + 10^{-12}) \right)$$
This concentrates rank capacity where spectral information is dense, independent of weight scale.

### Step 3: Downstream Activation Error Validation
Reconstruction validation is performed against the real activation effect rather than parameter Frobenius norm:
$$\text{err}_{act} = \frac{\| W X^T - A B X^T \|_F}{\| W X^T \|_F + 10^{-9}}$$
Layers are only discarded to fallback if their actual downstream output is degraded.

---

## 3. Results on DistilBERT / SST-2 (Rank 128)

Source CSV: `results/distilbert_sst2_sst2_sweep/summary_adasvd_fixed.csv`  
Source JSON: `results/distilbert_sst2_sst2_sweep/adasvd_stats.json`

| Stage | Accuracy | Compression | Fallback | Status | Avg Reconstruction Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Before Fix** | 50.92% | 1.00x | Yes (All layers fell back) | FAILED | > 0.60 |
| **After 3-Step Fix** | **88.76%** | **1.1612x** (13.88% reduction) | **0 / 24 (No fallback)** | **PASS** | **0.1704** |

### Additional Benchmark Metrics (from CSV):
- **Validation Loss**: 0.3365
- **F1 Score**: 0.8937 (Macro F1: 0.8872)
- **Precision / Recall**: 0.8904 / 0.8869
- **FLOPs Reduction**: 65.66%
- **Original Parameters**: 66,955,010
- **Compressed Parameters**: 57,660,674
- **Parameters Saved**: 9,294,336
- **Inference Speed**: 145,909.58 tokens/sec (0.88 ms/token latency)
- **Evaluation Elapsed Time**: 21.43 seconds
