# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

### 2.1.1 Classical SVD

Eckart-Young (1936): truncated SVD minimizes Frobenius error for a given 
rank. MinimalAURORA uses SVD with **energy-based rank selection** rather 
than fixed rank.

### 2.1.2 Randomized SVD

Halko et al. (2011): faster SVD via random projections. Not needed for 
our layer sizes.

### 2.1.3 Weighted Low-Rank

Golub et al. (1987), Srebro & Jaakkola (2003). MinimalAURORA uses 
unweighted SVD (unlike A3/AASVD).

## 2.2 Transformer Compression

### 2.2.1 ASVD / AASVD / A3

Companion modules:
- **ASVD**: activation-weighted SVD, fixed rank.
- **AASVD**: full covariance, streaming calibration.
- **A3**: Fisher diagonal, adaptive rank retry.

**MinimalAURORA differs:**
- No activation weighting (simpler).
- Energy threshold instead of error retry.
- Uses `nn.Linear` for factorization (not explicit U/S/V).

### 2.2.2 SDRM

SDRM (companion): budget-proportional rank allocation. Uses parameter 
ratio instead of energy threshold.

### 2.2.3 Res-SVD

Res-SVD (companion): two-level SVD with spectral LoRA initialization. 
MinimalAURORA uses zero-init LoRA.

### 2.2.4 SVD-Linear

SVD-Linear (companion): SVD + kernelized linear attention. Combines 
compression with complexity reduction. MinimalAURORA keeps softmax.

## 2.3 Parameter-Efficient Fine-Tuning

### 2.3.1 LoRA

Hu et al. (2021): $B = 0$ init. MinimalAURORA uses standard LoRA.

### 2.3.2 PiSSA

Meng et al. (2024): LoRA init from principal directions. Complementary 
to Res-SVD approach.

## 2.4 Energy-Based Rank Selection

### 2.4.1 Energy Threshold

Common in PCA (Jolliffe, 2002): keep components explaining ≥95% of 
variance. MinimalAURORA applies this to SVD of weights.

### 2.4.2 Applications

- **Image compression:** JPEG uses similar energy thresholding.
- **Recommender systems:** truncated SVD for latent factors.
- **NLP:** vocab projection.

## 2.5 Position of Our Work

| Aspect | ASVD | AASVD | A3 | Res-SVD | MinimalAURORA |
|---|---|---|---|---|---|
| Weighting | Activation | Full cov | Fisher diag | Unweighted | **Unweighted** |
| Rank selection | Fixed | Fixed cap | Adaptive retry | Fixed | **Energy** |
| LoRA init | Zero | Zero | Zero | Spectral | **Zero** |
| Complexity | Simple | Complex | Medium | Medium | **Simple** |
| Multi-arch | Yes | Yes | Yes | Yes | **Yes** |

## 2.6 Comparison to Companion Modules

MinimalAURORA is a **baseline-quality** compression method with the 
simplest possible machinery. It trades algorithmic sophistication for 
implementation clarity and reliability.

---