# 4. Method

## 4.1 Overview

A3 replaces each dense layer with a factorized form that is weighted by activation statistics. The method has three stages: calibration, factorization, and recovery. Calibration collects activation statistics using a small sample of task data. Factorization applies a weighted SVD so that important directions survive. Recovery then adjusts the remaining trainable parameters with a small LoRA stage.

## 4.2 Activation-Aware Weighting

A3 estimates an importance vector or covariance matrix from input activations. In the diagonal case, this is equivalent to weighting the weight matrix by the empirical Fisher-like energy terms. In the more general covariance form, the weighted transform is

$$
W_{\mathrm{weighted}} = W C_X^{1/2}
\tag{4.1}
$$

where $C_X = \mathbb{E}[x x^\top]$ is the activation covariance. This step rebalances the matrix so the factorization spends capacity on statistically important directions.

## 4.3 Streaming Covariance Estimation

The covariance is accumulated online over batches using a streaming estimate, which avoids storing the full calibration set in memory. This means the method scales comfortably to larger models while maintaining a low constant-time memory overhead.

## 4.4 Factorization and Reconstruction

The weighted matrix is factorized via SVD, and the compressed representation is kept at rank $r$ if the approximation error is acceptable. We then reconstruct the layer as

$$
\tilde{W} = U_r \Sigma_r V_r^\top
\tag{4.2}
$$

with the right factor deweighted when necessary to return to the original parameter space. The implementation includes a strict reconstruction check and a fallback to the original layer if the approximation is too unstable.

## 4.5 LayerNorm Recalibration

Because compression can change the activation statistics, A3 recalibrates LayerNorm parameters after factorization. The goal is to restore the original activation scale and reduce the distribution shift introduced by the approximation.

## 4.6 Complexity

The dominant calibration cost is proportional to the activation dimension and grows roughly quadratically in the hidden size, which is manageable for the project's transformer backbones. The actual compressed layer then reduces parameter count from $d_{\text{in}}d_{\text{out}}$ to $r(d_{\text{in}} + d_{\text{out}})$.
