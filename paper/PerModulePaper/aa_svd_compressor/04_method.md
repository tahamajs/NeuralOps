# 4. Method Formulation

## 4.1 Streaming Covariance Accumulation
During calibration over batches $t = 1, \dots, T$, `aa_svd_compressor` hooks target linear layers and maintains a running Exponential Moving Average (EMA) of activation covariance:
$$C_X^{(t)} = \beta C_X^{(t-1)} + (1 - \beta) \frac{1}{B} \sum_{i=1}^B X_i^\top X_i$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 5 -->
where default decay factor $\beta = 0.9$ (configurable via `ema_beta`).

## 4.2 Symmetric Whitening and Factorization
To condition the weight matrix without numerical instability, the symmetric covariance $C_X$ is regularized and decomposed via eigendecomposition in FP32 precision:
$$C_{\text{sym}} = \frac{1}{2}(C_X + C_X^\top) = Q \Lambda Q^\top$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 390-391 -->
$$\Lambda_{\text{reg}} = \operatorname{clamp}(\Lambda, \epsilon, \infty), \quad \epsilon = 10^{-6} \cdot \max(\Lambda) + 10^{-8}$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py lines 392-393 -->
$$C_X^{1/2} = Q \Lambda_{\text{reg}}^{1/2} Q^\top, \quad C_X^{-1/2} = Q \Lambda_{\text{reg}}^{-1/2} Q^\top$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py lines 394-395 -->

The preconditioned weight is formed as:
$$\tilde{W} = W C_X^{1/2}$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 401 -->
Performing standard SVD on $\tilde{W}$:
$$\tilde{W} = \tilde{U} \tilde{\Sigma} \tilde{V}^\top$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 406 -->
The low-rank factors are constructed with symmetric singular value distribution:
$$U_r = \tilde{U}_{:, :r} \tilde{\Sigma}_r^{1/2}, \quad V_r = (\tilde{\Sigma}_r^{1/2} \tilde{V}_{:r, :}^\top) C_X^{-1/2}$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py lines 408-414 -->

## 4.3 Fallback and Relative Error Guard
Before layer replacement, relative Frobenius reconstruction error is evaluated:
$$\text{err} = \frac{\|W - U_r V_r\|_F}{\max(\|W\|_F, 10^{-12})}$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 417 -->
If $\text{err} > \text{max\_error}$ (default 0.50) or if the parameter constraint $r(d_{\text{in}} + d_{\text{out}}) \ge d_{\text{in}} d_{\text{out}}$ is violated, the layer is safely retained in its original uncompressed form.
