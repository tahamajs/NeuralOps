# 4. Method Formulation
The compression algorithm identifies the minimal integer $r$ satisfying:
$$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau
<!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 100 -->

<!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 100 -->
$$subject to the rank floor constraint:
$$r = \max(r, r_{\min}), \quad r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{\text{in}}, d_{\text{out}}) \rfloor)
<!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 100 -->

<!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 100 -->
$$Factorization is performed with activation whitening $M = W \operatorname{Cov}(X)^{1/2}$ and symmetric square-root singular value distribution $U_r = U_{:, :r} \sqrt{\Sigma_r}$ and $V_r = (\sqrt{\Sigma_r} V_{:r, :}^\top) \operatorname{Cov}(X)^{-1/2}$.
