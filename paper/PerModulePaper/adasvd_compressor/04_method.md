# 4. Method Formulation
1. **Spectral Importance**: Layer budget is proportional to $\|W\|_* = \sum_i \sigma_i$. <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 47 -->
2. **Symmetric Factorization**: $A = U_{:, :r} \sqrt{\Sigma_r}, \quad B = \sqrt{\Sigma_r} V_{:r, :}^	op$. <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 48 -->
3. **Alternating Least Squares**: Refine $A$ and $B$ to minimize $\|W - A B\|_F^2$; revert to SVD factors if ALS error increases. <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 54 -->
4. **SVDLinear Substitution**: Replace dense projections with factorized modules. <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 50 -->
