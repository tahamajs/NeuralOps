# 4. Method Formulation
1. **Activation Covariance Whitening**:
   $$M = W \cdot \operatorname{Cov}(X)^{1/2}$$ <!-- source: Codes/framework/compression_engine/adaptive_ranks.py line 9 -->
2. **Dynamic Energy Thresholding**:
   Solve for minimal $r$ satisfying $rac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge 	au$ with floor $r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{	ext{in}}, d_{	ext{out}}) floor)$. <!-- source: Codes/framework/compression_engine/adaptive_ranks.py lines 5, 7 -->
3. **Symmetric Singular Value Distribution**:
   $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V_{:r, :}^	op) \cdot \operatorname{Cov}(X)^{-1/2}$$ <!-- source: Codes/framework/compression_engine/adaptive_ranks.py line 11 -->
4. **Reconstruction Verification**: Verify $\|W - U_r V_r\|_F / \|W\|_F \le \delta_{\max}$ (default 0.20). <!-- source: Codes/framework/compression_engine/adaptive_ranks.py line 81 -->
