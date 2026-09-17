# 4. Method Formulation
1. **FactorizedLinear Projections**:
   $$y = 	ext{lin}_2(	ext{lin}_1(x)) + rac{lpha}{r_{	ext{lora}}} (x A_{	ext{lora}}^	op) B_{	ext{lora}}^	op$$ <!-- source: Codes/framework/attentions/aegis_hybrid.py lines 53-55 -->
2. **Local Sliding Window Branch**: Dense softmax attention masked to $|i - j| \le W$. <!-- source: Codes/framework/attentions/aegis_hybrid.py line 35 -->
3. **Global Low-Rank Branch**: Factorized linear attention.
4. **Learned Gating Fusion**: Output is dynamically weighted via a sigmoid gate initialized with bias 1.8. <!-- source: Codes/framework/attentions/aegis_hybrid.py line 41 -->
