# 4. Method Formulation
1. **$Z$-Update**: $Z^{(t+1)} = \Pi_{\mathcal{S}}(W^{(t)} + U^{(t)})$ <!-- source: Codes/framework/compression_engine/admm_pruner.py line 10 -->
2. **Intermediate $W$**: $W^{(t+1/2)} = rac{1}{1 + ho^{(t)}} \left( W_0 + ho^{(t)} (Z^{(t+1)} - U^{(t)}) ight)$ <!-- source: Codes/framework/compression_engine/admm_pruner.py line 11 -->
3. **Distillation Step**: $W^{(t+1)} = W^{(t+1/2)} - \eta \lambda_{	ext{distill}} 
abla_W (	ext{MSE}_{	ext{distill}})$ <!-- source: Codes/framework/compression_engine/admm_pruner.py line 12 -->
4. **Dual Update**: $U^{(t+1)} = U^{(t)} + W^{(t+1)} - Z^{(t+1)}$ <!-- source: Codes/framework/compression_engine/admm_pruner.py line 13 -->
5. **Dynamic $ho$**: $ho^{(t+1)}$ scales up if $\|r^{(t+1)}\| > \mu \|s^{(t+1)}\|$ and scales down if $\|s^{(t+1)}\| > \mu \|r^{(t+1)}\|$. <!-- source: Codes/framework/compression_engine/admm_pruner.py line 15 -->
