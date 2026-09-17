# 4. Method Formulation
1. **FactorisedLinear Projections**: Projections factorize base weights with auxiliary LoRA parameterization. <!-- source: Codes/framework/attentions/aegis_v1.py lines 65-75 -->
2. **Split-Head Partition**: $H_{	ext{local}} = H // 2$ heads evaluate sliding-window softmax, $H_{	ext{global}} = H - H_{	ext{local}}$ evaluate linear kernel attention.
3. **Adaptive Context Gating**: Dynamic gating weights combine local and global output tensors.
