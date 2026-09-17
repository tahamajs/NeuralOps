# 4. Method Formulation
1. **Factorised Linear Projections**: Projections execute as frozen base layers plus trainable LoRA adapters.
2. **Local Sliding Window**: Window size $W=128$.
3. **Global Linear Attention**: Associative kernel attention with positive feature maps.
4. **Token Gating Network**: 2-layer projection predicts dynamic mixing coefficients.
