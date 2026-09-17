# 4. Method Formulation
1. **FactorisedLowRankProjection**: Enforces $r=128$ physical rank reduction across $W_Q, W_K, W_V, W_O$. <!-- source: Codes/framework/attentions/aegis_v6.py lines 60-70 -->
2. **Local Sliding Window**: Exact softmax attention bounded to $|i - j| \le W$.
3. **Global Linear Attention**: Associative kernel evaluation with numerical stability guards.
4. **Adaptive Gating**: Context fusion initialized to prioritize high-fidelity local attention.
