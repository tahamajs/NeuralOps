# 4. Method Formulation
1. **$W$-Update**: $W^{(t+1)} = rg\min_W f(W) + rac{ho_1}{2} \|W - Z^{(t)} + U_1^{(t)}\|_F^2 + rac{ho_2}{2} \|W - Q^{(t)} + U_2^{(t)}\|_F^2$
2. **$Z$-Update**: Projection onto 2:4 structured or block sparsity patterns. <!-- source: Codes/framework/compression_engine/admm_compressor.py line 21 -->
3. **$Q$-Update**: Nearest neighbor quantization.
4. **Dual Updates**: $U_1^{(t+1)} = U_1^{(t)} + W^{(t+1)} - Z^{(t+1)}, \quad U_2^{(t+1)} = U_2^{(t)} + W^{(t+1)} - Q^{(t+1)}$.
5. **Monotonic Tracker**: Preserves the iterate with lowest reconstruction error. <!-- source: Codes/framework/compression_engine/admm_compressor.py line 58 -->
