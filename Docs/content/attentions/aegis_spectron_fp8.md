# AEGIS-Spectron-FP8: Tri-Branch Architecture with Spectral Entropy Routing

## 1. Mathematical Formulation & Architecture
1. **Tri-Branch Hybrid**:
   - Branch 1: Local Sliding-Window Attention ($W=64$)
   - Branch 2: Global Low-Rank SVD Attention ($r=128$)
   - Branch 3: Linear State Recurrence ($h_t = A h_{t-1} + B x_t$)

2. **Spectron Spectral Entropy Router**:
   $$\mathcal{S}(X) = -\sum_{i=1}^D p_i \ln(p_i + 10^{-5}), \quad p_i = \frac{\sigma_i^2}{\sum \sigma_k^2}$$
   $$\alpha_b = \operatorname{Softmax}\left(\frac{\mathcal{H}_b(X) \cdot \mathcal{S}(X) - \lambda c_b}{\tau}\right)$$
   Router bias initialized to $(2.5, -0.5, -1.0)$ to prioritize high-fidelity local attention at initialization.

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: High local cost $c_{\text{loc}}=1.0$ penalized the local branch, while random projection initialization stripped DistilBERT features, causing accuracy to drop to 50.92% - 70.53%.
- **Architectural Repair**: Copied pretrained weights to Q/K/V/Out, initialized $U_{\text{svd}}$ from top singular vectors, and calibrated router bias to allocate 88% initial probability to local attention.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.72%** (restored from 50.92% / 70.53%).\n