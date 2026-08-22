# SOA Attention: Shared Orthogonal Attention with Latent Bottlenecks

## 1. Mathematical Formulation & Architecture
Factorizes Q, K, and V projections into low-rank representations initialized via truncated Singular Value Decomposition (SVD):
$$W \approx U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^T, \quad r=64$$
Multi-head attention computation:
$$A = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M_{\text{pad}}\right), \quad O = (A V) W_O$$

## 2. Repair Upgrades
- Integrated `from_distilbert_pretrained`.
- Increased default bottleneck rank to 64.
- Safe FP16 masking using $-10000.0$.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.60%**\n