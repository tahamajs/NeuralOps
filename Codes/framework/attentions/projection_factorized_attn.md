# Projection Factorized Attention (`projection_factorized_attn`)

## Architectural Overview
`ProjectionFactorizedAttention` is an alias of [`SVDLinearAttention`](svd_linear_attention.md) implementing truncated Singular Value Decomposition low-rank projection factorizations across Query, Key, Value, and Output linear layers.

## Key Properties
- **Registry Key:** `"projection_factorized_attn"`
- **Mathematical Form:** $W \approx U V$, where $U \in \mathbb{R}^{D_{\text{out}} \times r}, V \in \mathbb{R}^{r \times D_{\text{in}}}$
- **Complexity:** Projections: $\mathcal{O}(B \cdot N \cdot r \cdot D)$, Attention: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$
- **Pretrained Transfer:** Symmetrically decomposed singular values $U \sqrt{\Sigma}$ and $\sqrt{\Sigma} V^\top$ via `UniversalAdapter`.
- **Empirical SST-2 Accuracy:** 89.22% (Top Performer).

See [`svd_linear_attention.md`](svd_linear_attention.md) for full mathematical derivations, tensor trace, and benchmark analyses.
