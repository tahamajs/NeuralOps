# SVD Projection Attention (`svd_proj_attn`)

## Architectural Overview
`SVDProjectionAttention` is an alias of [`SVDProjectionAttention`](svd_projection_attention.md) providing factorized low-rank projections ($X V^\top U^\top + b$) with dynamic singular subspace reconstruction.

## Key Properties
- **Registry Key:** `"svd_proj_attn"`, `"svd_projection_attention"`
- **Mathematical Form:** $y = ((x V^\top) \odot \Sigma) U^\top + b$
- **Complexity:** $\mathcal{O}(B \cdot N \cdot r \cdot D)$ FLOPs per projection
- **Pretrained Transfer:** Symmetrically decomposed via `UniversalAdapter`.

See [`svd_projection_attention.md`](svd_projection_attention.md) for full mathematical derivations, tensor trace, and benchmark analyses.
