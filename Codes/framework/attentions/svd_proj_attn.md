# SVD Projection Attention (`svd_proj_attn`)

## Architectural Overview
`SVDProjectionAttention` is an alias of [`SVDProjectionAttention`](svd_projection_attention.md) providing factorized low-rank projections ($X V^\top U^\top + b$) with dynamic singular subspace reconstruction.

## Key Properties
- **Registry Key:** `"svd_proj_attn"`, `"svd_projection_attention"`
- **Mathematical Form:** $y = ((x V^\top) \odot \Sigma) U^\top + b$
- **Complexity:** $\mathcal{O}(B \cdot N \cdot r \cdot D)$ FLOPs per projection
- **Pretrained Transfer:** Symmetrically decomposed via `UniversalAdapter`.

See [`svd_projection_attention.md`](svd_projection_attention.md) for full mathematical derivations, tensor trace, and benchmark analyses.

## Implementation status

The implementation for this page is [svd_proj_attn.py](./svd_proj_attn.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
