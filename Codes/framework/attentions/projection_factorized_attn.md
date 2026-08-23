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

## Implementation status

The implementation for this page is [projection_factorized_attn.py](./projection_factorized_attn.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
