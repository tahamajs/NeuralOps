# SVD-LLM V2 Attention (`svd_llm_v2`)

## Architectural Overview
`SVDLLMV2Attention` is an alias of [`SVDLLMV2Attention`](svd_llm_v2_attention.md) implementing truncation-aware SVD parameterization with explicit $U, \Sigma, V^H$ matrix buffers.

## Key Properties
- **Registry Key:** `"svd_llm_v2"`, `"svd_llm_v2_attention"`
- **Mathematical Form:** $Q = ((X V_q^H) \odot S_q) U_q^\top + b_q$
- **Complexity:** $\mathcal{O}(B \cdot N \cdot r \cdot D)$ factorized projections, $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$ attention.
- **Pretrained Transfer:** Truncated SVD initialized via `UniversalAdapter` preserving singular spectrum energy.

See [`svd_llm_v2_attention.md`](svd_llm_v2_attention.md) for full mathematical derivations, tensor trace, and benchmark analyses.

## Implementation status

The implementation for this page is [svd_llm_v2.py](./svd_llm_v2.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
