# SVD-LLM V2 Attention (`svd_llm_v2`)

## Architectural Overview
`SVDLLMV2Attention` is an alias of [`SVDLLMV2Attention`](svd_llm_v2_attention.md) implementing truncation-aware SVD parameterization with explicit $U, \Sigma, V^H$ matrix buffers.

## Key Properties
- **Registry Key:** `"svd_llm_v2"`, `"svd_llm_v2_attention"`
- **Mathematical Form:** $Q = ((X V_q^H) \odot S_q) U_q^\top + b_q$
- **Complexity:** $\mathcal{O}(B \cdot N \cdot r \cdot D)$ factorized projections, $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$ attention.
- **Pretrained Transfer:** Truncated SVD initialized via `UniversalAdapter` preserving singular spectrum energy.

See [`svd_llm_v2_attention.md`](svd_llm_v2_attention.md) for full mathematical derivations, tensor trace, and benchmark analyses.

---

## Complete Source Code Implementation (`svd_llm_v2.py`)

```python
r"""
SVD-LLM V2 Attention Alias.
Maps to `framework.attentions.svd_llm_v2_attention.SVDLLMV2Attention`.
"""

from framework.attentions.svd_llm_v2_attention import (
    SVDLLMV2Attention,
    inject_svd_llm_v2,
)

__all__ = ["SVDLLMV2Attention", "inject_svd_llm_v2"]
```
