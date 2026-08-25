# SVD Linear Decay Gated Attention (`svd_linear_decay`)

## Architectural Overview
`DecayGatedLinearAttention` is an alias and recurrent mode of [`SVDLinearAttention`](svd_linear_attention.md) featuring non-negative kernel linear attention equipped with data-dependent exponential decay recurrence.

## Key Properties
- **Registry Key:** `"svd_linear_decay"`
- **Mathematical Form:** $S_t = \exp(-\text{softplus}(\text{gate}(x_t))) S_{t-1} + \phi(K_t)^\top V_t$
- **Complexity:** $\mathcal{O}(B \cdot N \cdot D \cdot d_k)$ time, $\mathcal{O}(B \cdot D \cdot d_k)$ state space.
- **Pretrained Transfer:** Symmetrically initialized zero-shot from pre-trained multi-head attention weights.

See [`svd_linear_attention.md`](svd_linear_attention.md) for full mathematical derivations, tensor trace, and benchmark analyses.

---

## Complete Source Code Implementation (`svd_linear_decay.py`)

```python
r"""
SVD Linear Decay Gated Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.
"""

from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
)

DecayGatedLinearAttention = SVDLinearAttention

__all__ = ["DecayGatedLinearAttention", "SVDLinearAttention", "SVDLinear"]
```
