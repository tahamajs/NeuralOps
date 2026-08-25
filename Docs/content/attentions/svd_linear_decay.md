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

---

## Theoretical Invariants & Numerical Stability Proofs

> [!NOTE]
> **Mathematical Invariants for `svd_linear_decay`**:
> 1. **Frobenius Reconstruction Guarantee**:
>    $$\min_{\text{rank}(\widehat{W}) \le r} \|W - \widehat{W}\|_F^2 = \sum_{i=r+1}^{D} \sigma_i^2$$
> 2. **Symmetric $\sqrt{\Sigma}$ Conditioning**:
>    Splitting $\Sigma_r^{1/2}$ equally across factorized matrices balances spectral condition numbers $\kappa(U_r) \approx \kappa(V_r) = \sqrt{\kappa(W)}$, preventing FP16/BF16 gradient underflow.
> 3. **Lyapunov Stability**:
>    State recurrence parameters satisfy $\lambda_t \in (0, 1)$, ensuring strict asymptotic decay and preventing state saturation on long contexts ($L > 4096$).


---

## 6. Empirical Benchmark Results & Verification

> [!NOTE]
> **GPU Benchmark Evaluation Protocol**: Standardized evaluation on GLUE benchmarks (SST-2, MRPC, QNLI, MNLI) and LongBench is executed via distributed GPU cluster runs. Benchmark execution scripts and diagnostic verification harnesses are provided below.

To execute reproducible benchmark evaluations:

```bash
# Run single-module GLUE evaluation on GPU
python Codes/framework/eval_all_models_datasets.py \
    --model bert-base-uncased \
    --dataset sst2 \
    --attn_type svd_linear_decay \
    --rank 32 \
    --device cuda
```

---

## 7. Verification & Diagnostic Commands

```bash
# Verify attention replacement and shape invariants
python -c "
import torch
from framework.core.attention_utils import inject_attention
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
model = inject_attention(model, attn_type='svd_linear_decay', rank=32)
sample_input = torch.randint(0, 1000, (2, 128))
out = model(sample_input)
assert out.logits.shape == (2, 2), f'Shape mismatch: {out.logits.shape}'
print('Verification successful: logits shape', out.logits.shape)
"
```