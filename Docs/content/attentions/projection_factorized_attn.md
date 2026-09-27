# Projection Factorized SVD Linear Attention

## Overview

**Projection Factorized SVD Linear Attention** decomposes the heavy projection matrices of standard multi-head self-attention ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$) into truncated Singular Value Decomposition (SVD) factors. By factorizing each weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ into rank-$r$ components $\mathbf{U} \in \mathbb{R}^{d_{\text{in}} \times r}$ and $\mathbf{V} \in \mathbb{R}^{r \times d_{\text{out}}}$, the mechanism reduces parameter count and FLOP requirements while preserving attention capacity via linear kernel evaluation.

### Key Characteristics
- **Low-Rank Factorization:** Each projection layer is factorized as $\mathbf{W} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$, drastically shrinking memory footprint during inference and training.
- **Linear Attention Compatibility:** Seamlessly interoperates with kernelized feature maps $\phi(\cdot)$ to achieve $O(N)$ computational complexity with respect to sequence length $N$.
- **Seamless Layer Injection:** Drop-in replacement for standard self-attention modules across DistilBERT, BERT, RoBERTa, and GPT architectures with automatic weight initialization from pretrained singular vectors.

---

## Mathematical Formulation

For an attention projection matrix $\mathbf{W} \in \mathbb{R}^{d \times d}$:

1. **Singular Value Truncation:**
   $$\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \approx \mathbf{A} \mathbf{B}$$
   where $\mathbf{A} = \mathbf{U}_r \mathbf{\Sigma}_r^{1/2} \in \mathbb{R}^{d \times r}$ and $\mathbf{B} = \mathbf{\Sigma}_r^{1/2} \mathbf{V}_r^T \in \mathbb{R}^{r \times d}$, with $r \ll d$.

2. **Factorized Linear Attention:**
   $$\mathbf{Q} = (\mathbf{X} \mathbf{A}_Q) \mathbf{B}_Q, \quad \mathbf{K} = (\mathbf{X} \mathbf{A}_K) \mathbf{B}_K, \quad \mathbf{V} = (\mathbf{X} \mathbf{A}_V) \mathbf{B}_V$$
   $$\mathbf{Y} = \frac{\phi(\mathbf{Q}) \left(\phi(\mathbf{K})^T \mathbf{V}\right)}{\phi(\mathbf{Q}) \sum_{j} \phi(\mathbf{K})_j^T + \epsilon}$$

3. **Output Projection:**
   $$\mathbf{O} = (\mathbf{Y} \mathbf{A}_O) \mathbf{B}_O$$

---

## Configuration & Hyperparameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rank` | `int` | `64` | Truncation rank for the low-rank projection factors |
| `feature_map` | `str` | `"elu"` | Kernel feature map applied to queries and keys (`"elu"`, `"relu"`, `"identity"`) |
| `eps` | `float` | `1e-6` | Numerical stabilization term in the denominator |
| `bias` | `bool` | `False` | Whether low-rank projection stages include learnable bias vectors |

---

## Usage Examples

```python
import torch
from framework.attentions.projection_factorized_attn import (
    ProjectionFactorizedAttention,
    inject_projection_factorized_attn,
)

# Standalone instantiation
attn = ProjectionFactorizedAttention(
    d_model=768,
    num_heads=12,
    rank=64
)

x = torch.randn(2, 64, 768)
out = attn(x)

# Model injection
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
injected_model = inject_projection_factorized_attn(model=model, rank=64)
```

---

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 90.60% | **84.75%** | -5.85% | 0.8504 | 1.20x | 6 | [svd-linear-attention-distilbert-sst2-projection-factorized-attn](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-projection-factorized-attn) | 2026-09-27 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **53.90%** | -27.90% | 0.5017 | 1.20x | 6 | N/A | 2026-09-19 |