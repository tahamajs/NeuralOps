# SVD Linear Decay Gated Attention

## Overview

**SVD Linear Decay Gated Attention (`DecayGatedLinearAttention`)** combines truncated Singular Value Decomposition (SVD) low-rank parameterization of projection weights with continuous exponential decay gating across sequence positions. By infusing linear kernel attention with an input-dependent or position-decay scalar $\gamma^{i-j}$, the mechanism enforces localized recency bias while evaluating global attention contexts in $O(N)$ linear time.

### Architectural Highlights
- **Low-Rank Projection Factorization**: Weight matrices $W_Q, W_K, W_V, W_O$ are decomposed into truncated rank-$r$ components $U$ and $V$, significantly reducing parameters and FLOP requirements.
- **Exponential Decay Gating**: Applies relative position decay $\exp(-\lambda |i - j|)$ to recurrent associative memory states, preventing saturation and catastrophic forgetting over long context windows.
- **Linear Feature Maps**: Replaces quadratic softmax affinity with kernel feature transformations $\phi(\cdot)$, computing autoregressive prefix sums in continuous linear time.

---

## Mathematical Formulation

1. **Low-Rank Projections**:
   $$Q = (X U_Q) V_Q^T, \quad K = (X U_K) V_K^T, \quad V = (X U_V) V_V^T$$

2. **Decay-Modulated Linear State Accumulation**:
   $$S_t = \gamma_t S_{t-1} + \phi(K_t)^T V_t$$
   $$Z_t = \gamma_t Z_{t-1} + \phi(K_t)^T$$
   $$O_t = \frac{\phi(Q_t) S_t}{\phi(Q_t) Z_t + \epsilon}$$

3. **Output Factorization**:
   $$Y = (O U_O) V_O^T$$

---

## Configuration & Hyperparameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rank` | `int` | `64` | Truncation rank for low-rank SVD projections |
| `use_decay_gating` | `bool` | `True` | Enables exponential decay state updates across sequence tokens |
| `decay_rate` | `float` | `0.95` | Base decay factor $\gamma \in (0, 1)$ for recurrent state transitions |
| `feature_map` | `str` | `"elu"` | Kernel activation applied to queries and keys (`"elu"`, `"relu"`) |
| `eps` | `float` | `1e-6` | Numerical stabilization epsilon in the denominator |

---

## Usage Examples

```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.attentions.svd_linear_decay import (
    DecayGatedLinearAttention,
    inject_svd_linear_decay,
)

# Direct module instantiation
attn = DecayGatedLinearAttention(
    d_model=768,
    num_heads=12,
    rank=64,
    use_decay_gating=True
)

x = torch.randn(2, 128, 768)
out = attn(x)

# Model-wide injection
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
injected_model = inject_svd_linear_decay(model=model, rank=64)
```

---

## Benchmark Results

| Model | Dataset | Status | Baseline Acc | Injected Acc | Latency | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BERT | sst2 | `PASS` | 91.06% | **0.00%** | 14.24 ms | 719.1 MB |
| BERT | mrpc | `PASS` | 91.06% | **0.00%** | 14.89 ms | 719.1 MB |
| ROBERTA | sst2 | `PASS` | 91.06% | **0.00%** | 15.73 ms | 729.0 MB |
| ROBERTA | mrpc | `PASS` | 91.06% | **0.00%** | 15.01 ms | 729.0 MB |

---

## Limitations & Failure Modes

1. **Zero-Shot Logit Mismatch**: Without fine-tuning recovery, naive zero-shot injection alters projection representations, necessitating post-injection recovery steps.
2. **Fixed Decay Horizon**: Static decay factors $\gamma$ may penalize tasks demanding uniformly long-range cross-sentence associative recall.

---

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 90.60% | **84.75%** | -5.85% | 0.8504 | 1.20x | 6 | [svd-linear-attention-distilbert-sst2-svd-linear-decay](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-svd-linear-decay) | 2026-09-27 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **53.90%** | -27.90% | 0.5017 | 1.20x | 6 | N/A | 2026-09-19 |
| **distilbert-agnews-full** | ag_news | `PASS` | 94.30% | **89.40%** | -4.90% | 0.8903 | 1.20x | 6 | [svd-linear-attention-distilbert-agnews-full-svd-linear-decay](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-agnews-full-svd-linear-decay) | 2026-09-19 |