# Adaptive Span Attention

A plug‑in attention module for DistilBERT that learns a per‑head span length, masking out distant tokens beyond that span.

## Paper
**Adaptive Attention Span in Transformers**  
Sukhbaatar et al., ACL 2019

## What it does
Instead of every attention head attending over the full context, each head gets a learnable scalar that defines its maximum look‑back distance.  
During inference, tokens outside that span are masked with `-inf`, effectively limiting each head’s receptive field.

## Implementation
- `AdaptiveSpanAttention` replaces the standard self‑attention in a DistilBERT layer.
- Projection weights (`q, k, v, out`) are copied from the pre‑trained module (zero‑shot init).
- The span parameter is stored as a logit, converted via sigmoid and scaled by `max_span`.
- Masking is **hard** – entries outside the span become `-1e9` (no gradient to span parameter).

## How to use
```python
from framework.core.registry import ATTENTION_REGISTRY
# In your compression config, set:
comp_config.adaptive_span_rank = 32   # kept for API compatibility, currently unused
comp_config.dropout = 0.1
comp_config.max_span = 128
```

The injection function is registered under the name `"adaptive_span"` and is called automatically when you select this attention type in the framework.

Example manual injection (for a single layer):
```python
new_attn = AdaptiveSpanAttention(
    model_config=model_config,
    rank=32,
    dropout=0.1,
    max_span=128,
    pretrained_attn=layer.attention   # original attention module
)
layer.attention = new_attn
```

## Configuration parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `adaptive_span_rank` | 32 | Not used by this module (kept for interface consistency) |
| `dropout` | 0.1 | Dropout on attention weights |
| `max_span` | 128 | Maximum possible span in token positions |

## Important notes
- **Non‑differentiable mask**: The current implementation uses a hard threshold, so the span parameters **do not receive gradients** during normal backpropagation. To actually train the spans you need to implement a soft mask (see original paper) or use a straight‑through estimator.
- **Zero‑shot init**: Requires the original attention module to have attributes `q_lin, k_lin, v_lin, out_lin`.
- **Padding**: Standard attention masking for padding tokens is applied after the span mask.

## Dependencies
- PyTorch
- Framework’s registry (`ATTENTION_REGISTRY`) – the injection function expects the compression configuration to be accessible.

---

*Keep it simple; adapt the mask function if you want end‑to‑end span learning.*