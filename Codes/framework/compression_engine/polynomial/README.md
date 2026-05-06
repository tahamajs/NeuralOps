
# PADRe – Polynomial Attention with Dropout Regularization

A softmax‑free attention mechanism that replaces the exponential function with a **polynomial kernel**:  
`attn = (1 + S/order)^order`  
This allows for **faster computation** and **possible linearisation** via tensor‑train or random features, while maintaining competitive accuracy.

## Key Features

- **No exponentials** – uses only multiplications and powers, reducing latency on some hardware.
- **Configurable order** – higher order (e.g., 4) approximates softmax more closely; lower order (2) is faster.
- **Stability** – input scores are recentered (`S = S - max(S)`) and clamped to avoid negative bases.
- **Drop‑in** – registered as `"padre"`, can be injected into any DistilBERT / BERT model.

## Usage

```python
from framework.attentions.padre_attention import PADReAttention

# Create with custom polynomial order
attn = PADReAttention(config, order=3, temperature=1.0)

# Or inject via registry
from framework.attentions.core.registry import ATTENTION_REGISTRY
ATTENTION_REGISTRY["padre"](layer, model_config, comp_config, ...)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `order` | 2 | Degree of the polynomial kernel |
| `temperature` | 1.0 | Scaling factor applied to scores before polynomial |

## How It Works

1. Compute scaled dot‑product scores `S = QK^T / sqrt(d) / temperature`.
2. Shift scores so maximum is 0 (for numerical stability).
3. Compute polynomial: `P = (1 + S / order)^order`, clamping base to `>=0`.
4. Normalise: `attn = P / sum(P)`.
5. Apply dropout and aggregate values.

## Advantages Over Softmax

- The polynomial can be **expanded into a sum of monomials**, making it possible to use linear attention tricks (e.g., `(Q @ K^T)^2 = (Q @ Q^T) @ (K @ K^T)`).
- On CPUs and edge devices, polynomial evaluation can be faster than exponential.
- Dropout directly on attention weights remains fully compatible.

## Related Work

- **Polynomial Networks** (E. DeCastro et al.)
- **Linformer**, **Performer** – other softmax alternatives.
- **PADRe** paper (placeholder).

---

*Part of the Efficient Transformer Toolkit – softmax is just one choice.*
