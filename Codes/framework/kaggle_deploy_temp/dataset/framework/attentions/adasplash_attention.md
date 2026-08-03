# AdaSplash Attention

Replaces softmax with **entmax (α=1.5)** – a sparse attention mapping that can output exact zeros.

## What it does
Standard attention uses softmax, which always assigns some probability to every token.  
Entmax with α=1.5 (the “1.5‑entmax”) produces **sparse outputs** – many tokens get exactly zero weight, while only the most relevant ones receive positive probability. This acts as an implicit, differentiable top‑k selection.

## Method
The attention scores are computed as usual (`QKᵀ / √d`), but instead of softmax, the iterative `entmax15` algorithm is used.

```
1. Start with a softmax distribution.
2. Repeat:
   - Take element‑wise square root.
   - Renormalise.
3. Converges to a sparse probability vector.
```

Only the few highest‑scoring positions survive; the rest become zero.

## Implementation
- `entmax15(scores, dim, n_iter)` – differentiable iterative algorithm.
- `AdaSplashAttention` – plug‑in replacement for the standard attention module.
- Q, K, V, O projections are standard `nn.Linear` layers (trainable).
- Works with padding mask (sets masked scores to `-inf`).

## How to use
Your compression framework uses the name `"adasplash"`:

```python
# In your config:
comp_config.alpha = 1.5        # default, do not change (only 1.5 implemented)
comp_config.n_iter = 10        # number of entmax iterations
```

The injection is automatic when the attention type is set to `"adasplash"`.

Manual injection:
```python
attn = AdaSplashAttention(
    config=model_config,
    rank=None,       # kept for interface compatibility
    alpha=1.5,
    n_iter=10
)
layer.attention = attn
```

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.5 | Entmax α (only 1.5 implemented) |
| `n_iter` | 10 | Number of convergence iterations |

## Notes
- **Sparsity** – great for long sequences and interpretability; most tokens are ignored after a few layers.
- **Non‑standard attention** – the output is not a pure probability distribution (some values zero, others larger).
- **Implementation** – only α=1.5 is coded; other values would require different algorithms.

---

*Simple: swap softmax for sparse entmax. That’s it.*