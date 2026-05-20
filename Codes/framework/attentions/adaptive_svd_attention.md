# Adaptive SVD Attention

A low‑rank factorisation of the attention projections, with a tiny trainable residual on top.

## What it does
Instead of keeping the full `(hidden_size × all_head_size)` matrices, each projection (`Q, K, V, O`) is split into two parts:

- **SVD part** – frozen/low‑rank approximation taken directly from the pretrained weight.  
- **Residual part** – a pair of very small matrices (`A` and `B`) that are trained to add back what the SVD missed.

Total parameters = SVD rank + residual rank, which is usually much smaller than the original weight.

## Method
1. For each original weight matrix \( W \) (e.g. the pretrained `q_lin.weight`), run a truncated SVD:  
   \( W \approx U_r S_r V_r^T \)  
   We set \( U_r \in \mathbb{R}^{\text{out}\times r} \), \( V_r \in \mathbb{R}^{r\times \text{in}} \) with \( r = \text{svd\_rank} \).  
2. Freeze these `U` and `V` as the base approximation.  
3. Add two small learnable matrices `A` (out_features × residual_rank) and `B` (residual_rank × in_features) whose product `A @ B` is added to the output. This residual part is **fully trainable**.  
4. Biases are copied from the original linear layers and left trainable.  
5. The whole operation is cheap:  
   `y = U @ (V @ x) + A @ (B @ x) + bias`

## Implementation

### `AdaptiveSVDLinear`
A drop‑in replacement for a single `nn.Linear` that holds the `U, V` SVD factors and the `A, B` residual adapter.

**Key methods:**
- `set_svd_factors(U, V)` – copies the SVD factors into the module (used during initialisation).
- `forward(x)` – computes `U(Vx) + A(Bx) + bias`.

Initialisation:  
- `U`, `V` get orthogonal init, then are overwritten by SVD.  
- `A`, `B` use Kaiming uniform init.

### `AdaptiveSVDAttention`
The full attention module that replaces DistilBERT’s self‑attention. It creates four `AdaptiveSVDLinear` instances for Q, K, V, O, initialises their SVD parts from the pretrained weights, and performs the usual scaled dot‑product attention.

## How to use
Set your compression config values:
```python
comp_config.svd_rank = 32         # rank of the frozen SVD part
comp_config.residual_rank = 8     # rank of the trainable residual adapter
comp_config.dropout = 0.1
```

The injection function is registered under `"adaptive_svd_attn"`. It’s called automatically when you select this attention type.

Manual injection for one layer:
```python
new_attn = AdaptiveSVDAttention(
    model_config=model_config,
    svd_rank=32,
    residual_rank=8,
    dropout=0.1,
    pretrained_attn=layer.attention   # original module
)
layer.attention = new_attn
```

## Configuration parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_rank` | 32 | Rank of the frozen SVD approximation |
| `residual_rank` | 8 | Rank of the trainable residual adapter (set to 0 for pure SVD, no training) |
| `dropout` | 0.1 | Attention dropout |

## Important notes
- **Zero‑shot quality**: The SVD part gives a very good starting point (often >95% of the original weight’s energy). The residual adapter fixes the remaining error during fine‑tuning.
- **Trainable parameters**: Only `A`, `B` (and biases) are updated; `U`, `V` stay frozen. This makes the module extremely parameter‑efficient.
- **SVD precision**: The SVD is computed in float32 for stability, then cast to the target dtype. Rank is clipped to the actual matrix dimensions.
- **Missing bias**: If the original linear layer has no bias, the adapter is also created without one.

---

*Simple: frozen SVD + tiny trainable fix. Good for compression and fast fine‑tuning.*