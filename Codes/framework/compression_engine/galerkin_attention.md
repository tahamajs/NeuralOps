# Galerkin Transformer: Petrov‑Galerkin Projection Attention

**Registered as `galerkin`**  
*Low‑rank linear attention via random sketching of keys and values, achieving O(N·rank) complexity while preserving pretrained weights through zero‑shot initialization.*

---

## Overview

The Galerkin Attention module provides an efficient alternative to standard multi‑head self‑attention by projecting the key and value sequences onto a low‑rank subspace using a **random sketching matrix**. Instead of computing a full N×N attention map, the module computes an N×rank attention matrix, where `rank` is a user‑specified parameter much smaller than the sequence length N. This reduces the per‑token complexity from O(N²·d) to O(N·rank·d), making it attractive for long sequences while retaining compatibility with existing Transformer architectures.

The module is designed as a drop‑in replacement for DistilBERT’s attention layers, capable of **zero‑shot initialization**: it copies all projection weights (query, key, value, output) from a pretrained standard attention layer, so no retraining is required to start using it. The name *Galerkin* refers to the Petrov‑Galerkin projection principle, where the attention operation is approximated by projecting queries onto a reduced basis formed by the sketched keys.

The module is registered under `"galerkin"` in the attention registry, and an injection function is provided to replace attention layers in a DistilBERT model during compression.

---

## Methodology

### 1. Standard Attention vs. Galerkin Approximation

Standard multi‑head attention for a sequence of length N and head dimension D computes:

```
Q, K, V ∈ R^{B×H×N×D}
A = softmax( Q·K^T / √D ) ∈ R^{B×H×N×N}
out = A·V
```

The key–value multiplication `Q·K^T` yields an N×N matrix, dominating the cost at O(N²·D). To avoid this, Galerkin Attention introduces a **sketching matrix** **S** ∈ ℝ^{N×r} (where r ≪ N), drawn from a Gaussian distribution (scaled by 1/√N for normalization). The keys and values are first projected:

```
K_sketch = K·S  ∈ ℝ^{B×H×r×D}
V_sketch = V·S  ∈ ℝ^{B×H×r×D}
```

Then attention is computed between the full query Q and the sketched keys:

```
scores = Q·K_sketch^T / √D  ∈ ℝ^{B×H×N×r}
probs = softmax(scores)
out = probs·V_sketch      ∈ ℝ^{B×H×N×D}
```

The complexity now becomes O(N·r·D + N·r·D) = O(N·r·D), which is linear in sequence length when r is fixed. The sketching matrix effectively reduces the number of key–value tokens from N to r, approximating the full attention.

### 2. Random Sketching and Normalization

The sketching matrix **S** ∈ ℝ^{N×r} is generated randomly for each forward pass (or could be stored as a buffer for reproducibility). Each entry is drawn from a normal distribution with standard deviation `1/√N`. This scaling ensures that the projection `K·S` roughly preserves the norm of K, keeping the attention scores at a similar scale. The sketch is applied equally to all heads and batch elements.

The choice of Gaussian sketching is motivated by the Johnson–Lindenstrauss lemma, which guarantees that pairwise distances are approximately preserved after such a random projection. Here it is applied to the key/value vectors, reducing the sequence dimension while maintaining the ability of queries to attend to the most relevant (now compressed) information.

### 3. Zero‑Shot Initialization from Pretrained Weights

The module constructor requires a `pretrained_attn` object (the original DistilBERT attention layer). It directly copies the weights and biases from `pretrained_attn.q_lin`, `k_lin`, `v_lin`, and `out_lin` into its own linear projections. This means the Galerkin attention starts with the exact same linear transformations as the full attention; only the attention scoring changes. This allows a model to be immediately switched to the efficient mode without any additional training, hence “zero‑shot”.

### 4. Mask Handling

If an `attention_mask` is provided (e.g., for padding), it is applied to the scores before softmax. The mask is expected as a boolean tensor where `True` indicates valid tokens. The code converts any mask of shape `(B, N)` to `(B, 1, 1, N)` and sets scores of invalid positions to `-1e9`. This ensures that padding tokens do not affect the attention probabilities, which are computed over the sketched dimension.

---

## Implementation Details

### GalerkinAttention Class

```python
class GalerkinAttention(nn.Module):
    def __init__(self, model_config, rank=64, dropout=0.1, pretrained_attn=None):
        ...
```

- **`model_config`**: Contains `num_attention_heads` and `hidden_size`.
- **`rank`**: The size of the sketched dimension (default 64). This is the number of sketched key‑value “virtual tokens”.
- **`dropout`**: Dropout probability applied to attention probabilities.
- **`pretrained_attn`**: An existing attention module (with `q_lin`, `k_lin`, `v_lin`, `out_lin`) from which weights are copied.

**`_reshape`** helper transforms from `(B, N, H*D)` to `(B, H, N, D)`.

**`forward`**:

1. Projects hidden states to Q, K, V and reshapes to multi‑head.
2. Generates sketching matrix `sketch` of shape `(N, rank)` with `torch.randn` scaled by `1/√N`.
3. Applies sketch: `K_sketch = einsum('bhnd,nr->bhrd', K, sketch)`, similarly for V.
4. Computes scores `Q @ K_sketch.T / sqrt(D)` → shape `(B, H, N, r)`.
5. Applies optional mask to scores, then softmax over the last dimension (the r sketched keys).
6. Computes context `probs @ V_sketch`.
7. Transposes, flattens, and passes through output projection.
8. Returns `(output, probs if output_attentions else None)`.

Note: The code uses `randn` for the sketch each forward pass. For reproducibility, the random seed could be set externally. The scaling factor `1/√N` may be suboptimal for variable sequence lengths; a learnable scale or normalization based on actual N could be more robust.

### Injection Function

```python
@ATTENTION_REGISTRY.register("galerkin")
def inject_galerkin(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank = getattr(comp_config, 'galerkin_rank', 64)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = GalerkinAttention(model_config, rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn
```

This function is called by the DistilBERT compression framework to replace the original attention layer in a transformer block. It takes the `old_attn` as argument to enable zero‑shot weight transfer.

---

## Usage and Configuration

To use Galerkin attention, set the attention type to `"galerkin"` in the compression configuration and optionally specify the rank.

### Example

```python
comp_config.attention_type = "galerkin"
comp_config.galerkin_rank = 32   # use rank 32
comp_config.dropout = 0.1        # standard dropout
```

Then inject the attention into the model through the framework. The pretrained model’s weights are automatically copied.

### Parameters

| Parameter          | Default | Description                                                                                      |
|--------------------|---------|--------------------------------------------------------------------------------------------------|
| `galerkin_rank`    | `64`    | Number of sketched key‑value “tokens”. Smaller values yield faster computation but more approximation. |
| `dropout`          | `0.1`   | Dropout rate on attention probabilities (applied after softmax).                                  |

**Note**: The sketching matrix is generated randomly each forward pass, which can cause slight variance in outputs. For deterministic inference, one could fix the random seed or store the sketch as a buffer computed once per sequence length. The current implementation trades determinism for simplicity.

---

## Discussion

### Advantages

- **Linear complexity**: The attention computation scales as O(N·r·D) instead of O(N²·D), enabling longer sequences with controlled memory.
- **Zero‑shot usability**: Inherits pretrained weights directly, so no fine‑tuning is required to switch to this attention mechanism, though accuracy may degrade slightly.
- **Drop‑in replacement**: Compatible with existing Transformer architectures; only the attention sub‑layer is modified.
- **Conceptually simple**: The random sketching is easy to implement and understand.

### Limitations

- **Random projection variance**: The stochastic nature of the sketch may lead to inconsistent attention patterns across runs, potentially hurting accuracy.
- **Rank sensitivity**: The rank must be chosen appropriately; too low and the model loses important information, too high and the speed‑up diminishes.
- **Mask handling nuance**: The mask is applied after the sketch, meaning that the sketch still includes contributions from padding tokens (though they are masked from attention scores). The projection `K_sketch = K·S` mixes information from all positions, including padding, into the sketched keys. This could cause leakage of padding information into the attention. A proper implementation should mask K and V before applying the sketch or use a mask‑aware projection.
- **No training‑free guarantee**: While weights are copied, the change in attention mechanism may require fine‑tuning to recover performance on specific tasks.
- **Sequence length dependence**: The sketch dimension is fixed, but the scaling `1/√N` depends on N; for variable‑length batches, the effective scaling changes per sequence, which might be suboptimal.

### Comparison with Other Linear Attentions

| Method            | Approach                              | Complexity | Retains Pretrained Weights |
|-------------------|---------------------------------------|------------|----------------------------|
| Galerkin (this)   | Random sketch of K, V                | O(N·r·D)  | Yes (copied)               |
| Linformer         | Learned projection of K, V            | O(N·k·D)  | No (requires training)     |
| Performer         | Random features for kernel estimation  | O(N·r·D)  | No                         |
| Linear Transformer| Kernelized softmax                     | O(N·D²)   | No                         |

Galerkin is unique in its ability to directly transfer pretrained weights, making it a quick zero‑shot linear attention option.

---

## Conclusion

The Galerkin Transformer attention module (registered as `"galerkin"`) offers a practical linear‑complexity attention mechanism for DistilBERT models via random sketching. By projecting keys and values onto a lower‑dimensional subspace, it dramatically reduces the cost of long‑sequence processing. Its zero‑shot initialization from existing pretrained weights allows immediate deployment without training, making it a convenient tool for exploratory research and real‑world applications where long context is needed. Future improvements could include a more robust handling of padding in the sketch, learnable or deterministic sketching, and integration with fine‑tuning strategies to recover any accuracy drop.