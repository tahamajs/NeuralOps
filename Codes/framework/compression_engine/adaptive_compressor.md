# Adaptive Transformer Compressor (Adaptive SVD)

**Registered as `adaptive_svd`**  
*Automatic, energy‑based low‑rank compression of full multi‑head attention layers.*

---

## Overview

The Adaptive Transformer Compressor is a post‑training compression module that automatically identifies and compresses all `nn.MultiheadAttention` layers in a model using an **energy‑threshold SVD** approach. Unlike fixed‑rank methods, it adapts the compression rank per attention layer based on the singular value spectrum of the weight matrices, retaining just enough components to preserve a specified fraction of the total energy (e.g., 95%). The original attention layers are replaced with lightweight `CompressedAttentionLayer` instances that implement the factorized projections, leading to real parameter reduction and potential inference speed‑ups.

The module is integrated into the compression framework via the `"adaptive_svd"` entry point and can be used with any PyTorch model containing `nn.MultiheadAttention` sub‑modules.

---

## Methodology

### Energy‑Threshold Rank Selection

For each projection matrix **W** (query, key, value, output) inside a multi‑head attention layer, the singular values **σ₁ ≥ σ₂ ≥ … ≥ σ\_r** are computed. The cumulative energy up to rank `k` is defined as:

```
E(k) = ∑_{i=1}^{k} σ_i² / ∑_{i=1}^{r} σ_i²
```

Given an energy threshold `τ` (e.g., 0.95), the algorithm selects the smallest rank `k` such that `E(k) ≥ τ`. This ensures that at least `τ × 100%` of the matrix’s Frobenius norm is preserved. The effective rank is then bounded below by `min_rank` (default 16) to avoid excessively small, unstable factorizations.

This adaptive scheme tailors the compression to each layer: layers with a rapid singular value decay can be compressed more aggressively, while layers with a flatter spectrum retain higher ranks.

### Factorization and Replacement

Once the rank is determined, the full weight matrix is factorized into two smaller matrices:

```
W ≈ A @ B    with    A ∈ ℝ^{d_out × k}, B ∈ ℝ^{k × d_in}
```

The factorized projections are encapsulated in a `CompressedAttentionLayer`. This layer replaces the original multi‑head attention module, internally implementing:

- **Query/Key/Value projections**: Each original linear projection (which combined multiple heads) is decomposed into a low‑rank pair. During forward, the input is first projected by `B` (rank‑reduction) then by `A` (expansion), mimicking the full projection at a fraction of the cost.
- **Output projection**: Similarly factorized.
- **Attention computation**: The actual scaled dot‑product attention is performed unchanged, but with the reduced intermediate dimensions.

The result is a drop‑in replacement with significantly fewer parameters and FLOPs.

### Integration into the Pipeline

The compressor scans the model for all `nn.MultiheadAttention` modules. For each one found:

1. A `CompressedAttentionLayer` is instantiated, passing the original module, the energy threshold, and the minimum rank.
2. The original attention layer is replaced in‑place within the model hierarchy.
3. Compression statistics (original vs. new parameter counts) are logged.

The replacement is done via `setattr` on the parent module, using the module hierarchy path.

---

## Implementation Details

### AdaptiveTransformerCompressor Class

Inherits from `BaseCompressor`.

```python
class AdaptiveTransformerCompressor(BaseCompressor):
    def __init__(self, model, energy_threshold=0.95, min_rank=16, **kwargs):
        super().__init__(model, name="adaptive_svd", **kwargs)
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank
```

- **`energy_threshold`**: Fraction of singular value energy to retain (default 0.95 → 95%).
- **`min_rank`**: Minimum rank allowed to prevent over‑aggressive compression (default 16).

The `compress()` method:

1. Iterates over all named modules.
2. If a module is an instance of `nn.MultiheadAttention`, it attempts to wrap it into `CompressedAttentionLayer`.
3. The parent module is accessed via `get_submodule`, and the child is replaced.
4. Statistics are recorded.
5. After processing, `_compressed_params` is updated and the compressed model is returned along with stats.

**Dependency**: The method relies on `CompressedAttentionLayer` from `framework.attentions.compressed_attention`. This layer is expected to internally handle the SVD, energy computation, and factorization.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("adaptive_svd")
def adaptive_svd_factory(model, comp_config):
    energy_threshold = getattr(comp_config, 'energy_threshold', 0.95)
    min_rank = getattr(comp_config, 'min_rank', 16)
    compressor = AdaptiveTransformerCompressor(
        model, energy_threshold=energy_threshold, min_rank=min_rank
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory retrieves configuration parameters from `comp_config`. These can be set by the user to control compression aggressiveness.

### `CompressedAttentionLayer` (Conceptual)

While the full implementation of `CompressedAttentionLayer` is not shown here, the typical design is:

- For each of **q_proj, k_proj, v_proj, out_proj**, compute the truncated SVD with rank determined by the energy threshold.
- Replace each linear layer with a `nn.Sequential` of two linear layers: a down‑projection to rank `k` and an up‑projection to original dimension.
- Adjust the attention logic (head splitting) to work with the factorized dimensions.
- Optional: fuse head‑wise operations for additional speed.

The use of `CompressedAttentionLayer` ensures that the model not only has fewer parameters but also performs actual low‑rank computations at runtime.

---

## Usage and Configuration

Set the compression method to `"adaptive_svd"` and optionally configure the energy threshold and minimum rank.

### Example

```python
comp_config.comp_method = "adaptive_svd"
comp_config.energy_threshold = 0.90   # retain 90% of energy
comp_config.min_rank = 8             # never go below rank 8
```

Then invoke the compression pipeline. The compressor will automatically detect all multi‑head attention layers, compress them, and return the compact model.

### Parameters

| Parameter             | Default | Description                                                                                                                                 |
|-----------------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `energy_threshold`    | 0.95    | Fraction of the singular value energy to preserve. Higher values give less compression but higher fidelity.                                   |
| `min_rank`            | 16      | Minimum rank allowed for any projection. Prevents degenerate cases.                                                                          |
| (model requirement)   | –       | Model must contain `nn.MultiheadAttention` layers (e.g., from `torch.nn`). If the model uses custom attention, it will not be compressed.    |

**Note**: The compressor only processes modules that are exactly `nn.MultiheadAttention`. If your model uses a different attention class (even if functionally equivalent), you need to either adapt the filter or ensure the attention layers are instances of `nn.MultiheadAttention`.

---

## Discussion

### Advantages

- **Fully automatic**: No need to specify ranks per layer. The energy threshold provides a principled, data‑independent criterion.
- **Layer‑wise adaptation**: Layers with redundant information receive higher compression, while important layers retain more capacity.
- **True runtime speed‑up**: The replacement with `CompressedAttentionLayer` yields actual reductions in FLOPs and memory during inference, not just parameter count.
- **No calibration required**: The SVD is performed purely on the weight matrices; no data is needed. This makes the method fast and deterministic.
- **Stable lower bound**: The `min_rank` prevents extreme compression that could destabilize the model.

### Limitations

- **Dependence on a custom layer**: The `CompressedAttentionLayer` must be correctly implemented and tested. Its quality directly impacts the compressed model’s performance.
- **Only for `nn.MultiheadAttention`**: If the model uses a different attention implementation (e.g., from HuggingFace `transformers` or custom blocks), the compressor will not touch them. A broader version could be implemented with a more flexible filter.
- **Loss of head‑wise flexibility**: Standard multi‑head attention learns head‑specific projections; after global low‑rank factorization, the independence of heads may be blurred, potentially losing some representational capacity. However, many works have shown that this is rarely a problem in practice.
- **No fine‑tuning**: Like other post‑training compression methods, there may be a drop in accuracy. The energy threshold can be tuned to trade off accuracy and size; optionally, a few steps of fine‑tuning can recover performance.
- **Implementation specifics**: The success of the compression depends on how `CompressedAttentionLayer` factorizes the combined projection (which normally maps to `num_heads * head_dim`). The factorization must correctly handle the head splitting and concatenation logic. If implemented naively, it could lead to incorrect outputs. The provided snippet assumes a robust underlying implementation.

### Comparison with A3 and AASVD

| Method | Rank Selection | Data‑Aware | Replaces Weights | Runtime Speed‑up |
|--------|----------------|------------|------------------|------------------|
| A3 | Fixed | No | Overwrites weight | No (dense matrix) |
| AASVD | Fixed | Yes (activation anchor) | Overwrites weight | No |
| Adaptive SVD | Energy‑threshold per layer | No | Replaces with factorized layer | Yes |

The Adaptive SVD compressor is the only one among the three that structurally changes the model to achieve real inference speed‑ups, while also automatically selecting the rank per layer.

---

## Conclusion

The Adaptive Transformer Compressor offers a powerful “set‑and‑forget” tool to compress multi‑head attention layers with minimal configuration. By leveraging an energy threshold on the singular value spectrum, it intelligently allocates different compression ratios across layers, resulting in a compact model that can run faster without any additional training. Its integration via the `"adaptive_svd"` factory makes it a straightforward choice for shrinking transformer models in the DistilBERT compression ecosystem.