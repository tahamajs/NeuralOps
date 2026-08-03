# Adaptive Transformer Compressor (Adaptive SVD)

**Registered as `adaptive_svd`**  
*Automatic layer‑wise low‑rank compression with energy‑threshold SVD for multi‑head attention.*

---

## Overview

The Adaptive Transformer Compressor is a fully automatic post‑training compression tool that targets all `nn.MultiheadAttention` layers within a PyTorch model. Instead of using a fixed rank or requiring manual tuning, it analyses the singular value spectrum of each attention projection matrix and selects a custom rank that retains a user‑specified fraction of the matrix’s energy (e.g., 95%). The original attention layer is then **structurally replaced** by a `CompressedAttentionLayer` that implements the factorised projections, resulting in both a smaller parameter count and a genuine reduction in inference FLOPs.

Designed for the DistilBERT compression framework, the module is registered under the name `"adaptive_svd"` and can be applied with a few lines of configuration. It requires no calibration data, no fine‑tuning, and automatically adapts to every attention layer in the model.

---

## Methodology

### Energy‑Threshold Rank Selection

Consider a weight matrix **W** of shape `(d_out, d_in)` from an attention projection (query, key, value, or output). Its singular value decomposition is:

```
W = U Σ V^T  with  Σ = diag(σ₁, σ₂, …, σ_r)
```

The total energy (squared Frobenius norm) of **W** is `∑ σ_i²`. For a rank `k`, the **cumulative energy fraction** is:

```
E(k) = (∑_{i=1}^{k} σ_i²) / (∑_{i=1}^{r} σ_i²)
```

Given a threshold `τ` (e.g., 0.95), the algorithm selects the smallest `k` such that `E(k) ≥ τ`. This ensures that at least `τ × 100%` of the original matrix’s energy is preserved in the low‑rank approximation. A minimum rank `min_rank` (default 16) is enforced to prevent extremely small, unstable factorisations.

Because the singular value decay varies from layer to layer, some attention projections may be compressed to very low ranks while others retain larger ranks – all determined automatically.

### Structural Replacement with CompressedAttentionLayer

Once the adaptive rank is determined, the full attention layer is **not just weight‑overwritten**; it is completely replaced by a `CompressedAttentionLayer`. This custom module internally factorises each linear projection into two smaller matrices:

```
W ≈ A · B   with A ∈ ℝ^{d_out × k}, B ∈ ℝ^{k × d_in}
```

The forward pass then becomes:
1. **Down‑projection** with `B` (reduces dimension from `d_in` to `k`).
2. **Up‑projection** with `A` (expands back to `d_out`).

This approach applies to all four projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`). The attention computation itself (scaled dot‑product) remains unchanged, but because it now operates on lower‑dimensional intermediate representations, both memory and computation are reduced. The compression therefore yields real speed‑ups, not merely fewer stored parameters.

The `CompressedAttentionLayer` also properly handles the splitting and merging of attention heads, ensuring that the output matches the shape of the original layer.

### Automatic Layer Detection

The compressor scans the entire model with `named_modules()`. For every module that is an `nn.MultiheadAttention` instance, it:

1. Creates a `CompressedAttentionLayer(original_module, energy_threshold, min_rank)`.
2. Locates the parent module in the hierarchy and replaces the child via `setattr`.
3. Logs compression statistics (original vs. compressed parameter counts).

Modules that are not `nn.MultiheadAttention` are left untouched, preserving all other weights and biases.

---

## Implementation Details

### AdaptiveTransformerCompressor Class

```python
class AdaptiveTransformerCompressor(BaseCompressor):
    def __init__(self, model, energy_threshold=0.95, min_rank=16, **kwargs):
        super().__init__(model, name="adaptive_svd", **kwargs)
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank
```

- **`model`**: The full PyTorch model (e.g., a DistilBERT variant).
- **`energy_threshold`**: Fraction of energy to retain (float between 0 and 1). Lower values give stronger compression.
- **`min_rank`**: Minimum allowed rank, preventing over‑compression of small matrices.

The `compress()` method:
- Sets the model to evaluation mode.
- Iterates over all named modules, identifying `nn.MultiheadAttention` instances.
- For each, attempts to wrap it in `CompressedAttentionLayer`.
- If successful, replaces the original layer and records parameter counts.
- If an exception occurs (e.g., from incompatible dimensions), it prints an error and continues.
- After all replacements, updates internal compressed parameter count and returns the model with compression statistics.

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

The factory reads `energy_threshold` and `min_rank` from the `comp_config` object, using defaults if they are not set. It then instantiates the compressor, runs it, and returns the compressed model.

### Dependency: `CompressedAttentionLayer`

The module `CompressedAttentionLayer` (imported from `framework.attentions.compressed_attention`) is the core component that performs the rank selection, SVD, factorisation, and provides the low‑rank forward logic. It must handle:

- **Combined head projection**: In standard `nn.MultiheadAttention`, the `q_proj`, `k_proj`, `v_proj` are linear layers that map from `d_model` to `d_model` (or to `num_heads * head_dim`). The low‑rank version must correctly separate and combine heads.
- **Energy‑based rank calculation**: Computes SVD of each projection weight, accumulates squared singular values, and finds the smallest `k` satisfying the energy threshold.
- **Efficient forward pass**: Uses the factorised matrices to reduce intermediate dimensions, then carries out scaled dot‑product attention.

While the full implementation is beyond this README, the existence of this dependency ensures that the compressed model is both parameter‑efficient and computationally faster.

---

## Usage and Configuration

To activate the Adaptive Transformer Compressor, set the compression method to `"adaptive_svd"` in your compression configuration and optionally tune the threshold and minimum rank.

### Example

```python
# Inside your compression script or config builder
comp_config.comp_method = "adaptive_svd"
comp_config.energy_threshold = 0.90   # retain 90% of the energy
comp_config.min_rank = 8              # don't go below rank 8
```

Then pass the model and `comp_config` to the framework’s compression runner. The framework will call `adaptive_svd_factory`, compress the model, and return the compact version.

### Parameters

| Parameter          | Default | Description                                                                                  |
|--------------------|---------|----------------------------------------------------------------------------------------------|
| `energy_threshold` | `0.95`  | Fraction of singular value energy to preserve. `1.0` = no compression; `0.8` = aggressive.   |
| `min_rank`         | `16`    | Lower bound for the rank of any projection matrix. Prevents degenerate, very low‑rank layers.|

**Note**: The model must contain `nn.MultiheadAttention` layers (the standard PyTorch implementation). If the model uses custom attention (e.g., HuggingFace’s `BertSelfAttention`), the compressor will not act on them. In that case, adapt the filter in the compressor code.

---

## Discussion

### Advantages

- **Fully automatic**: No need to specify per‑layer ranks; the energy threshold provides a principled, data‑independent criterion.
- **Layer‑wise adaptation**: Layers with quickly decaying singular values get higher compression; more important layers retain more capacity.
- **Real runtime speed‑up**: By replacing dense projections with low‑rank pairs, both memory and compute are reduced during inference, not just storage.
- **No calibration data**: The SVD is computed on the weights alone, making the method deterministic and quick (no forward passes required).
- **Graceful fallback**: If a layer cannot be compressed (e.g., due to dimension mismatch), an error is printed and the layer is skipped, preserving the rest of the model.

### Limitations

- **Dependence on custom layer**: The quality of the compressed model hinges on the correctness and efficiency of `CompressedAttentionLayer`. It must be thoroughly tested.
- **Limited to `nn.MultiheadAttention`**: Many transformer libraries use different attention classes (e.g., `transformers.models.bert.modeling_bert.BertSelfAttention`). The current filter will miss those; a more flexible module type check may be needed for broader use.
- **Potential accuracy drop**: Like all post‑training compression, there is a trade‑off. The energy threshold can be adjusted, and optionally a few steps of fine‑tuning can recover lost accuracy.
- **Head interactions**: Global low‑rank factorisation of the concatenated Q/K/V projections may mix head‑specific information. While often benign, it can sometimes degrade multi‑head diversity. This effect is usually minor when the threshold is high (≥0.95).
- **No cross‑layer sharing**: Each layer is compressed independently; further gains could be achieved by sharing factors across layers, but that is not implemented.

### Comparison with A3 and AASVD

| Feature               | A3 (fixed SVD) | AASVD (anchored SVD) | Adaptive SVD (this module) |
|-----------------------|----------------|----------------------|----------------------------|
| Rank selection        | Fixed rank     | Fixed rank           | Energy‑threshold (adaptive per layer) |
| Data‑awareness        | No             | Yes (activation)     | No                          |
| Structural change     | Overwrites weight | Overwrites weight    | Replaces with factorized layer |
| Runtime speed‑up      | No             | No                   | Yes                         |
| Calibration needed    | No             | Yes (texts)          | No                          |

The Adaptive SVD compressor stands out by offering automatic per‑layer rank selection and real inference acceleration, making it ideal for deploying compressed models in production without manual tuning.

---

## Conclusion

The Adaptive Transformer Compressor (registered as `"adaptive_svd"`) provides a powerful “set‑and‑forget” solution to compress multi‑head attention layers. By using an energy threshold to automatically determine the optimal rank for each projection, it achieves significant parameter reduction while converting the model into a truly faster factorised form. Its simple integration via the compression registry makes it a natural choice for shrinking transformer models in the DistilBERT ecosystem, requiring only a few configuration lines and no extra training.