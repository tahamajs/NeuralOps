# ERC‑SVD: Error‑Controlled SVD Compression

**Registered as `erc`**  
*Automatically selects the rank per linear layer by preserving a specified fraction of the singular value energy, controlling the approximation error.*

---

## Overview

ERC‑SVD (Error‑Controlled SVD) is a post‑training compression method that compresses each linear layer’s weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup> using truncated SVD, with the rank determined individually for each layer by a user‑specified **allowed error** criterion. Instead of fixing a global rank, the method guarantees that the relative Frobenius norm error (squared) of the approximation does not exceed `allowed_error`. Concretely, it retains the smallest number of singular values such that the cumulative energy (sum of squared singular values) of the kept components is at least `1 – allowed_error` of the total energy.

This provides a principled, layer‑adaptive compression that respects a uniform error budget across the model. Layers with a rapid decay of singular values can be compressed aggressively, while layers with a flat spectrum retain higher rank to keep the error within bounds. The method requires no calibration data, working purely on the weight matrices, and is extremely simple to apply.

The module is registered under `"erc"` in the DistilBERT compression framework and can be used out‑of‑the‑box with a single configuration parameter.

---

## Methodology

### 1. Singular Value Energy and Cumulative Distribution

For a weight matrix **W**, let its SVD be:

```
W = U Σ V^T
```

where `Σ = diag(σ₁, σ₂, …, σ_r)` and `σ₁ ≥ σ₂ ≥ … ≥ σ_r ≥ 0`. The total **energy** (squared Frobenius norm) of the matrix is:

```
E_total = ∑_{i=1}^r σ_i²
```

The cumulative energy fraction up to rank `k` is:

```
E_cum(k) = (∑_{i=1}^k σ_i²) / E_total
```

This function increases from 0 to 1 as `k` goes from 0 to `r`.

### 2. Rank Selection from Allowed Error

The user specifies a positive `allowed_error` (e.g., `0.01`). The goal is to find the smallest rank `r*` such that the discarded energy is at most `allowed_error` times the total energy:

```
1 - E_cum(r*) ≤ allowed_error
```

Equivalently:

```
E_cum(r*) ≥ 1 - allowed_error
```

If `allowed_error = 0.01`, we retain 99% of the total singular value energy. The rank `r*` is determined by scanning the cumulative energy array until the threshold is met:

```python
energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
idx = torch.where(energy >= (1 - self.allowed_error))[0]
rank = idx[0].item() + 1 if len(idx) > 0 else 1
```

If no singular value reaches the threshold (which can only happen if `allowed_error` is very small and even the full rank doesn’t suffice), the rank is set to 1 to prevent a zero‑rank edge case.

### 3. Low‑Rank Approximation

Once `r*` is determined, the weight is approximated by keeping the first `r*` singular vectors and values:

```
W_approx = U_r* Σ_r* V_r*^T
```

This approximation is optimal in the Frobenius norm for the given rank, and the relative error (squared) is exactly `1 - E_cum(r*) ≤ allowed_error`.

### 4. Per‑Layer Adaptive Compression

The same procedure is applied independently to every `nn.Linear` module. Because the singular value distribution varies across layers, each layer ends up with a different rank, all satisfying the same relative error bound. This results in an overall compressed model where the error contributed by each linear transformation is controlled.

---

## Implementation Details

### ERCCompressor Class

Inherits from `BaseCompressor`.

```python
class ERCCompressor(BaseCompressor):
    def __init__(self, model, allowed_error=0.01, **kwargs):
        super().__init__(model, name="erc", **kwargs)
        self.allowed_error = allowed_error
```

- **`allowed_error`**: the maximum allowed fraction of energy to be discarded (default `0.01`, i.e., preserve 99% of energy). Must be a positive float, typically less than 1.

#### `_compress_layer(W)` method
1. Takes a weight tensor **W**, converts to float32.
2. Computes full SVD via `torch.linalg.svd(W.float(), full_matrices=False)`.
3. Computes cumulative energy array.
4. Finds the smallest index `idx[0]` where cumulative energy ≥ `1 - allowed_error`.
5. Sets `rank = idx[0] + 1` (since index is 0‑based). If no such index, rank = 1.
6. Returns the truncated reconstruction `U[:, :rank] @ diag(S[:rank]) @ Vt[:rank, :]`.

The `compress()` method:
- Loops over all `nn.Linear` modules.
- Replaces each `module.weight.data` with the output of `_compress_layer(module.weight.data)`. The bias is left unchanged.
- Updates `_compressed_params` and returns the model with compression statistics.

**Note**: The SVD is performed on GPU via PyTorch (the weight is not moved to CPU). This may be memory‑intensive for huge matrices, but it keeps the operation in‑place. The resulting weight remains a dense matrix of the same shape; the compression is in the effective rank, which can be extracted later for factorized storage.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("erc")
def erc_factory(model, comp_config):
    allowed_error = getattr(comp_config, 'erc_allowed_error', 0.01)
    compressor = ERCCompressor(model, allowed_error)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `erc_allowed_error` from `comp_config` (default `0.01`) and applies ERC‑SVD compression.

---

## Usage and Configuration

Set the compression method to `"erc"` and optionally adjust the allowed error.

### Example

```python
comp_config.comp_method = "erc"
comp_config.erc_allowed_error = 0.02   # allow 2% energy loss per layer
```

Then run the compression pipeline. Each linear layer’s weight will be replaced with a low‑rank approximation satisfying the error bound.

### Parameters

| Parameter              | Default | Description                                                                                      |
|------------------------|---------|--------------------------------------------------------------------------------------------------|
| `erc_allowed_error`    | `0.01`  | Maximum fraction of singular value energy to discard per layer. Must be between 0 and 1.         |

**Note**: A smaller allowed error results in higher ranks (less compression) and better accuracy; a larger error leads to more aggressive compression.

---

## Discussion

### Advantages

- **Error‑driven**: The compression rank is directly linked to a meaningful accuracy‑related metric (energy preservation). Users can tune the allowed error to balance accuracy and compression without guessing ranks.
- **Per‑layer adaptation**: Each layer gets its own rank, respecting its individual singular value decay. This tends to compress “redundant” layers more and keep important layers intact.
- **No calibration data**: Works purely on weights, making it fast and easy to apply.
- **Simplicity**: The code is concise and easy to understand or modify.
- **Provable error bound**: For the weight matrix itself, the Frobenius error is bounded; this often translates to limited degradation in the layer’s output for typical inputs (though not a guarantee).

### Limitations

- **Dense weight output**: The compressed weight is still a full dense matrix, so there is no immediate reduction in parameter count or FLOPs. To achieve actual memory and computational savings, the low‑rank factors must be extracted and stored separately (e.g., replacing the layer with two smaller linear layers).
- **SVD cost**: Full SVD is computed for every linear layer; for very large models, this can be time‑consuming. Approximate SVD or randomized algorithms could mitigate this but are not used.
- **Independent per‑layer**: The error bound is local to each weight matrix; interactions between layers (e.g., compounding of errors) are not considered. The final output error could be larger than `allowed_error`.
- **Bias not compressed**: Biases remain at full precision, which may cause a slight misalignment with the compressed weights.
- **Numerical issues**: If singular values decay very slowly, the cumulative energy threshold may require almost full rank, yielding little compression. This is a feature, not a bug, but it means that models with poor low‑rank structure will not compress well with this method.
- **Minimum rank of 1**: In the edge case where no rank meets the threshold (e.g., allowed_error close to 0 and many tiny singular values), the rank is forced to 1, which could cause severe degradation.

### Comparison with Other SVD‑Based Compressors

| Method             | Rank Selection Criterion        | Per‑Layer Adaptation | Error Guarantee     |
|--------------------|---------------------------------|----------------------|---------------------|
| ERC‑SVD (this)     | Cumulative energy ≥ 1‑error     | Yes                  | Yes (per‑layer Frobenius) |
| A3 (simple SVD)    | Fixed rank                      | No                   | No                  |
| Adaptive SVD       | Energy threshold (≥ threshold)  | Yes                  | Yes (implicitly)    |
| AdaSVD (importance)| Proportional to nuclear norm    | Yes (budget‑based)   | No                  |

ERC‑SVD is essentially a simpler, more explicit version of Adaptive SVD (energy threshold). Both use the same principle, but ERC‑SVD frames it as an error tolerance rather than an energy retention threshold, which is equivalent (threshold = 1 – error). The difference is minor; ERC‑SVD is a clean implementation of the energy‑based rank selection.

---

## Conclusion

ERC‑SVD (registered as `"erc"`) offers a straightforward, principled approach to post‑training weight compression via SVD with an error‑controlled rank selection. By specifying a maximum allowed energy loss, users can automatically determine the optimal rank for each linear layer, achieving a compressed model with bounded local approximation error. The method’s simplicity and lack of calibration data requirements make it an excellent baseline for low‑rank compression in the DistilBERT framework. To realise actual inference speed‑ups, the low‑rank factors should be extracted and used to replace the dense weight with two smaller linear layers, a step that can be added as a post‑processing phase. Future enhancements could include incorporating activation statistics to refine the error measure, and using approximate SVD for faster compression of large models.