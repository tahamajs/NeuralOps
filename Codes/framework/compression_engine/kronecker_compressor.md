# Kronecker Product Compressor: Factorizing Weight Matrices as Kronecker Products

**Registered as `kronecker`**  
*Replaces standard linear layers with a Kronecker product decomposition, drastically reducing parameters by exploiting factorisable dimensions.*

---

## Overview

The Kronecker Product Compressor compresses a model by replacing every `nn.Linear` layer with a `KroneckerLinear` module—a drop‑in replacement that approximates the original weight matrix **W** ∈ ℝ<sup>m×n</sup> as the **Kronecker product** of two smaller matrices **A** and **B**:

```
W ≈ A ⊗ B
```

Here, **A** ∈ ℝ<sup>m₁×n₁</sup> and **B** ∈ ℝ<sup>m₂×n₂</sup> are chosen such that `m = m₁·m₂` and `n = n₁·n₂`. The Kronecker product reduces the number of parameters from `m·n` to `m₁·n₁ + m₂·n₂`, which can be orders of magnitude smaller when the dimensions have non‑trivial factorisations. This approach is especially effective for layers with power‑of‑two or highly composite shapes (e.g., 768 = 2⁸·3, which allows many factorisations). If the dimensions cannot be cleanly factorised, or the Kronecker decomposition is not feasible, the compressor falls back to a low‑rank SVD approximation (inside `KroneckerLinear.from_linear`).

The compressor is registered under `"kronecker"` and is extremely simple to use: it requires no configuration parameters, no calibration data, and automatically processes all linear layers in a model.

---

## Methodology

### 1. Kronecker Product Decomposition

The Kronecker product **A ⊗ B** is defined as the block matrix:

```
A ⊗ B = [ a₁₁·B   a₁₂·B   …   a₁ₙ₁·B ]
        [ a₂₁·B   a₂₂·B   …   a₂ₙ₁·B ]
        [   …       …     …      …    ]
        [ aₘ₁·B   aₘ₂·B   …   aₘₙ₁·B ]
```

If **W** can be exactly expressed as **A ⊗ B**, then the linear transformation `y = x·W^T` can be computed efficiently using the identity:

```
x · (A ⊗ B)^T = B · X · A^T
```

where **X** is a reshaped version of **x**. This leads to two smaller matrix multiplications, reducing both parameters and FLOPs.

In practice, exact Kronecker factorisation is rare; the goal is to find **A** and **B** that minimise `‖W – A ⊗ B‖_F`. This can be done by:

- Reshaping **W** into a tensor of shape `(m₁, m₂, n₁, n₂)`.
- Transposing to `(m₁, n₁, m₂, n₂)` and flattening the first and last two dimensions to obtain a matrix of size `(m₁·n₁) × (m₂·n₂)`.
- Applying SVD to this matrix to find a low‑rank approximation, then splitting the resulting factors back into **A** and **B**.

The actual implementation is handled by `KroneckerLinear.from_linear(module)`, which attempts to find a suitable factorisation; if the dimensions do not allow a good Kronecker split (e.g., primes), it falls back to a standard low‑rank SVD decomposition.

### 2. Parameter Reduction

If **W** has dimensions `m × n` and is factorised as `A ∈ ℝ^{m₁×n₁}, B ∈ ℝ^{m₂×n₂}` with `m = m₁·m₂, n = n₁·n₂`, then the total parameter count for the two small matrices is `m₁·n₁ + m₂·n₂`. For a square 768×768 matrix, factorising as `m₁=n₁=8` and `m₂=n₂=96` gives `8·8 + 96·96 = 64 + 9216 = 9280` parameters, vs the original 589,824—a ~63× reduction. Even with moderate factor sizes, the savings are significant.

### 3. Forward Pass in KroneckerLinear

The `KroneckerLinear` module stores the two small matrices **A** and **B** (and optionally a bias). In the forward pass, the computation is performed via the Kronecker product identity, avoiding the materialisation of the full weight matrix. This yields both parameter storage reduction and computational speed‑up, especially for large layers.

---

## Implementation Details

### KroneckerCompressor Class

```python
class KroneckerCompressor(BaseCompressor):
    def __init__(self, model, **kwargs):
        super().__init__(model, name="kronecker", **kwargs)
```

The compressor takes no extra hyperparameters. The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. For each, creates a `KroneckerLinear.from_linear(module)`.
3. Replaces the original module with the new one via `setattr` on the parent.
4. After processing, updates the compressed parameter count.

**Note**: The replacement is unconditional; every linear layer is converted. If a layer’s dimensions cannot be factorised nicely, the fallback inside `KroneckerLinear.from_linear` will provide an alternative (likely low‑rank SVD). The compressor itself does not need to handle special cases.

### KroneckerLinear (from `.kronecker_linear`)

Although the full code is not shown, we can outline its expected behaviour:

- **`from_linear(linear)`**: Analyses the shape `(out_features, in_features)` and tries to find a factorisation `out = m₁*m₂, in = n₁*n₂` that minimises the Kronecker product error or falls back to a rank‑determined SVD. This method returns a `KroneckerLinear` instance with `A` and `B` as parameters or buffers.
- **`forward(x)`**: Computes the linear transformation using the Kronecker identity for efficiency.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("kronecker")
def kronecker_factory(model, comp_config):
    compressor = KroneckerCompressor(model)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

No configuration parameters are needed; the compressor works out‑of‑the‑box.

---

## Usage and Configuration

To apply Kronecker compression, simply set the compression method to `"kronecker"`:

```python
comp_config.comp_method = "kronecker"
```

Then run the compression pipeline. The model will be transformed with Kronecker‑factorised linear layers.

### Parameters

No extra parameters are required. The compressor is fully automatic.

---

## Discussion

### Advantages

- **Massive parameter reduction**: Kronecker product can compress square matrices by 10–100× when dimensions have good factorisations.
- **Computational speed‑up**: The forward pass uses two smaller matrix multiplies, reducing total FLOPs.
- **No calibration data**: Works solely on the weight matrix structure.
- **Seamless drop‑in**: The replacement layers expose the same `forward` signature; the rest of the model code remains unchanged.
- **Automatic fallback**: If Kronecker factorisation is not suitable, the internal logic resorts to low‑rank SVD, ensuring some compression is always achieved.

### Limitations

- **Depends on factorisable dimensions**: For prime dimensions (e.g., 769), the Kronecker product may be forced to use factors like 1×769, which offers no compression. The fallback SVD then takes over, but that yields only a moderate rank reduction.
- **Fallback quality unknown**: The details of the `from_linear` method (not shown) determine the actual compression and accuracy trade‑off. If the fallback is a fixed‑rank SVD, it might be suboptimal for some layers.
- **May require fine‑tuning**: The approximation error can degrade performance; fine‑tuning is often needed.
- **Implementation dependency**: The compressor’s behaviour entirely depends on the `KroneckerLinear` implementation; any bugs or limitations there will propagate.

### Comparison with Other Low‑Rank Methods

| Method              | Factorisation       | Dimension Requirement     | Parameter Reduction Mechanism |
|---------------------|---------------------|----------------------------|-------------------------------|
| Kronecker (this)    | A ⊗ B               | Needs composite dimensions | Kronecker product factorisation |
| SVD (A3)            | U Σ V^T             | None                       | Truncated SVD                 |
| CUR                 | C @ U @ R           | None                       | Column/row selection          |
| HashedNet           | Hash table          | None                       | Weight sharing                |

Kronecker compression is unique in its exploitation of Kronecker structure, which can yield extremely compact representations when the dimensions are amenable, but its performance degrades to generic low‑rank for other shapes.

---

## Conclusion

The Kronecker Product Compressor (registered as `"kronecker"`) offers a powerful, fully automatic method to compress linear layers by decomposing weight matrices into Kronecker products of two smaller matrices. Its simplicity—zero configuration, no calibration—makes it an attractive first choice for models with highly composite hidden sizes. While the effectiveness depends on the underlying `KroneckerLinear` implementation and the factorisability of dimensions, the approach promises substantial reductions in both parameters and computation, with the flexibility to fall back to standard low‑rank compression when necessary. This compressor exemplifies the kind of mathematical structure that can be leveraged for efficient transformer deployment.