# Kronecker Product Linear Layer: Parameter‑Efficient Factorization via A ⊗ B

**Standalone module**  
*Approximates the weight matrix of a linear layer as the Kronecker product of two smaller matrices, achieving drastic parameter reduction when the dimensions factorise favourably.*

---

## Abstract

The `KroneckerLinear` module replaces a standard dense linear layer with a low‑parameter factorisation of its weight matrix **W** ∈ ℝ<sup>m×n</sup> as the **Kronecker product** of two smaller matrices **A** ∈ ℝ<sup>m₁×n₁</sup> and **B** ∈ ℝ<sup>m₂×n₂</sup> where `m = m₁·m₂` and `n = n₁·n₂`. When such a factorisation is possible (i.e., the dimensions are composite), the number of trainable parameters drops from `m·n` to `m₁·n₁ + m₂·n₂`. For a typical 768×768 layer this can yield a ∼60× reduction. The module automatically attempts to factorise the dimensions using a heuristic, and for cases where exact factorisation fails, it falls back to a low‑rank SVD approximation with a comparable parameter budget. The forward pass reconstructs the full weight via the Kronecker product or, in the fallback, via a matrix product, maintaining the same functional interface as `nn.Linear`. This documentation provides the full mathematical foundation, the nearest Kronecker product algorithm used for weight initialisation, the forward computation, and usage notes.

---

## 1. Introduction

Neural network compression techniques based on matrix factorisation can substantially reduce the memory and computational footprint of linear layers. Among these, the **Kronecker product decomposition** is particularly attractive when the layer dimensions exhibit a non‑trivial factorisation. If the weight matrix can be expressed as **A ⊗ B**, the linear transformation can be performed without ever materialising the full matrix, reducing both parameters and FLOPs. This module implements a drop‑in replacement for `nn.Linear` that automatically tries to factorise the weight into a Kronecker product, using a classical algorithm to recover the optimal factors from a pretrained weight. When exact Kronecker structure is absent, it employs a truncated SVD to achieve a similar degree of compression.

---

## 2. Methodology

### 2.1 Kronecker Product Decomposition

The **Kronecker product** of two matrices **A** ∈ ℝ<sup>p×q</sup> and **B** ∈ ℝ<sup>r×s</sup> is the block matrix

```
A ⊗ B = [ a₁₁·B   a₁₂·B   …   a₁q·B ]
        [ a₂₁·B   a₂₂·B   …   a₂q·B ]
        [   …       …     …     …    ]
        [ aₚ₁·B   aₚ₂·B   …   aₚq·B ]
```

If the dimensions of **W** can be written as `m = m₁·m₂` and `n = n₁·n₂`, we aim to find **A** ∈ ℝ<sup>m₁×n₁</sup> and **B** ∈ ℝ<sup>m₂×n₂</sup> such that

```
W ≈ A ⊗ B .
```

The total number of parameters then becomes `m₁·n₁ + m₂·n₂` instead of `(m₁·m₂)·(n₁·n₂)`.

### 2.2 Dimension Factorisation

The function `factorize_dim(dim)` finds a pair of factors `(a, b)` with `a·b = dim` and `a` as close as possible to √dim. This yields near‑square factors that balance the sizes of **A** and **B**. The module uses these factors to define the shapes of **A** and **B**:

```python
in_a, in_b = factorize_dim(in_features)
out_a, out_b = factorize_dim(out_features)
```

If `in_a * in_b == in_features` and `out_a * out_b == out_features`, the layer can be represented exactly as a Kronecker product; otherwise, the fallback is activated.

### 2.3 Nearest Kronecker Product Approximation (Weight Initialisation)

Given a pretrained weight matrix **W** from a standard `nn.Linear` layer, the class method `from_linear` computes the optimal **A** and **B** (in the Frobenius norm) that minimise `‖W – A ⊗ B‖_F`. This is a known problem, and the solution can be obtained via a **rank‑1 SVD** of a permuted version of **W**.

The algorithm:

1. Reshape **W** to a 4D tensor of shape `(out_a, out_b, in_a, in_b)`.
2. Permute the axes to `(out_a, in_a, out_b, in_b)` and flatten the first two and last two dimensions, yielding a matrix `W_unfold` of size `(out_a·in_a, out_b·in_b)`.
3. Perform SVD on `W_unfold`, keeping only the first singular value and the corresponding left and right singular vectors:
   ```
   U, S, V^T = svd(W_unfold)
   u₁ = U[:, 0]   ∈ ℝ^{out_a·in_a}
   v₁ = V[0, :]   ∈ ℝ^{out_b·in_b}
   σ₁ = S[0]
   ```
4. The rank‑1 approximation `σ₁ u₁ v₁^T` is the best Kronecker factorisation of the unfolded matrix. Reshape `u₁` into `(out_a, in_a)` to obtain **A**, and `v₁` into `(out_b, in_b)` to obtain **B**. (The scalar σ₁ can be absorbed arbitrarily into either factor; the implementation simply uses `u₁` and `v₁` as the factors.)
5. Set the module’s `A` and `B` parameters to these reshaped vectors.

This procedure yields the exact nearest Kronecker product (up to scaling) for the given factorisation; if **W** is exactly a Kronecker product, the approximation error will be zero (except for the truncation of the singular value if multiple components exist, but the rank‑1 assumption forces a Kronecker product form).

### 2.4 Fallback to Low‑Rank SVD

When the dimensions cannot be factorised exactly (e.g., one of the features is prime), the module resorts to a standard truncated SVD. It chooses a rank `r = max(1, min(in_features, out_features) // 4)` to achieve a roughly comparable parameter reduction. The weight is decomposed as:

```
W ≈ U @ V^T ,   U ∈ ℝ^{out×r}, V ∈ ℝ^{r×in}
```

with parameter count `r·(out + in)`. Initialisation in this case uses the SVD of the original weight.

### 2.5 Forward Computation

#### Kronecker product path (`can_decompose = True`)

The forward pass reconstructs the full weight matrix explicitly via `torch.kron(A, B)` and then calls `F.linear(x, W, bias)`. This is simple but materialises a large matrix, so it does **not** reduce computational cost or peak memory during inference. A more efficient implementation would use the identity

```
(A ⊗ B) x = vec( B · X · A^T )
```

where **x** is reshaped to `(in_b, in_a)` (for a single sample) and the result is reshaped back, thereby avoiding the large Kronecker product entirely. The current code is designed for clarity and moderate dimensions; future optimisations could incorporate this technique.

#### SVD fallback path (`can_decompose = False`)

The weight is formed by `U @ V` and the linear operation applied normally. Again, this materialises the full weight, so no FLOP reduction is achieved. However, storing **U** and **V** instead of **W** still reduces the model’s parameter footprint.

---

## 3. Implementation Details

### `KroneckerLinear` Class

```python
class KroneckerLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32, device=None):
        ...
```

**Key attributes when `can_decompose`**:
- `self.A`: `nn.Parameter` of shape `(out_a, in_a)`
- `self.B`: `nn.Parameter` of shape `(out_b, in_b)`
- `self.bias`: optional parameter

**Key attributes in fallback**:
- `self.U`, `self.V`: low‑rank factors
- `self.rank`: determined as `max(1, min(in, out)//4)`

**`from_linear(cls, linear)`**: initialises a `KroneckerLinear` from a pretrained `nn.Linear`. If the dimensions factorise, it computes the nearest Kronecker product via the SVD of the unfolded matrix; otherwise, it performs a truncated SVD of the weight.

**`forward(x)`**: computes the full weight matrix and applies `F.linear`.

### `KroneckerCompressor` (registered as `"kronecker"`)

The compressor simply replaces every `nn.Linear` in the model with a `KroneckerLinear.from_linear(module)`. It requires no hyperparameters and is registered under `"kronecker"`.

---

## 4. Usage and Configuration

The `KroneckerLinear` can be used standalone or via the compressor.

### Standalone usage

```python
from kronecker_linear import KroneckerLinear

# Create from an existing layer
original = nn.Linear(768, 768)
kron = KroneckerLinear.from_linear(original)
output = kron(input_tensor)
```

### Through the compression framework

Set `comp_config.comp_method = "kronecker"` and run the pipeline. No additional parameters are needed.

### Parameters (for the compressor)

None – the compressor is fully automatic.

**Important**: After compression, fine‑tuning is strongly recommended to recover accuracy, as the Kronecker approximation (or SVD fallback) may degrade performance.

---

## 5. Discussion

### Advantages

- **Massive parameter reduction**: For factorisable dimensions, the number of parameters drops dramatically (e.g., 768 = 8 × 96 → A:8×8, B:96×96 → 64+9216 = 9280 params vs. 589,824).
- **Simple, training‑free conversion**: `from_linear` provides an immediate approximation without any data.
- **Fallback ensures coverage**: When Kronecker is not possible, SVD still provides some compression.
- **Easy integration**: The module behaves exactly like `nn.Linear`.

### Limitations

- **No computational speed‑up**: The forward pass materialises the full weight matrix, so FLOPs are unchanged; only parameter storage is reduced. To gain inference acceleration, the forward method should be rewritten to use the Kronecker product identity (for the decomposable case) or sequential low‑rank multiplication (for the SVD fallback).
- **Performance loss**: The nearest Kronecker product is a very restricted approximation; many weight matrices do not factorise well, leading to a larger reconstruction error.
- **Inflexible factorisation**: The heuristic `factorize_dim` always chooses the closest‑to‑square factorisation, which may not be optimal for all architectures. A more flexible approach could search over factor pairs to minimise reconstruction error.
- **Rank‑1 assumption**: The nearest Kronecker product method keeps only a single singular vector, discarding all other components. If the unfolded matrix has significant energy in higher singular values, the approximation will be poor.
- **Fallback rank choice**: The arbitrary `rank = min(in, out)//4` may be too small or too large; an energy‑based criterion would be more robust.

### Comparison with Other Factorisations

| Method                | Factorisation Type    | Parameter Count           | Inference Speed‑up (naïve) |
|-----------------------|-----------------------|---------------------------|----------------------------|
| Kronecker (this)      | A ⊗ B                | `m₁n₁ + m₂n₂`            | No (full matrix built)     |
| Low‑rank SVD (A3)     | U Σ V^T              | `r·(m+n)`                 | No (weight overwritten)    |
| CUR                   | C @ U @ R            | `r·(m+n) + r²`           | Yes (sequential multiply)  |

Kronecker is uniquely suited for dimensions with a clear Kronecker structure, offering extreme compression ratios at the cost of a larger approximation error.

---

## 6. Conclusion

The `KroneckerLinear` module provides a ready‑to‑use drop‑in replacement for linear layers that compresses the weight matrix via a Kronecker product factorisation. Its automatic dimension splitting, nearest Kronecker product initialisation, and SVD fallback make it a versatile tool for parameter reduction. While the current forward pass does not reduce inference time, the module’s compact representation is valuable for memory‑constrained environments, and it can serve as a foundation for further optimisations (e.g., efficient Kronecker‑based multiplication). The accompanying `KroneckerCompressor` integrates seamlessly into the DistilBERT compression framework, enabling one‑click compression of existing models.