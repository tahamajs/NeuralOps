# CURLinear: Drop‑In Replacement for `nn.Linear` via CUR Decomposition

**Standalone module**  
*Factorises a linear layer’s weight matrix into actual columns and rows, reducing parameter count while preserving the original forward interface.*

---

## Overview

`CURLinear` is a `nn.Module` that directly replaces a standard `nn.Linear` layer by decomposing its weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup> into three smaller matrices:

```
W ≈ C · U · R
```

where:
- **C** ∈ ℝ<sup>d_out × r</sup> contains `r` carefully selected **columns** of **W**,
- **R** ∈ ℝ<sup>r × d_in</sup> contains `r` carefully selected **rows** of **W**,
- **U** ∈ ℝ<sup>r × r</sup> is a small square “core” matrix that links the two,

and `r` is a user‑specified rank (≤ min(d_out, d_in)). This is known as the **CUR decomposition** (or pseudo‑skeleton approximation). Because **C** and **R** are formed from actual data vectors of the original weight, the factorization is highly interpretable.

The module stores only **C**, **U**, and **R**, drastically reducing the number of parameters from `d_out·d_in` to `r·(d_out + d_in + r)`. During the forward pass, the full weight is reconstructed on‑the‑fly as `W_approx = C @ U @ R`, and the linear transformation proceeds as usual. This approach is particularly attractive when memory footprint is the primary concern, and the layer can remain compatible with any downstream code expecting an `nn.Linear`-like object.

The class can be instantiated directly from an existing `nn.Linear` using the `from_linear` class method (or the constructor given the linear layer). The column/row selection uses **leverage score sampling** (norm‑based probabilities) by default, providing a near‑optimal low‑rank approximation in practice.

---

## Methodology

### 1. CUR Decomposition

For a matrix **W** of size `m × n`, a rank‑`r` CUR decomposition writes:

```
W ≈ C · U · R
```

- **C** = `W[:, cols]` – an `m × r` matrix formed by picking `r` columns of **W**.
- **R** = `W[rows, :]` – an `r × n` matrix formed by picking `r` rows of **W**.
- **U** = `C⁺ · W · R⁺` – an `r × r` core matrix, where `C⁺` and `R⁺` are pseudo‑inverses.

This decomposition is exact if **W** has rank ≤ `r` and the chosen columns and rows span the full column and row spaces. In the low‑rank approximation setting, the quality depends heavily on which columns/rows are selected.

### 2. Leverage Score Sampling

To select the most informative columns and rows, we use **leverage scores**—the squared Euclidean norms of the columns/rows. The sampling probabilities are proportional to these norms:

- **Column probability**: `p_j = ‖W[:, j]‖² / Σ_k ‖W[:, k]‖²`
- **Row probability**: `q_i = ‖W[i, :]‖² / Σ_k ‖W[k, :]‖²`

We then sample `r` columns and `r` rows **without replacement** according to these distributions. Columns/rows with larger norms (i.e., larger “energy”) are more likely to be selected, leading to a better approximation of the original matrix.

If `method='uniform'`, all columns/rows are equally likely, which is simpler but typically less accurate.

### 3. Core Matrix Computation

After selecting **C** and **R**, the optimal core **U** that minimizes `‖W – C·U·R‖_F` is:

```
U = C⁺ · W · R⁺
```

where `C⁺` is the Moore‑Penrose pseudo‑inverse (computed via `torch.linalg.pinv`) and similarly for `R⁺`. This gives the best least‑squares reconstruction for the chosen column and row indices.

### 4. Forward Pass

The `forward` method computes:

```python
W_approx = self.C @ self.U @ self.R   # shape (out_features, in_features)
y = x @ W_approx.T                     # standard linear transform
if self.bias is not None:
    y = y + self.bias
```

Thus, the layer behaves exactly like the original `nn.Linear`—it accepts a tensor of shape `(*, in_features)` and outputs `(*, out_features)`—but its weight matrix is not stored as a dense `d_out × d_in` tensor. Instead, it is recomputed at each forward pass from the three small matrices. While this reconstruction costs `O(r·d_out·d_in + r²·(d_out+d_in))` operations, the overall forward computation remains `O(b·d_out·d_in)` because the final matrix multiply dominates. Therefore, the primary benefit is **storage reduction**, not inference speed‑up (unless the matrix multiply is replaced by a sequential low‑rank product, as in the full CUR Compressor described elsewhere).

---

## Implementation Details

### Constructor

```python
class CURLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, method: str = 'leverage'):
```

**Arguments**:
- `linear`: an existing `nn.Linear` layer whose weight and bias will be used.
- `rank`: desired rank `r`. The actual rank used is `min(rank, in_features, out_features)`.
- `method`: `'leverage'` (default) for norm‑based sampling, `'uniform'` for equal probability.

The constructor performs the following steps:

1. Extracts the weight matrix **W** (converts to float32, moves to CPU).
2. Determines effective rank `r`.
3. Samples columns:
   - Computes column norms, normalises to probabilities.
   - Samples `r` indices without replacement.
   - Builds **C** = `W[:, cols]`.
4. Samples rows similarly → **R** = `W[rows, :]`.
5. Computes core **U** = `pinv(C) @ W @ pinv(R)` using PyTorch’s `linalg.pinv`.
6. Registers **C**, **U**, **R** as buffers (non‑trainable, but part of the module’s state).
7. Handles bias: if the original had a bias, it is copied as a trainable parameter; otherwise, no bias.

### Buffers and Parameters

- `self.C` (buffer): `(out_features, rank)`
- `self.U` (buffer): `(rank, rank)`
- `self.R` (buffer): `(rank, in_features)`
- `self.bias` (parameter or None)

All buffers are stored in float32 on the model’s device. The original weight is discarded.

### Forward Method

Reconstructs `W_approx` each call. Note that this is not the most efficient way to use CUR for inference; a production implementation would compute `y = x @ R^T @ U^T @ C^T` sequentially to reduce FLOPs, but the current design keeps the interface identical to `nn.Linear` and relies on the same dense matrix multiplication, merely re‑materializing the weight. This is a trade‑off: simplicity and compatibility over maximum speed.

---

## Usage and Configuration

`CURLinear` can be used standalone or as part of the `CURCompressor` pipeline. To directly replace an `nn.Linear`:

```python
import torch.nn as nn
from curlinear import CURLinear

# Original layer
linear = nn.Linear(768, 768)

# Replace with rank‑64 CUR approximation
cur_layer = CURLinear(linear, rank=64, method='leverage')

# Now use like a normal Linear
x = torch.randn(32, 768)
y = cur_layer(x)   # same shape as before
```

For a full model compression, the `CURCompressor` (registered as `"cur"`) automates this replacement across all linear layers.

### Choosing the Rank

- The parameter compression ratio is roughly `r·(d_out+d_in+r) / (d_out·d_in)`. For a typical 768×768 matrix, rank 64 yields about 13% of the original parameters.
- Higher ranks preserve more accuracy but reduce compression.
- Rank can be chosen based on a validation metric or using an energy threshold on the singular values (not built into this class).

### Determinism

The sampling uses `np.random.choice` without an explicit seed. For reproducibility, set the NumPy random seed before instantiation:

```python
np.random.seed(42)
cur_layer = CURLinear(linear, rank=64)
```

---

## Discussion

### Advantages

- **Significant parameter reduction**: The storage footprint of the weight drops dramatically, enabling deployment on memory‑constrained devices.
- **Interpretable factors**: **C** and **R** are actual columns/rows of the original weight, which can be inspected to understand which input features or output neurons are retained.
- **Drop‑in compatibility**: The module subclasses `nn.Linear`‑like interface (though it does not inherit from `nn.Linear` directly, it mimics its `forward` signature). It can replace any linear layer without changing the surrounding code.
- **No training data required**: The decomposition uses only the weight matrix.
- **Leverage score sampling**: Provides a principled, near‑optimal column/row selection that works well in practice.

### Limitations

- **No inference speed‑up**: The forward pass still performs a dense matrix multiplication of size `(batch, in) × (in, out)`, because `W_approx` is reconstructed as a full tensor. To actually reduce FLOPs, the computation must be restructured as `x @ R^T @ U^T @ C^T`, which is not done here. Thus, this module is primarily for **storage compression**, not latency reduction.
- **Non‑deterministic**: Random sampling leads to slightly different approximations each run. For critical deployments, one may want to evaluate multiple runs and pick the best, or use a deterministic selection method.
- **Rank must be specified manually**: There is no automatic rank selection based on energy or accuracy constraints.
- **Fixed approximation**: Once created, the factors are frozen; there is no built‑in fine‑tuning mechanism. However, one could set the buffers as parameters and train further.
- **Pseudo‑inverse computation**: For large `r`, `torch.linalg.pinv` can be numerically unstable or slow, though `r` is typically small.

### Comparison with Direct SVD‑based Low‑Rank

| Aspect                  | CURLinear (this)                     | SVD‑based low‑rank (e.g., two smaller layers) |
|-------------------------|--------------------------------------|-----------------------------------------------|
| Factors                 | Actual columns/rows of W             | Abstract orthogonal bases                     |
| Interpretability        | High                                 | Low                                           |
| Parameter count         | `r·(d_out+d_in+r)`                  | `r·(d_out+d_in)` (two matrices)              |
| Reconstruction overhead | Reconstructs dense W each forward    | No reconstruction; sequential low‑rank multiply |
| Inference speed‑up      | None (dense matmul)                  | Yes (sequential low‑rank)                     |
| Storage only            | Yes                                  | Yes (if stored as two smaller layers)         |

If inference speed is important, the sequential low‑rank multiplication should be used (as in the full `CURCompressor` that creates a custom forward). `CURLinear` is a simpler, storage‑focused alternative that keeps the dense matmul.

---

## Conclusion

`CURLinear` provides an easy‑to‑use, drop‑in replacement for `nn.Linear` that compresses the weight matrix using CUR decomposition and leverage score sampling. It dramatically reduces the number of stored parameters while preserving the original layer’s interface, making it ideal for shrinking model checkpoints before storage or transmission. While it does not accelerate inference, its simplicity and interpretability make it a valuable tool in the model compression ecosystem. For a full integration that also reduces computation, the accompanying `CURCompressor` (which uses a more efficient forward) is recommended.