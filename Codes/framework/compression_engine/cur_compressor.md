# CUR Compressor: Column‑Row Low‑Rank Decomposition with Leverage Score Sampling

**Registered as `cur`**  
*Replaces linear layers with a factored CUR representation, reducing parameters and FLOPs.*

---

## Overview

The CUR Compressor decomposes each selected `nn.Linear` layer into a product of three matrices: **C**, **U**, and **R**, such that:

```
W ≈ C · U · R
```

where:
- **C** ∈ ℝ<sup>d_out × r</sup> contains **r actual columns** of the original weight matrix **W**,
- **R** ∈ ℝ<sup>r × d_in</sup> contains **r actual rows** of **W**,
- **U** ∈ ℝ<sup>r × r</sup> is a small “core” matrix that connects them, computed via pseudo‑inverses.

This factorization is known as **CUR decomposition** (or pseudo‑skeleton approximation). Unlike SVD, which produces abstract basis vectors, CUR uses real data vectors from the original matrix. The columns and rows are selected using **leverage score sampling**, which prioritises vectors with larger norms, capturing the most important variation in the weight matrix.

The module replaces the original `nn.Linear` layer with a custom `CURLinear` layer that stores **C**, **U**, and **R** as buffers and performs the forward pass efficiently as:

```
y = ((x · R^T) · U^T) · C^T + bias
```

Because `r` is much smaller than `d_in` and `d_out`, this approach reduces both the parameter count (from `d_out·d_in` to `r·(d_out + d_in + r)`) and the computational cost of the matrix multiplication.

The compressor is registered under `"cur"` and can be applied to all linear layers or filtered by name. It requires no calibration data, relying purely on the weight matrix for column/row selection.

---

## Methodology

### 1. CUR Decomposition

For a matrix **W** ∈ ℝ<sup>m×n</sup>, a rank‑`r` CUR decomposition approximates it as:

```
W ≈ C · U · R
```

- **C** = `W[:, cols]` : an `m × r` matrix formed by selecting `r` columns of **W**.
- **R** = `W[rows, :]` : an `r × n` matrix formed by selecting `r` rows of **W**.
- **U** = **C⁺ · W · R⁺** : the `r × r` core matrix, where `·⁺` denotes the Moore‑Penrose pseudo‑inverse.

This decomposition is exact if **W** has rank ≤ `r` and the selected columns and rows span the row and column spaces. In practice, for a low‑rank approximation, the selection of columns and rows is critical.

### 2. Leverage Score Sampling

To pick the most informative columns and rows, we use **leverage scores** (statistical leverage). The leverage score of the `i`‑th column (or row) is proportional to the squared Euclidean norm of that column (or row). These scores indicate how much influence a column has on the matrix.

Formally:

- **Column sampling probability**:  
  `p_j = ‖W[:, j]‖² / ∑_k ‖W[:, k]‖²`
- **Row sampling probability**:  
  `q_i = ‖W[i, :]‖² / ∑_k ‖W[k, :]‖²`

We then sample `r` columns (and rows) **without replacement** according to these probabilities. Columns/rows with larger norms are more likely to be selected, leading to a better low‑rank approximation for a given rank. This is known to produce near‑optimal CUR decompositions under the Frobenius norm.

If `method='uniform'`, all columns/rows have equal probability, which is simpler but usually less accurate.

### 3. Core Matrix Computation

Once **C** and **R** are chosen, the core **U** is solved to minimize `‖W – C·U·R‖_F`. The optimal **U** is:

```
U = C⁺ · W · R⁺
```

where `C⁺ = (C^T C)^{-1} C^T` and `R⁺ = R^T (R R^T)^{-1}` are the pseudo‑inverses. This ensures that the product **C·U·R** is the best approximation of **W** for the given column and row selections.

### 4. Forward Computation in CURLinear

Given an input **x** ∈ ℝ<sup>b×n</sup>, the original linear layer computes `y = x · W^T + bias`. Using the CUR factorization:

```
y = x · (C · U · R)^T + bias
  = x · (R^T · U^T · C^T) + bias
```

To avoid recomputing the product `R^T · U^T · C^T` as a dense matrix (which would be `n×m` and defeat the purpose), we compute sequentially:

1. **xR** = `x @ R^T`     → shape `(b, r)`
2. **xRU** = `xR @ U^T`   → shape `(b, r)`
3. **out** = `xRU @ C^T`  → shape `(b, m)`

This requires only `b·n·r + b·r² + b·r·m` operations, which is significantly cheaper than the original `b·n·m` when `r` is small.

The layer stores **C**, **U**, **R** as non‑trainable buffers (registered with `register_buffer`), meaning they persist on the device but are not updated during fine‑tuning. The bias is kept as a trainable parameter, preserving the original bias if present.

### 5. Replacement Logic

The compressor iterates over all `nn.Linear` modules. For each:
- It computes the maximum possible rank `max_rank = min(in_features, out_features)`.
- The effective rank `r = min(self.rank, max_rank)`.
- If `r >= max_rank`, no compression is applied (the layer is left as is, since no reduction would be achieved).
- Otherwise, `CURLinear.from_linear(module, r, method)` is called to create the replacement.
- The parent module is located, and the child is replaced via `setattr`.

A filter `target_names` can restrict compression to layers whose names contain specific substrings (e.g., `'attention'`). This allows selective compression.

---

## Implementation Details

### CURLinear Class

```python
class CURLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=None, method='leverage', device=None):
        ...
```

- **Buffers**: `C` (out × r), `U` (r × r), `R` (r × in) are registered as buffers (non‑trainable).
- **Bias**: If provided, stored as `nn.Parameter`.
- **Forward**: Performs the three‑step low‑rank multiplication as described.

**`from_linear(cls, linear, rank, method)`**:
1. Extracts weight **W** (float32, CPU).
2. Samples columns and rows using leverage scores (or uniform).
3. Builds **C**, **R**, computes **U** = `C⁺ @ W @ R⁺`.
4. Instantiates `CURLinear` and copies the matrices to buffers.
5. Returns the new layer, moved to the original device.

The use of `np.random.choice` makes the sampling non‑deterministic; the approximation can vary across runs. A seed could be set externally for reproducibility.

### CURCompressor Class

Inherits from `BaseCompressor`.

```python
class CURCompressor(BaseCompressor):
    def __init__(self, model, rank=64, method='leverage', target_names=None, **kwargs):
        ...
```

- **`rank`**: Base rank (default 64). Adjusted per layer if `max_rank` is smaller.
- **`method`**: `'leverage'` or `'uniform'`.
- **`target_names`**: List of substrings to filter layers (e.g., `['attn', 'mlp']`). If empty, all Linear layers are compressed.

`compress()`:
- Loops over named modules, filters by type `nn.Linear` and optional `target_names`.
- Skips layers where `r >= max_rank`.
- Replaces with `CURLinear`; errors are caught and printed.
- Updates parameter count using `_count_params` (which now should reflect the reduced parameters in CURLinear).
- Returns the model and stats.

**`_get_parent_module`** static method navigates the model hierarchy to replace the child layer.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("cur")
def cur_compressor_factory(model, comp_config):
    rank = getattr(comp_config, 'cur_rank', 64)
    method = getattr(comp_config, 'cur_method', 'leverage')
    target_names = getattr(comp_config, 'cur_target_names', None)
    compressor = CURCompressor(model, rank=rank, method=method, target_names=target_names)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Configuration parameters:
- `cur_rank`
- `cur_method`
- `cur_target_names` (optional list)

The factory returns the compressed model directly.

---

## Usage and Configuration

### Example

```python
comp_config.comp_method = "cur"
comp_config.cur_rank = 32
comp_config.cur_method = "leverage"        # or "uniform"
comp_config.cur_target_names = ["attention", "mlp"]  # optional filter
```

Then pass to the compression pipeline. The model’s targeted linear layers will be replaced with `CURLinear` instances.

### Parameters

| Parameter            | Default     | Description                                                                                                         |
|----------------------|-------------|---------------------------------------------------------------------------------------------------------------------|
| `cur_rank`           | `64`        | Desired rank for the CUR decomposition. Lower = more compression, higher accuracy.                                   |
| `cur_method`         | `"leverage"` | Sampling method: `"leverage"` (norm‑based probabilities) or `"uniform"`.                                            |
| `cur_target_names`   | `None`      | List of substrings to match layer names (e.g., `['attn', 'q_proj']`). If `None` or empty, compress all Linear layers.|

**Note**: The randomness in column/row selection means the compression result is non‑deterministic unless a global random seed is set. This stochasticity can actually be beneficial if multiple compressions are performed and the best is chosen, but for reproducibility, fix the seed.

---

## Discussion

### Advantages

- **Real‑data basis**: The factors consist of actual columns and rows of the original weight matrix, which can be more interpretable than SVD factors.
- **Parameter & FLOP reduction**: The number of parameters drops from `d_out·d_in` to `r·(d_out + d_in + r)`, and the forward FLOPs are reduced from `b·d_out·d_in` to `b·r·(d_in + d_out + r)`. For moderate `r`, this yields substantial speed‑up.
- **Leverage score sampling**: Statistically sound column/row selection leads to approximations close to the optimal truncated SVD in practice, with the added benefit of sparsity in the selection (only `2r` columns/rows need to be stored from the original matrix, but here we store the full factors).
- **No calibration data**: Works purely on the weight matrix.
- **Selective compression**: The `target_names` filter allows focusing on the most parameter‑heavy layers (e.g., attention projections) while leaving others untouched.

### Limitations

- **Non‑deterministic**: The random sampling introduces variability; for critical applications, one may need to run multiple times and select the best (by evaluating a metric).
- **Fixed rank per layer**: All selected layers use the same `cur_rank` (or capped by min dimension). A per‑layer adaptive rank (based on singular value decay) could improve the accuracy/size trade‑off.
- **No fine‑tuning**: The CUR layers are not trained after replacement; the weights are frozen. Fine‑tuning the core **U** (and optionally **C**, **R**) could recover lost accuracy.
- **Replacement overhead**: The replacement creates a new `CURLinear` object; the original `nn.Linear` is discarded. This is fine for storage, but during inference, the three‑step multiplication may have a slight kernel launch overhead compared to a single large matrix multiply, especially for very small ranks. However, the overall reduction in total operations usually dominates.
- **Bias is kept as parameter**: If bias is large, it remains in full precision; quantizing it could be added separately.
- **Leverage sampling assumes non‑negative norms**: For weight matrices with all‑zero columns, the probability computation needs a small epsilon (already present).

### Comparison with Other Low‑Rank Compressors

| Feature                | CUR (this)                 | SVD (A3/CR)              | Adaptive SVD            |
|------------------------|----------------------------|--------------------------|--------------------------|
| Basis origin           | Actual columns/rows of W   | Abstract singular vectors| Abstract singular vectors|
| Factorisation format   | C, U, R                    | U, Σ, V                  | Two small linear layers  |
| Parameter reduction    | Yes (buffers)              | No (overwrites weight)   | Yes (low‑rank layers)    |
| Inference speed‑up     | Yes (sequential low‑rank)  | No                       | Yes                      |
| Sampling               | Leverage scores            | N/A                      | N/A                      |
| Deterministic          | No (stochastic)            | Yes                      | Yes                      |

CUR offers a unique combination of interpretability, real‑data factors, and immediate inference acceleration without the need to extract separate low‑rank layers manually.

---

## Conclusion

The CUR Compressor (registered as `"cur"`) provides a practical implementation of CUR decomposition for model compression. By selecting informative columns and rows via leverage score sampling and computing a core matrix, it replaces standard linear layers with a compact, computationally efficient `CURLinear` module. The reduction in parameters and FLOPs is immediate and requires no retraining or calibration data. With its stochastic nature and interpretable factors, CUR is a powerful tool for compressing large transformer models, especially when combined with selective layer targeting. Future enhancements could include per‑layer adaptive rank selection, deterministic column selection (e.g., DEIM), and support for fine‑tuning the core matrices to recover accuracy.