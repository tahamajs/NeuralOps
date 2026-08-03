# CUR‑MoE Compressor: CUR Decomposition for Mixture‑of‑Experts Layers

**Registered as `cur_moe`**  
*Low‑rank compression of expert weight matrices in MoE modules using CUR decomposition with leverage score sampling.*

---

## Overview

The CUR‑MoE Compressor is a post‑training compression technique that specifically targets the **expert layers** inside a Mixture‑of‑Experts (MoE) architecture. MoE models contain multiple parallel feed‑forward “experts”, each a small neural network, making them particularly parameter‑heavy. CUR‑MoE compresses each 2D weight matrix (usually linear projections inside each expert) by replacing it with a low‑rank **CUR decomposition**, drastically reducing the number of effective parameters while preserving the original layer structure.

Unlike the general `CURCompressor` (which replaces entire layers with a custom `CURLinear` module), CUR‑MoE performs a **weight overwrite**: it decomposes the weight matrix **W** into `W ≈ C · U · R`, where **C** and **R** are actual columns and rows of **W**, and then writes the dense product back into the parameter tensor. This means the expert layers remain standard `nn.Linear` objects—no structural change—but their weight matrices are forced to be low‑rank. The method requires no calibration data, relying solely on the weight values, and uses leverage score sampling to select the most informative columns and rows.

The module is designed for the DistilBERT compression framework and is registered under `"cur_moe"`. It can be applied to any model that contains modules with an `experts` attribute (i.e., typical MoE implementations).

---

## Methodology

### 1. CUR Decomposition (Recap)

For a weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup>, a rank‑`r` CUR decomposition expresses it as:

```
W ≈ C · U · R
```

where:
- **C** = `W[:, cols]` contains `r` selected columns of **W** (size d_out × r),
- **R** = `W[rows, :]` contains `r` selected rows of **W** (size r × d_in),
- **U** = **C⁺ · W · R⁺** is the `r × r` core matrix, computed via pseudo‑inverses.

This decomposition is exact if **W** has rank ≤ `r` and the chosen columns/rows span the column and row spaces. For low‑rank approximation, the selection of columns and rows is critical.

### 2. Leverage Score Sampling

CUR‑MoE selects columns and rows using **leverage scores**, which are proportional to the squared Euclidean norms:

- **Column sampling probability**:  
  `p_j = ‖W[:, j]‖² / Σ_k ‖W[:, k]‖²`
- **Row sampling probability**:  
  `q_i = ‖W[i, :]‖² / Σ_k ‖W[k, :]‖²`

We sample `r` columns and `r` rows **without replacement** according to these probabilities. This tends to pick columns/rows with larger magnitudes, which carry more information, leading to a near‑optimal approximation in practice.

If `method='uniform'`, all columns/rows are equally likely.

### 3. Low‑Rank Approximation via Overwrite

After obtaining **C**, **R**, and **U**, the algorithm forms the full approximated weight:

```
W_approx = C · U · R
```

This dense matrix of shape (d_out, d_in) is then written back into `param.data`, replacing the original weight. Because **W_approx** is the product of three matrices with inner dimension `r`, its rank is at most `r`, achieving compression. The parameter tensor itself remains a dense `d_out × d_in` matrix, so the memory footprint does not decrease unless the tensor is stored in a factored form—but the effective information content is reduced, which can be exploited by subsequent sparsification or quantization.

### 4. Targeting MoE Experts

The compressor searches the model for modules that possess an `experts` attribute—a typical pattern in MoE implementations (e.g., `MixtureOfExperts` container in Fairseq, or custom MoE layers). For each such module, it iterates over all sub‑modules listed in `module.experts`. Inside each expert, it looks for parameter tensors that:
- Have `'weight'` in their name,
- Are 2‑dimensional (i.e., weight matrices, not biases).

These are likely the linear projections (e.g., `w1`, `w2` in a feed‑forward network). For each qualifying weight, the CUR approximation is applied as described.

This approach compresses only the expert weights, leaving the gating network and other shared parts untouched, focusing the compression budget where the bulk of parameters reside.

---

## Implementation Details

### CURMoECompressor Class

```python
class CURMoECompressor(BaseCompressor):
    def __init__(self, model, rank=64, method='leverage', **kwargs):
        super().__init__(model, name="cur_moe", **kwargs)
        self.rank = rank
        self.method = method
```

- **`rank`**: Desired rank `r` for CUR decomposition (default 64). Actual rank is capped by `min(d_out, d_in)`.
- **`method`**: `'leverage'` (default) or `'uniform'`.

The `compress()` method:

1. Iterates over all named modules in the model.
2. Checks if the module has an `experts` attribute (using `hasattr`).
3. For each expert in `module.experts` (expected to be an iterable of `nn.Module`):
   - Iterates over `expert.named_parameters()`.
   - If parameter name contains `'weight'` and the tensor is 2D:
     - Moves weight to CPU, converts to float32 NumPy.
     - Determines effective `r = min(rank, d_out, d_in)`.
     - Samples columns and rows based on leverage scores (or uniform).
     - Forms **C**, **R**.
     - Computes pseudo‑inverses via `np.linalg.pinv`.
     - Computes **U** = `C_pinv @ W @ R_pinv`.
     - Forms `W_approx = C @ U @ R`.
     - Writes back to `param.data`, preserving original dtype and device.
4. Updates compressed parameter count and returns model with stats.

**Note**: The code uses NumPy on CPU for the decomposition. This is acceptable for post‑training compression but may be slow for very large expert matrices. A PyTorch‑only implementation could be faster but is not implemented here.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("cur_moe")
def cur_moe_factory(model, comp_config):
    rank = getattr(comp_config, 'cur_moe_rank', 64)
    method = getattr(comp_config, 'cur_moe_method', 'leverage')
    compressor = CURMoECompressor(model, rank=rank, method=method)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `cur_moe_rank` and `cur_moe_method` from `comp_config`. The compressed model is returned directly.

---

## Usage and Configuration

To apply CUR‑MoE compression, set the compression method to `"cur_moe"` and provide the desired rank and method.

### Example

```python
comp_config.comp_method = "cur_moe"
comp_config.cur_moe_rank = 32
comp_config.cur_moe_method = "leverage"
```

Then run the compression pipeline. The model’s MoE expert weight matrices will be overwritten with their rank‑32 CUR approximations.

### Parameters

| Parameter          | Default      | Description                                                                 |
|--------------------|--------------|-----------------------------------------------------------------------------|
| `cur_moe_rank`     | `64`         | Target rank for CUR decomposition. Lower = more compression, less accuracy. |
| `cur_moe_method`   | `'leverage'` | Column/row selection method: `'leverage'` (norm‑based) or `'uniform'`.      |

**Important**: The model must contain modules with an `experts` attribute. If no such modules exist, the compressor does nothing. The compressor assumes that `module.experts` is an iterable of sub‑modules whose parameters can be accessed via `expert.named_parameters()`.

---

## Discussion

### Advantages

- **Targeted compression**: Focuses on the most parameter‑intensive part of MoE models—the expert layers—leaving the router and shared components unchanged.
- **No architectural change**: The expert modules remain standard `nn.Linear` layers; no custom code needed for inference. This makes it compatible with existing runtimes.
- **Interpretable basis**: The use of actual columns and rows of the weight matrix provides a degree of interpretability; one can inspect which features are retained.
- **No calibration data**: Purely weight‑based, so it’s fast and deterministic up to the random sampling.
- **Leverage score sampling**: Provides a principled, near‑optimal selection of columns and rows for the given rank.

### Limitations

- **Dense matrix storage**: After compression, the weight tensors are still dense `d_out × d_in`; the parameter count **does not decrease** in terms of stored numbers. The compression is only in the “effective rank” sense, which can be used for further compression (e.g., factorization, sparsity) but does not save memory on its own. This is a crucial difference from the `CURCompressor` that replaces the layer with a parameter‑efficient `CURLinear`.
- **Overwrite only**: The original weight information is permanently lost; there is no fine‑tuning or recovery unless the original weights are saved separately.
- **Non‑deterministic**: The random sampling can produce slightly different results across runs. Reproducibility requires setting a NumPy seed.
- **Assumes specific MoE structure**: The compressor expects `module.experts` to be an iterable of modules. If the model uses a different naming convention or structure (e.g., a list of linear layers stored in a dictionary), it may not be compressed. The detection logic may need adaptation for other MoE implementations.
- **Numerical stability**: Pseudo‑inverse computation can be sensitive; for ill‑conditioned matrices, the approximation may be poor.
- **No fine‑tuning**: Accuracy may drop; fine‑tuning the compressed model is recommended to recover performance.

### Comparison with Related Compressors

| Feature                     | CUR‑MoE (this)              | CUR Compressor (full)         | A3 (SVD overwrite)         |
|-----------------------------|-----------------------------|-------------------------------|----------------------------|
| Target                      | MoE expert weights          | All Linear layers (optional)  | Attention Linear layers    |
| Decomposition               | CUR (columns+rows)          | CUR (columns+rows)            | SVD                        |
| Structural change           | No (overwrites weight)      | Yes (replaces with CURLinear) | No (overwrites weight)     |
| Parameter reduction         | No (dense tensor)           | Yes (factorized buffers)      | No (dense tensor)          |
| Interpretable factors       | Yes (actual columns/rows)   | Yes                           | No                         |

CUR‑MoE is best suited when you want to quickly reduce the effective rank of expert weight matrices without modifying the model architecture, perhaps as a pre‑step before applying further compression or fine‑tuning.

---

## Conclusion

The CUR‑MoE Compressor (registered as `"cur_moe"`) provides a lightweight, targeted approach to compress the expert layers in Mixture‑of‑Experts models. By overwriting each weight matrix with its rank‑`r` CUR approximation using leverage score sampling, it reduces the effective information content while keeping the model structure intact. The method requires no training data and integrates seamlessly into the compression framework. While it does not reduce storage directly (the weight remains dense), it opens the door for subsequent factorization or sparsification and offers an interpretable decomposition. Future improvements could include support for different MoE architectures, deterministic column selection, and optional fine‑tuning of the compressed weights.