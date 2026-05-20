# CUR‑MoE: Simple CUR Decomposition for Expert Layers (Name‑Based Selection)

**Registered as `cur_moe`**  
*Lightweight low‑rank approximation of MoE expert linear layers using random column/row sampling.*

---

## Overview

This variant of the CUR‑MoE Compressor provides a straightforward method to compress the linear layers inside Mixture‑of‑Experts (MoE) blocks by approximating their weight matrices with a **CUR decomposition**. Unlike the previous CUR‑MoE (which searched for modules with an `experts` attribute and used leverage score sampling), this version identifies expert layers by their **name** (must contain the substring `'expert'`) and selects columns and rows **uniformly at random**. The trade‑off is simplicity and speed: no dependency on module attributes, no computation of column/row norms, and a deterministic (up to random seed) but not accuracy‑optimised selection.

The module overwrites the weight matrix of each matching `nn.Linear` with its low‑rank CUR approximation, keeping the layer structure intact. The decomposition is performed by a helper function `cur_compress` that builds **C**, **R**, and the core **U**, then reconstructs the dense weight. The module is registered under `"cur_moe"` and integrates with the DistilBERT compression framework via a factory function.

---

## Methodology

### 1. CUR Decomposition (Randomised)

For a weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup>, a rank‑`r` CUR decomposition approximates:

```
W ≈ C · U · R
```

where:
- **C** = **W[:, col_idx]** contains `r` columns of **W** (size d_out × r),
- **R** = **W[row_idx, :]** contains `r` rows of **W** (size r × d_in),
- **U** = **C⁺ · W · R⁺** is the `r × r` core matrix, computed via pseudo‑inverses.

In this implementation, the column and row indices are chosen **randomly without replacement** using `torch.randperm`, i.e., a uniform distribution over all possible columns/rows. This is a stark simplification: while leverage score sampling (proportional to squared norms) yields a near‑optimal approximation, random selection is much faster but can result in a larger approximation error for the same rank. The rationale is to provide a quick, baseline compression that can be useful when computational budget is extremely limited or when the final accuracy will be restored via subsequent fine‑tuning.

### 2. Core Matrix Computation

Given the selected columns and rows, the optimal **U** (minimising `‖W – C·U·R‖_F`) is:

```
U = C⁺ · W · R⁺
```

where `C⁺` and `R⁺` are the Moore‑Penrose pseudo‑inverses, computed via `torch.linalg.pinv` on float32 copies of the matrices. The function `cur_compress` then reconstructs the full approximated weight:

```
W_approx = (C @ U @ R).to(original_dtype)
```

and returns it. The original weight tensor is overwritten with this dense low‑rank matrix.

### 3. Expert Layer Detection

The compressor scans all modules in the model. It selects an `nn.Linear` layer if its name (the full dotted path, e.g., `'encoder.layer.0.moe.expert1.linear1'`) contains the substring `'expert'`. This heuristic is simple and works for many MoE naming conventions (e.g., Fairseq, HuggingFace Transformers with expert modules). However, it may miss experts that are not named accordingly, and it could also inadvertently target non‑expert layers whose names happen to contain “expert”. A more robust approach would inspect the module attributes, but this method is chosen for its minimal assumptions.

---

## Implementation Details

### `cur_compress` function

```python
def cur_compress(weight, rank):
    out_dim, in_dim = weight.shape
    col_idx = torch.randperm(in_dim)[:rank]
    row_idx = torch.randperm(out_dim)[:rank]
    C = weight[:, col_idx]
    R = weight[row_idx, :]
    C_pinv = torch.linalg.pinv(C.float())
    R_pinv = torch.linalg.pinv(R.float())
    U = C_pinv @ weight.float() @ R_pinv
    return (C @ U @ R).to(weight.dtype)
```

- **Input**: `weight` – a 2D tensor; `rank` – integer.
- **Process**: Randomly selects `rank` columns and rows, computes pseudo‑inverses in float32, constructs core **U**, and returns the dense product cast back to the original dtype.
- **Note**: The entire operation is performed on the same device as the weight tensor; no CPU offloading. This may be faster but could use GPU memory for the SVD‑like pseudo‑inverses.

### `CURMoECompressor` Class

```python
class CURMoECompressor(BaseCompressor):
    def __init__(self, model, rank=32, **kwargs):
        super().__init__(model, name="cur_moe", **kwargs)
        self.rank = rank
```

- **`rank`**: Desired CUR rank (default 32). Applied directly (capped by min dimension inside `cur_compress`? Actually not capped – `cur_compress` does not check; it will use `rank` columns/rows from `randperm` which works as long as `rank <= min(d_out, d_in)`. If `rank` exceeds a dimension, `randperm` will raise an error. The caller should ensure `rank` is appropriate.)

`compress()`:
1. Iterates over all named modules.
2. If the module is `nn.Linear` and `'expert' in name`:
   - Replaces `module.weight.data` with the result of `cur_compress(module.weight.data, self.rank)`.
3. Updates compressed params and returns model with stats.

**Potential issue**: Biases are left untouched. If the layer has a bias, it remains unchanged, which might cause a mismatch but is typical.

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

The factory reads `cur_moe_rank` and `cur_moe_method` from `comp_config`. Note that the `method` parameter is accepted but **not used** by the `CURMoECompressor` class (which has no `method` attribute). This appears to be a leftover from a previous version or an intended future extension. As a result, the sampling is always uniform random regardless of `cur_moe_method`.

---

## Usage and Configuration

Set the compression method to `"cur_moe"` and optionally specify the rank.

### Example

```python
comp_config.comp_method = "cur_moe"
comp_config.cur_moe_rank = 16
```

Then run the compression pipeline. All linear layers whose names contain `'expert'` will be compressed to rank 16.

### Parameters

| Parameter        | Default | Description                                                                      |
|------------------|---------|----------------------------------------------------------------------------------|
| `cur_moe_rank`   | `64`    | Rank for CUR decomposition. Should be ≤ min(in_features, out_features) of the layer. |
| `cur_moe_method` | `'leverage'` | **Ignored** in this implementation; sampling is always uniform random.           |

**Important**: The `rank` must be ≤ the smallest dimension of the weight matrix; otherwise `torch.randperm` will throw an error. For safety, ensure the rank is chosen conservatively.

---

## Discussion

### Advantages

- **Extremely simple**: Only a few lines of code; easy to understand and modify.
- **Fast**: Random column/row selection is O(d_out + d_in) compared to O(d_out·d_in) for norm computation; pseudo‑inverse is done on small r×? matrices.
- **No structural changes**: The layers remain standard `nn.Linear`; the model is still just a collection of ordinary modules.
- **Name‑based filtering**: Works out‑of‑the‑box for many MoE implementations that follow the `'expert'` naming convention.
- **In‑place GPU operation**: The `cur_compress` function runs on the same device, avoiding CPU transfer.

### Limitations

- **Random sampling**: Uniform selection can produce poor approximations, especially for weight matrices with highly non‑uniform importance across columns/rows. The approximation error can be significantly larger than with leverage score sampling or SVD.
- **Method parameter ignored**: The factory accepts `method` but does not use it, which can be confusing and limits configurability.
- **Name‑based detection brittle**: If experts are named differently (e.g., `'moe.experts.0.linear1'` without the substring `'expert'`?), the layer will be missed. Conversely, a non‑expert layer named `'some_expert_thing'` will be compressed erroneously.
- **No rank sanity check**: If `rank` is larger than any dimension, `randperm` will raise an error; the compressor should cap the rank internally.
- **No bias handling**: Bias is not compressed or adjusted, which may be acceptable but could be improved.
- **Dense weight overwrite**: Like the other CUR‑MoE variant, the weight remains a full dense matrix; there is no reduction in parameter count or memory usage. The benefit is purely in the “effective rank” – information content is lowered, which might help with later steps but doesn't shrink the model.
- **Numerical stability**: Pseudo‑inverse of randomly selected columns might be ill‑conditioned, leading to a poor core **U**.

### Comparison with Other CUR‑MoE Implementations

| Aspect                   | This version (name‑based, random) | Previous CUR‑MoE (attribute‑based, leverage) |
|--------------------------|-----------------------------------|---------------------------------------------|
| Expert detection         | `'expert' in name`                | `hasattr(module, 'experts')` then iterate   |
| Column/row selection     | `randperm` (uniform random)       | Leverage scores (norm‑based probabilities)  |
| Use of `method` param    | Ignored                           | Used (`'leverage'` or `'uniform'`)          |
| Computation location     | On‑device (GPU)                   | CPU via NumPy                               |
| Parameter reduction      | None (dense weight)               | None (dense weight)                         |

The current version prioritises simplicity and speed over accuracy, making it suitable for rapid experimentation or as a lightweight baseline. For better accuracy, the leverage‑score variant or an SVD‑based approach would be preferable.

---

## Conclusion

This name‑based CUR‑MoE Compressor (registered as `"cur_moe"`) offers a dead‑simple method to inject low‑rank structure into the expert layers of MoE models. By randomly selecting columns and rows, it provides a fast, calibration‑free weight overwrite that can serve as a starting point for further compression or fine‑tuning. While its accuracy may lag behind more sophisticated selection methods, its minimalistic design makes it a valuable educational tool and a quick‑to‑deploy baseline in the DistilBERT compression ecosystem. Future improvements could incorporate proper leverage sampling, adaptive rank selection, and factorized storage to achieve real parameter savings.