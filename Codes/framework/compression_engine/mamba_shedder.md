# Mamba‑Shedder: Sensitivity‑Based Structured Pruning

**Registered as `mamba_shedder`**  
*Prunes input features of linear layers based on activation sensitivity (approximate Fisher information), reducing the model’s width while preserving the most informative features.*

---

## Overview

Mamba‑Shedder is a post‑training structured pruning method that reduces the input dimensionality of each `nn.Linear` layer by removing features that are least “sensitive” according to the activation statistics on a calibration dataset. The sensitivity of an input feature is defined as the square root of the mean squared activation, which serves as a lightweight proxy for the diagonal of the Fisher information matrix. Features with low sensitivity contribute little to the variance of the layer’s input and can be pruned with minimal impact. The pruning is performed column‑wise on the weight matrix, keeping the top `(1 – sparsity)` fraction of input columns with the highest sensitivity scores.

The compressor requires a tokenizer (attached to the model) and a **calibration data loader** (provided as an argument to `compress()`), making it data‑dependent. It is registered under `"mamba_shedder"` and is inspired by the idea of “shedding” unnecessary dimensions, akin to the parameter reduction philosophy in state‑space models like Mamba (hence the name).

---

## Methodology

### 1. Activation Collection

Given a calibration dataset (e.g., a few hundred tokenized sentences), the model is run in evaluation mode. Forward hooks attached to every `nn.Linear` module capture their **input tensors** (pre‑activation) for each batch. All inputs are concatenated along the batch and sequence dimensions, resulting in a matrix **X** ∈ ℝ<sup>N × d_in</sup>, where `N` is the total number of tokens and `d_in` is the layer’s input feature size.

### 2. Sensitivity Metric

For each linear layer, the sensitivity of input feature `j` is defined as the root mean square of its activations:

```
s_j = sqrt( (1/N) Σ_{i=1}^N X_{ij}^2 )
```

This is equivalent to the square root of the diagonal of the empirical uncentered covariance matrix of the inputs. The quantity `s_j²` is an estimate of the Fisher information (under a Gaussian output model with identity noise) associated with the corresponding weight column. Larger values indicate that the input feature is more variable and thus more “important” for capturing data dynamics.

The code computes this as:

```python
fisher = (act_flat ** 2).mean(dim=0)   # mean squared activation
sensitivity[name] = fisher.sqrt()
```

### 3. Structured Pruning

For a layer with weight **W** ∈ ℝ<sup>d_out × d_in</sup>, the algorithm selects the `k = int(d_in * (1 - sparsity))` input features with the highest sensitivities. Let `keep_idx` be the indices of those features, sorted by descending sensitivity. The weight matrix is pruned by keeping only the corresponding columns:

```
W' = W[:, keep_idx]   → shape (d_out, k)
```

The bias vector **b** ∈ ℝ<sup>d_out</sup> is **incorrectly** pruned by the same indices in the provided code (`bias.data[keep_idx]`), which would try to select elements from the bias according to the input feature indices, leading to a size mismatch unless `d_out` equals the number of kept indices (which it generally does not). This is a bug. The bias should remain unchanged, as it corresponds to output channels, not input features.

### 4. Propagation of Reduced Dimensions

Because the input feature dimension is reduced, subsequent layers that receive the output of this layer as input must also have their input dimensions pruned consistently. However, the current implementation does **not** propagate these changes; it merely prunes each layer independently based on its own input activations. This will lead to shape mismatches in the network unless all layers are pruned such that the output dimension of one layer matches the input dimension of the next. The code appears to assume that the calibration loader is used only for sensitivity estimation, and that each layer is pruned independently without adjusting downstream layers—this is a critical flaw that makes the compressed model unusable.

A proper structured pruning pipeline must either prune all layers globally to align dimensions, or sequentially propagate the pruned indices.

---

## Complexity Analysis

### A. Compression Time Complexity

- **Forward pass for calibration**: O(L · B · N · D²) where L is number of layers, B is calibration batch count, N is sequence length, D is hidden size. This is the cost of running the model once over the calibration set.
- **Sensitivity computation**: For each layer, O(d_in · N) to compute mean squared values (after flattening). This is cheap.
- **Sorting**: O(d_in log d_in) per layer.
- Overall compression time is dominated by the calibration forward pass.

### B. Inference Complexity

After compression (if dimensions were correctly aligned), each linear layer would have shape `(d_out, k)` instead of `(d_out, d_in)`, reducing the matrix multiplication cost from `O(d_out · d_in)` per token to `O(d_out · k)`, where `k = d_in * (1 - sparsity)`. Thus FLOPs are reduced by a factor of `1/(1 - sparsity)` for each layer.

However, the current buggy implementation does **not** actually achieve this because biases are pruned incorrectly and subsequent layers’ dimensions are not aligned. If these issues were fixed, the computational savings would be proportional to the sparsity.

### C. Memory Usage

**During compression**: Activation storage for each linear layer requires memory proportional to `N · d_in`. For large calibration sets, this can be significant but manageable.

**After compression**: The weight matrices become smaller (fewer columns), reducing the model’s parameter count and memory footprint.

---

## Implementation Details

### MambaShedderCompressor Class

```python
class MambaShedderCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, sparsity=0.2, device="cuda", **kwargs):
        ...
```

- **`tokenizer`**: Required for tokenizing calibration data? Actually the `calib_loader` is passed to `compress()`, so the tokenizer may not be used directly in the current code. It’s stored but not called; the calib_loader is expected to yield already‑tokenized batches.
- **`sparsity`**: Fraction of input features to remove (default 0.2 → keep 80%).
- **`device`**: Device for computation.

`_collect_activations(calib_loader)`:
- Registers hooks on all `nn.Linear` modules to save the input tensors.
- Runs the model on the calibration loader.
- Returns a dict of layer name → concatenated input activations.

`_compute_sensitivity(activations)`:
- For each layer, flattens the activation tensor to (N, d_in), computes mean squared activation along N, then square root.

`compress(self, calib_loader)`:
1. Calls `_collect_activations(calib_loader)`.
2. Computes sensitivities.
3. For each linear layer:
   - Gets its weight, computes `n_keep = int(weight.shape[1] * (1 - sparsity))`.
   - Sorts sensitivities descending, takes top `n_keep` indices.
   - Prunes weight columns: `module.weight.data = weight[:, keep_idx]`.
   - **Bug**: Also prunes bias with `module.bias.data = module.bias.data[keep_idx]` (should be left alone).
4. Updates `_compressed_params` and returns model.

**Critical flaw**: The compressor does not adjust subsequent layers to match the new input dimensions, nor does it handle cross‑layer dimension alignment. The resulting model will very likely raise a shape mismatch error at the next module that expects the original input size.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("mamba_shedder")
def mamba_shedder_factory(model, comp_config):
    sparsity = getattr(comp_config, 'mamba_shedder_sparsity', 0.2)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("mamba_shedder requires tokenizer attached as model.tokenizer")
    compressor = MambaShedderCompressor(model, tokenizer, sparsity)
    compressed_model, _ = compressor.compress()   # BUG: missing calib_loader argument!
    return compressed_model
```

The factory calls `compressor.compress()` without the required `calib_loader` parameter, causing a runtime error. This makes the module unusable through the standard pipeline without modification. A corrected factory should accept a `calib_loader` from the configuration or environment.

---

## Usage and Configuration

Currently, the compressor cannot be used via the factory because of the missing argument. To use it, one must instantiate the compressor directly and call `compress(calib_loader)` manually.

### Example (with direct usage)

```python
from mamba_shedder import MambaShedderCompressor
from torch.utils.data import DataLoader

# Assume calib_loader yields batches of tokenized data
compressor = MambaShedderCompressor(model, tokenizer, sparsity=0.3)
compressed_model, stats = compressor.compress(calib_loader)
```

### Parameters (if factory were fixed)

| Parameter                        | Default | Description                                                                 |
|----------------------------------|---------|-----------------------------------------------------------------------------|
| `mamba_shedder_sparsity`         | `0.2`   | Fraction of input features to prune. 0 = no pruning, 1 = all pruned.       |
| (attached) `model.tokenizer`     | required| Tokenizer for the model (currently not used, but checked).                  |

**Warning**: The method is currently broken due to multiple bugs (bias pruning, dimension misalignment, factory missing argument). Do not use without thorough fixing.

---

## Discussion

### Advantages (Intended)

- **Data‑driven pruning**: Sensitivity based on actual activation variance ensures that the most active features are kept.
- **Structured reduction**: Pruning entire input columns reduces both parameters and computation.
- **Simple metric**: Fisher‑like sensitivity is cheap to compute.

### Limitations (Actual)

- **Factory broken**: The factory function raises an error because `compress()` is called without `calib_loader`.
- **Bias pruning error**: The bias is incorrectly indexed with `keep_idx`, leading to shape mismatch or incorrect values.
- **No dimension alignment**: The model becomes inconsistent after pruning; downstream layers still expect the original input size. This is the most critical flaw and makes the compressed model non‑functional.
- **Tokenizer unused**: The stored tokenizer is not used; the calibration data must be pre‑tokenized and provided as a loader.
- **No handling of residual connections**: In transformer architectures, pruning input features of a sub‑layer would break the residual addition, which expects the same hidden size. This is not addressed.

### Comparison with Other Structured Pruners

| Method            | Sensitivity Metric        | Dimension Alignment  | Status           |
|-------------------|---------------------------|----------------------|------------------|
| Mamba‑Shedder     | sqrt(mean squared input)  | None (broken)        | Buggy / Non‑functional |
| FLAP              | Weighted input variance   | Sequential (buggy)   | Buggy            |
| FLAT‑LLM          | Output activation PCA     | None (broken)        | Buggy            |
| HybridKV          | Weight variance (static/dynamic) | N/A (prunes output rows) | Functional |

Mamba‑Shedder is conceptually similar to FLAP but uses a simpler sensitivity metric. However, its current implementation is incomplete and unsafe for use.

---

## Conclusion

Mamba‑Shedder (registered as `"mamba_shedder"`) presents a straightforward idea for sensitivity‑based structured pruning of linear layers. Despite its promising simplicity, the implementation suffers from multiple critical bugs—including a missing calibration loader in the factory, erroneous bias pruning, and a total lack of cross‑layer dimension alignment—that render it unusable in its current state. To become a viable compression tool, the module needs a complete rework: the factory must accept a calibration loader, bias pruning must be removed or corrected, and a mechanism to propagate pruned dimensions across the network (respecting residual connections) must be implemented. As it stands, it serves more as a template for future development than a functioning compressor.