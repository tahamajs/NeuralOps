# FLAT‑LLM: Activation‑Aware Structured Pruning via PCA on Output Dimensions

**Registered as `flat_llm`**  
*Prunes entire output neurons (rows of the weight matrix) by retaining the dimensions with the highest activation variance, determined via PCA on calibration data.*

---

## Overview

FLAT‑LLM (Activation‑Aware Dimension Pruning via PCA) is a post‑training structured pruning method that reduces the **output dimensionality** of each linear layer. It collects the output activations of every `nn.Linear` module from a calibration dataset, computes the covariance matrix of those activations, and selects the top‑`k` output dimensions with the largest eigenvalues (i.e., highest variance). The weight matrix and bias are then pruned to keep only those rows, effectively discarding the least variable output neurons. The result is a model with thinner layers, where the number of output features per layer is reduced by a factor `keep_ratio`, leading to a direct reduction in parameters and computational cost.

This technique is particularly suitable for layers where the output activations exhibit a low‑rank structure—many neurons may carry redundant information. By focusing on the most fluctuating (high‑variance) dimensions, FLAT‑LLM preserves the expressive power while shrinking the model. The module is registered under `"flat_llm"` and requires a tokenizer (attached to the model) and a small set of calibration texts.

---

## Methodology

### 1. Activation Capture

The model is run in evaluation mode on a set of calibration texts (tokenized and padded). Forward hooks attached to every `nn.Linear` layer record their **output tensors** (post‑linear, pre‑activation function if any). After one forward pass, the collected outputs are moved to CPU. For a layer with output features `d_out`, the gathered activations **Y** have shape `(N, d_out)` after flattening batch and sequence dimensions.

### 2. Covariance and PCA

We compute the `d_out × d_out` covariance matrix of the output activations:

```
C = (1/(N-1)) (Y - μ)^T (Y - μ)
```

where `μ` is the mean of each output feature across the `N` samples. In the code, `np.cov(act_flat.T)` is used, which computes the covariance matrix directly (with default normalization by `N-1`). The eigendecomposition of **C** gives:

```
C = V Λ V^T
```

with eigenvalues `λ₁ ≥ λ₂ ≥ … ≥ λ_{d_out}` representing the variance along each principal component. The indices are sorted in descending order of eigenvalues.

### 3. Dimension Selection

Given a `keep_ratio` (e.g., 0.7), we retain the top `k = max(1, ⌈keep_ratio · d_out⌉)` dimensions with the largest eigenvalues. These correspond to the output directions where the activations fluctuate the most, thus carrying the most discriminative information. The pruned set of indices is:

```
idx = argsort(eigenvalues)[::-1][:k]
```

### 4. Weight and Bias Pruning

The linear layer’s weight matrix **W** (shape `d_out × d_in`) and bias **b** (shape `d_out`) are pruned by keeping only the selected rows:

```
W' = W[idx, :]
b' = b[idx]
```

This reduces the layer’s output dimension from `d_out` to `k`. The new weight shape is `(k, d_in)`. Note that this changes the **output** of the layer, which means the subsequent layers must have their **input** dimensions adapted accordingly. In a typical transformer, the input dimension of the next layer (e.g., layer norm, then linear) expects the full `d_out`. Simply pruning rows without adjusting downstream layers will cause a shape mismatch. However, in certain architectures (like feed‑forward networks where the output of one layer directly feeds the input of the next, and there is no residual connection), the pruning can be propagated by also pruning the corresponding input channels of the next layer—but the current implementation does **not** handle this. It is therefore assumed that the user will use this compressor only on models where such structural changes are tolerable, or combine it with a mechanism to align dimensions.

---

## Implementation Details

### FLATLLMCompressor Class

```python
class FLATLLMCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, keep_ratio=0.7, calibration_texts=None, **kwargs):
        super().__init__(model, name="flat_llm", **kwargs)
        self.tokenizer = tokenizer
        self.keep_ratio = keep_ratio
        self.calibration_texts = calibration_texts or [
            "The quick brown fox jumps over the lazy dog."
        ]
```

- **`tokenizer`**: Required to encode calibration texts.
- **`keep_ratio`**: Fraction of output dimensions to retain (default 0.7, i.e., keep 70%, prune 30%).
- **`calibration_texts`**: List of calibration strings (default: one generic sentence).

The `compress()` method:

1. Sets model to `eval()`.
2. Registers forward hooks on all `nn.Linear` layers. The hooks save the module’s **output** into a dictionary (moving to CPU).
3. Tokenizes calibration texts, moves inputs to the model’s device.
4. Runs a forward pass with `torch.no_grad()`.
5. Removes hooks.
6. For each linear layer that has recorded activations:
   - Flattens output to `(N, d_out)` float32 NumPy array.
   - Computes covariance matrix `np.cov(act_flat.T)`.
   - Computes eigenvalues via `np.linalg.eigh(cov)` (note: `eigh` is for symmetric matrices, returns eigenvalues ascending; the code later sorts in descending order).
   - Determines `keep_dims = max(1, int(total_dims * self.keep_ratio))`.
   - Sorts eigenvalues descending, takes indices of top `keep_dims`.
   - Prunes weight: `module.weight.data = W[idx, :]` (keeps rows idx).
   - Prunes bias: `module.bias.data = module.bias.data[idx]`.
7. Updates compressed parameter count and returns model with stats.

**Critical caveat**: The method does **not** adjust any subsequent layer’s input dimensions or handle residual connections. In its current form, the pruned model will almost certainly encounter a shape mismatch in the very next module that expects the original `d_out` as input. This is a known limitation and makes the compressor unsuitable for direct use on typical transformer models without additional propagation logic.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("flat_llm")
def flat_llm_factory(model, comp_config):
    keep_ratio = getattr(comp_config, 'flat_llm_keep_ratio', 0.7)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("flat_llm requires tokenizer attached as model.tokenizer")
    compressor = FLATLLMCompressor(model, tokenizer, keep_ratio=keep_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory expects the tokenizer attached as `model.tokenizer` and `flat_llm_keep_ratio` in `comp_config`.

---

## Usage and Configuration

Attach the tokenizer, set compression method to `"flat_llm"`, and specify the keep ratio.

### Example

```python
model.tokenizer = tokenizer
comp_config.comp_method = "flat_llm"
comp_config.flat_llm_keep_ratio = 0.6   # keep 60% of output neurons
```

Then run the compression pipeline. Note that the model will likely be broken due to dimension mismatch; further steps (like pruning corresponding input channels of next layers) are required but not implemented.

### Parameters

| Parameter               | Default | Description                                                                                         |
|-------------------------|---------|-----------------------------------------------------------------------------------------------------|
| `flat_llm_keep_ratio`   | `0.7`   | Fraction of output dimensions to retain. Must be between 0 and 1.                                   |
| (model attribute) `tokenizer` | Required | Tokenizer for processing calibration texts.                                                      |

---

## Discussion

### Advantages

- **Data‑driven pruning**: The importance of an output neuron is determined by how much its activation varies, capturing actual information content.
- **Structured reduction**: Removing whole output neurons directly reduces weight matrix size and subsequent computation (if dimensions are aligned).
- **Simple PCA approach**: Computationally light (covariance + eigen on small matrices) and easy to understand.
- **No iterative retraining**: Purely analytical, given a calibration set.

### Limitations

- **Broken dimension alignment**: The compressed model is not immediately usable because the output dimension of one layer is reduced, but the next layer still expects the original input size. This is the main flaw. A complete implementation would need to also prune the **input** channels of downstream layers (or insert projection layers) to match the new dimensions. Without this, the model will raise shape errors.
- **Residual connections not handled**: In transformer blocks with skip connections, pruning output neurons of a sub‑layer will break the addition with the residual stream, which expects the original hidden size. FLAT‑LLM does not address this.
- **Covariance computation assumes sufficient samples**: If the calibration set is too small, the covariance matrix may be rank‑deficient, leading to unreliable eigenvalue ranking. The default single‑sentence calibration is likely insufficient; a more representative dataset should be used.
- **Deterministic but calibration‑dependent**: The choice of calibration texts can strongly influence which neurons are pruned. Different calibration sets may yield different compressed models.
- **Bias pruning**: The bias is pruned along with the weight rows. This is correct because the output dimension shrinks, so the bias vector must shrink too. No issues there.

### Potential Use Cases

FLAT‑LLM is best suited for **feed‑forward sub‑networks** within a transformer (e.g., the intermediate layer of an FFN) where the output dimension can be reduced if the subsequent layer (the output projection of the same FFN) is adjusted accordingly—i.e., prune the intermediate dimension of the FFN and also prune the input dimension of the next linear layer. This is essentially structured pruning of the whole hidden dimension, which can be done if one ensures that the same set of indices is pruned from all layers that share that dimension. The current code could be extended to collect indices from the activation of the intermediate layer and then apply those indices to both the first linear (output) and the second linear (input) of the FFN. However, this is left for future work.

### Comparison with Other Structured Pruners

| Method            | Granularity       | Importance Criterion            | Handles Dimension Alignment | Calibration |
|-------------------|-------------------|----------------------------------|-----------------------------|-------------|
| FLAT‑LLM (this)   | Output neurons    | Activation variance (PCA)        | No (needs external propagation) | Yes         |
| FLAP              | Input features    | Weighted input variance          | Attempts sequential propagation (buggy) | Yes         |
| CacheNotes        | Output neurons    | Task heuristic + weight norm     | No                          | No (task type) |

FLAT‑LLM is a clean implementation of output neuron pruning via PCA, but its lack of dimension alignment makes it a module that must be used as part of a larger pruning pipeline.

---

## Conclusion

FLAT‑LLM (registered as `"flat_llm"`) demonstrates the concept of activation‑aware structured pruning using principal component analysis. By retaining the output dimensions with the highest activation variance, it effectively compresses linear layers with a data‑informed criterion. However, the current implementation is incomplete: it does not propagate the dimension reduction to subsequent layers, leaving the model in an inconsistent state. To be practically useful, it must be integrated into a framework that handles dimension alignment (e.g., prunes the corresponding input channels of downstream layers and manages residual connections). As such, this module serves as a foundational building block for more comprehensive structured pruning pipelines in the DistilBERT ecosystem. Future enhancements could include automatic propagation, handling of skip connections, and support for per‑layer keep ratios based on energy retention.