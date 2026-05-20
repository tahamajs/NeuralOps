# FLAP: Fluctuation‑Based Adaptive Structured Pruning

**Registered as `flap`**  
*Structured pruning of input features guided by input variance and weight importance, propagating reduced dimensions through the network.*

---

## Overview

FLAP (Fluctuation‑based Adaptive Structured Pruning) is a post‑training compression method that removes entire input features from linear layers based on their **fluctuation importance**. The importance of an input feature (column of the weight matrix) is measured by combining the variance of that feature’s activations (captured from calibration data) with the squared norm of the corresponding weight column. The idea is that a feature is important if it exhibits high variance (i.e., it is actively used) **and** the associated weights are large. The pruning keeps the top `(1 – sparsity)` fraction of columns, effectively reducing the layer’s input dimensionality. By propagating the pruned input through the model layer by layer, the method achieves a consistent reduction of activation dimensions throughout the network, leading to smaller and faster models.

The module is designed as a drop‑in compressor for the DistilBERT framework, registered under `"flap"`. It requires a tokenizer (attached to the model) and a small amount of calibration data—though the current implementation uses a trivial repetition of three sentences as a placeholder.

---

## Methodology

### 1. Fluctuation Metric

For a given linear layer with weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup> and calibration input **X** ∈ ℝ<sup>N × d_in</sup>, the importance of the `j`‑th input feature is computed as:

```
s_j = (∑_{i=1}^{d_out} W_{ij}^2) × Var(X[:, j])
```

This is the **WIFV (Weighted Input Feature Variance)** metric. It multiplies the squared L2 norm of the weight column (how much the layer depends on that input) by the variance of that input feature across calibration samples. Features that are both highly variable and have large associated weights are deemed more important.

Two other variants are also available in the code:
- **IFV** (Input Feature Variance): `s_j = Var(X[:, j])` – purely based on activation variance.
- **WIFN** (Weighted Input Feature Norm): `s_j = (1/d_out) Σ_i |W_{ij}| × √Var(X[:, j])` – a mean over absolute weights times standard deviation.

By default, the compressor uses WIFV.

### 2. Column Pruning

After computing the scores, the algorithm selects the top `k = ⌈(1 – sparsity) × d_in⌉` columns with the highest importance. The weight matrix is pruned to keep only those columns:

```
W' = W[:, keep_idx]
```

and the bias is kept unchanged (since it is of size `d_out`, not affected by input dimension reduction). The pruned weight now has shape `(d_out, k)`, reducing the number of parameters from `d_out × d_in` to `d_out × k`.

### 3. Activation Propagation

Because the input dimension of one layer is the output dimension of the previous one, pruning must be propagated through the network to maintain compatibility. The compressor processes layers sequentially. Starting with the initial input embeddings (obtained from calibration texts), it:
- Prunes the first linear layer’s weight columns.
- Updates the calibration input by selecting the same columns (`flat_inputs = flat_inputs[:, keep_idx]`), thereby reducing its feature count to `k`.
- Feeds this reduced input to the next linear layer, and repeats.

This ensures that every layer sees an input that matches its pruned weight matrix. However, this simplistic propagation **ignores residual connections**, layer normalisation, and other branch structures present in typical transformer architectures. A full FLAP implementation would need to handle those carefully (often by pruning coordinated sets of layers to keep residual dimensions aligned). The current code serves as a conceptual demonstration.

---

## Implementation Details

### FLAPCompressor Class

Inherits from `BaseCompressor`.

```python
class FLAPCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, sparsity=0.2, device="cuda", nsamples=1024, **kwargs):
        super().__init__(model, name="flap", **kwargs)
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.device = device
        self.nsamples = nsamples
```

- **`sparsity`**: Fraction of input features to remove (default 0.2 → keep 80%).
- **`device`**: Device to run calibration on (default `"cuda"`).
- **`nsamples`**: Number of calibration samples to generate (default 1024, created by repeating 3 sentences).

The `compress()` method:

1. Creates a calibration dataset by repeating three generic sentences to reach `nsamples` (a crude approximation, intended as placeholder).
2. Tokenizes and extracts input embeddings via `model.get_input_embeddings()`.
3. Flattens the embeddings to shape `(total_tokens, d_model)` → `flat_inputs`.
4. Iterates over each encoder layer (`model.encoder.layer`). For each `nn.Linear` inside the layer:
   - Calls `_compute_fluctuation_metric(weight, flat_inputs)` to obtain importance scores.
   - Computes `n_keep = int(weight.shape[1] * (1 - sparsity))`.
   - Selects the top indices via `np.argsort(scores)[-n_keep:]`.
   - Prunes the weight: `module.weight.data = weight[:, keep_idx]`.
   - If bias exists, it incorrectly does `module.bias.data = module.bias.data[keep_idx]`—this is a bug (bias length should remain `out_features`, not `in_features`). The code as written would shrink the bias along the wrong dimension, likely causing a runtime error. This is a known issue.
   - Updates `flat_inputs = flat_inputs[:, keep_idx]` for the next modules.
5. Updates compressed parameter count and returns the model.

**Critical Issues**:
- Bias pruning is erroneous; it should remain unpruned.
- Residual connections are not adjusted, leading to shape mismatches in networks with skip connections.
- The sequential propagation assumes a purely feed‑forward structure without layer norm or shared weights, which is unrealistic for standard transformers.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("flap")
def flap_compressor_factory(model, comp_config):
    sparsity = getattr(comp_config, 'flap_sparsity', 0.2)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("flap requires tokenizer attached as model.tokenizer")
    compressor = FLAPCompressor(model, tokenizer, sparsity=sparsity)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory requires the tokenizer to be attached as `model.tokenizer`. Configuration parameters are read from `comp_config` with defaults.

---

## Usage and Configuration

Attach the tokenizer, set `comp_config.comp_method = "flap"`, and specify sparsity.

### Example

```python
model.tokenizer = tokenizer
comp_config.comp_method = "flap"
comp_config.flap_sparsity = 0.3   # remove 30% of input features per layer
```

Then run the compression pipeline.

### Parameters

| Parameter          | Default | Description                                                                                     |
|--------------------|---------|-------------------------------------------------------------------------------------------------|
| `flap_sparsity`    | `0.2`   | Fraction of input features to prune (0.0 = none, 1.0 = all).                                    |
| `flap_metric`      | `"WIFV"` (hardcoded) | Importance metric; not exposed via config. Modify code for IFV or WIFN.                   |
| `flap_nsamples`    | `1024`  | Number of calibration samples (currently created by repeating three fixed sentences).            |
| (model attribute) `tokenizer` | Required | Tokenizer for creating calibration input. |

**Note**: The calibration data generation is a placeholder; for meaningful results, replace it with a representative dataset.

---

## Discussion

### Advantages

- **Structured pruning**: Removes entire input features, leading to smaller weight matrices and reduced computation. This can directly speed up inference (unlike unstructured sparsity) if the model is reshaped.
- **Data‑informed importance**: Uses activation variance, which captures real data dynamics, combined with weight magnitude for a balanced measure.
- **Simple concept**: Easy to understand and implement; the sequential propagation is straightforward in a feed‑forward chain.

### Limitations

- **Residual connections broken**: The method does not handle skip connections, making it unsuitable for modern transformers without further adaptation. In practice, FLAP would need to prune dimensions consistently across parallel branches (e.g., attention and FFN in a residual block).
- **Bias pruning bug**: The code erroneously prunes the bias vector along the input dimension, which will cause errors. The bias should remain length `out_features`.
- **Unrealistic calibration**: The calibration set is created by repeating three sentences, providing very limited activation statistics. A proper implementation would use a real dataset.
- **Sequential dependency**: The order of pruning depends on the arbitrary traversal order of modules within a layer; if a layer has multiple linear modules, the input is successively pruned, which may not correctly reflect the actual dataflow (e.g., Q/K/V projections all receive the same input, so they should be pruned uniformly rather than sequentially with the pruned input from the previous projection affecting the next). The current loop prunes them one after another, causing a cascade of dimension reductions within the same layer, which is incorrect.
- **No fine‑tuning**: The pruned model is likely broken due to shape mismatches; it cannot be used for inference without fixing residual paths and other structural issues.

### Comparison with Other Structured Pruners

| Method           | Granularity | Importance Criterion          | Handles Residuals | Data Needed     |
|------------------|-------------|-------------------------------|-------------------|-----------------|
| FLAP (this)      | Columns     | WIFV (input var × weight L2)  | No (broken)       | Yes (tokenizer) |
| AST (2:4)       | Weights     | Magnitude per block           | N/A               | No              |
| CacheNotes       | Neurons     | Task‑aware heuristic + norm   | No (neuron pruning) | No (task type) |

FLAP’s column pruning is a promising idea, but this implementation is a very early sketch and should not be used as‑is in any production pipeline.

---

## Conclusion

FLAP (registered as `"flap"`) demonstrates the concept of fluctuation‑based structured pruning where input features are removed according to a combined variance–weight importance metric. While the core idea is sound and well‑motivated, the provided code is a simplistic placeholder that does not handle the complexities of real transformer architectures (residuals, parallel branches, layer norm) and contains significant bugs. As such, it serves as an educational example or a starting point for a more robust implementation. To create a functional FLAP compressor, one would need to incorporate proper handling of skip connections, correct bias pruning, real calibration data, and coordinated pruning across parallel sub‑layers. Future work can build upon this skeleton to deliver a truly effective structured pruning tool for the DistilBERT ecosystem.