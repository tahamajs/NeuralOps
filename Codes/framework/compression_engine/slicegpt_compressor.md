# SliceGPT PCA‑Based Slicing Compressor

**Registered as `slicegpt_pca`**  
*A simplified, illustrative implementation of the SliceGPT idea: uses PCA on a reference layer’s activations to determine how many input dimensions to keep, then uniformly prunes the weight matrices of all linear layers by trimming their input features.*

---

## Overview

SliceGPT is a compression technique that leverages principal component analysis (PCA) to reduce the hidden dimension of a Transformer model. The original SliceGPT method applies an orthogonal transformation derived from PCA to the weight matrices and embeddings, effectively “slicing off” the least important directions and reducing the model’s width without changing the network structure. This implementation is a minimal, stand‑alone approximation of that idea: it collects the output activations of a reference layer (or the first linear module) over a calibration set, performs PCA, and then prunes the **input columns** of every `nn.Linear` layer to keep only the top `k` principal components that explain a fraction `(1 – sparsity)` of the variance. The result is a model where all layers have a reduced input dimension, but **no adjustment** is made to the output dimensions of preceding layers or to embedding sizes. Consequently, this code is a conceptual placeholder that demonstrates the PCA‑guided dimension selection but **does not produce a directly usable compressed model** due to dimension mismatches across layers.

The compressor is registered under `"slicegpt_pca"` and requires a calibration data loader. It is not meant for production use as‑is; a full SliceGPT implementation would also apply the PCA projection matrix to the weights in a more sophisticated, dimension‑aligned manner.

---

## Methodology

### 1. Activation Collection and PCA

A forward hook is attached to a **single reference layer** — preferably the first encoder layer (`'encoder.layer.0'`). If that layer is not found, the hook falls back to the first `nn.Linear` encountered. The model is run on the calibration loader, and the **output activations** of that module are stored. All collected activation vectors are concatenated and reshaped to a 2D matrix **X** ∈ ℝ<sup>N×D</sup>, where `N` is the total number of tokens and `D` is the hidden size.

Principal component analysis (PCA) is then performed on **X** using scikit‑learn’s `PCA(n_components=D)`. The explained variance ratios `λ_i / Σ λ_j` are computed for each principal component, and the cumulative sum is calculated:

```
C(k) = ∑_{i=1}^{k} explained_variance_ratio_i
```

Given a user‑specified **sparsity** `s` (e.g., 0.3), the algorithm retains enough components to preserve a total variance fraction of `1 – s`. Concretely, the number of kept dimensions `k` is:

```
k = min{ i | C(i) ≥ 1 – s } + 1
```

Thus, if `s = 0.3`, we keep enough components to explain at least 70% of the total variance.

### 2. Uniform Weight Pruning

Once `k` is determined, the compressor iterates over **all `nn.Linear` modules** in the model and modifies their weight tensors:

```
W' = W[:, :k]     (shape out_features × k)
```

If a bias exists, it is also truncated:

```
b' = b[:k]        (should be of length out_features, not in_features; this is a bug)
```

Note: the bias of a linear layer is of size `out_features`. Truncating it to `k` (the input dimension) is conceptually wrong unless `out_features` equals `k`, which generally is not the case. This is a coding error. The intention is to prune the **input features**, not the output neurons. The bias should remain unchanged.

### 3. Intended (but Missing) Global Alignment

In the genuine SliceGPT algorithm, the PCA is used to compute a semi‑orthogonal matrix **Q** ∈ ℝ<sup>D×k</sup> that projects the hidden states to the lower‑dimensional space, and then both the embedding matrix and all weight matrices are transformed accordingly to maintain consistency. The approach here is a rough simplification that only truncates the input dimension, and thus breaks the model’s dimensional integrity. It is presented as a starting point for understanding the PCA‑based dimension selection mechanism.

---

## Complexity Analysis

### A. Compression Time Complexity

- **Activation collection**: One forward pass over the calibration dataset. Complexity depends on model size and dataset size.
- **PCA**: O(N·D·min(N,D) + D³) using standard SVD. For N ≫ D (many tokens), the dominating factor is O(N·D²). For a typical calibration set of a few thousand tokens and D=768, this is fast.
- **Weight pruning**: O(L·D) for iterating over all linear layers and slicing tensors – negligible.

### B. Inference Complexity (if the model were functional)

If dimension alignment were correct, the weight matrices would have shape `(out_features, k)` instead of `(out_features, D)`, reducing the per‑token FLOPs of each linear layer from `out_features · D` to `out_features · k`. The speed‑up factor would be approximately `D / k`. However, because this implementation does not adjust the output dimensions of preceding layers, the model cannot run correctly.

### C. Memory Usage

**After compression**: The weight matrices have fewer columns (`k` instead of `D`), reducing the parameter count. The total parameters become roughly `(k/D)` of the original linear layer parameters. Biases remain unchanged in the correct implementation. However, due to dimension mismatches, the model is not loadable for inference.

---

## Implementation Details

### SliceGPTPCACompressor Class

```python
class SliceGPTPCACompressor(BaseCompressor):
    def __init__(self, model, calibration_loader, sparsity=0.3, **kwargs):
        super().__init__(model, name="slicegpt_pca", **kwargs)
        self.calibration_loader = calibration_loader
        self.sparsity = sparsity
```

- **`calibration_loader`**: PyTorch DataLoader yielding input batches (dictionaries of tensors).
- **`sparsity`**: Fraction of variance to **discard**. Must be in [0,1). Default 0.3 → keep 70% variance.

**`compress()` method**:

1. Attaches a forward hook to the first encoder layer (or first `nn.Linear`).
2. Passes all calibration batches through the model, collecting the output activations of that single module.
3. Removes the hook.
4. Concatenates activations, runs PCA with `n_components=D`.
5. Computes cumulative explained variance; finds `k` where cumsum ≥ 1 – sparsity.
6. For every `nn.Linear`:
   - Replaces `weight` with `weight[:, :k]`.
   - Replaces `bias` with `bias[:k]` (bug: should be untouched).
7. Updates `_compressed_params` and returns the model with stats.

**Critical flaw**: Bias truncation and the lack of output dimension alignment mean the resulting model will have shape mismatches when loaded or run.

### Factory Function (not shown but presumably similar)

Typically, a factory would read `slicegpt_pca_sparsity` and `calibration_loader` and create the compressor.

---

## Usage and Configuration

Set compression method to `"slicegpt_pca"` and provide a calibration loader.

### Example

```python
comp_config.comp_method = "slicegpt_pca"
comp_config.sparsity = 0.4   # discard 40% variance → keep 60%
```

Then run the compression pipeline. **Warning**: The compressed model will be inconsistent; this is a proof‑of‑concept only.

### Parameters

| Parameter       | Default | Description                                                                                 |
|-----------------|---------|---------------------------------------------------------------------------------------------|
| `sparsity`      | `0.3`   | Fraction of activation variance to discard. Higher values → more compression (smaller `k`). |

**Note**: The calibration loader must yield batches in the same format as the model’s forward method expects (e.g., `input_ids`, `attention_mask`).

---

## Discussion

### Advantages

- **Conceptually simple**: Demonstrates the idea of using PCA on activations to find the most important dimensions.
- **Data‑aware**: The pruning is guided by actual data statistics, not just weight magnitudes.
- **Uniform reduction**: All layers receive the same new input dimension, which can be convenient if the model’s architecture supports global dimension reduction.

### Limitations

- **Non‑functional model**: The implementation does not adjust the output dimensions of layers or the embedding matrix, resulting in shape mismatches and an unusable compressed model. The bias truncation is erroneous. This code is purely illustrative.
- **Single reference layer**: PCA is performed on only one layer’s activations, assuming that the principal directions are consistent across the entire network. In practice, different layers may have different importance distributions.
- **No post‑processing**: A full SliceGPT implementation would apply the PCA projection matrix to all weights and embeddings to keep the model internally consistent. This code simply trims columns, which is insufficient.
- **Requires calibration data**: Unlike weight‑only methods, this compressor needs a representative dataset to capture activation statistics.
- **Scikit‑learn dependency**: PCA is performed on CPU via scikit‑learn, which may be slow for very large activation matrices.

### Comparison with Proper SliceGPT

| Feature                  | This simplified version         | Full SliceGPT                    |
|--------------------------|---------------------------------|----------------------------------|
| Dimension alignment      | None (breaks model)             | Orthogonal transformation applied globally |
| Bias handling            | Incorrectly truncated           | Bias projected or kept unchanged |
| Embedding matrix         | Not touched                     | Also projected to reduced dim    |
| Output neurons           | Not pruned (rows unchanged)     | Rows are also pruned via PCA     |
| Use case                 | Conceptual demo                 | Production‑ready compression     |

---

## Conclusion

The SliceGPT PCA‑Based Slicing Compressor (registered as `"slicegpt_pca"`) provides a rough sketch of the dimension reduction idea behind SliceGPT. It successfully determines the number of principal dimensions to keep based on activation variance, but fails to produce a consistent compressed model due to a lack of global dimension alignment and a bias truncation bug. It serves as an educational tool for understanding PCA‑guided compression and as a starting point for building a complete SliceGPT implementation. For practical model compression, a full version that applies the PCA projection matrix to all relevant weight matrices is required. Future work should address these gaps and integrate the proper transformation steps.