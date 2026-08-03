# KMeans‑Based Weight Clustering Compressor

**Registered as `kmeans`**  
*Compresses linear layers by clustering their weight values into a small set of centroids, storing only the centroids and an index map instead of the full weight matrix.*

---

## Overview

The KMeans Compressor replaces each `nn.Linear` layer with a `KMeansLinear` module that represents the original weight matrix using **k‑means clustering**. The full float weight values are partitioned into a user‑specified number of clusters, and each weight is replaced by its cluster’s centroid. The compressed layer stores only:

- A vector of `n_clusters` centroid values (floats),
- An index tensor (integers) of the same shape as the original weight, indicating which centroid each weight corresponds to,
- The bias (if present).

During the forward pass, the full weight matrix is reconstructed on‑the‑fly by indexing into the centroid vector. This reduces the storage for the weight from `out_features × in_features` floating‑point numbers to `n_clusters` floats plus `out_features × in_features` integers (which can be stored with fewer bits). While the number of stored elements is not drastically reduced (actually, the index map has the same number of entries), the entropy is much lower because the values are restricted to a small set, enabling further compression with techniques like Huffman coding. Moreover, the index map can be stored with only ⌈log₂(n_clusters)⌉ bits per entry, offering a significant bit‑width reduction.

The module is designed for the DistilBERT compression framework and is registered under `"kmeans"`. It requires the `scikit‑learn` library for the k‑means clustering step, which is performed once during compression.

---

## Methodology

### 1. Weight Clustering via k‑Means

Given the weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup> of a linear layer, all entries are flattened into a 1D array. The k‑means algorithm is applied to this array (treated as a set of 1D points) to find `k` centroids that minimize the within‑cluster sum of squared errors. Mathematically:

```
min_{c₁,…,c_k} Σ_i min_j ‖W_i – c_j‖²
```

where `W_i` are the flattened weight values and `c_j` are the centroids. The algorithm used is the standard k‑means from `sklearn.cluster.KMeans` with a fixed random seed for reproducibility (`random_state=42`).

The result is:
- `centroids`: a 1D array of `k` floating‑point numbers,
- `labels`: an array of the same shape as **W** (reshaped from the flat labels), containing the cluster index for each weight position.

### 2. Parameter Storage Reduction

The original weight requires `d_out * d_in` floating‑point numbers. The compressed representation consists of:

- `k` floating‑point numbers (centroids),
- `d_out * d_in` integers (cluster indices).

If the indices are stored as a dense integer tensor (e.g., `long` by default, 64 bits), the total storage may even increase. However, the real compression comes from:

- **Quantization**: The weights are effectively quantized to `k` distinct values. For small `k` (e.g., 256 = 8 bits), each index requires only 8 bits, meaning the weight matrix can be stored with `d_out * d_in * 8` bits instead of `d_out * d_in * 32` bits (if float32). This is a 4× reduction.
- **Further lossless compression**: Since many adjacent weights might belong to the same cluster (especially after pruning or in structured matrices), the index map can be compressed with run‑length encoding or Huffman coding.
- **Potential for low‑bit inference**: At runtime, the centroids can be stored in full precision, while the indices are fetched in low‑bit format and used to gather the centroids.

### 3. Forward Pass Reconstruction

The `KMeansLinear.forward()` method reconstructs the weight matrix on‑the‑fly:

```python
W = self.centroids[self.indices].to(x.dtype)
return F.linear(x, W, self.bias)
```

The indexing operation produces a dense weight matrix of the same shape as the original, which is then used in a standard linear transformation. This approach keeps the forward interface identical to `nn.Linear`, but the computation is still a full matrix multiplication; no FLOP reduction occurs—only storage compression. However, in a memory‑constrained environment, the ability to avoid storing a full float matrix can be beneficial.

### 4. Compression Process

The `KMeansCompressor` replaces all `nn.Linear` layers with `KMeansLinear`. The steps:

1. For each `nn.Linear`:
   - Fit k‑means on the flattened weight (CPU, NumPy via scikit‑learn).
   - Create a `KMeansLinear` instance with the computed centroids, indices, and bias.
   - Replace the module in the model.
2. Update the internal parameter count: the parameter count now includes centroids (k floats), bias, and indices (stored as a buffer, not counted as a parameter? The buffer is registered, but `_count_params` likely counts only `nn.Parameter` tensors, so indices may not be included. This could lead to an underestimation of the actual memory. We'll note that.)

---

## Implementation Details

### KMeansLinear Class

```python
class KMeansLinear(nn.Module):
    def __init__(self, linear: nn.Linear, n_clusters: int):
        ...
```

- **`linear`**: original linear layer.
- **`n_clusters`**: number of clusters for k‑means.

The constructor:
- Flattens the weight to 1D, runs `KMeans`.
- Saves `centroids` as a buffer (float32 tensor of size `(n_clusters,)`).
- Saves `indices` as a long tensor buffer of shape `(out_features, in_features)`.
- Copies bias if present.

**`forward(x)`**:
- Reconstructs weight via `centroids[indices]`, casts to input dtype, applies `F.linear`. Note: the code snippet uses `F.linear` but `F` is not imported; in actual code, `torch.nn.functional` must be imported.

### KMeansCompressor Class

```python
class KMeansCompressor(BaseCompressor):
    def __init__(self, model, n_clusters=256, **kwargs):
        super().__init__(model, name="kmeans", **kwargs)
        self.n_clusters = n_clusters
```

`compress()`:
- Iterates over all `nn.Linear`.
- Creates `KMeansLinear` replacement.
- Uses `_get_parent` to navigate and set the new layer.
- Updates `_compressed_params`.

**Note**: `_compressed_params` will count parameters in `KMeansLinear` (centroids as buffer? If centroids are registered via `register_buffer`, they are not `nn.Parameter` and might not be counted. This could incorrectly report a large parameter reduction. In practice, one would count the number of elements of centroids (k) plus bias. The indices buffer consumes memory but is not a parameter; a more accurate accounting would consider total memory bytes.)

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("kmeans")
def kmeans_factory(model, comp_config):
    n_clusters = getattr(comp_config, 'kmeans_clusters', 256)
    compressor = KMeansCompressor(model, n_clusters=n_clusters)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The configuration parameter `kmeans_clusters` sets the number of clusters (default 256).

---

## Usage and Configuration

To compress a model using k‑means clustering, set the compression method to `"kmeans"` and optionally adjust the number of clusters.

### Example

```python
comp_config.comp_method = "kmeans"
comp_config.kmeans_clusters = 128   # 128 distinct weight values per layer
```

Then run the compression pipeline. All linear layers will be replaced with `KMeansLinear` instances.

### Parameters

| Parameter          | Default | Description                                                                                               |
|--------------------|---------|-----------------------------------------------------------------------------------------------------------|
| `kmeans_clusters`  | `256`   | Number of clusters (centroids) per linear layer. Smaller values give higher compression but more accuracy loss. |

**Note**: The k‑means fitting uses `random_state=42`; results are deterministic. The method requires `scikit‑learn` installed.

---

## Discussion

### Advantages

- **Quantization via clustering**: Replaces the continuous weight values with a small codebook, enabling low‑bit storage.
- **Preserves linear interface**: The forward pass is still a dense matrix multiplication; no change to downstream logic.
- **Simple concept**: Easy to understand and implement.
- **Fine‑tuning possible**: The centroids can be trained further; the indices remain fixed (or could be jointly optimized).
- **Extreme compression potential**: For large matrices, using e.g., 256 clusters reduces the effective weight storage to 8 bits per element plus the small codebook.

### Limitations

- **Storage of indices**: The index map is stored as a `long` tensor by default, which may actually increase memory unless a custom storage format is used (e.g., packing 8‑bit indices). In the current implementation, the indices buffer uses 64 bits per entry, more than float32, so the *memory* of the layer might increase. This is a critical flaw if the goal is memory reduction without custom packing.
- **No FLOP reduction**: The forward pass still computes a full matrix multiply; there is no speed‑up.
- **Requires sklearn**: The k‑means step relies on scikit‑learn, which may not be available in all environments.
- **Cluster number must be chosen**: Too few clusters cause large approximation error; too many reduce compression.
- **All layers treated equally**: Same number of clusters for all layers, which may be suboptimal.
- **No activation awareness**: Clustering is purely based on weight values; some weights may be more important than others but are treated equally.

### Comparison with Other Quantization Methods

| Method               | Quantization Type   | Storage Format            | Inference Speed‑up |
|----------------------|---------------------|---------------------------|--------------------|
| KMeans (this)        | Clustering (non‑uniform) | Centroids + indices (dense int) | No (dense matmul) |
| AstroQuantizer       | Uniform min‑max     | Dense floats with limited levels | No              |
| Huffman coding       | Lossless            | Bit‑strings               | No                |

KMeans is a form of non‑uniform quantization where the codebook is learned from the weight distribution. However, the storage format in this implementation does not realize the bit‑width reduction unless the indices are packed.

---

## Conclusion

The KMeans‑based weight clustering compressor (registered as `"kmeans"`) provides an alternative quantization approach by replacing weight values with a small set of centroids learned via k‑means. While the concept enables significant reduction in the *entropy* of the weight matrix, the current implementation stores the index map in a high‑bit integer format, which may negate memory savings unless further compression is applied. It serves as a foundation for more advanced compression pipelines where the indices are packed or used with custom low‑bit kernels. Users should be aware of the memory trade‑off and consider pairing this compressor with a lossless coding step. Future improvements could include native support for low‑bit index storage and optional joint fine‑tuning of centroids.