# KVTC: KV Cache Transform Coding – PCA Decorrelation + Adaptive Quantization + Entropy Coding

**Registered as `kvtc`**  
*Compresses linear layer weights using a transform coding pipeline: PCA-based low‑rank approximation, uniform quantization, and entropy coding estimation for compact weight storage.*

---

## Overview

KVTC (KV Cache Transform Coding) applies ideas from classical signal transform coding (such as JPEG) to compress the weight matrices of linear layers. The method, inspired by the paper *“KV Cache Transform Coding for Compact Storage in LLM Inference”* (Staniszewski & Lancucki, ICLR 2026), consists of three stages:

1. **Decorrelation via PCA**: The weight matrix is decomposed with SVD, and only the principal components that retain a specified fraction of the total energy (e.g., 95%) are kept. This yields a low‑rank approximation that captures the most informative structure.
2. **Adaptive quantization**: The low‑rank weight is then uniformly quantized to a user‑specified bit‑width (e.g., 4 bits), using its min‑max range.
3. **Entropy coding estimation**: A simple entropy coder computes an **effective compression ratio** by comparing the original 32‑bit float representation to the information content of the quantized weights, based on the number of unique quantized values.

This pipeline reduces the storage requirements for the weight, first by discarding low‑energy singular components, and then by representing the remaining values with fewer bits. The module is designed for the DistilBERT compression framework and is registered under `"kvtc"`. It works purely on the weight tensors without calibration data.

---

## Methodology

### 1. PCA Decorrelation via SVD

For a weight matrix **W** ∈ ℝ<sup>m×n</sup>, the singular value decomposition (SVD) is performed:

```
W = U Σ V^T
```

with singular values `σ₁ ≥ σ₂ ≥ … ≥ σ_r ≥ 0`. The total energy is `E = Σ σ_i²`. We keep the smallest rank `k` such that the cumulative energy reaches a fraction `pca_energy` (e.g., 0.95):

```
k = min { t | (Σ_{i=1}^{t} σ_i²) / E ≥ pca_energy }
```

This rank is then bounded between 4 and `min(m,n)` to avoid degenerate cases. The low‑rank approximation is:

```
W_low = U_k Σ_k V_k^T
```

where **U_k** contains the first `k` left singular vectors, **Σ_k** is the diagonal matrix of the top‑`k` singular values, and **V_k^T** the corresponding right vectors. This step removes noise and less important variation, concentrating the information into a smaller dimensional subspace.

### 2. Adaptive Quantization

The low‑rank matrix **W_low** is quantized using a uniform affine scheme. Let `min_val` and `max_val` be its minimum and maximum entries. To avoid division by zero when all entries are equal, a small epsilon is added if `max_val == min_val`. The quantization step size is:

```
scale = (max_val - min_val) / (2^b - 1)
```

where `b` is the target bit‑width (e.g., 4). Each element `w` is mapped to:

```
w_q = round( (w - min_val) / scale ) * scale + min_val
```

This restricts `W_low` to at most `2^b` distinct levels (though due to min‑max asymmetry, the actual number may be slightly less). The quantized weight replaces the original in the model, stored in the original floating‑point dtype but with low entropy.

### 3. Entropy Coding Estimation

The `SimpleEntropyCoder` class provides a theoretical compression ratio estimate. It counts the number of unique values in the quantized tensor. If the number of unique values is `U`, the information content per element is `log₂(U)` bits (assuming uniform distribution). The original weight was stored with 32 bits (float32). The estimated compression ratio is:

```
compression_ratio = 32 / log₂(max(U, 1))
```

For example, if after 4‑bit quantization there are exactly 16 unique levels, the ratio is `32 / 4 = 8`. The compressor records the average ratio across all compressed layers for diagnostics. This ratio is not used to further compress the weights; it merely indicates the potential savings if an entropy code (e.g., Huffman) were applied.

---

## Implementation Details

### KVTCCompressor Class

```python
class KVTCCompressor(BaseCompressor):
    def __init__(self, model, bits=4, pca_energy=0.95, **kwargs):
        super().__init__(model, name="kvtc", **kwargs)
        self.bits = bits
        self.pca_energy = pca_energy
```

- **`bits`**: Quantization bit‑width (default 4). Determines the number of quantization levels (≤ 2^bits).
- **`pca_energy`**: Fraction of singular value energy to retain (default 0.95). Controls the rank of the low‑rank approximation.

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. For each weight tensor **W** (float32):
   - Computes SVD.
   - Determines rank `k` via cumulative energy threshold.
   - Forms low‑rank approximation `W_low`.
   - Quantizes to `self.bits` bits using min‑max range.
   - Computes compression ratio estimate via `SimpleEntropyCoder`.
   - Writes `W_quant` back to `module.weight.data` in original dtype.
3. Records average compression ratio in `self._compression_stats`.
4. Updates `_compressed_params` and returns model with stats.

**Note**: Biases are not compressed. The SVD is performed on GPU via PyTorch; no CPU offloading. The final weight remains a full dense matrix; only the effective bit precision is reduced.

### SimpleEntropyCoder

A static helper that computes `32 / log2(unique_vals)` as an estimate.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("kvtc")
def kvtc_factory(model, comp_config):
    bits = getattr(comp_config, 'kvtc_bits', 4)
    pca_energy = getattr(comp_config, 'kvtc_pca_energy', 0.95)
    compressor = KVTCCompressor(model, bits=bits, pca_energy=pca_energy)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Configuration parameters `kvtc_bits` and `kvtc_pca_energy` can be set in `comp_config`.

---

## Usage and Configuration

To apply KVTC compression, set the compression method to `"kvtc"` and optionally tune the bit‑width and energy retention.

### Example

```python
comp_config.comp_method = "kvtc"
comp_config.kvtc_bits = 3          # 3‑bit quantization
comp_config.kvtc_pca_energy = 0.9  # keep 90% of energy
```

Then run the compression pipeline. All linear layer weights will be replaced by their quantized low‑rank approximations.

### Parameters

| Parameter           | Default | Description                                                                                            |
|---------------------|---------|--------------------------------------------------------------------------------------------------------|
| `kvtc_bits`         | `4`     | Target bit‑width for uniform quantization. Lower values give higher compression but more accuracy loss.|
| `kvtc_pca_energy`   | `0.95`  | Fraction of singular value energy to retain. Higher values keep more rank (less compression).          |

**Note**: After compression, the weights are still stored as floats but with few distinct values. Further lossless compression (e.g., Huffman) can be applied to the stored tensors to realise the estimated compression ratio.

---

## Discussion

### Advantages

- **Combines low‑rank and quantization**: The PCA step first removes redundant directions, making the subsequent quantization more effective (fewer unique values).
- **Energy‑adaptive rank**: Each layer gets its own rank based on the singular value spectrum, preventing excessive degradation of important layers.
- **No calibration data**: Works purely on weight matrices.
- **Simple and fast**: The SVD and quantization are performed once; no iterative steps.
- **Informative metric**: The estimated compression ratio gives immediate feedback on the potential for further entropy coding.

### Limitations

- **No actual entropy coding**: The weights are not actually stored as Huffman‑encoded bits; the ratio is just an estimate. To achieve the projected storage reduction, a post‑processing step is required.
- **Dense weight output**: The weight remains a full dense matrix; inference speed is unchanged. Memory savings come only if the representation is later converted to a compressed format.
- **Quantization range based on low‑rank matrix**: Using the min‑max of the low‑rank weight (not the original) may lead to suboptimal clipping if the low‑rank approximation altered the range. The original min‑max might be more representative, but the low‑rank matrix is the one being quantized.
- **Bias untouched**: Biases remain in full precision, which is a minor inconsistency.
- **Rank floor of 4**: For very small layers or tight energy thresholds, a minimum rank of 4 is enforced, which might be too high and waste parameters.

### Comparison with Other Compression Methods

| Method              | Stages                           | Output Format        | Potential for Further Lossless Compression |
|---------------------|----------------------------------|----------------------|--------------------------------------------|
| KVTC (this)         | PCA + quantization + entropy est.| Dense float (few levels) | High (Huffman, etc.)                       |
| KMeans              | Clustering (k‑means)             | Centroids + indices  | Moderate (indices can be packed)           |
| AstroQuantizer      | Uniform quantization             | Dense float (few levels) | High                                       |
| A3 (SVD)            | Truncated SVD                    | Dense float          | Low                                        |

KVTC is unique in its explicit integration of an entropy coding estimation step and its data‑free adaptive rank selection.

---

## Conclusion

KVTC (registered as `"kvtc"`) brings a full transform coding pipeline to weight compression: PCA decorrelation for redundancy reduction, adaptive quantization to lower bit precision, and an entropy coder to measure the achievable compression. This holistic approach yields both effective low‑rank and low‑bit representations, and it provides a clear metric for further compression stages. Its simple, calibration‑free nature makes it easy to apply, while the energy‑based rank selection tailors compression to each layer. For users aiming to maximise storage savings in deployed models, KVTC serves as an excellent pre‑processing step before applying lossless entropy encoding. Future extensions could incorporate actual entropy coding and joint fine‑tuning of the quantized weights.