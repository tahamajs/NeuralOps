# LSC: Fourier‑Based Low‑Rank Compression

**Registered as `lsc`**  
*Compresses weight matrices by discarding low‑magnitude frequency components in the 2D Fourier domain, preserving only the top‑k strongest spectral modes.*

---

## Overview

LSC (Low‑rank Spectral Compression) applies the 2D discrete Fourier transform (DFT) to each linear layer’s weight matrix, retains only the `k = rank²` frequency components with the largest magnitude, zeros out the rest, and transforms back to the spatial domain. The resulting weight matrix is a dense approximation of the original, now containing only the most energetic spectral patterns. The method is purely algebraic, requires no calibration data, and treats each layer independently. It is especially suited when weight matrices exhibit pronounced periodic or smooth structures that concentrate energy in a few Fourier modes.

The compressor is registered under `"lsc"` in the DistilBERT compression framework and is controlled by a single rank parameter (the square root of the number of preserved frequency components).

---

## Methodology

### 1. Two‑Dimensional Discrete Fourier Transform

For a weight matrix **W** ∈ ℝ<sup>M×N</sup>, the 2D DFT is:

```
F(ω₁, ω₂) = Σ_{m=0}^{M-1} Σ_{n=0}^{N-1} W[m, n] · exp( -2πi (ω₁ m / M + ω₂ n / N) )
```

This decomposes the weight matrix into a sum of complex sinusoidal basis functions, each with a magnitude and phase. The magnitudes indicate the strength of each 2D frequency component.

### 2. Truncation by Magnitude

After computing the 2D FFT, the algorithm retains the `r = rank²` frequency bins with the **largest absolute values** (magnitudes). All other bins are set to zero. The selection is global across all frequencies, not restricted to low or high frequencies, so it adapts to the weight’s specific spectral content.

Let **F** be the complex FFT matrix; define a mask **M** where `M[ω₁,ω₂] = 1` if the magnitude `|F[ω₁,ω₂]|` is among the top‑`r` largest, else `0`. The truncated spectrum is:

```
F_trunc = F ⊙ M
```

### 3. Inverse Transform

The compressed weight matrix is obtained by applying the inverse 2D DFT to the truncated spectrum, taking the real part:

```
W_compressed = Re( IFFT2(F_trunc) )
```

Because the original weight matrix is real, the retained frequencies (which are not Hermitian‑symmetric after this arbitrary truncation) would yield a complex output; discarding the imaginary part introduces an additional approximation error, but it is typically small.

The final weight matrix has the same shape as the original and contains only those spatial patterns that can be represented by the `r` strongest Fourier modes.

### 4. Rank Interpretation

The parameter `rank` determines the number of kept frequency components `k = rank²`. A larger `rank` preserves more modes, leading to a better approximation but less compression (in terms of information content). Since the weight matrix after compression is still stored as a full dense matrix, the memory footprint does **not** change—only the effective number of degrees of freedom is reduced. This can be exploited later if the Fourier coefficients are stored explicitly (i.e., only the `k` complex coefficients and their positions are saved), but the current implementation does not do that.

---

## Complexity Analysis

### A. Compression Time Complexity

For a weight matrix of size `M × N`:

- **2D FFT**: O(M N log(M N)) using the fast Fourier transform. In practice, `np.fft.fft2` costs O(M N log(M N)).
- **Magnitude sorting**: O(M N log(M N)) to sort all magnitudes and pick the top‑`r`.
- **IFFT**: O(M N log(M N)).
Overall time per layer is O(M N log(M N)). For typical hidden sizes (e.g., 768 × 768), this is fast.

### B. Inference Complexity

After compression, the weight matrix remains a full dense matrix of shape `M × N`. The forward pass still performs a standard matrix multiplication `y = x·W^T + bias`, which costs O(batch · M · N) FLOPs—exactly the same as before compression. There is **no inference speed‑up** from this method alone.

### C. Memory Usage

**During compression**:
- The full weight matrix (M N floats) and its complex FFT (2 · M N floats if stored as complex) require O(M N) memory.
- The sorting of magnitudes requires additional O(M N) temporary storage.
Total peak memory is a few times the weight matrix size.

**After compression**:
- The weight is still stored as a dense matrix of M N floats. There is **no reduction in parameter count or model size on disk**. The compression is only in terms of the number of independent spectral components; the storage format does not exploit the sparsity in the frequency domain.

If one were to store only the `r` complex coefficients and their indices, the storage could be reduced from `M N` floats to roughly `r · (2 + 2)` (complex value + two integer coordinates) = `4 r` numbers, which for `r ≪ M N` yields massive savings. However, such a sparse Fourier representation is not implemented here.

---

## Implementation Details

### LSCCompressor Class

```python
class LSCCompressor(BaseCompressor):
    def __init__(self, model, rank=32, **kwargs):
        super().__init__(model, name="lsc", **kwargs)
        self.rank = rank
```

- **`rank`**: Square root of the number of retained frequency components (default 32, i.e., keep `32² = 1024` modes).

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. Extracts the weight tensor, moves to CPU, converts to float32 NumPy array.
3. Computes `W_fft = np.fft.fft2(W)` (complex 2D FFT).
4. Computes magnitudes `np.abs(W_fft)`, flattens, and finds the indices of the top `rank*rank` largest magnitudes.
5. Creates a binary mask `M` of the same shape as `W_fft`, zero everywhere, then sets `mask.flat[flat_idx] = 1`.
6. Applies mask: `W_fft_masked = W_fft * mask`.
7. Inverse FFT: `W_comp = np.fft.ifft2(W_fft_masked).real`.
8. Writes the compressed weight back to `module.weight.data`, preserving original dtype and device.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("lsc")
def lsc_factory(model, comp_config):
    rank = getattr(comp_config, 'lsc_rank', 32)
    compressor = LSCCompressor(model, rank=rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The configuration parameter `lsc_rank` can be set in `comp_config`.

---

## Usage and Configuration

To apply LSC compression, set `comp_config.comp_method = "lsc"` and optionally adjust the rank.

### Example

```python
comp_config.comp_method = "lsc"
comp_config.lsc_rank = 16   # keep 256 strongest frequencies
```

Then run the compression pipeline. All linear layers’ weights will be replaced by their Fourier‑truncated approximations.

### Parameters

| Parameter    | Default | Description                                                                                              |
|--------------|---------|----------------------------------------------------------------------------------------------------------|
| `lsc_rank`   | `32`    | Square root of the number of retained frequency components. Higher values preserve more spectral detail. |

**Note**: The number of kept frequencies is `rank * rank`. For `rank=32`, this is 1024 modes, which is far less than the total `768*768=589,824` pixels. The compression in terms of degrees of freedom is therefore ≈ 576×. However, the actual stored weight size does not change.

---

## Discussion

### Advantages

- **Efficient compression of periodic patterns**: If the weight matrix has strong repetitive structures (e.g., from convolutional‑like patterns), the Fourier domain can represent them very compactly.
- **Global frequency selection**: The method automatically picks the most energetic frequencies, which can be anywhere in the spectrum.
- **Fast compression**: O(M N log(M N)) per layer using highly optimized FFT libraries.
- **No calibration data**: Works purely on weights.
- **Simple**: Straightforward to implement.

### Limitations

- **No storage or inference benefit**: The weight matrix remains dense; there is no reduction in memory or compute. The method only reduces the *effective rank* in a spectral sense, which may be used later for analysis or as a pre‑processing step for other compression.
- **Arbitrary truncation breaks Hermitian symmetry**: Real‑valued signals require Hermitian symmetry in the Fourier domain. By zeroing out components arbitrarily, that symmetry is lost, and the inverse FFT yields a complex signal. Taking only the real part introduces further error. A better approach would preserve symmetry, e.g., by keeping frequency pairs that are symmetric.
- **Fixed number of kept frequencies**: The same `rank²` components are kept for all layers, disregarding their actual spectral concentration.
- **Sensitivity to magnitude ordering**: Small changes in magnitudes near the cutoff can flip which frequencies are kept, leading to instability.
- **No explicit low‑rank structure**: The resulting matrix does not have a factorised low‑rank form; it is just a filtered version of the original. Consequently, it cannot be directly used for accelerated inference via smaller matrix multiplications.

### Potential Extensions

- **Store sparse Fourier representation**: Instead of the dense weight, save only the top Fourier coefficients (complex values + coordinates). Then, at inference, compute the linear transformation by taking the IFFT of the sparse spectrum—though this is typically much slower unless the number of coefficients is extremely small.
- **Symmetry‑preserving truncation**: Keep frequency pairs `(ω₁,ω₂)` and `(-ω₁,-ω₂)` together to ensure a real output.
- **Per‑layer adaptive `rank`**: Use an energy threshold on the Fourier magnitudes to determine how many coefficients to keep.

---

## Conclusion

LSC (registered as `"lsc"`) offers a novel spectral view of weight compression: by keeping only the strongest Fourier components, it approximates the weight matrix with a few dominant 2D sinusoids. While the current implementation does not reduce model size or inference cost, it provides a unique perspective on the spatial structure of weights and can serve as a pre‑processing step for more advanced compression techniques. Future work could transform this into a storage‑efficient format by keeping only the sparse Fourier coefficients and performing fast Fourier‑based inference.