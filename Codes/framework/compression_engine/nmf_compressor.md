# NMF: Non‑Negative Matrix Factorization Approximation

**Registered as `nmf`**  
*Approximates each linear layer’s weight matrix using Non‑Negative Matrix Factorization, enforcing a parts‑based, additive decomposition that can lead to more interpretable low‑rank factors.*

---

## Overview

NMF (Non‑Negative Matrix Factorization) compresses a weight matrix **W** ∈ ℝ<sup>M×N</sup> by first shifting all entries to be strictly positive (to meet the non‑negativity requirement of NMF), then decomposing the shifted matrix into the product of two smaller non‑negative matrices **H** ∈ ℝ<sup>M×r</sup> and **C** ∈ ℝ<sup>r×N</sup> such that:

```
W_shifted ≈ H · C
```

After the decomposition, the original shift is reversed to obtain the final compressed weight. Because NMF enforces non‑negativity on the factors, the approximation is **additive** and tends to capture local, parts‑based patterns in the data, which can be more interpretable than unconstrained SVD. The rank `r` is a user‑specified hyperparameter, and the factorization is computed using scikit‑learn’s NMF implementation. The final weight matrix is reconstructed as a dense matrix of the original shape, preserving the layer’s interface.

The compressor is registered under `"nmf"` and requires no calibration data. It uses scikit‑learn’s `NMF` solver, running on CPU.

---

## Methodology

### 1. Non‑Negativity Transformation

NMF requires the input matrix to have all entries ≥ 0. Since weight matrices can contain negative values, a global shift is applied:

```
W_shifted = W - min(W) + ε
```

where `ε = 1e-5` is a small constant to ensure strict positivity (avoid zeros that may cause issues in multiplicative update algorithms). This transformation simply translates the matrix values without changing the structural relationships.

### 2. Non‑Negative Matrix Factorization

We seek a low‑rank approximation of **W_shifted** of the form:

```
W_shifted ≈ H · C    with H ∈ ℝ^{M×r}, C ∈ ℝ^{r×N}, H_ij ≥ 0, C_ij ≥ 0
```

The decomposition is performed by scikit‑learn’s `NMF` with `n_components=r`, using a random initialization and a maximum of 200 iterations. The algorithm minimizes the Frobenius norm ||W_shifted – H·C||_F via multiplicative update rules.

### 3. Reconstruction of the Original Weight

Once the non‑negative factors are obtained, the low‑rank approximation in the original value space is obtained by shifting back:

```
W_compressed = H · C + min(W) - ε
```

This yields a dense matrix of the same shape as **W** but with rank ≤ r. The shift ensures that the reconstructed weight is roughly in the original numerical range.

### 4. Why NMF?

Unlike SVD, NMF produces factors with no negative entries. This forces the approximation to be a sum of non‑negative parts, which can be beneficial:

- **Interpretability**: Each row of **H** indicates how much each of the `r` basis patterns (rows of **C**) contributes to the corresponding output neuron.
- **Sparsity**: NMF often induces sparsity in the factors, leading to many near‑zero entries and potential for further compression.
- **Different geometry**: The non‑negativity constraint may capture a different kind of redundancy than SVD, sometimes preserving more useful structure for certain layers.

However, NMF typically requires more iterations and can converge to local minima, so the approximation quality may vary across runs or layers.

---

## Complexity Analysis

### A. Compression Time Complexity

For each linear layer of size M×N:

- **NMF** via scikit‑learn uses multiplicative updates, each iteration costing O(M·N·r). With a maximum of 200 iterations, the total time per layer is roughly O(200·M·N·r). For M=N=768, r=64, this is about 200·768·768·64 ≈ 7.5×10⁹ operations, which may take a second or two on CPU per layer. For a model with many linear layers (e.g., 72), total compression time can be on the order of minutes.
- **Shift and reconstruction**: O(M·N), negligible.

### B. Inference Complexity

After compression, the weight matrix is stored as a full dense M×N matrix. The forward pass still performs matrix multiplication of size `y = x·W^T`, costing O(batch·M·N) FLOPs—no reduction. To achieve inference speed‑up, one would need to store and use the factor matrices **H** and **C** explicitly, performing `y = (x·C^T)·H^T`, which would cost O(batch·r·(M+N)). The current code does not do this; it reconstructs the dense weight.

### C. Memory Usage

**During compression**: The weight matrix (M×N floats), the shifted copy, and the NMF factors (H: M×r, C: r×N) require memory of roughly M·N + 2·r·(M+N) floats. For large layers, this is dominated by the weight matrix itself.

**After compression**: The parameter count remains unchanged (full dense weight). There is **no immediate storage reduction**. If the factors were stored instead, the total parameters would reduce from M·N to r·(M+N).

---

## Implementation Details

### NMFCompressor Class

```python
class NMFCompressor(BaseCompressor):
    def __init__(self, model, rank=64, **kwargs):
        super().__init__(model, name="nmf", **kwargs)
        self.rank = rank
```

- **`rank`**: Number of NMF components (default 64). Must be ≤ min(M,N).

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. Converts the weight to float32, moves to CPU, NumPy array.
3. Shifts the matrix to be positive: `W_shifted = W - W.min() + 1e-5`.
4. Runs `skNMF(n_components=self.rank, init='random', max_iter=200)`.
5. Obtains `H = nmf.fit_transform(W_shifted)` (shape M×r) and `C = nmf.components_` (shape r×N).
6. Reconstructs: `W_comp = H @ C + min_val - 1e-5`.
7. Writes back to `module.weight.data` in the original dtype and device.

**Note**: The random initialization may lead to slightly different results across runs; the random seed is not fixed. For reproducibility, one may set the NumPy random seed beforehand.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("nmf")
def nmf_factory(model, comp_config):
    rank = getattr(comp_config, 'nmf_rank', 64)
    compressor = NMFCompressor(model, rank=rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `nmf_rank` from `comp_config`. The default rank is 64.

---

## Usage and Configuration

Set compression method to `"nmf"` and optionally adjust the rank.

### Example

```python
comp_config.comp_method = "nmf"
comp_config.nmf_rank = 32
```

Then run the compression pipeline. All linear layer weights will be replaced by their NMF‑based approximations.

### Parameters

| Parameter   | Default | Description                                                     |
|-------------|---------|-----------------------------------------------------------------|
| `nmf_rank`  | `64`    | Number of NMF components per layer. Lower = more compression.   |

---

## Discussion

### Advantages

- **Non‑negativity constraint**: Forces an additive, parts‑based decomposition that can produce sparser, more interpretable factors.
- **Avoids negative values in factors**: This can be beneficial for subsequent quantization or for layers where negative correlations are less meaningful.
- **No calibration data**: Works purely on the weight matrix.
- **Different approximation space**: May capture patterns that SVD misses due to orthogonality constraints.

### Limitations

- **Longer compression time**: NMF iterative optimization is slower than SVD (especially for many layers).
- **Local minima**: NMF is non‑convex and may converge to different solutions depending on initialization. Accuracy may vary.
- **Dense output**: Like SVD‑based methods, the weight remains a full dense matrix; no memory or FLOP reduction unless factors are explicitly stored.
- **Shift distortion**: The additive shift changes the matrix before factorization; reversing it does not completely undo the effect on the approximation error because NMF operates on the shifted values. The relative importance of weights can be slightly distorted if the shift is large.
- **Positive bias**: The small ε and shift may cause a bias in the reconstructed weights, especially for layers where the minimum is very negative.
- **Not online**: Uses scikit‑learn’s CPU‑only NMF, not GPU.

### Comparison with SVD‑Based Compressors

| Method         | Decomposition | Constraint | Speed | Interpretability | Output      |
|----------------|---------------|------------|-------|------------------|-------------|
| NMF (this)     | NMF           | Non‑negativity | Slower | Higher (parts‑based) | Dense weight |
| A3 (SVD)       | SVD           | None        | Fast  | Lower            | Dense weight |
| CUR            | Column/row    | None        | Fast  | High (real data)  | Dense or factors |

NMF is a niche tool for scenarios where non‑negative factors are desired, e.g., to interface with hardware that only supports non‑negative coefficients, or to obtain interpretable weight patterns.

---

## Conclusion

NMF (registered as `"nmf"`) provides an alternative low‑rank approximation method for weight compression, enforcing non‑negativity through a shift+factor+unshift pipeline. While its compression speed is slower than SVD, it offers a different inductive bias that may preserve certain structures better and yields sparse, interpretable factors. The method is easy to apply via the factory, requiring only a rank parameter. For actual parameter savings and inference speed‑up, the factor matrices should be extracted and used as separate linear layers. Future improvements could include direct non‑negative layer replacements and GPU‑accelerated NMF solvers.