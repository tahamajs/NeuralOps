# SAES‑SVD: Joint Intra‑ and Inter‑Layer Error Compensation

**Registered as `saes`**  
*Applies truncated SVD to each linear layer’s weight matrix; designed as a placeholder for a future method that would correct approximation errors jointly across layers.*

---

## Overview

SAES‑SVD (Structured Adaptive Error Suppression via SVD) is a post‑training weight compression module that, in its current form, performs a simple per‑layer truncated Singular Value Decomposition. For each `nn.Linear` layer, the weight matrix **W** is decomposed via SVD and only the top `r` singular components are retained, resulting in a low‑rank approximation. The parameter `correcting_steps` is provided but not yet used—it is intended for a future mechanism that would propagate and compensate for the approximation error across multiple layers, thereby “suppressing” the accumulated distortion. The current implementation thus serves as a minimal baseline for SVD‑based compression, and the `correcting_steps` flag prepares the interface for a more advanced joint error correction algorithm to be added later.

The compressor is registered under `"saes"` and can be applied to any model with `nn.Linear` layers. It requires no calibration data and works purely on the weight tensors.

---

## Methodology

### 1. Standard Truncated SVD (Current Implementation)

For a weight matrix **W** ∈ ℝ<sup>M×N</sup>, the full SVD is computed:

```
W = U Σ V^H
```

where **U** ∈ ℝ<sup>M×K</sup>, **Σ** ∈ ℝ<sup>K×K</sup> (diagonal), **V^H** ∈ ℝ<sup>K×N</sup>, with `K = min(M,N)`. A target rank `r` is specified (default 64). The low‑rank approximation is:

```
W_approx = U_r Σ_r V_r^H
```

where **U_r** contains the first `r` columns of **U**, **Σ_r** is the top‑left `r×r` block of **Σ**, and **V_r^H** contains the first `r` rows of **V^H**. This is the optimal rank‑`r` approximation in Frobenius norm.

The compressed weight is stored as a full dense matrix; no factorised form is kept. Biases are left unchanged.

### 2. Intended Joint Error Compensation (Future)

The name *SAES‑SVD* suggests a scheme where the approximation error introduced in one layer is **compensated** by adjusting the weights of subsequent layers. For example, the residual **E = W – W_approx** could be propagated forward and the next layer’s weight could be modified to minimise the final output error. The parameter `correcting_steps` would control the number of iterative refinement passes across the entire network. This would lead to better overall accuracy for a given total parameter budget, because errors that would compound are explicitly corrected.

In the current placeholder, this logic is not implemented; the `correcting_steps` parameter is stored but not used. All layers are compressed independently.

---

## Complexity Analysis

### A. Compression Time Complexity

- **Per layer**: Full SVD of an M×N matrix via `np.linalg.svd` costs O(M·N·min(M,N)). For a typical 768×768 matrix, this is roughly 4.5×10⁸ operations, taking a few milliseconds on a modern CPU.
- **Total**: For a model with `L` linear layers, total time is O(L·D³) where D is the hidden size. For DistilBERT (L~72 matrices of 768×768), compression takes a few seconds.

### B. Inference Complexity

After compression, the weight matrix remains dense of the same shape. The forward pass is unchanged: `y = x·W_approx^T + bias`. **No reduction in FLOPs or latency** occurs. The model’s memory footprint also remains unchanged unless the weight is subsequently factorised into two smaller matrices.

### C. Memory Usage

- **During compression**: The weight matrix (M×N floats), and the SVD results (U, Σ, Vh) require a few times the weight size. For a single layer, this is a few megabytes.
- **After compression**: The weight tensor is still M×N floats. Parameter count is unchanged.

---

## Implementation Details

### SAESCompressor Class

```python
class SAESCompressor(BaseCompressor):
    def __init__(self, model, rank=64, correcting_steps=3, **kwargs):
        super().__init__(model, name="saes", **kwargs)
        self.rank = rank
        self.correcting_steps = correcting_steps
```

- **`rank`**: Number of singular components to keep (default 64). Must be between 1 and `min(M,N)`.
- **`correcting_steps`**: (Unused) Intended number of inter‑layer error correction iterations.

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. Converts the weight to float32, moves to CPU, NumPy array.
3. Computes full SVD.
4. Truncates to rank `r = min(rank, min(W.shape))`.
5. Reconstructs `W_approx = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]`.
6. Writes `W_approx` back to the module’s weight, preserving the original dtype and device.
7. Updates the internal parameter count (`_compressed_params`) – note that the number of parameters is the same as the original dense weight.
8. Returns the compressed model and statistics.

### Factory Function (typical, not shown in snippet)

A factory function would be registered under `"saes"` to enable use via the compression configuration:

```python
@COMPRESSOR_REGISTRY.register("saes")
def saes_factory(model, comp_config):
    rank = getattr(comp_config, 'saes_rank', 64)
    correcting_steps = getattr(comp_config, 'saes_correcting_steps', 3)
    compressor = SAESCompressor(model, rank=rank, correcting_steps=correcting_steps)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

---

## Usage and Configuration

Set compression method to `"saes"` and optionally adjust the rank.

### Example

```python
comp_config.comp_method = "saes"
comp_config.saes_rank = 32
```

Then run the compression pipeline. All linear layer weights will be replaced by their rank‑32 SVD approximations.

### Parameters

| Parameter                 | Default | Description                                                                                   |
|---------------------------|---------|-----------------------------------------------------------------------------------------------|
| `saes_rank`               | `64`    | Target rank of the truncated SVD. Lower values give more compression but higher approximation error. |
| `saes_correcting_steps`   | `3`     | **Currently unused.** Reserved for future joint error compensation iterations.                  |

---

## Discussion

### Advantages

- **Extremely simple**: The core of the method is a single SVD call, making it easy to understand, debug, and integrate.
- **No calibration data**: Works out‑of‑the‑box on any pretrained model.
- **Optimal in Frobenius norm**: For a fixed rank, the truncated SVD provides the best low‑rank approximation in terms of the Frobenius norm.
- **Foundation for future improvements**: The `correcting_steps` parameter indicates that the module is designed to be extended with inter‑layer error correction, a potentially powerful technique.

### Limitations

- **No actual error compensation**: The current code does not implement any joint layer correction, so the name is aspirational.
- **Dense weight output**: The weight remains a full dense matrix; there is no reduction in memory or computation. The rank reduction is only “effective” unless the matrix is stored in factorised form.
- **Global fixed rank**: All layers receive the same rank, which may be suboptimal; some layers may need higher rank than others.
- **Bias not compressed**: Biases are not adjusted, which may cause a slight mismatch.
- **Correcting_steps ignored**: Users expecting a sophisticated error‑suppression algorithm will be disappointed; the module currently is equivalent to a basic SVD compressor (like A3).

### When to Use

- As a quick baseline for SVD compression.
- As a starting point for implementing a full SAES‑SVD algorithm that propagates residuals.
- In combination with a subsequent factorisation step to achieve real parameter reduction.

### Comparison with Similar Compressors

| Method       | Decomposition | Per‑Layer Adaptivity | Error Compensation | Output Format      |
|--------------|---------------|----------------------|--------------------|--------------------|
| SAES‑SVD (this) | Truncated SVD | No                   | None (placeholder) | Dense weight       |
| A3           | Truncated SVD | No                   | None               | Dense weight       |
| ERC‑SVD      | Truncated SVD | Yes (energy threshold)| None               | Dense weight       |
| OATS         | L + S decomposition | Yes (iterative)  | None               | Dense weight       |

SAES‑SVD is essentially identical to A3 but with a placeholder parameter for future use.

---

## Conclusion

SAES‑SVD (registered as `"saes"`) provides a minimal but extensible compression module based on truncated SVD. While its current behaviour is that of a simple per‑layer low‑rank approximation, the architecture is prepared for the addition of inter‑layer error compensation via the `correcting_steps` parameter. This makes it a convenient scaffold for researchers who want to experiment with joint error correction techniques without rewriting the entire compression pipeline. For immediate practical compression, one can use it as a no‑fuss SVD compressor, but to achieve real memory and computational savings, a factorised storage and custom linear layer must be employed. Future work will implement the actual error propagation and compensation, turning the placeholder into a full‑fledged SAES‑SVD method.