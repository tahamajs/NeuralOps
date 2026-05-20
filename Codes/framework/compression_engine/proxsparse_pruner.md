# ProxSparse: Regularized Learning of Semi‑Structured Sparsity

**Registered as `proxsparse`**  
*Iterative soft‑thresholding with an adaptive percentile threshold, gradually pushing the weight matrix toward a sparse pattern while preserving proximity to the original dense weights.*

---

## Overview

ProxSparse is a post‑training pruning method that induces sparsity in the weight matrices of linear layers by repeatedly applying a **soft‑thresholding operator** based on the current magnitude distribution. Unlike hard thresholding that simply zeroes out weights below a fixed percentile, ProxSparse uses a **proximal update**: at each iteration, it computes a sparse target via soft‑thresholding (the proximal operator of the ℓ₁ norm) at a threshold that is the `(1‑sparsity)`‑quantile of the current absolute weights. The current weight is then moved toward this sparse target with a step size `λ`. After several such iterations, the final sparse weight is the last soft‑thresholded version. This approach balances the sparsity constraint with fidelity to the original weight, often yielding better accuracy than one‑shot magnitude pruning at the same sparsity level.

The compressor is registered under `"proxsparse"` and requires no calibration data. It operates purely on the weight tensors and is controlled by three parameters: target sparsity, step size, and number of iterations.

---

## Methodology

### 1. Soft‑Thresholding and Proximal Operators

For a given threshold θ ≥ 0, the **soft‑thresholding operator** S_θ : ℝ → ℝ is defined as:

```
S_θ(w) = sign(w) · max(|w| – θ, 0)
```

This is the proximal operator for the ℓ₁ norm with parameter θ, i.e., it solves:

```
S_θ(w) = argmin_z { ½‖w – z‖² + θ ‖z‖₁ }
```

Soft‑thresholding shrinks all weights toward zero by the amount θ; weights with magnitude less than θ become exactly zero, while larger weights are reduced in magnitude.

### 2. Percentile‑Based Threshold

To enforce a given sparsity fraction `s` (e.g., 0.5 means 50% sparsity), the threshold θ must be set such that exactly a fraction `s` of the entries are zero after soft‑thresholding. Since soft‑thresholding zeros entries with |w| ≤ θ, we need θ to be the `100·(1‑s)`‑th percentile of the absolute weight values. That is, if we want to keep a fraction `(1‑s)` of entries non‑zero, we set θ to the smallest value such that at least `s` fraction of |w| are ≤ θ. Using `np.percentile(np.abs(W), 100 * (1 - self.sparsity))` does exactly this: it returns the threshold below which `(1‑s)` of the absolute values fall. Then all entries with |W| ≤ θ are set to zero, achieving the desired sparsity after soft‑thresholding.

### 3. Iterative Proximal Update

Let **W₀** be the original weight matrix. ProxSparse performs the following iteration for `t = 0, 1, …, steps‑1`:

```
θ_t = Percentile_{100·(1‑s)}( |W_t| )
Z_t = S_{θ_t}(W_t)                     (sparse target)
W_{t+1} = W_t – λ (W_t – Z_t)          (proximal gradient step)
```

where λ ∈ (0, 1] is the step size (`lambda_prox`). This update can be rewritten as a convex combination:

```
W_{t+1} = (1 – λ) W_t + λ Z_t
```

Thus, the weight matrix is slowly blended toward its soft‑thresholded version. Over iterations, the threshold adapts because **W_t** changes, converging to a matrix that is approximately sparse and close to the original. After the loop, the final compressed weight is taken as **Z_steps**, the last sparse target.

The process is reminiscent of the iterative shrinkage‑thresholding algorithm (ISTA) for solving a regularized least‑squares problem: `minimize ½‖W – W₀‖_F² + γ ‖W‖₁` but with a constraint on the number of non‑zeros rather than a fixed γ. By adjusting the threshold each iteration to meet the exact sparsity, the algorithm finds a local solution that respects the cardinality constraint while remaining close to **W₀**.

### 4. Why Not Simple Hard Thresholding?

One‑shot magnitude pruning (keeping the top‑k weights) ignores the effect of pruning on the remaining weights. Soft‑thresholding and the iterative update allow weights to “redistribute” their magnitudes: entries that are moderately important but just below the threshold can be reduced rather than killed, while very important entries survive. The blending via λ provides a smooth path, avoiding an abrupt change that could destroy the weight structure.

---

## Complexity Analysis

### A. Compression Time Complexity

For each linear layer of size `M × N` (output × input):

- **Percentile computation**: O(M N log(M N)) using sorting (numpy’s `percentile` with interpolation likely uses partial sort, which is O(M N log(M N)) in worst case). In practice, for 768×768 ≈ 590k entries, sorting is fast.
- **Soft‑thresholding and update**: O(M N) per iteration.
- Overall per layer: O(steps · M N log(M N)).

With `steps = 5`, the time is dominated by a few sorts per layer. For a full model with many layers, compression time is on the order of seconds.

### B. Inference Complexity

After compression, the weight matrix remains dense but contains zeros. The forward pass still performs a dense matrix multiplication of size `M × N`. **No FLOP reduction** is achieved; unstructured sparsity does not speed up dense linear algebra on standard hardware without specialized sparse kernels. The benefit is purely in terms of storage when combined with sparse matrix formats.

### C. Memory Usage

- **During compression**: A copy of the weight (float32) and temporary arrays for sorting. Memory is O(M N).
- **After compression**: The weight tensor is still `M × N` floats. Storage is not reduced unless the matrix is stored in a sparse format (e.g., COO) and the inference engine supports sparse operations. The parameter count as reported by PyTorch will remain the same.

---

## Implementation Details

### ProxSparsePruner Class

```python
class ProxSparsePruner(BaseCompressor):
    def __init__(self, model, sparsity=0.5, lambda_prox=0.01, steps=5, **kwargs):
        super().__init__(model, name="proxsparse", **kwargs)
        self.sparsity = sparsity
        self.lambda_prox = lambda_prox
        self.steps = steps
```

- **`sparsity`**: Target fraction of weights to be zeroed (default 0.5). The fraction kept is `1 - sparsity`.
- **`lambda_prox`**: Step size for the proximal update (default 0.01). Higher values move faster toward sparsity but may cause instability.
- **`steps`**: Number of iterations (default 5). More steps allow better convergence but increase time.

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. Converts weight to float32, moves to CPU NumPy array.
3. Runs `steps` iterations:
   - Computes threshold via `np.percentile(np.abs(W), 100 * (1 - self.sparsity))`.
   - Computes `W_prox = np.where(np.abs(W) > threshold, W - np.sign(W) * threshold, 0)`.
   - Updates `W = W - self.lambda_prox * (W - W_prox)`.
4. After loop, assigns `module.weight.data = torch.tensor(W_prox, dtype=module.weight.dtype, device=module.weight.device)`.
5. Updates `_compressed_params` and returns model with stats.

**Note**: The final weight is `W_prox` (the last soft‑thresholded version), not the final blended `W`. This ensures exact adherence to the sparsity constraint, as `W_prox` is guaranteed to have the desired sparsity.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("proxsparse")
def proxsparse_factory(model, comp_config):
    sparsity = getattr(comp_config, 'proxsparse_sparsity', 0.5)
    lambda_prox = getattr(comp_config, 'proxsparse_lambda', 0.01)
    steps = getattr(comp_config, 'proxsparse_steps', 5)
    compressor = ProxSparsePruner(model, sparsity=sparsity, lambda_prox=lambda_prox, steps=steps)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `proxsparse_sparsity`, `proxsparse_lambda`, and `proxsparse_steps` from `comp_config`. All have sensible defaults.

---

## Usage and Configuration

Set compression method to `"proxsparse"` and adjust hyperparameters as needed.

### Example

```python
comp_config.comp_method = "proxsparse"
comp_config.proxsparse_sparsity = 0.6   # 60% sparsity
comp_config.proxsparse_lambda = 0.05    # step size
comp_config.proxsparse_steps = 10       # more iterations
```

Then run the compression pipeline. All linear layer weights will be pruned to the target sparsity using the iterative proximal scheme.

### Parameters

| Parameter                 | Default | Description                                                                                     |
|---------------------------|---------|-------------------------------------------------------------------------------------------------|
| `proxsparse_sparsity`     | `0.5`   | Target fraction of zero weights. 1 = all zero, 0 = dense.                                        |
| `proxsparse_lambda`       | `0.01`  | Step size for the proximal update. Must be between 0 and 1.                                       |
| `proxsparse_steps`        | `5`     | Number of iterations. More steps lead to smoother transition but longer compression time.         |

---

## Discussion

### Advantages

- **Soft transition**: The iterative blending avoids the abrupt information loss of one‑shot hard thresholding, potentially preserving more accuracy at the same sparsity level.
- **Adaptive threshold**: The threshold is recomputed each iteration based on the current weight distribution, allowing the pruning pattern to evolve.
- **No data required**: Works purely on weights.
- **Simple and interpretable**: The algorithm is an intuitive proximal method for cardinality‑constrained approximation.
- **Controllable speed‑accuracy trade‑off**: `lambda_prox` and `steps` allow tuning how aggressively sparsity is enforced.

### Limitations

- **Unstructured sparsity**: Zeroed weights are scattered individually, offering no computational speed‑up on dense hardware. Sparse matrix support is needed to realize inference benefits.
- **Heuristic convergence**: There is no guarantee that the iterative process converges to the optimal approximation; it may oscillate or get stuck in a local minimum.
- **Fixed global sparsity**: All layers receive the same sparsity fraction, which may be suboptimal; important layers may need lower sparsity.
- **Percentile ties**: If many entries share the same magnitude at the threshold, the number of non‑zeros may deviate slightly from the exact fraction.
- **Dense output**: The weight remains stored as a dense matrix; memory savings require explicit sparse storage.

### Comparison with Other Pruners

| Method               | Strategy                              | Adaptivity | Iterative | Output          |
|----------------------|---------------------------------------|------------|-----------|-----------------|
| ProxSparse (this)    | Soft‑thresholding + proximal blending | Yes (percentile per step) | Yes | Dense with zeros |
| Magnitude pruning    | Keep top‑k, zero others               | No         | No        | Dense with zeros |
| Bonsai (IMP style)   | Progressive hard thresholding         | No (fixed schedule) | Yes | Dense with zeros |
| ADMM pruner          | ADMM with hard thresholding           | No         | Yes       | Dense with zeros |

ProxSparse’s soft‑thresholding approach provides a smoother, more nuanced pruning than hard thresholding, potentially retaining more information at the boundaries of the threshold.

---

## Conclusion

ProxSparse (registered as `"proxsparse"`) introduces a regularized learning perspective to post‑training pruning. By iteratively moving weights toward a soft‑thresholded target with a fixed sparsity, it achieves a balance between model size reduction and accuracy retention. The method is lightweight, data‑free, and easy to configure. While it does not directly reduce computational cost, it produces sparse weight matrices that can be exploited by sparse inference engines. Future enhancements could include per‑layer sparsity allocation, structured sparsity patterns, and integration with sparse matrix libraries for immediate speed‑up.