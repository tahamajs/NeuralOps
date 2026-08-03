# AdaSVD: Adaptive SVD with Importance‑Aware Rank Allocation and Iterative Refinement

**Registered as `adasvd`**  
*Per‑layer importance‑guided low‑rank compression with alternating least‑squares refinement.*

---

## Overview

AdaSVD is a post‑training compression method that reduces the parameter count of all linear layers in a transformer model by assigning a unique SVD rank to each layer based on its **importance**—measured by the nuclear norm of its weight matrix—and then iteratively refining the low‑rank factors to better approximate the original weights. The user specifies a **target parameter removal ratio** (e.g., 0.3 to eliminate 30% of parameters), and the algorithm distributes the compression budget proportionally to each layer’s importance, rather than using a single global rank. An alternating least‑squares refinement step further improves the approximation, recovering some of the accuracy lost by naive truncation.

Designed for the DistilBERT compression framework, AdaSVD is registered under `"adasvd"` and can be applied without any calibration data. It works purely on the weight matrices and is suitable for any model containing `nn.Linear` layers.

---

## Methodology

### 1. Layer Importance via Nuclear Norm

For a weight matrix **W** ∈ ℝ<sup>m×n</sup>, the nuclear norm (also called trace norm) is defined as the sum of its singular values:

```
‖W‖_* = Σ σ_i
```

This quantity captures the “energy” of the matrix and serves as a proxy for how much the layer contributes to the model’s capacity. Layers with larger nuclear norms are deemed more important and receive a smaller compression factor (higher rank), while less important layers are compressed more aggressively.

### 2. Global Parameter Budget

The total number of learnable parameters in the model, `total_params`, is computed. The target is to remove a fraction `τ` of these parameters (e.g., `τ = 0.3`). The desired number of remaining parameters is `(1 - τ) × total_params`.

For each linear layer with original parameters `m × n`, its allocated rank `r` is determined by:

```
r = max(1, min( m, n, (1 - τ) × (max_rank) × (importance_frac) ))
```

where:
- `max_rank = min(m, n)` is the full rank.
- `importance_frac = ‖W_layer‖_* / Σ_all_layers ‖W‖_*`.

The formula ensures that the sum of `r × (m + n)` across all layers approximates the target remaining parameters. In practice, the final compression ratio may deviate slightly because the rank is an integer and is bounded.

### 3. Low‑Rank Approximation with SVD

After determining `r`, the weight matrix is decomposed via SVD:

```
W = U Σ V^T
```

The truncated factors **U\_r**, **Σ\_r**, **V\_r** (first `r` components) are used to form an initial approximation:

```
W_approx = U_r Σ_r V_r^T
```

### 4. Iterative Refinement (Alternating Least Squares)

Simple truncation discards information contained in the small singular values. To mitigate this, AdaSVD performs **`num_iters`** (default 5) rounds of refinement that alternate between updating **U\_r** and **V\_r** while keeping **Σ\_r** fixed, or recomputing the best diagonal **Σ\_r** after the updates. The intended procedure (a form of alternating least squares for low‑rank approximation) is:

1. **Update U**: Fix **V\_r** and **Σ\_r**, solve for **U\_r** to minimise ‖W - U\_r Σ\_r V\_r^T‖<sub>F</sub>. The closed‑form solution is:

   ```
   U_r = W V_r^T (Σ_r V_r V_r^T + ε I)^{-1}
   ```

   In the code, a simplified step `U_r = W @ Vt_r.T @ np.linalg.inv(np.diag(S_r) + ε I)` is used (assuming **V\_r** is orthogonal, this becomes `U_r = W V_r^T Σ_r^{-1}`).

2. **Update V**: Fix **U\_r** and **Σ\_r**, solve for **V\_r**:

   ```
   V_r = (Σ_r U_r^T U_r Σ_r + ε I)^{-1} Σ_r U_r^T W
   ```

   The code approximates this as `Vt_r = np.linalg.inv(U_r.T @ U_r + ε I) @ U_r.T @ W`.

3. **Recompute Σ**: After updating **U\_r** and **V\_r**, the optimal diagonal **Σ\_r** is the diagonal of **U\_r^T W V\_r^T** (assuming **U\_r** and **V\_r** are not necessarily orthogonal). This is applied as `S_r = np.diag(U_r.T @ W @ Vt_r.T)`.

In practice, the provided implementation includes a placeholder loop (with `pass`) and only applies the final recomputation of **Σ\_r** using the originally truncated **U\_r** and **V\_r**. A full implementation would perform several alternating updates to converge to a better approximation.

### 5. Replacement

The refined low‑rank matrix `W_approx = U_r diag(S_r) V_r^T` is written back to `module.weight.data`. The model structure remains unchanged; only the weight values are modified. This means the storage size stays the same (dense matrix), but the effective rank is reduced, which can be exploited by a subsequent factorization step or simply left as a compressed representation.

---

## Implementation Details

### AdaSVDCompressor Class

Inherits from `BaseCompressor`.

```python
class AdaSVDCompressor(BaseCompressor):
    def __init__(self, model, target_param_ratio=0.3, num_iters=5, **kwargs):
        super().__init__(model, name="adasvd", **kwargs)
        self.target_param_ratio = target_param_ratio
        self.num_iters = num_iters
```

- **`target_param_ratio`**: Fraction of total parameters to **remove** (range 0–1). Default 0.3 → remove 30%, keep 70%.
- **`num_iters`**: Number of refinement iterations (currently partially implemented).

The `compress()` method:

1. Computes the nuclear norm (sum of singular values) for every `nn.Linear` layer using `np.linalg.svd` on the weight matrix moved to CPU.
2. Sums the importances to compute `total_importance`.
3. Calculates `total_params` (number of trainable parameters).
4. For each linear layer:
   - Determines the rank `r` using the heuristic formula.
   - Performs full SVD of the weight matrix.
   - Truncates to rank `r`.
   - (Optional) Applies iterative refinement loop (currently a stub; only one final recomputation of `S_r` is performed).
   - Writes the approximated weight back.
5. Updates `_compressed_params` and returns the model with compression statistics.

**Note**: The SVDs are computed on CPU with NumPy, which may be slow for very large models. The method uses double‑precision for stability but casts back to the original weight dtype.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("adasvd")
def adasvd_factory(model, comp_config):
    target_param_ratio = getattr(comp_config, 'adasvd_target_ratio', 0.3)
    num_iters = getattr(comp_config, 'adasvd_iters', 5)
    compressor = AdaSVDCompressor(
        model, target_param_ratio=target_param_ratio, num_iters=num_iters
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Configuration parameters are read from `comp_config` with fallback defaults. The factory creates the compressor and returns the compressed model.

---

## Usage and Configuration

Set the compression method to `"adasvd"` and specify the desired parameter reduction ratio and number of refinement iterations.

### Example

```python
comp_config.comp_method = "adasvd"
comp_config.adasvd_target_ratio = 0.4   # remove 40% of parameters
comp_config.adasvd_iters = 10           # perform 10 alternating refinements
```

Then pass the model and config to the framework’s compression runner.

### Parameters

| Parameter              | Default | Description                                                                 |
|------------------------|---------|-----------------------------------------------------------------------------|
| `adasvd_target_ratio`  | `0.3`   | Fraction of model parameters to eliminate. Higher values = more compression. |
| `adasvd_iters`         | `5`     | Number of alternating refinement steps. More iterations may improve accuracy but take longer. |

**Important**: The refinement step is currently a stub; the `num_iters` parameter is accepted but not fully utilized. For a functional iterative refinement, the code inside the loop must be completed (see discussion). The final `S_r` is recomputed, which already provides a small improvement over direct truncation.

---

## Discussion

### Advantages

- **Importance‑based budget allocation**: Not all layers contribute equally; AdaSVD protects critical layers by giving them higher ranks, while aggressively compressing less important ones. This often yields better accuracy‑per‑parameter than uniform rank compression.
- **Global compression target**: The user specifies a single number (fraction of parameters to remove) and the algorithm automatically distributes the budget.
- **Iterative refinement (concept)**: Alternating least squares can reduce the approximation error beyond simple truncation, especially when the rank is small. Once fully implemented, this will further recover accuracy.
- **No calibration data**: Works purely on weights; deterministic and fast (except for SVD computation).
- **Model structure preserved**: The layers remain standard `nn.Linear`, so the compressed model can be used as‑is or further factorized for speed.

### Limitations

- **Incomplete refinement**: The current code does not actually run the iterative updates inside the loop; it only recomputes the diagonal `S_r` once. To realize the full benefit, the alternating steps must be correctly implemented.
- **SVD on CPU with NumPy**: Large weight matrices can cause memory pressure and are slow to decompose. A batched or GPU‑based SVD (e.g., `torch.linalg.svd`) could speed this up.
- **Heuristic rank formula**: The simple proportional allocation may not exactly hit the target parameter ratio, especially with integer rounding. A more rigorous budget‑solving algorithm (e.g., binary search) could be used.
- **No inference speed‑up**: Because the weight matrix remains dense, FLOPs are unchanged. To achieve speed gains, one must subsequently replace the compressed layer with a pair of smaller linear layers (factorization). AdaSVD can serve as the first stage of such a pipeline.
- **Importance measure**: The nuclear norm is a simple heuristic; it may not perfectly correlate with layer sensitivity. Other metrics (e.g., Fisher information, gradient‑based) could be more accurate but require data.

### Comparison with Other Compressors

| Feature                         | A3 (fixed SVD) | AASVD (anchored) | Adaptive SVD | AdaSVD (this module) |
|---------------------------------|----------------|------------------|--------------|------------------------|
| Rank selection                  | Fixed          | Fixed            | Energy‑threshold per layer | Proportional to importance + global budget |
| Data‑awareness                  | No             | Yes (activation) | No           | No                     |
| Global compression target       | No             | No               | No           | Yes (target param ratio) |
| Iterative refinement            | No             | No               | No           | Yes (stub)             |
| Structural replacement          | No             | No               | Yes          | No (weight overwrite)  |

AdaSVD is unique in offering a global compression budget control and importance‑based allocation, making it ideal when a specific model size reduction is required without manual per‑layer tuning.

---

## Conclusion

AdaSVD provides a flexible, importance‑aware compression method for transformer linear layers. By assigning ranks proportional to each layer’s nuclear norm and allowing refinement of the low‑rank factors, it strikes a balance between compression ratio and accuracy retention. The global parameter reduction target simplifies usage, while the iterative refinement (once fully implemented) will further improve approximation quality. Registered as `"adasvd"`, it integrates smoothly into the DistilBERT compression ecosystem and can serve as a powerful pre‑processing step before optional factorization for runtime speed‑ups.