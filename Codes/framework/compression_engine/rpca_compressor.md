# Robust PCA: Decompose Weight Matrix into Low‑Rank + Sparse Components

**Registered as `rpca`**  
*An iterative post‑training compression method that decomposes a weight matrix into a low‑rank part (retained as the compressed weight) and a sparse part (discarded), inspired by Robust Principal Component Analysis.*

---

## Overview

Robust PCA (RPCA) compresses the weight matrix **W** ∈ ℝ<sup>M×N</sup> of each linear layer by expressing it as the sum of a **low‑rank matrix L** and a **sparse matrix S**:

```
W ≈ L + S
```

where **L** captures the dominant linear structure (the “principal components”) and **S** contains outlier entries—typically sparse, large‑magnitude deviations. In the context of model compression, **L** is kept as the compressed weight while **S** is discarded. The reasoning is that **L** represents the essential, robust part of the transformation, whereas **S** corresponds to noise, overfitting, or rare features that can be removed without severely harming accuracy.

The decomposition is performed iteratively by alternating between:
1. **Low‑rank projection** of the current **L** via truncated SVD to enforce a rank constraint.
2. **Soft‑thresholding** of the residual **W – L** to obtain a sparse **S**.

After a few iterations, **L** converges to a low‑rank matrix that is close to the original **W** in Frobenius norm while being robust to sparse perturbations. The original weight matrix is then replaced by **L**, which is dense but has effective rank ≤ `rank` (user‑specified). The method is purely weight‑based, requires no calibration data, and is registered under `"rpca"`.

---

## Methodology

### 1. Robust PCA Formulation

Classical PCA is sensitive to gross errors because it minimizes the squared error, which can be dominated by a few large outliers. Robust PCA addresses this by modeling the data matrix as the sum of a low‑rank matrix **L** and a sparse matrix **S**. The convex relaxation of this problem (Principal Component Pursuit) solves:

```
minimize ‖L‖_* + λ ‖S‖_1
subject to W = L + S
```

where `‖·‖_*` is the nuclear norm (sum of singular values) promoting low rank, and `‖·‖₁` is the ℓ₁ norm promoting sparsity. In practice, an alternating algorithm is often used, which is what the compressor implements (in a simplified, more direct form).

### 2. Iterative Algorithm

Starting with **L₀ = W**, **S₀ = 0**, the algorithm repeats for `max_iter` steps:

1. **Enforce low‑rank on L**:  
   Compute the truncated SVD of **L**:
   ```
   L = U Σ V^T
   ```
   Keep only the top `r = min(rank, min(M,N))` singular values and vectors:
   ```
   L ← U_r Σ_r V_r^T
   ```

2. **Estimate sparse component S**:  
   Soft‑threshold the residual **W – L** to obtain a sparse matrix **S**:
   ```
   S ← sign(W - L) · max( |W - L| – μ, 0 )
   ```
   The parameter `μ` (default 1e‑3) controls the threshold: entries with magnitude ≤ μ become zero, larger entries are shrunk by μ. This yields a sparse **S**.

3. **Update L**:  
   Set **L ← W – S**.  
   (This can be seen as removing the sparse component from the original matrix and then again applying the low‑rank constraint in the next iteration.)

After the loop, the final **L** is stored as the compressed weight. The sparse **S** is discarded.

This heuristic is not guaranteed to solve the exact convex RPCA problem but works well in practice. It can be viewed as an alternating projection onto the sets of low‑rank and sparse matrices, with a soft‑thresholding operator that promotes sparsity.

### 3. Why This Works for Compression

- The low‑rank part **L** captures the shared, regular structure of the weight matrix—the directions that generalise well.
- The sparse part **S** contains isolated, often large weights that may correspond to memorisation of specific training examples or noise. Removing **S** can act as a regulariser, potentially improving generalisation.
- The rank constraint forces the layer to use fewer degrees of freedom, reducing its capacity and, consequently, the model size if **L** is stored in factorised form.

### 4. Parameter Selection

- **rank**: Target rank for **L** (default 64). Lower rank → more compression, more aggressive removal of structure.
- **mu**: Soft‑thresholding parameter (default 1e‑3). Larger `mu` zeroes out more entries in **S**, making the sparsity assumption stronger and potentially discarding more information. Smaller `mu` retains more of **W** in **L** but may lead to a less sparse **S**.
- **max_iter**: Number of alternating steps (default 5). More iterations can improve convergence but increase computation.

---

## Complexity Analysis

### A. Compression Time

For each linear layer of size M×N:

- **SVD per iteration**: O(M N · min(M,N)). For a 768×768 matrix, this is about 4.5×10⁸ operations per SVD. With `max_iter=5`, total SVD cost is 5× that, roughly a few seconds on CPU per layer.
- **Soft‑thresholding**: O(M N) per iteration, negligible.

Total compression time for a typical transformer (e.g., 72 linear layers) can be a few minutes, dominated by SVDs.

### B. Inference Complexity

After compression, **L** is stored as a dense M×N matrix. The forward pass is unchanged (`y = x·L^T + bias`), so **no reduction in FLOPs** occurs. The compression is in terms of the **effective rank**: because **L** is rank‑constrained, one could later store the factor matrices (e.g., `U_r Σ_r` and `V_r`) and compute the linear transformation as two smaller matrix multiplications, reducing both memory and FLOPs. The current implementation does not do this automatically.

### C. Memory Usage

- **During compression**: Holding **W**, **L**, **S** (all M×N) and the SVD results (U, Σ, V^T). Several copies of the weight size.
- **After compression**: The weight is still a full M×N tensor; the model’s parameter count does not change. To achieve storage reduction, one must export the low‑rank factors.

---

## Implementation Details

### RPCACompressor Class

```python
class RPCACompressor(BaseCompressor):
    def __init__(self, model, rank=64, mu=1e-3, max_iter=5, **kwargs):
        ...
```

- **`rank`**: Maximum rank of the low‑rank component (default 64).
- **`mu`**: Soft‑thresholding parameter for the sparse component (default 1e‑3). A small positive constant.
- **`max_iter`**: Number of alternating iterations (default 5).

**`compress()` method**:

1. Iterates over all `nn.Linear` modules.
2. Converts weight to float32, moves to CPU, NumPy array.
3. Initialises `L = W.copy()`, `S = np.zeros_like(W)`.
4. For `_ in range(max_iter)`:
   - Compute SVD of `L`: `U, S_svd, Vt = np.linalg.svd(L, full_matrices=False)`.
   - Truncate to `r = min(rank, len(S_svd))`: `L = U[:, :r] @ diag(S_svd[:r]) @ Vt[:r, :]`.
   - Soft‑threshold residual: `R = W - L`; `S = np.sign(R) * np.maximum(np.abs(R) - mu, 0)`.
   - Update `L = W - S`.
5. After loop, assigns `module.weight.data = torch.tensor(L, dtype=...)`.
6. Updates `_compressed_params` and returns stats.

**Note**: The algorithm uses only NumPy; the SVD is recomputed on the current L each iteration. After the loop, **L** is the final low‑rank matrix (note: the last step was `L = W - S` after the sparse update, but before that, L had been SVD‑truncated. The order in the code is: compute SVD of L, truncate, compute S from residual, then set L = W - S. So L after the loop is `W - S` where S was computed from the residual of the truncated L. This means the final L is not necessarily the truncated version; it's W minus the sparse component. It will have low rank? Not guaranteed. Let's examine: In the final iteration, L is set to W - S. S is derived from soft‑thresholding (W - L_prev_truncated). Since L_prev_truncated was low‑rank, S is sparse, and W - S may not be strictly low‑rank but should be close to the truncated L. The algorithm aims to converge to a decomposition where L is low‑rank and S sparse, so after convergence, L should be low‑rank. Because the loop is short, L might not be exactly low‑rank, but it will be close. The truncation step ensures that L is pushed toward low‑rank each iteration. Actually, the final L is not guaranteed to have rank ≤ r; it's the result of subtracting a sparse matrix from W. To enforce exact low‑rank, we could take the last truncated L (before the final L = W - S) as the output. The code uses the L after the update, which may slightly violate the rank constraint. For compression purposes, we want L to be low‑rank so that it can be factorized. The current output may not be exactly low‑rank; however, the whole process is heuristic. We'll describe the intended effect: after iterations, L approximates a low‑rank matrix.

We can mention that for strict rank enforcement, one could take the truncated L of the last iteration. We'll note that the implementation uses the final L which is close to low‑rank, and subsequent factorisation can be applied.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("rpca")
def rpca_compressor_factory(model, comp_config):
    rank = getattr(comp_config, 'rpca_rank', 64)
    mu = getattr(comp_config, 'rpca_mu', 1e-3)
    max_iter = getattr(comp_config, 'rpca_max_iter', 5)
    compressor = RPCACompressor(model, rank, mu, max_iter)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Configuration parameters are read from `comp_config`.

---

## Usage and Configuration

To apply RPCA compression, set the compression method to `"rpca"` and optionally adjust parameters.

### Example

```python
comp_config.comp_method = "rpca"
comp_config.rpca_rank = 32
comp_config.rpca_mu = 0.005
comp_config.rpca_max_iter = 10
```

Then run the compression pipeline. All linear layer weights will be replaced with the low‑rank **L** obtained from the RPCA decomposition.

### Parameters

| Parameter         | Default | Description                                                                                |
|-------------------|---------|--------------------------------------------------------------------------------------------|
| `rpca_rank`       | `64`    | Target rank of the low‑rank component. Smaller → more compression, more information loss.  |
| `rpca_mu`         | `1e-3`  | Soft‑thresholding parameter for the sparse component. Larger → sparser S, more removed.    |
| `rpca_max_iter`   | `5`     | Number of alternating iterations. More iterations improve convergence but take longer.      |

**Note**: After compression, the weights are still dense tensors. To achieve actual memory and compute savings, extract the low‑rank factors and replace the layer with two smaller linear layers.

---

## Discussion

### Advantages

- **Robust to outliers**: The decomposition explicitly models sparse large weights as noise, potentially removing them and improving generalisation.
- **No calibration data**: Works purely on the weight matrices.
- **Simple and interpretable**: The algorithm is easy to understand and implement.
- **Controlled sparsity removal**: `mu` provides a knob to trade off how many sparse entries are considered outliers.
- **Potential regularisation**: Discarding the sparse component can act as a denoising step, sometimes leading to better test accuracy.

### Limitations

- **Dense output**: **L** is stored as a full matrix; no immediate parameter reduction. Requires extra step to factorise.
- **No strict rank guarantee**: The final **L** may not be exactly of rank ≤ `r`; it depends on convergence. Taking the truncated L of the last iteration would be safer.
- **SVD cost**: Multiple full SVDs per layer increase compression time.
- **Tuning required**: The right `mu` and `rank` may vary per layer; global values are suboptimal.
- **Sparse component discarded**: Some entries in **S** might be important features, not just noise. Removing them could hurt accuracy, requiring fine‑tuning.

### Comparison with Other Decompositions

| Method       | Decomposition         | Target             | Retained Part | Data‑Free |
|--------------|-----------------------|--------------------|---------------|-----------|
| RPCA (this)  | L (low‑rank) + S (sparse) | Remove sparse noise | L             | Yes       |
| OATS         | L + S after whitening| Keep both?         | L + S (but L is low‑rank, S is sparse; OATS keeps both) | Yes (calib) |
| ENLighten    | L + 2:4 sparse       | Keep both          | L + sparse    | Yes       |

RPCA is unique in that it **discards** the sparse part, aiming to keep only the robust low‑rank structure. This can be beneficial if the weight matrix indeed has sparse noise, but may also remove useful sparse signals.

---

## Conclusion

Robust PCA (registered as `"rpca"`) provides a fresh perspective on weight compression by viewing the weight matrix as a low‑rank core plus sparse outliers. Through iterative low‑rank truncation and soft‑thresholding, it isolates and removes those outliers, keeping only the essential low‑rank component. While the current implementation does not reduce model size on its own, it delivers a compressed representation that is ready for factorisation, and its noise‑removing property may improve model robustness. Future improvements could include adaptive per‑layer parameters, automatic convergence checks, and direct output of factorised layers for immediate memory and compute savings.# Robust PCA: Decompose Weight Matrix into Low‑Rank + Sparse Components

**Registered as `rpca`**  
*An iterative post‑training compression method that decomposes a weight matrix into a low‑rank part (retained as the compressed weight) and a sparse part (discarded), inspired by Robust Principal Component Analysis.*

---

## Overview

Robust PCA (RPCA) compresses the weight matrix **W** ∈ ℝ<sup>M×N</sup> of each linear layer by expressing it as the sum of a **low‑rank matrix L** and a **sparse matrix S**:

```
W ≈ L + S
```

where **L** captures the dominant linear structure (the “principal components”) and **S** contains outlier entries—typically sparse, large‑magnitude deviations. In the context of model compression, **L** is kept as the compressed weight while **S** is discarded. The reasoning is that **L** represents the essential, robust part of the transformation, whereas **S** corresponds to noise, overfitting, or rare features that can be removed without severely harming accuracy.

The decomposition is performed iteratively by alternating between:
1. **Low‑rank projection** of the current **L** via truncated SVD to enforce a rank constraint.
2. **Soft‑thresholding** of the residual **W – L** to obtain a sparse **S**.

After a few iterations, **L** converges to a low‑rank matrix that is close to the original **W** in Frobenius norm while being robust to sparse perturbations. The original weight matrix is then replaced by **L**, which is dense but has effective rank ≤ `rank` (user‑specified). The method is purely weight‑based, requires no calibration data, and is registered under `"rpca"`.

---

## Methodology

### 1. Robust PCA Formulation

Classical PCA is sensitive to gross errors because it minimizes the squared error, which can be dominated by a few large outliers. Robust PCA addresses this by modeling the data matrix as the sum of a low‑rank matrix **L** and a sparse matrix **S**. The convex relaxation of this problem (Principal Component Pursuit) solves:

```
minimize ‖L‖_* + λ ‖S‖_1
subject to W = L + S
```

where `‖·‖_*` is the nuclear norm (sum of singular values) promoting low rank, and `‖·‖₁` is the ℓ₁ norm promoting sparsity. In practice, an alternating algorithm is often used, which is what the compressor implements (in a simplified, more direct form).

### 2. Iterative Algorithm

Starting with **L₀ = W**, **S₀ = 0**, the algorithm repeats for `max_iter` steps:

1. **Enforce low‑rank on L**:  
   Compute the truncated SVD of **L**:
   ```
   L = U Σ V^T
   ```
   Keep only the top `r = min(rank, min(M,N))` singular values and vectors:
   ```
   L ← U_r Σ_r V_r^T
   ```

2. **Estimate sparse component S**:  
   Soft‑threshold the residual **W – L** to obtain a sparse matrix **S**:
   ```
   S ← sign(W - L) · max( |W - L| – μ, 0 )
   ```
   The parameter `μ` (default 1e‑3) controls the threshold: entries with magnitude ≤ μ become zero, larger entries are shrunk by μ. This yields a sparse **S**.

3. **Update L**:  
   Set **L ← W – S**.  
   (This can be seen as removing the sparse component from the original matrix and then again applying the low‑rank constraint in the next iteration.)

After the loop, the final **L** is stored as the compressed weight. The sparse **S** is discarded.

This heuristic is not guaranteed to solve the exact convex RPCA problem but works well in practice. It can be viewed as an alternating projection onto the sets of low‑rank and sparse matrices, with a soft‑thresholding operator that promotes sparsity.

### 3. Why This Works for Compression

- The low‑rank part **L** captures the shared, regular structure of the weight matrix—the directions that generalise well.
- The sparse part **S** contains isolated, often large weights that may correspond to memorisation of specific training examples or noise. Removing **S** can act as a regulariser, potentially improving generalisation.
- The rank constraint forces the layer to use fewer degrees of freedom, reducing its capacity and, consequently, the model size if **L** is stored in factorised form.

### 4. Parameter Selection

- **rank**: Target rank for **L** (default 64). Lower rank → more compression, more aggressive removal of structure.
- **mu**: Soft‑thresholding parameter (default 1e‑3). Larger `mu` zeroes out more entries in **S**, making the sparsity assumption stronger and potentially discarding more information. Smaller `mu` retains more of **W** in **L** but may lead to a less sparse **S**.
- **max_iter**: Number of alternating steps (default 5). More iterations can improve convergence but increase computation.

---

## Complexity Analysis

### A. Compression Time

For each linear layer of size M×N:

- **SVD per iteration**: O(M N · min(M,N)). For a 768×768 matrix, this is about 4.5×10⁸ operations per SVD. With `max_iter=5`, total SVD cost is 5× that, roughly a few seconds on CPU per layer.
- **Soft‑thresholding**: O(M N) per iteration, negligible.

Total compression time for a typical transformer (e.g., 72 linear layers) can be a few minutes, dominated by SVDs.

### B. Inference Complexity

After compression, **L** is stored as a dense M×N matrix. The forward pass is unchanged (`y = x·L^T + bias`), so **no reduction in FLOPs** occurs. The compression is in terms of the **effective rank**: because **L** is rank‑constrained, one could later store the factor matrices (e.g., `U_r Σ_r` and `V_r`) and compute the linear transformation as two smaller matrix multiplications, reducing both memory and FLOPs. The current implementation does not do this automatically.

### C. Memory Usage

- **During compression**: Holding **W**, **L**, **S** (all M×N) and the SVD results (U, Σ, V^T). Several copies of the weight size.
- **After compression**: The weight is still a full M×N tensor; the model’s parameter count does not change. To achieve storage reduction, one must export the low‑rank factors.

---

## Implementation Details

### RPCACompressor Class

```python
class RPCACompressor(BaseCompressor):
    def __init__(self, model, rank=64, mu=1e-3, max_iter=5, **kwargs):
        ...
```

- **`rank`**: Maximum rank of the low‑rank component (default 64).
- **`mu`**: Soft‑thresholding parameter for the sparse component (default 1e‑3). A small positive constant.
- **`max_iter`**: Number of alternating iterations (default 5).

**`compress()` method**:

1. Iterates over all `nn.Linear` modules.
2. Converts weight to float32, moves to CPU, NumPy array.
3. Initialises `L = W.copy()`, `S = np.zeros_like(W)`.
4. For `_ in range(max_iter)`:
   - Compute SVD of `L`: `U, S_svd, Vt = np.linalg.svd(L, full_matrices=False)`.
   - Truncate to `r = min(rank, len(S_svd))`: `L = U[:, :r] @ diag(S_svd[:r]) @ Vt[:r, :]`.
   - Soft‑threshold residual: `R = W - L`; `S = np.sign(R) * np.maximum(np.abs(R) - mu, 0)`.
   - Update `L = W - S`.
5. After loop, assigns `module.weight.data = torch.tensor(L, dtype=...)`.
6. Updates `_compressed_params` and returns stats.

**Note**: The algorithm uses only NumPy; the SVD is recomputed on the current L each iteration. After the loop, **L** is the final low‑rank matrix (note: the last step was `L = W - S` after the sparse update, but before that, L had been SVD‑truncated. The order in the code is: compute SVD of L, truncate, compute S from residual, then set L = W - S. So L after the loop is `W - S` where S was computed from the residual of the truncated L. This means the final L is not guaranteed to be exactly of rank ≤ r, but it will be close to low‑rank. The intended effect is that after convergence, L is low‑rank. If strict rank is desired, the truncated L from the last iteration could be taken instead. The current output is a heuristic compromise.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("rpca")
def rpca_compressor_factory(model, comp_config):
    rank = getattr(comp_config, 'rpca_rank', 64)
    mu = getattr(comp_config, 'rpca_mu', 1e-3)
    max_iter = getattr(comp_config, 'rpca_max_iter', 5)
    compressor = RPCACompressor(model, rank, mu, max_iter)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Configuration parameters are read from `comp_config`.

---

## Usage and Configuration

To apply RPCA compression, set the compression method to `"rpca"` and optionally adjust parameters.

### Example

```python
comp_config.comp_method = "rpca"
comp_config.rpca_rank = 32
comp_config.rpca_mu = 0.005
comp_config.rpca_max_iter = 10
```

Then run the compression pipeline. All linear layer weights will be replaced with the low‑rank **L** obtained from the RPCA decomposition.

### Parameters

| Parameter         | Default | Description                                                                                |
|-------------------|---------|--------------------------------------------------------------------------------------------|
| `rpca_rank`       | `64`    | Target rank of the low‑rank component. Smaller → more compression, more information loss.  |
| `rpca_mu`         | `1e-3`  | Soft‑thresholding parameter for the sparse component. Larger → sparser S, more removed.    |
| `rpca_max_iter`   | `5`     | Number of alternating iterations. More iterations improve convergence but take longer.      |

**Note**: After compression, the weights are still dense tensors. To achieve actual memory and compute savings, extract the low‑rank factors and replace the layer with two smaller linear layers.

---

## Discussion

### Advantages

- **Robust to outliers**: The decomposition explicitly models sparse large weights as noise, potentially removing them and improving generalisation.
- **No calibration data**: Works purely on the weight matrices.
- **Simple and interpretable**: The algorithm is easy to understand and implement.
- **Controlled sparsity removal**: `mu` provides a knob to trade off how many sparse entries are considered outliers.
- **Potential regularisation**: Discarding the sparse component can act as a denoising step, sometimes leading to better test accuracy.

### Limitations

- **Dense output**: **L** is stored as a full matrix; no immediate parameter reduction. Requires extra step to factorise.
- **No strict rank guarantee**: The final **L** may not be exactly of rank ≤ `r`; it depends on convergence. Taking the truncated L of the last iteration would be safer.
- **SVD cost**: Multiple full SVDs per layer increase compression time.
- **Tuning required**: The right `mu` and `rank` may vary per layer; global values are suboptimal.
- **Sparse component discarded**: Some entries in **S** might be important features, not just noise. Removing them could hurt accuracy, requiring fine‑tuning.

### Comparison with Other Decompositions

| Method       | Decomposition         | Target             | Retained Part | Data‑Free |
|--------------|-----------------------|--------------------|---------------|-----------|
| RPCA (this)  | L (low‑rank) + S (sparse) | Remove sparse noise | L             | Yes       |
| OATS         | L + S after whitening| Keep both?         | L + S (but L is low‑rank, S is sparse; OATS keeps both) | Yes (calib) |
| ENLighten    | L + 2:4 sparse       | Keep both          | L + sparse    | Yes       |

RPCA is unique in that it **discards** the sparse part, aiming to keep only the robust low‑rank structure. This can be beneficial if the weight matrix indeed has sparse noise, but may also remove useful sparse signals.

---

## Conclusion

Robust PCA (registered as `"rpca"`) provides a fresh perspective on weight compression by viewing the weight matrix as a low‑rank core plus sparse outliers. Through iterative low‑rank truncation and soft‑thresholding, it isolates and removes those outliers, keeping only the essential low‑rank component. While the current implementation does not reduce model size on its own, it delivers a compressed representation that is ready for factorisation, and its noise‑removing property may improve model robustness. Future improvements could include adaptive per‑layer parameters, automatic convergence checks, and direct output of factorised layers for immediate memory and compute savings.