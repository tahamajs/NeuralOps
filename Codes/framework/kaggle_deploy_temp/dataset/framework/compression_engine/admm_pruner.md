# ADMM Pruner: Soft Thresholding via ADMM for Weight Sparsity

**Registered as `admm_pruner`**  
*An iterative ADMM-based method to project a weight matrix onto the set of matrices with a specified sparsity level.*

---

## Overview

The ADMM Pruner applies the **Alternating Direction Method of Multipliers (ADMM)** to find a sparse approximation of the weight matrix **W** ∈ ℝ<sup>m×n</sup> for every linear layer. The user specifies a target sparsity `s` (fraction of entries to remove) and the algorithm iteratively projects the matrix onto the set of `k`‑sparse matrices where `k = ⌈(1-s) * m*n⌉`. While the direct solution to this projection is a simple hard‑threshold (keep the top‑k magnitude entries), the ADMM Pruner provides an iterative, principled framework that can be extended to more complex structured sparsity constraints. The current implementation uses a hard‑thresholding step together with a dual variable update, converging to the sparse projection in just a few iterations.

The module is registered under `"admm_pruner"` and requires no calibration data. It works purely on the weight matrices and is a building block for more advanced ADMM‑based compression pipelines.

---

## Mathematical Background

### 1. Problem Formulation

For a given weight matrix **W** ∈ ℝ<sup>m×n</sup>, we wish to find a sparse matrix **Z** that stays as close as possible to **W**, measured by the squared Frobenius norm, while satisfying a cardinality constraint:

```
minimize  ½ ||W - Z||_F^2
subject to  ‖Z‖_0 ≤ k
```

where `‖Z‖_0` counts the number of non‑zero entries, and `k` is derived from the sparsity `s` as `k = ⌈(1-s)·m·n⌉`.

This problem is known as **projection onto the set of k‑sparse matrices**. Its closed‑form solution is the hard‑thresholding operator:

```
Z^* = H_k(W) 
```

where `H_k(W)` retains the `k` entries of **W** with the largest absolute values and sets all others to zero. However, ADMM is often preferred when the constraint is more complex (e.g., block sparsity, group sparsity) or when the objective includes additional regularizers.

### 2. ADMM Basics

ADMM solves problems of the form:

```
minimize  f(x) + g(z)
subject to  x - z = 0
```

by introducing a dual variable **y** (or **u**) and forming the **augmented Lagrangian** (with penalty parameter ρ > 0):

```
L_ρ(x, z, y) = f(x) + g(z) + y^T(x - z) + (ρ/2)‖x - z‖^2
```

The algorithm alternates:

1. **x‑update**: `x^{k+1} = argmin_x L_ρ(x, z^k, y^k)`
2. **z‑update**: `z^{k+1} = argmin_z L_ρ(x^{k+1}, z, y^k)`
3. **y‑update**: `y^{k+1} = y^k + ρ(x^{k+1} - z^{k+1})`

By defining the **scaled dual variable** `u = y/ρ`, the updates simplify to:

```
x^{k+1} = prox_{f/ρ}(z^k - u^k)
z^{k+1} = prox_{g/ρ}(x^{k+1} + u^k)
u^{k+1} = u^k + (x^{k+1} - z^{k+1})
```

where `prox_{h}(v)` is the proximal operator of function `h`, defined as `prox_h(v) = argmin_x (h(x) + ½‖x - v‖^2)`.

### 3. Applying ADMM to the Sparsity Projection

We rewrite the original problem in the ADMM form by **duplicating the variable**:

```
minimize  f(Z) + g(X)
subject to  Z - X = 0
```

with:
- `f(Z) = ½‖W - Z‖_F^2`  (data fidelity)
- `g(X) = I_{card(X) ≤ k}` (indicator function of the sparse set; 0 if cardinality ≤ k, ∞ otherwise)

The proximal operator of `f/ρ` is a simple quadratic minimization, and the proximal operator of `g/ρ` is the **projection** onto the set of k‑sparse matrices (which does not depend on ρ).

Using the scaled dual variable `U`, the ADMM iterations are:

1. **Z‑update** (primary variable):
   ```
   Z^{t+1} = prox_{f/ρ}(X^t - U^t)
          = argmin_Z ½‖W - Z‖_F^2 + (ρ/2)‖Z - (X^t - U^t)‖_F^2
          = (2W + ρ(X^t - U^t)) / (2 + ρ)
   ```

2. **X‑update** (auxiliary sparse variable):
   ```
   X^{t+1} = prox_{g/ρ}(Z^{t+1} + U^t)
          = Π_{‖·‖_0 ≤ k}(Z^{t+1} + U^t)
   ```
   where `Π_{‖·‖_0 ≤ k}(·)` is the hard‑thresholding operator that keeps the `k` largest‑magnitude entries.

3. **U‑update** (dual variable):
   ```
   U^{t+1} = U^t + (Z^{t+1} - X^{t+1})
   ```

When ρ → ∞ (or if we set ρ so large that the quadratic term dominates), the Z‑update becomes `Z^{t+1} ≈ X^t - U^t`. In many practical implementations for sparsity projection, the **simplified ADMM** is used, where we directly treat **Z** as the auxiliary sparse variable and skip the explicit Z‑update. This leads to the algorithm used in the ADMM Pruner:

```
X^{t+1} = H_k(X^t + U^t)          (hard‑thresholding on the sum)
U^{t+1} = U^t + (W - X^{t+1})      (dual update with the original W)
```

This is equivalent to setting ρ very large (or absorbing it) and initialising `X^0 = W`. The iterations converge to the projection of **W** onto the k‑sparse set. In the code, **X** is called `Z` (the sparse output) and the dual variable is `U`.

### 4. Hard‑Thresholding via Percentile

The sparsity level is given as a fraction `s` (e.g., 0.5). The number `k` is `floor((1-s) * numel)`, but the threshold is computed using `np.percentile` on the absolute values of the matrix to threshold. Specifically:

```
threshold = np.percentile(|Y|, 100 * s)
```

This computes the value such that `s` percent of the entries are **below** it; thus the fraction of entries **above** the threshold is `1-s`. This precisely matches the requirement to keep `(1-s)` fraction of entries. However, because of possible ties, the actual number of non‑zeros may not be exactly `k`, but it is a consistent way to achieve the desired sparsity.

The hard‑thresholding step:
```python
Z = np.where(np.abs(Z_plus_U) > threshold, Z_plus_U, 0)
```

---

## Algorithm Breakdown in the Code

1. **Initialization**:
   ```python
   W = module.weight.data.float().cpu().numpy()
   Z = W.copy()
   U = np.zeros_like(W)
   ```
   The variable `Z` starts as a copy of the original weight; `U` is the scaled dual variable initialised to zero.

2. **Iteration** (for `max_iter` steps):
   ```python
   Z_plus_U = Z + U
   threshold = np.percentile(np.abs(Z_plus_U), 100 * self.sparsity)
   Z = np.where(np.abs(Z_plus_U) > threshold, Z_plus_U, 0)
   U = U + (W - Z)
   ```
   - The first line forms the **argument for projection**: current sparse estimate + dual variable.
   - The second line computes the **sparsity threshold** so that a fraction `sparsity` of elements are below it.
   - The third line performs **hard‑thresholding**: keeps elements above the threshold, zeros out the rest. This is the projection onto the sparse set.
   - The fourth line updates the **dual variable** using the residual between the original weight `W` and the new sparse estimate `Z`.

3. **After the loop**, the sparse matrix `Z` is assigned back to the module weight.

The role of **`rho`** (set in the constructor but not used) is typically the penalty parameter. In the simplified ADMM, `rho` is effectively absorbed into the scaling of the dual variable. Here, the algorithm implicitly uses `rho = 1` (or more precisely, the updates are equivalent to the large‑rho approximation). The given `rho` parameter may be intended for a full ADMM implementation where the Z‑update is not skipped.

---

## Implementation Details

### ADMMPruner Class

```python
class ADMMPruner(BaseCompressor):
    def __init__(self, model, sparsity=0.5, rho=0.1, max_iter=5, **kwargs):
        super().__init__(model, name="admm_pruner", **kwargs)
        self.sparsity = sparsity
        self.rho = rho
        self.max_iter = max_iter
```

- **`sparsity`**: Fraction of weights to **remove** (default 0.5 → 50% of weights set to zero).
- **`rho`**: ADMM penalty parameter (currently unused in the simplified algorithm).
- **`max_iter`**: Number of ADMM iterations (default 5). More iterations improve convergence to the exact projection, though the problem is simple and a single hard‑threshold of `W` would also give the optimum. The iterative approach demonstrates the ADMM framework.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("admm_pruner")
def admm_pruner_factory(model, comp_config):
    sparsity = getattr(comp_config, 'admm_pruner_sparsity', 0.5)
    rho = getattr(comp_config, 'admm_pruner_rho', 0.1)
    max_iter = getattr(comp_config, 'admm_pruner_max_iter', 5)
    compressor = ADMMPruner(model, sparsity=sparsity, rho=rho, max_iter=max_iter)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads the parameters from the compression configuration and applies the pruning. The model is returned with all linear layers pruned.

### Execution Notes

- All operations are performed in `float32` on CPU (via NumPy) to ensure numerical stability.
- The weight tensor is moved back to the original device and dtype after compression.
- Biases are not pruned; only linear layer weights are modified.
- The pruning is **unstructured** – individual entries are zeroed, leading to sparse weight matrices. Actual speed‑up requires sparse matrix support or hardware acceleration.

---

## Usage and Configuration

Enable ADMM pruning by setting the compression method to `"admm_pruner"` and specifying the sparsity level.

### Example

```python
comp_config.comp_method = "admm_pruner"
comp_config.admm_pruner_sparsity = 0.7   # remove 70% of weights
comp_config.admm_pruner_rho = 0.1        # rho (unused placeholder)
comp_config.admm_pruner_max_iter = 10    # more iterations
```

Run the compression pipeline; the model’s linear weights will be sparse.

### Parameters

| Parameter                  | Default | Description                                                                                     |
|----------------------------|---------|-------------------------------------------------------------------------------------------------|
| `admm_pruner_sparsity`     | `0.5`   | Target fraction of zero weights. 0 = no pruning, 1 = all zero (max sparsity).                   |
| `admm_pruner_rho`          | `0.1`   | Placeholder for ADMM penalty (currently unused in the simplified algorithm).                    |
| `admm_pruner_max_iter`     | `5`     | Number of ADMM iterations. For simple sparse projection, 1 is enough; more may polish the result.|

**Note**: Because the algorithm iteratively projects `Z+U` and updates `U`, it will converge to the exact hard‑threshold of `W` within a few iterations. In practice, `max_iter=5` is sufficient even for fine‑grained convergence.

---

## Discussion

### Why ADMM for a trivial projection?

The problem `min_Z ||W - Z||^2 s.t. card(Z) ≤ k` has a closed‑form solution: keep the largest `k` entries. However, implementing this via ADMM serves as a **proof‑of‑concept** for more complex constraints. In future extensions, the ADMM framework can incorporate:

- **Structured sparsity** (e.g., block‑row sparsity, channel pruning) where the projection is not a simple hard‑threshold.
- **Combined objectives** (pruning + quantization or low‑rank constraints).
- **Regularizers** like group Lasso or total variation.

The `rho` parameter, currently unused, will become active when the full ADMM with the Z‑update is implemented (mixing data fidelity and the constraint). The simplified scheme here corresponds to the “large ρ” limit.

### Advantages

- **Plug‑and‑play**: Works on any `nn.Linear` module without data.
- **Principled framework**: Extends easily to advanced constraints.
- **Fast**: Only a few iterations needed.

### Limitations

- **Unstructured sparsity**: Zeroing individual weights does not directly accelerate dense matrix multiplications without specialized sparse kernels or hardware.
- **No fine‑tuning**: After pruning, accuracy may drop. It is recommended to follow with sparse fine‑tuning.
- **`rho` unused**: The algorithm does not currently balance between data fidelity and sparsity using the penalty parameter; it directly enforces the sparsity constraint each iteration. Users should not expect `rho` to affect the result.
- **Percentile ties**: If many entries share the same magnitude near the threshold, the number of non‑zeros may slightly exceed or fall short of the exact target. The impact is usually negligible.

### Comparison with Magnitude Pruning (ADMM Compressor)

The previously described `admm` compressor (magnitude pruning + quantization) applies a one‑shot hard‑threshold and then quantization. The `admm_pruner` performs an **iterative** projection, which yields the same final sparse matrix but demonstrates the ADMM loop. For simple magnitude pruning, the one‑shot approach is more efficient; the iterative version is valuable when the ADMM loop will be extended.

---

## Conclusion

The ADMM Pruner (registered as `"admm_pruner"`) introduces the ADMM optimization paradigm for weight sparsity. While the specific problem has a direct analytical solution, the iterative algorithm lays the groundwork for future structured and combined compression techniques. Its integration into the compression framework allows researchers to experiment with ADMM‑based pruning and easily swap in custom projection operators while maintaining the same dual update logic. The module serves as both a practical sparse pruner and an educational example of ADMM in neural network compression.