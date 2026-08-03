# MGIE‑SVD with Total Rank Budget

**Registered as `mgie_budget`**  
*Allocates a global rank budget across all linear layers proportionally to the entropy of their singular value distributions, then compresses each layer via truncated SVD.*

---

## Overview

MGIE‑SVD Budget is a post‑training compression method that distributes a fixed **total rank budget** (sum of ranks across all linear layers) among the layers based on the **entropy** of their singular value spectra. The idea is that a layer with a flat (high‑entropy) singular value distribution carries more diverse information and thus deserves a higher rank, while a layer with a sharply decaying spectrum (low entropy) can be compressed more aggressively. This approach avoids per‑layer hyperparameter tuning and guarantees that the overall parameter count after low‑rank factorisation (if applied) stays within a desired bound.

The compressor computes the singular values of each `nn.Linear` weight matrix, normalizes them to a probability distribution, calculates the Shannon entropy, and then assigns each layer a rank proportional to its entropy, respecting the total budget. Finally, each weight is replaced by its optimal rank‑`r` approximation via truncated SVD. The module works purely on weight matrices and requires no calibration data.

It is registered under `"mgie_budget"` in the DistilBERT compression framework and is controlled by a single hyperparameter: the total rank budget.

---

## Methodology

### 1. Singular Value Distribution and Entropy

For a weight matrix **W** ∈ ℝ<sup>M×N</sup>, let its full SVD be:

```
W = U Σ V^T,   Σ = diag(σ₁, σ₂, …, σ_K),   K = min(M,N)
```

where σ₁ ≥ σ₂ ≥ … ≥ σ_K ≥ 0. The singular values are normalised to sum to 1:

```
p_i = σ_i / (∑_{j=1}^{K} σ_j),   i = 1,…,K
```

This forms a discrete probability distribution over the singular components. The **Shannon entropy** (in bits) of this distribution is:

```
H = – Σ_{i=1}^{K} p_i log₂(p_i + ε)
```

where ε is a small value to avoid log(0). A higher entropy indicates a more uniform spread of singular values, implying the weight matrix contains information spread across many directions. A low entropy means that a few large singular values dominate, and the rest can be discarded with little loss.

### 2. Rank Budget Allocation

Given a user‑specified **total rank budget** `R_total` (e.g., 1000), the ranks are distributed proportionally to the entropies:

```
r_i = max( 1, ⌊ (H_i / Σ_j H_j) × R_total ⌋ )
```

If all entropies sum to zero (unlikely for real weights), each layer is assigned a rank of 1. Each `r_i` is further capped by the maximum possible rank `min(M_i, N_i)` of that layer.

The resulting rank vector `{r_i}` satisfies `Σ r_i ≈ R_total` (after capping and rounding). This ensures that the total number of “effective parameters” (if one were to store the SVD factors) remains controlled.

### 3. Low‑Rank Approximation

For each linear layer, the weight is replaced by its truncated SVD of rank `r_i`:

```
W_approx = U_{r_i} Σ_{r_i} V_{r_i}^T
```

where `U_{r_i}` contains the first `r_i` left singular vectors, `Σ_{r_i}` the top `r_i` singular values, and `V_{r_i}^T` the first `r_i` right singular vectors. This is the best rank‑`r_i` approximation in Frobenius norm.

The new weight is stored as a full dense matrix of the original shape; the compression is only in the sense that its effective rank has been reduced. No actual parameter savings occur in the model’s stored tensors unless the factorised form is extracted explicitly.

---

## Complexity Analysis

### A. Compression Time Complexity

For each linear layer of size M × N:

- **Full SVD**: O(M N · min(M,N)) using standard algorithms. For a 768×768 matrix, this is roughly O(768³) ≈ 4.5×10⁸ operations, which is fast on a modern CPU (a few milliseconds).
- **Entropy computation**: O(K) where K = min(M,N).
- **Truncation and reconstruction**: O(r M N) but typically absorbed into SVD.

The total compression time is dominated by the SVDs over all L linear layers. For a typical transformer (e.g., 12 layers, each with 4 linear projections in attention + 2 in FFN = 6*12 = 72 matrices), the total SVD time is on the order of a few seconds on CPU.

### B. Inference Complexity

After compression, the weight matrices remain dense, so the forward pass still performs matrix multiplications of the original size. **No inference speed‑up** is achieved by this method alone. To realise computational gains, the low‑rank factors must be extracted and used as two smaller linear layers, reducing FLOPs from O(M N) to O(r (M+N)). The budgeted rank allocation provides a principled way to choose those ranks.

### C. Memory Usage

**During compression**: The SVD of each matrix is computed on CPU using NumPy, which requires storing the weight matrix (M×N floats) and the decomposition products (U, Σ, V^T). Peak memory is a few times the weight size.

**After compression**: The model parameters remain the same count (full dense weights). No storage reduction occurs unless the factorised representation is stored.

If one stores the factors `U_r Σ_r` (M×r) and `V_r` (r×N) instead of the full M×N matrix, the total parameter count reduces to `Σ r_i (M_i + N_i)`. With the global rank budget `R_total`, this sum is controlled, achieving overall model compression.

---

## Implementation Details

### MGIESVDBudgetCompressor Class

```python
class MGIESVDBudgetCompressor(BaseCompressor):
    def __init__(self, model, total_rank_budget=1000, **kwargs):
        super().__init__(model, name="mgie_budget", **kwargs)
        self.total_rank_budget = total_rank_budget
```

- **`total_rank_budget`**: Sum of ranks across all linear layers (default 1000). Must be a positive integer.

The `compress()` method:

1. Iterates over all `nn.Linear` modules, computing:
   - Full SVD of the weight (on CPU via NumPy).
   - Singular value distribution and its Shannon entropy (base 2).
   - Maximum possible rank = `min(out_features, in_features)`.
   Stores `(name, module, max_rank, entropy)`.
2. Computes total entropy. Allocates ranks proportionally, with a minimum of 1.
3. For each layer:
   - Caps the allocated rank to `max_rank`.
   - Computes SVD again (could reuse the earlier decomposition to save time; currently recomputes).
   - Truncates and writes back `W_approx`.

**Note**: The SVD is performed twice per layer (once for entropy, once for truncation). This is redundant; a single SVD could serve both purposes. The implementation can be optimised by storing the SVD results.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("mgie_budget")
def mgie_budget_factory(model, comp_config):
    total_rank_budget = getattr(comp_config, 'total_rank_budget', 1000)
    compressor = MGIESVDBudgetCompressor(model, total_rank_budget=total_rank_budget)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The configuration parameter is read from `comp_config.total_rank_budget`.

---

## Usage and Configuration

To compress a model using MGIE‑SVD Budget, set the compression method to `"mgie_budget"` and optionally specify the total rank budget.

### Example

```python
comp_config.comp_method = "mgie_budget"
comp_config.total_rank_budget = 800   # sum of all ranks will be approx. 800
```

Then run the compression pipeline. All linear layers will be approximated with truncated SVDs whose ranks sum to roughly 800.

### Parameters

| Parameter              | Default | Description                                                                                   |
|------------------------|---------|-----------------------------------------------------------------------------------------------|
| `total_rank_budget`    | `1000`  | Total desired sum of ranks across all compressed linear layers. Larger budget = higher accuracy. |

**Note**: After compression, the model’s weight tensors are still full size. To actually reduce parameter count, one must factorise the weights into two smaller matrices. This compressor provides the rank assignment needed for that subsequent step.

---

## Discussion

### Advantages

- **Global budget control**: A single hyperparameter sets the overall compression level; the algorithm automatically distributes capacity where it is most needed (high‑entropy layers).
- **No calibration data**: Works purely on weight statistics.
- **Information‑theoretic basis**: Entropy of singular values is a natural measure of the complexity of a weight matrix.
- **Principled allocation**: Layers with more diverse singular spectra get higher ranks, avoiding the one‑size‑fits‑all approach of fixed‑rank methods.

### Limitations

- **No direct parameter reduction**: Weights remain dense; extra steps required for actual memory or FLOP savings.
- **Double SVD computation**: The current code performs SVD twice per layer, increasing runtime. Can be easily fixed by caching.
- **Rank sum not exact**: Due to integer rounding and capping to max rank, the actual sum of ranks may be less than the budget.
- **Entropy assumes positive singular values**: The normalisation `p_i = σ_i / sum(σ)` is valid; entropy calculation uses `scipy.stats.entropy` which handles zero probabilities. No issue.
- **No per‑layer importance weighting**: Every layer’s entropy is treated equally; some layers may be more critical to the final output. A weighted allocation could be superior.

### Comparison with Other Rank‑Allocation Methods

| Method               | Allocation Criterion           | Calibration | Per‑Layer Adaptivity |
|----------------------|--------------------------------|-------------|----------------------|
| MGIE‑SVD Budget      | Entropy of singular values     | No          | Yes (proportional)   |
| EntropyBudgetAlloc.  | Whitened singular value entropy| Yes (data)  | Yes (global ranking) |
| AdaSVD               | Nuclear norm proportion        | No          | Yes (budget‑based)   |
| FRanDI (energy mode) | Energy threshold               | No          | Yes                 |

MGIE‑SVD Budget is unique in using the pure entropy of the weight’s own singular spectrum without any activation data, providing a simple data‑free global budget allocation.

---

## Conclusion

MGIE‑SVD Budget (registered as `"mgie_budget"`) offers an elegant, information‑theoretic approach to distributing a total rank budget across the layers of a transformer. By assigning higher ranks to layers with flatter singular value distributions, it aims to preserve the most complex weight structures while meeting a global compression target. The method requires no calibration and integrates easily into the framework. While it does not directly reduce model size, it provides the optimal rank configuration needed for a subsequent low‑rank factorisation step, making it a valuable planning tool for model compression. Future improvements could include caching the SVD to avoid duplicate work, incorporating layer importance weights, and automatically extracting the factorised form for actual parameter savings.