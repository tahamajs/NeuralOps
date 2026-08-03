# ARMOR Pruner: 2:4 Sparse Core + Low‑Rank Error Correction

**Registered as `armor_pruner`**  
*Structured sparsity with a low‑rank residual approximation for accurate weight compression.*

---

## Overview

The ARMOR Pruner (Approximate Low‑Rank Correction with Sparse Core) compresses the weight matrix of each linear layer by splitting it into two components:

1. **Sparse core**: A structured sparse matrix where, within each block of 4 consecutive rows, every row retains only its top‑2 magnitude weights (≈ 50% sparsity). This yields a highly regular sparsity pattern that can be accelerated on specialised hardware.
2. **Low‑rank correction**: A dense, low‑rank matrix that captures the residual error between the original weight and the sparse core. This correction is obtained by truncated SVD of the residual, using a fixed rank (default 16).

The final weight is the sum of these two components:  

```
W_armor = W_sparse + ΔW_lowrank
```

The name *ARMOR* reflects the idea of a “hard” sparse skeleton complemented by a “soft” dense correction, aiming to retain model accuracy while enabling significant compression and potential hardware efficiency. The module is registered under `"armor_pruner"` and requires no calibration data—it operates purely on the weight tensors.

---

## Mathematical Formulation

### 1. Structured Sparsity: Row‑wise Top‑k per Block

Let **W** ∈ ℝ<sup>N×M</sup> be a weight matrix. Partition the rows into blocks of size `B` (default 4). For each block `W_block = W[i:i+B, :]`, we enforce that **each row** within the block retains only the `k` largest magnitude entries, with `k = 2`. This is achieved via a row‑wise hard‑threshold:

For row `r` in the block, let the absolute values be `a = |W_block[r,:]|`. Sort `a` ascending and pick the `k`‑th largest value `θ_r = a[-k]`. The sparse mask is:

```
mask[r, j] = 1  if |W_block[r,j]| ≥ θ_r,  else 0
```

Then `W_sparse[r,j] = W_block[r,j] * mask[r,j]`. Because the threshold is the `k`‑th largest absolute value, exactly `k` entries per row are preserved (except for ties, which may keep more; the algorithm uses `≥`). The resulting matrix has approximately 50% non‑zero entries when `k=2` and `B=4`, but the pattern is **not** the classic NVIDIA 2:4 (which would enforce exactly 2 non‑zeros in every contiguous group of 4 elements *within a column*). Instead, it is a row‑wise, block‑aligned sparsity that can still be exploited by some sparse matrix accelerators.

### 2. Residual and Low‑Rank Correction

The sparse core discards a significant amount of information. The residual:

```
R = W - W_sparse
```

contains the removed weights. To recover some of the lost signal, we approximate `R` by a low‑rank matrix using truncated SVD:

```
R ≈ U_r Σ_r V_r^T
```

where:
- **U\_r** ∈ ℝ<sup>N×r</sup> are the first `r` left singular vectors,
- **Σ\_r** ∈ ℝ<sup>r×r</sup> is diagonal with the top `r` singular values,
- **V\_r^T** ∈ ℝ<sup>r×M</sup> are the first `r` right singular vectors,
- `r = min(16, rank(R))` is the chosen correction rank.

The correction matrix `ΔW = U_r Σ_r V_r^T` is dense but has low rank, so it adds only a modest number of effective parameters (r·(N+M) instead of N·M). The final compressed weight is:

```
W_armor = W_sparse + ΔW
```

### 3. Why This Decomposition?

- **Hardware efficiency**: The sparse core, with a regular pattern (exactly 2 non‑zeros per row in each block of 4 rows), can be stored and computed efficiently on GPUs that support block‑sparse formats.
- **Accuracy recovery**: The low‑rank correction captures the principal directions of the pruning error, preserving essential weight structure that simple magnitude pruning discards.
- **No training needed**: Both components are obtained algebraically, making this a post‑training compression technique.

---

## Algorithm in Detail

For each `nn.Linear` module:

1. **Compute sparse core**:
   - Iterate over row blocks of size `B` (default 4).
   - For each block, compute per‑row absolute values, determine the 2nd‑largest magnitude per row, and set all entries below that threshold to zero. (This may be simplified in practice: `threshold = np.sort(np.abs(blk), axis=1)[:, -2:]` gives the two largest values per row; using `threshold[:, 0]` as the cut‑off.)
2. **Compute residual**: `R = W - W_sparse`.
3. **Low‑rank correction**:
   - Perform full SVD of `R`: `U, S, Vt = np.linalg.svd(R, full_matrices=False)`.
   - Truncate to rank `r = min(16, len(S))`.
   - Form correction `ΔW = U[:, :r] @ diag(S[:r]) @ Vt[:r, :]`.
4. **Combine**: `W_armor = W_sparse + ΔW`.
5. **Replace** the original weight with `W_armor`.

The final weight is dense (the sum of a sparse matrix and a low‑rank matrix), so the model structure remains unchanged. However, the two components can be stored and computed separately for inference efficiency.

---

## Implementation Details

### ARMORPruner Class

```python
class ARMORPruner(BaseCompressor):
    def __init__(self, model, block_size=4, **kwargs):
        super().__init__(model, name="armor_pruner", **kwargs)
        self.block_size = block_size
```

- **`block_size`**: Size of the row block for the sparse pattern (default 4). The sparsity inside each block is fixed at `k = block_size // 2` (i.e., keep 2 out of 4).

The `compress()` method:

- Loops over all `nn.Linear` layers.
- Converts weight to CPU float32 NumPy array.
- Applies the sparse masking as described (row‑wise top‑2 within each block of 4 rows).
- Computes SVD of the residual.
- Constructs correction and final weight.
- Writes back to `module.weight.data`, preserving original dtype and device.

**Note**: The current implementation does not explicitly handle `k` as a parameter; it hardcodes `keep_top = 2` by taking `[:, -2:]` from the sorted absolute values. A more general version could parameterise `k`.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("armor_pruner")
def armor_pruner_factory(model, comp_config):
    block_size = getattr(comp_config, 'armor_block_size', 4)
    compressor = ARMORPruner(model, block_size=block_size)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The only configuration parameter is the block size. The method is registered under `"armor_pruner"`.

---

## Usage and Configuration

Apply ARMOR pruning by setting the compression method to `"armor_pruner"` in the compression configuration.

### Example

```python
comp_config.comp_method = "armor_pruner"
comp_config.armor_block_size = 4   # uses the default 2:4 pattern
```

Then run the compression pipeline. The model's linear layers will be replaced with the sparse‑plus‑low‑rank approximation.

### Parameters

| Parameter            | Default | Description                                                                       |
|----------------------|---------|-----------------------------------------------------------------------------------|
| `armor_block_size`   | `4`     | Row block size for the sparse pattern. The sparsity per row is fixed at 2 non‑zeros. |

**Important**: The rank of the low‑rank correction is currently fixed at `min(16, rank(residual))`. This could be made configurable in a future version.

---

## Discussion

### Advantages

- **Regular sparsity**: The block‑aligned row‑wise top‑2 pattern is more amenable to hardware acceleration than unstructured sparsity.
- **Better accuracy**: The low‑rank correction significantly reduces the accuracy drop compared to pure magnitude pruning. It acts as a compressed representation of the pruned weights.
- **No training**: Purely algebraic, works out‑of‑the‑box.
- **Decomposable**: The two components (sparse and low‑rank) can be stored separately, leading to parameter savings: instead of storing a full dense matrix, one can store the sparse indices/values (≈ 50% density) plus the two low‑rank factors (U and V). This can reduce storage from N·M to about 0.5·N·M + r·(N+M).

### Limitations

- **Not true 2:4 structured sparsity**: The standard NVIDIA 2:4 pattern requires exactly 2 non‑zeros in each contiguous group of 4 weights *along the column dimension* (or within a row with a sliding window of 4). This implementation only ensures exactly 2 non‑zeros per row per block of 4 rows, which is a different regular pattern. The effective sparse format may not map directly to current sparse tensor cores.
- **Residual SVD cost**: Full SVD of the residual matrix is expensive for large weight matrices (O(N·M·min(N,M))). For extremely large layers, an approximate SVD (e.g., randomized SVD) would be faster.
- **Fixed correction rank**: The rank is hardcoded to 16; layers with a larger residual might need a higher rank, while others could use lower. An adaptive rank selection based on energy threshold would improve flexibility.
- **Dense final weight**: The code writes back a dense matrix, not separate sparse+low‑rank structures. To actually achieve inference speed‑up, a custom layer that computes `output = input @ (W_sparse + U V)` would be needed. The current implementation only reduces storage if the matrix is stored in decomposed form.
- **Ties in magnitude**: Using `>=` for the threshold may retain more than 2 entries per row if several entries share the same magnitude as the second largest. This slightly increases density.

### Comparison with Other Pruners

| Feature                        | ADMM Pruner (simple sparse) | ARMOR Pruner (this)  |
|--------------------------------|-----------------------------|----------------------|
| Sparsity pattern               | Unstructured                | Structured (row‑wise top‑2 per block of 4 rows) |
| Error correction               | None                        | Low‑rank SVD of residual |
| Accuracy recovery              | Limited                     | Better, due to low‑rank correction |
| Hardware friendliness           | Low (unstructured)          | Moderate (structured, but not 2:4) |
| Computational overhead          | Small (percentile threshold) | Higher (full SVD)    |
| Storage format after compression | Dense (with zeros)          | Dense (but can be factorized) |

ARMOR is best suited when moderate compression is needed, and the low‑rank correction can be exploited either through a decomposed storage or as an intermediate step before fine‑tuning.

---

## Conclusion

The ARMOR Pruner (registered as `"armor_pruner"`) offers a compelling middle ground between unstructured pruning and full low‑rank factorization. By combining a structured sparse core with a low‑rank error correction, it preserves more of the original weight matrix’s information than pure magnitude pruning, while still offering potential for hardware‑efficient inference when the two components are stored separately. Its integration into the compression framework makes it a useful tool for experimentation with hybrid sparse‑plus‑low‑rank representations. Future enhancements could include true 2:4 block sparsity and adaptive rank selection for the correction.