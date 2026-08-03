# DeltaKV: Residual‑Based Weight Compression via Low‑Rank Reference + Sparse Residual

**Registered as `delta_kv`**  
*Inspired by KV‑cache compression, this method decomposes linear weights into a low‑rank “reference” and a sparse “residual”, achieving high compression with controlled accuracy loss.*

---

## Overview

DeltaKV is a post‑training weight compression technique that draws inspiration from the **DeltaKV** paper (*DeltaKV: Residual‑Based KV Cache Compression via Long‑Range Similarity*, Hao et al., arXiv:2602.08005). In the original work, the key‑value cache of a language model is compressed by storing only the **residuals** of tokens relative to the most similar historical references, drastically reducing storage. Here, the same philosophy is applied to the weight matrices of linear layers: each weight **W** is decomposed into a **low‑rank reference** (capturing the main structure) and a **sparse residual** (capturing important deviations). The result is a compressed representation that can be stored more efficiently—the low‑rank part as a pair of small matrices, and the sparse residual as a sparse matrix—or used as an intermediate step before further factorization.

The module is designed as a drop‑in compressor for the DistilBERT compression framework. It operates on all `nn.Linear` layers, requires no calibration data, and is purely algebraic. It is registered under `"delta_kv"` and controlled by two parameters: the rank of the reference and the sparsity fraction of the residual.

---

## Methodology

### 1. Low‑Rank Reference via Truncated SVD

For a weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup>, we first extract a low‑rank component **L** that captures the bulk of the matrix’s energy. This is done via truncated SVD:

```
W = U Σ V^T
```

We retain the top `r` singular values and vectors:

```
L = U_r Σ_r V_r^T
```

where:
- **U_r** ∈ ℝ<sup>d_out × r</sup>,
- **Σ_r** ∈ ℝ<sup>r × r</sup> (diagonal),
- **V_r^T** ∈ ℝ<sup>r × d_in</sup>.

The rank `r` is a user‑specified value (default 16). This low‑rank matrix **L** serves as the “reference” that captures the core linear mapping.

### 2. Residual Extraction

The residual **R** is the difference between the original weight and its low‑rank reference:

```
R = W - L
```

This residual contains the information that was discarded by the low‑rank approximation. In many layers, this residual is close to zero except for a few important entries.

### 3. Sparse Residual via Magnitude Thresholding

To compress the residual, we apply **magnitude‑based sparsification**: keep only the top `s` fraction of entries (by absolute value) and set the rest to zero. Formally:

```
R_sparse[i,j] = R[i,j]  if |R[i,j]| ≥ θ
                0       otherwise
```

where `θ` is the threshold chosen so that exactly `⌈s·d_out·d_in⌉` entries are retained. The sparsity fraction `s` (default 0.29, matching the KV cache compression ratio from the paper) controls how many residual entries survive.

### 4. Combined Compressed Weight

The final compressed weight is the sum of the low‑rank reference and the sparse residual:

```
W_compressed = L + R_sparse
```

Because **L** has rank `r` and **R_sparse** has only a fraction `s` of non‑zeros, the combined matrix has a compact description:
- **L** can be stored as two matrices `A = U_r Σ_r` (size d_out × r) and `B = V_r` (size r × d_in), requiring `r·(d_out + d_in)` parameters.
- **R_sparse** can be stored in a sparse format (e.g., COO) requiring approximately `2·s·d_out·d_in` entries for indices and values (for 2D coordinates). Together, the storage can be significantly less than the original `d_out·d_in`.

In the current implementation, however, the dense matrix `W_compressed` is written back to the layer’s weight, so the model parameter count does not decrease immediately. The value is in the decomposition itself, which can be later converted to a factorized form.

### 5. Analogy to DeltaKV Cache Compression

In DeltaKV, the KV cache is compressed by keeping a small set of historical reference tokens and storing only the **residual** of each new token relative to the most similar references. Here, the reference is the low‑rank matrix **L** (analogous to the reference tokens), and the residual **R_sparse** captures the additional per‑entry “corrections”. The choice of sparsity `s=0.29` directly mirrors the storage reduction to 29% reported in the KV cache paper.

---

## Implementation Details

### DeltaKVCompressor Class

Inherits from `BaseCompressor`.

```python
class DeltaKVCompressor(BaseCompressor):
    def __init__(self, model, rank=16, sparsity=0.29, **kwargs):
        super().__init__(model, name="delta_kv", **kwargs)
        self.rank = rank
        self.sparsity = sparsity
```

- **`rank`**: Rank of the low‑rank reference **L** (default 16). Smaller values increase compression but may lose more information.
- **`sparsity`**: Fraction of residual entries to retain (default 0.29). A smaller sparsity gives a sparser residual and higher overall compression.

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. For each weight tensor **W** (converted to float32):
   - Computes full SVD: `U, S, Vt = torch.linalg.svd(W, full_matrices=False)`.
   - Determines effective rank `r = min(self.rank, min(W.shape))`.
   - Forms low‑rank reference **L** = `(U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]`.
   - Computes residual **R** = `W - L`.
   - Flattens absolute residual, selects the top `keep_n = max(1, int(numel * self.sparsity))` largest entries.
   - Finds the threshold `θ` = the smallest value among those top entries.
   - Creates a binary mask `mask = (abs(R) >= θ).float()`.
   - Forms sparse residual `R_sparse = R * mask`.
   - Sets the new weight to `W_compressed = L + R_sparse`, cast to original dtype.
3. Updates internal compressed parameter count and returns the model with stats.

**Note**: The SVD is performed on GPU via PyTorch, which may be memory‑intensive for very large matrices. For such cases, a CPU offload or approximate SVD could be used, but is not implemented here.

### DeltaKVStore (Utility)

The code also includes a `DeltaKVStore` class that implements the actual KV cache compression logic described in the paper. It maintains a reference key‑value store, computes residuals, and keeps only significant residuals. However, this class is **not used** by the `DeltaKVCompressor`. It is provided as a utility for future applications or direct KV cache compression, but for weight compression, only the low‑rank + sparse residual method is applied. The compressor’s name `delta_kv` thus reflects the spirit, not the literal mechanism of the paper.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("delta_kv")
def delta_kv_factory(model, comp_config):
    rank = getattr(comp_config, 'delta_kv_rank', 16)
    sparsity = getattr(comp_config, 'delta_kv_sparsity', 0.29)
    compressor = DeltaKVCompressor(model, rank=rank, sparsity=sparsity)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Configuration parameters `delta_kv_rank` and `delta_kv_sparsity` can be set in `comp_config`. The factory creates the compressor and returns the compressed model.

---

## Usage and Configuration

To apply DeltaKV compression, set the compression method to `"delta_kv"` and optionally tune the rank and sparsity.

### Example

```python
comp_config.comp_method = "delta_kv"
comp_config.delta_kv_rank = 8         # smaller rank → more compression
comp_config.delta_kv_sparsity = 0.2   # keep only 20% of residual entries
```

Then run the compression pipeline. All linear layer weights will be replaced with their low‑rank + sparse residual approximations.

### Parameters

| Parameter              | Default | Description                                                                                               |
|------------------------|---------|-----------------------------------------------------------------------------------------------------------|
| `delta_kv_rank`        | `16`    | Rank of the low‑rank reference matrix **L**. Smaller = more compression.                                  |
| `delta_kv_sparsity`    | `0.29`  | Fraction of residual entries to keep. 0.0 = pure low‑rank, 1.0 = no sparsification (full residual).       |

**Note**: The effective compression ratio in terms of “non‑zero parameters” (if stored factorized) is approximately `(r·(d_out+d_in) + 2·s·d_out·d_in) / (d_out·d_in)`. For a 768×768 matrix, rank 16 and sparsity 0.29 gives about 0.08 + 0.58 ≈ 66% of original parameters—not a dramatic reduction because the sparse residual is still large. To achieve aggressive compression, a smaller sparsity and rank are needed; but then accuracy may drop significantly. This method is best thought of as a way to induce a **structured sparse + low‑rank** representation that can be exploited by specialized hardware, rather than a direct parameter‑reduction technique.

---

## Discussion

### Advantages

- **Dual structure**: Combines the benefits of low‑rank (global structure) and sparsity (local corrections). The reference captures the principal subspace, while the residual captures important outliers.
- **Inspiration from KV cache**: The concept of keeping only residuals relative to a reference is novel and has proven effective in the context of KV caches; this is an adaptation to weight compression.
- **No calibration data**: Works purely on the weight matrix.
- **Tunable**: The rank and sparsity provide two knobs to trade off between accuracy and compression.

### Limitations

- **Dense weight output**: The current implementation stores the result as a dense matrix; to actually reduce parameters or FLOPs, one must extract the low‑rank factors and the sparse residual explicitly and use them in a custom layer.
- **Sparse residual still large**: At sparsity=0.29, the residual matrix is 71% zero, which is unstructured sparsity. Special hardware or sparse linear algebra is required to accelerate computations with such a matrix; otherwise, the benefit is mostly in storage (if stored in sparse format).
- **Fixed rank for all layers**: All linear layers get the same rank and sparsity, which may not be optimal. Adaptive allocation (e.g., based on singular value decay) could improve results.
- **Threshold ties**: The magnitude thresholding may keep slightly more entries than the target fraction if many entries have identical magnitudes near the threshold.
- **Not end‑to‑end KV compression**: The `DeltaKVStore` class, which actually implements KV cache compression, is not integrated. Users interested in KV cache compression would need to instantiate and use that class separately in their inference loop.

### Comparison with Other Structured Compressors

| Feature                     | DeltaKV (this)                     | ARMOR (sparse core + low‑rank)    |
|-----------------------------|------------------------------------|-----------------------------------|
| Reference                   | Low‑rank SVD                       | Sparse core (top‑2 per row block) |
| Error correction            | Sparse residual (unstructured)     | Low‑rank correction               |
| Sparsity pattern            | Unstructured                       | Structured (row‑wise per block)   |
| Parameter reduction (stored) | If factorized: low‑rank + sparse   | If factorized: sparse + low‑rank  |
| Inspiration                 | KV cache compression               | General sparse + low‑rank         |

DeltaKV is essentially a mirrored version of ARMOR: ARMOR has a sparse core and low‑rank correction; DeltaKV has a low‑rank core and sparse correction. The choice depends on which component (low‑rank vs. sparse) better captures the weight matrix’s structure.

---

## Conclusion

DeltaKV (registered as `"delta_kv"`) brings the residual‑based compression philosophy from KV cache optimisation to weight matrix compression. By decomposing a weight into a low‑rank reference and a sparse residual, it creates a hybrid representation that can be stored compactly and potentially accelerated on mixed‑precision hardware. While the current implementation overwrites the weight with the dense composite matrix, the underlying decomposition is the key contribution, ready to be exploited by further factorization steps. The method’s simplicity and zero‑shot nature make it a versatile addition to the DistilBERT compression toolkit, especially for those looking to experiment with sparse + low‑rank combinations inspired by recent advances in KV cache compression.