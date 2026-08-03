# Tensor Train Compressor (TT)

**Registered as `tt`**  
*Replaces every `nn.Linear` layer with a Tensor Train (TT) factorised layer, dramatically reducing the number of parameters by representing the weight matrix as a chain of small core tensors, with controlled approximation error.*

---

## Overview

The Tensor Train Compressor applies the **Tensor Train decomposition**, also known as **Matrix Product States (MPS)** in physics, to the weight matrices of linear layers. Instead of storing a dense matrix **W** ∈ ℝ<sup>M×N</sup>, the method reshapes it into a high‑dimensional tensor and factorises it into a sequence of three‑dimensional **core tensors** **G**<sub>1</sub>, **G**<sub>2</sub>, …, **G**<sub>d</sub>. Each core is a small tensor of size `r_{k-1} × m_k × n_k × r_k`, where `M = ∏ m_k`, `N = ∏ n_k`, and the **TT‑ranks** `r_k` control the compression level. The forward pass performs a series of small tensor contractions instead of a large matrix‑vector product, leading to significant savings in both memory and computation when the TT‑ranks are low.

The compressor is designed for the DistilBERT compression framework, registered under `"tt"`. It automatically factorises every `nn.Linear` layer using a user‑specified TT‑rank (default 16). The factorisation is performed by the `TTLinearLayer` class (imported from `.tt_linear`), which provides `from_linear` to initialise the cores from the original weight matrix. This allows immediate compression without any calibration data, though fine‑tuning is recommended to recover the best accuracy.

---

## Methodology

### 1. Tensor Train Format for Matrices

A matrix **W** ∈ ℝ<sup>M×N</sup> can be reshaped into a `2d`‑dimensional tensor by factorising its dimensions:

```
M = m_1 · m_2 · … · m_d
N = n_1 · n_2 · … · n_d
```

and identifying the weight as a tensor **W** ∈ ℝ<sup>m_1×n_1 × … × m_d×n_d</sup> (after pairing input and output modes). The **Tensor Train (TT) decomposition** represents this tensor as a product of `d` core tensors:

```
W(i_1, j_1, …, i_d, j_d) = G_1[i_1, j_1] · G_2[i_2, j_2] · … · G_d[i_d, j_d]
```

where each core **G**<sub>k</sub> is a 4‑dimensional tensor of shape `r_{k-1} × m_k × n_k × r_k`. The boundary ranks are fixed: `r_0 = r_d = 1`. The intermediate ranks `r_1, …, r_{d-1}` are hyperparameters (commonly set to a single `tt_rank` value). The contraction is performed by summing over the intermediate indices (like a chain of matrix multiplications).

In the linear layer setting, the input vector **x** ∈ ℝ<sup>N</sup> is also reshaped into a tensor of shape `(n_1, …, n_d)`. The output vector **y** = **x**·**W**<sup>T</sup> (or **Wx** depending on convention) can be computed by a series of tensor contractions without ever reconstructing the full weight matrix. The computational cost is proportional to the sum of the core sizes, which is much smaller than `M·N` when the TT‑ranks are low.

### 2. Compression via TT Decomposition

Given an original weight matrix **W**, the TT decomposition is typically computed by a sequence of truncated SVDs (TT‑SVD) or cross‑approximation algorithms. The `TTLinearLayer.from_linear` method performs this decomposition, setting the target rank(s) to the specified `tt_rank`. The result is a set of core tensors that approximate **W** up to a controllable error. The number of parameters in the TT format is:

```
params_TT = Σ_{k=1}^{d} r_{k-1} · m_k · n_k · r_k
```

where `r_0 = r_d = 1`. For a square matrix where `M=N=∏ s_k` (with `m_k=n_k=s_k`), and using a uniform rank `r`, this becomes approximately `d · r² · s²` if each core is roughly square, which is drastically smaller than `(∏ s_k)²` for large `d` and small `r`.

### 3. Forward Pass in TTLinearLayer

The `TTLinearLayer` module stores the list of core tensors as trainable parameters (or buffers, if frozen). During inference, the linear transformation is performed by contracting the input vector (reshaped to a tensor) with the cores in sequence, accumulating the result into the output shape. This is typically achieved with a loop:

```
y = x_reshaped
for core in cores:
    y = tensor_contract(y, core)
```

The exact contraction order ensures that the number of operations remains proportional to the TT‑ranks and the dimensions. No full matrix is materialised, so the memory footprint is reduced and inference can be faster when ranks are low.

---

## Complexity Analysis

### A. Compression Time Complexity

The TT decomposition of a matrix of size M×N typically requires several SVDs on reshaped sub‑matrices. The complexity is roughly O( (M+N) · r² · max(m_k,n_k) ) but depends on the specific algorithm. For a 768×768 matrix with a rank of 16, the decomposition takes a few seconds on CPU. The overall compression time for a model with L linear layers is moderate (minutes for a full BERT‑size model).

### B. Inference Complexity

**Original dense layer**: O(batch · M · N) FLOPs.

**TTLinearLayer**: O( batch · Σ_k r_{k-1} · m_k · n_k · r_k ) operations. For example, with M=N=768 factorised as 4×4×4×4×3? Actually 768 can be factored in many ways; a typical choice might be `(8,8,12)` for M and N. The FLOPs scale with the TT‑ranks, which are much smaller than the dimensions. Consequently, inference can be faster, especially on hardware where memory bandwidth is the bottleneck.

### C. Memory Usage

- **Parameter count**: The original layer stores M·N numbers; the TT layer stores only the core tensors. For a moderate TT‑rank, the compression ratio (M·N / sum_core_sizes) can be 10–50×. Thus the model size shrinks considerably.
- **During inference**: The cores and the intermediate contractions use O( batch · max(r_k) · max(m_k) ) memory, which is usually less than O(batch · N) when r_k is small.

---

## Implementation Details

### TTCompressor Class

```python
class TTCompressor(BaseCompressor):
    def __init__(self, model: nn.Module, tt_rank: int = 16, **kwargs):
        super().__init__(model, name="tt_compressor", **kwargs)
        self.tt_rank = tt_rank
        self.compressed_layers = []
```

- **`tt_rank`**: Target TT‑rank for all intermediate bonds. The actual rank may be capped by the dimensions of the cores. Default 16.

**`compress()`**:
1. Iterates over all named modules.
2. For each `nn.Linear` that is not already a `TTLinearLayer`:
   - Calls `TTLinearLayer.from_linear(module, self.tt_rank)` to create a new layer initialised from the original weight.
   - Replaces the module in‑place via `setattr`.
   - Records the compression in `self.compressed_layers`.
3. Updates `_compressed_params` and returns the model with stats.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("tt")
def tt_compressor_factory(model: nn.Module, comp_config) -> nn.Module:
    tt_rank = getattr(comp_config, 'tt_rank', 16)
    compressor = TTCompressor(model, tt_rank=tt_rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `tt_rank` from the compression configuration. If not set, 16 is used.

### TTLinearLayer (from `.tt_linear`)

The `TTLinearLayer` class (not shown in detail) is expected to contain:
- A list of core tensors (parameters or buffers).
- A bias (copied from the original linear layer).
- A `forward` method that performs the TT‑contraction with the input.
- A `from_linear` classmethod that decomposes the weight matrix.

---

## Usage and Configuration

To apply Tensor Train compression, set the compression method to `"tt"` and optionally specify the TT‑rank.

### Example

```python
comp_config.comp_method = "tt"
comp_config.tt_rank = 12   # use rank 12 for all layers
```

Then run the compression pipeline. The model will be transformed; its linear layers will be replaced with `TTLinearLayer` instances.

### Parameters

| Parameter | Default | Description                                                                                                 |
|-----------|---------|-------------------------------------------------------------------------------------------------------------|
| `tt_rank` | `16`    | Maximum TT‑rank for the decomposition. Lower values give higher compression but larger approximation error. |

**Note**: The actual ranks used may be bounded by the factorisation of the input/output dimensions. If a dimension cannot be factorised into at least two factors, the TT decomposition might fall back to a single core (equivalent to a dense matrix) or raise an error, depending on the implementation. Ensure the hidden sizes are highly composite (e.g., 768 = 2⁸·3) for best results.

---

## Discussion

### Advantages

- **Massive parameter reduction**: TT factorisation can compress a weight matrix by orders of magnitude, making it suitable for extreme memory constraints.
- **Efficient inference**: The forward pass avoids full matrix multiplication, potentially yielding speed‑ups on CPU or edge devices.
- **No calibration data**: Works purely on weight matrices; one‑shot compression.
- **Fine‑tuning compatible**: The cores can be updated by gradient descent, allowing recovery of accuracy.
- **Well‑studied mathematical framework**: TT decomposition has strong theoretical guarantees (e.g., existence of low‑rank approximations for smooth tensors).

### Limitations

- **Dependency on dimensional factorisation**: The method requires that `in_features` and `out_features` can be factorised into multiple factors. Prime numbers or awkward dimensions reduce the effectiveness and may require padding or fallback.
- **Implementation complexity**: The `TTLinearLayer` must handle arbitrary factorisations and efficient contraction orders; a suboptimal implementation can be slow.
- **Rank must be chosen**: No automatic rank selection; users must experiment to find the best trade‑off.
- **Possible accuracy loss**: Without fine‑tuning, the truncated decomposition may degrade performance, especially for aggressive ranks.
- **Core initialisation quality**: The `from_linear` method’s decomposition accuracy depends on the TT‑SVD algorithm; if poorly implemented, the approximation may be suboptimal.

### Comparison with Other Low‑Rank Methods

| Method       | Decomposition Type        | Parameter Count Formula              | Inference Speed‑up | Requires Factorisation of Dims? |
|--------------|---------------------------|--------------------------------------|--------------------|----------------------------------|
| Tensor Train | TT (matrix product state) | Σ r_{k-1} m_k n_k r_k              | Yes (contractions) | Yes                              |
| SVD (SoKA)   | U Σ V^T                   | r·(M+N)                              | Yes (two matmuls) | No                               |
| Kronecker    | A ⊗ B                     | m₁n₁ + m₂n₂                         | Yes (via identity)| Yes                              |
| HashedNets   | Hash table                | B                                     | No (weight reconstruction) | No                         |

Tensor Train offers a more flexible factorisation that can capture hierarchical correlations in the weight matrix, often achieving better compression than a single low‑rank decomposition for the same parameter budget, but it requires careful dimension shaping.

---

## Conclusion

The Tensor Train Compressor (registered as `"tt"`) provides a powerful, mathematically elegant approach to model compression by representing weight matrices in the tensor train format. It delivers dramatic parameter reductions and the potential for accelerated inference, especially on hardware that can exploit the structured contractions. Its integration into the DistilBERT framework via a simple factory function makes it easy to apply. Future improvements could include automatic rank selection based on a target energy preservation, support for more sophisticated TT‑rounding algorithms, and optimised CUDA kernels for the TT‑linear operation. As always, fine‑tuning after compression is recommended to maximise accuracy.