# TTLinearLayer: Tensor Train (Matrix Product State) Linear Layer

**Standalone module**  
*Represents the weight matrix of a linear layer as a chain of small core tensors in the Tensor Train format, reducing parameter count dramatically and enabling efficient training and inference without ever storing the full weight.*

---

## Overview

TTLinearLayer is a drop‑in replacement for `nn.Linear` that compresses the weight matrix **W** ∈ ℝ<sup>out×in</sup> into the **Tensor Train (TT)** decomposition, also known as **Matrix Product State (MPS)** in quantum physics. The key idea is to reshape both input and output dimensions into multi‑dimensional grids:

```
in = ∏_{k=1}^{d} p_k,    out = ∏_{k=1}^{d} q_k
```

and represent **W** as a sequence of `d` small **core tensors** **G**<sub>k</sub> of size `r_{k-1} × p_k × q_k × r_k` (with `r_0 = r_d = 1`). The intermediate **TT‑ranks** `r_k` control the approximation accuracy and the total number of parameters. During a forward pass, the linear transformation `y = x W^T + b` can be performed by sequentially contracting the input (reshaped to the same grid) with the cores, without ever materialising the full weight matrix. This reduces the parameter count from `out·in` to Σ<sub>k</sub> r_{k-1}·p_k·q_k·r_k, often by orders of magnitude, and can accelerate inference on hardware that supports tensor contractions.

The class provides a constructor, a `from_linear` method that initialises the cores from a pretrained dense weight via TT‑SVD, and a `forward` method. In the current implementation, the forward pass reconstructs the full weight (for correctness) unless the TT contraction is explicitly implemented; a fallback for the common **d=2** (two‑factor) case is included, and higher dimensions require the `tensorly` library. The module is intended as both a practical compression tool and an educational reference.

---

## Methodology

### 1. Tensor Train Decomposition for Matrices

Consider a matrix **W** ∈ ℝ<sup>M×N</sup>. Choose factorisations:

```
M = m_1 · m_2 · … · m_d
N = n_1 · n_2 · … · n_d
```

(here, for compatibility with linear layer conventions, we rename `p_k = n_k` for the input factors and `q_k = m_k` for the output factors). The weight can be reshaped into a `2d`‑dimensional tensor:

```
Ŵ ∈ ℝ^{m_1×n_1 × m_2×n_2 × … × m_d×n_d}
```

The **Tensor Train (TT) decomposition** represents this tensor as a product of cores:

```
Ŵ(i₁,j₁,…,i_d,j_d) = G_1[i₁,j₁] · G_2[i₂,j₂] · … · G_d[i_d,j_d]
```

where each **G**<sub>k</sub> is a 4‑dim tensor of shape `r_{k-1} × m_k × n_k × r_k`, and the multiplication implies contraction over the intermediate rank indices (like a chain of matrix multiplications). The ranks `r_k` satisfy `r_0 = r_d = 1` and `1 ≤ r_k ≤ min( Π_{i≤k} m_i n_i, Π_{i>k} m_i n_i )`.

For a linear layer, the input **x** ∈ ℝ<sup>N</sup> is reshaped to `(n_1, n_2, …, n_d)` and the output `y = x·W^T + b` can be computed by contracting **x** with the core tensors, avoiding the full matrix. This yields memory and computational savings when the TT‑ranks are small.

### 2. Parameter Count and Compression

The number of parameters in the TT representation is:

```
params_TT = Σ_{k=1}^{d} r_{k-1} · p_k · q_k · r_k
```

Compare to the original `M·N`. For example, with `M=N=768`, factorised as `(p1,p2)=(24,32)`, `(q1,q2)=(24,32)`, and all intermediate ranks set to `r=16`:

- Cores: G₁: (1×24×24×16) = 9216, G₂: (16×32×32×1) = 16384, total ≈ 25,600 parameters.
- Original dense: 589,824 parameters → compression ratio ≈ 23×.

For larger `d`, the compression can be even greater.

### 3. TT‑SVD Initialisation from a Pretrained Weight

Given the original weight matrix **W**, the `from_linear` class method computes the TT decomposition using a sequence of truncated SVDs (TT‑SVD). The algorithm consists of:

1. Reshape **W** to the tensor `(q_1,…,q_d, p_1,…,p_d)`.
2. For `k = 1, …, d-1`:
   - Reshape the current residual to a matrix with rows corresponding to the combined left modes and columns to the combined right modes.
   - Perform truncated SVD to keep the top `r_k` singular components.
   - The left singular vectors become the core for mode `k`, the singular values and right singular vectors are folded into the next core.
3. The last core is the remaining right part, with `r_d = 1`.

This procedure is exact if the original tensor has TT‑ranks at most the chosen ranks; otherwise, it gives the optimal rank‑`r` approximation under certain orthogonality conditions. The current code supports `d=2` (common for moderate hidden sizes) natively; for higher `d`, the `tensorly` library is required.

### 4. Forward Pass (Implemented via Materialisation)

Because efficient TT‑contraction code is intricate, the current `forward` method reconstructs the full weight matrix **W_full** from the cores and then performs a standard matrix multiplication. This preserves correctness and simplicity, but it defeats the memory‑efficiency of TT during inference. The full weight reconstruction uses:

- For `d=2`, a direct `einsum` contraction:
  ```
  W_full = einsum('pqr,rPQ->pqPQ', core0, core1).permute(0,2,1,3).reshape(out, in)
  ```
- For `d>2`, if `tensorly` is installed, it could use `tt_to_tensor`; otherwise, it raises an error.

A production version would implement the sequential contraction directly to avoid the large intermediate matrix.

---

## Complexity Analysis

### A. Compression Time Complexity

- **TT‑SVD for d=2**: O(out·in·min(out,in)) due to a single SVD of the reshaped matrix. For 768×768, this is ~4.5×10⁸ operations, taking milliseconds.
- **For d>2**: the chain of SVDs involves several matrix operations of intermediate sizes, but overall complexity is comparable to a full SVD of the original matrix, often slightly higher but still manageable for post‑training compression.

### B. Inference Complexity (when materialising weight)

- **Original dense layer**: O(batch·out·in) FLOPs.
- **TTLinearLayer (current)**: same dense cost, plus the materialisation overhead O(out·in). No speed‑up; memory during forward includes the full weight matrix.

**If efficient TT contraction were used**, the complexity would be O(batch · Σ r_{k-1}·p_k·q_k·r_k), which is much lower than dense when TT‑ranks are small.

### C. Memory Usage

- **Parameters stored**: only the core tensors, drastically reduced.
- **During inference (current)**: the full weight matrix is reconstructed, temporarily using `out·in` memory, negating the storage benefit at runtime. A future optimised implementation would avoid this.

---

## Implementation Details

### TTLinearLayer Class

```python
class TTLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, tt_rank=16, bias=True, shape=None, dtype=None, device=None):
        ...
```

- **`tt_rank`**: Target TT‑rank for all intermediate bonds (default 16). The actual rank may be capped by dimension products.
- **`shape`**: Optional `(p_dims, q_dims)` tuple of tuples specifying the factorisation. If `None`, a balanced 2‑factor split is chosen automatically using divisors near the square roots. For example, 768 → (24,32) or (32,24) depending on the divisor search.
- **`bias`**: Whether to include a bias term.
- **Cores**: Stored as an `nn.ParameterList` of `nn.Parameter` tensors. The first core shape `(1, p1, q1, r1)`, intermediate cores `(r_{i-1}, p_i, q_i, r_i)`, last core `(r_{d-1}, p_d, q_d, 1)`.

**`_default_shape(in_f, out_f)`**: finds integer factors `(p1,p2)` and `(q1,q2)` that are closest to equal, minimising the difference between `p1` and `p2`. This ensures balanced core sizes.

**`from_linear(cls, linear_layer, tt_rank=16)`**:
1. Determines the factorisation shape via `_get_shape`.
2. Creates an empty `TTLinearLayer`.
3. Performs TT‑SVD:
   - For d=2 (the common case):
     - Reshapes the weight to `(q1, q2, p1, p2)`.
     - Unfolds to `(q1·p1, q2·p2)`.
     - Computes SVD, truncates to `tt_rank`.
     - Splits singular values via `sqrt(S)` to initialise two cores.
     - Shapes the first core as `(1, p1, q1, rank)` and second as `(rank, p2, q2, 1)`.
   - For d>2, attempts to use `tensorly`; if not installed, falls back to random initialisation with a warning.
4. Copies bias if present.

**`forward(x)`**:
- Flattens batch dimensions.
- Materialises the full weight via `_materialize_weight()`.
- Performs standard matrix multiplication and adds bias.
- Reshapes output to original batch shape.

**`_materialize_weight()`**:
- For d=2, uses `einsum` to contract the two cores into a full matrix.
- For d>2, requires `tensorly` to form the tensor and reshape.

### Integration

The `TTLinearLayer` can be used standalone or via the `TTCompressor` (registered as `"tt"`) which replaces all `nn.Linear` layers in a model.

---

## Usage and Configuration

### Standalone Replacement

```python
from tt_linear import TTLinearLayer
import torch.nn as nn

original = nn.Linear(768, 768)
compressed = TTLinearLayer.from_linear(original, tt_rank=32)

# Forward pass (materialises weight internally – memory heavy at runtime)
x = torch.randn(16, 768)
y = compressed(x)   # output shape (16, 768)
```

### Through the Compression Framework

```python
comp_config.comp_method = "tt"
comp_config.tt_rank = 12
```

Then run the compression pipeline. The model’s linear layers will be replaced with `TTLinearLayer` instances initialised from the pretrained weights.

### Parameters

| Parameter      | Default | Description                                                                                           |
|----------------|---------|-------------------------------------------------------------------------------------------------------|
| `tt_rank`      | `16`    | Target TT‑rank for all intermediate bonds. Lower values → more compression, larger approximation error. |
| (for factory) `tt_rank` | `16` | Read from `comp_config`.                                                                              |

**Note**: For dimensions that do not factorise nicely, the TT‑SVD may struggle or the rank may be capped. Highly composite hidden sizes (e.g., 768, 1024) work well.

---

## Discussion

### Advantages

- **Extreme parameter reduction**: TT can compress dense weight matrices by 10–50×, making models much smaller.
- **Mathematical elegance**: TT decomposition is well‑studied with optimality guarantees.
- **No calibration data**: The compression uses only the weight matrix, no forward passes needed.
- **Fine‑tuning friendly**: The cores are learnable parameters; after initialisation, the model can be fine‑tuned.
- **Scalable**: Can be extended to higher‑order tensors (d>2) for even greater compression.

### Limitations

- **Forward pass materialises weight**: The current implementation does not achieve inference speed‑up and temporarily uses as much memory as the original layer. An efficient TT contraction must be implemented for production.
- **d=2 factorisation hard‑coded**: The automatic shape only splits into two factors; higher‑order TT is only supported via external libraries.
- **Tensorly dependency for d>2**: Without tensorly, the decomposition cannot initialise properly and forward pass will fail.
- **Rank selection heuristic**: The TT‑rank is uniform across all bonds; in some cases, variable ranks could improve the trade‑off.
- **Reshape overhead**: The factorisation search and SVD are done on CPU (NumPy). This is fine for moderate sizes but could be slow for very large hidden dimensions.

### Comparison with Other Low‑Rank Methods

| Method         | Format            | Parameter Count          | Forward Efficiency     | Requires Factorisation? |
|----------------|-------------------|--------------------------|------------------------|--------------------------|
| Tensor Train   | TT‑cores          | Σ r_{k-1}p_k q_k r_k    | O(batch·sum core sizes) if optimised | Yes |
| SVD (SoKA)     | U Σ V             | r·(out+in)               | Two small matmuls      | No                       |
| Kronecker      | A ⊗ B             | m₁n₁ + m₂n₂              | Via Kronecker identity | Yes                      |
| HashedNets     | Hash table        | B                        | Indexing + matmul      | No                       |

TT offers a middle ground between simple low‑rank (SVD) and extremely compact hash‑based methods, with the ability to capture hierarchical correlations when dimensions are factorised.

---

## Conclusion

TTLinearLayer provides a robust implementation of Tensor Train decomposition for linear layers, enabling dramatic parameter compression from a single pretrained weight matrix. The `from_linear` initialisation via TT‑SVD yields a near‑optimal low‑rank TT representation, while the modular design allows future integration of efficient TT‑contraction kernels. As a drop‑in replacement for `nn.Linear`, it combines storage efficiency with the possibility of fine‑tuning, making it a valuable tool for deploying large Transformer models on memory‑constrained devices. For actual runtime speed‑ups, users are encouraged to implement the forward contraction without materialising the full weight, leveraging libraries such as `tensorly` or custom CUDA kernels.