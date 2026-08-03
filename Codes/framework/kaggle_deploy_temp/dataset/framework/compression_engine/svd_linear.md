# SVDLinearLayer: Low‑Rank Factorised Replacement for nn.Linear

**Standalone module**  
*Replaces a standard linear layer with an explicit low‑rank factorisation W ≈ U Σ V^T, where U, Σ, V are the top‑`r` singular components of the original weight matrix. The forward pass uses two small matrix multiplications without materialising the full weight, reducing both parameter count and computational cost.*

---

## Overview

`SVDLinearLayer` is a lightweight, drop‑in replacement for a pretrained `nn.Linear` module. It decomposes the weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup> into three factors:

```
W ≈ U_r · diag(σ_1, …, σ_r) · V_r^T
```

where `r` is the specified rank (≤ min(d_out, d_in)). The left singular vectors **U_r** ∈ ℝ<sup>d_out × r</sup>, the singular values **σ** ∈ ℝ<sup>r</sup>, and the right singular vectors **V_r** ∈ ℝ<sup>r × d_in</sup> are taken from the truncated SVD of the original weight. During inference, the linear transformation is performed as:

```
y = ( (x · V_r^T) ⊙ σ ) · U_r^T + bias
```

This sequential low‑rank multiplication avoids ever constructing the full `d_out × d_in` matrix, thereby reducing the number of stored parameters from `d_out·d_in` to `r·(d_out + d_in + 1)` (plus bias) and the computational cost per token from `O(d_out·d_in)` to `O(r·(d_out + d_in))`.

The module is ideal for post‑training compression: it initialises the factors directly from the SVD of the original layer’s weight, providing an optimal low‑rank approximation (in Frobenius norm) without any training. The resulting layer can be used as‑is or fine‑tuned further.

---

## Methodology

### 1. Truncated Singular Value Decomposition

Let **W** ∈ ℝ<sup>M×N</sup> be the weight matrix of a linear layer, with M = out_features, N = in_features. The full SVD is:

```
W = U Σ V^H
```

where **U** ∈ ℂ<sup>M×K</sup>, **Σ** ∈ ℝ<sup>K×K</sup> (diagonal), **V^H** ∈ ℂ<sup>K×N</sup>, and K = min(M,N). For a real matrix, **U** and **V** are real orthogonal matrices.

Given a target rank `r ≤ K`, the best rank‑`r` approximation in the Frobenius norm is obtained by keeping only the first `r` singular values and the corresponding singular vectors:

```
W ≈ U_r · Σ_r · V_r^T
```

where:
- **U_r** = U[:, :r] ∈ ℝ<sup>M×r</sup>
- **Σ_r** = diag(σ₁, …, σ_r) ∈ ℝ<sup>r×r</sup>
- **V_r^T** = (V^H)[:r, :] ∈ ℝ<sup>r×N</sup>

### 2. Parameterisation

The layer stores three trainable parameters (or frozen, if not fine‑tuned):
- `self.U` : shape (M, r) – left singular vectors
- `self.S` : shape (r,) – singular values (stored as a 1D tensor, interpreted as diagonal)
- `self.Vh` : shape (r, N) – right singular vectors (transposed)

The bias is copied unchanged from the original layer.

### 3. Forward Pass

Given an input **x** ∈ ℝ<sup>batch×N</sup>, the forward computation is:

```
h = x · Vh^T           # (batch, r)   → projects input onto the right singular subspace
h = h * self.S         # (batch, r)   → scales by singular values (element‑wise)
out = h · U^T          # (batch, M)   → maps back to output dimension
out = out + bias
```

This sequence is mathematically equivalent to:

```
out = x · (V_r Σ_r U_r^T)^T + bias = x · (U_r Σ_r V_r^T) + bias
```

but avoids forming the full `M×N` matrix explicitly. The computational complexity per token is:
- First multiplication: `batch × N × r`
- Element‑wise scaling: negligible
- Second multiplication: `batch × r × M`
Total: `O(batch·r·(M+N))`.

### 4. Initialisation from a Pretrained Layer

The constructor takes an existing `nn.Linear` and a rank. It computes the full SVD of the weight using `np.linalg.svd` (on CPU for simplicity), truncates to `r`, and assigns the components to `U`, `S`, `Vh`. This provides an optimal low‑rank approximation of the original weight, so the new layer immediately mimics the old one as closely as possible for that rank.

---

## Complexity Analysis

### A. Compression Time

- **SVD**: Computing the full SVD of an M×N matrix via `np.linalg.svd` costs O(M·N·min(M,N)). For a typical 768×768 matrix, this is roughly 4.5×10⁸ floating‑point operations, taking a few milliseconds on a modern CPU.
- **Layer construction**: O(1) to instantiate and copy data.
- Overall, replacing a single layer is practically instantaneous for moderate dimensions.

### B. Inference Complexity

**Original layer**: `y = x @ W^T + bias` → `O(batch·M·N)`.

**SVDLinearLayer**: `y = ((x @ Vh^T) * S) @ U^T + bias` → `O(batch·r·(M+N))`.

For example, with M=N=768, r=64, the original cost per token is ~589k operations, while the factorised cost is ~98k operations, a 6× reduction. For larger r, the speed‑up diminishes, but for moderate r (e.g., 16–64) the savings are substantial, especially on hardware where memory bandwidth is the bottleneck.

### C. Memory Usage

**Original**: stores a weight tensor of M·N floats (e.g., 589,824 floats for 768×768).

**SVDLinearLayer**: stores:
- U: M·r floats
- S: r floats
- Vh: r·N floats
- bias: M floats

Total: `r·(M+N) + r + M`. For M=N=768, r=64: parameters ≈ 64·1536 + 64 + 768 ≈ 98,176 + 832 ≈ 99,008, about **6× fewer** than the original 589,824. The memory footprint of the layer is thus significantly reduced.

---

## Implementation Details

### `SVDLinearLayer` Class

```python
class SVDLinearLayer(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int):
        ...
```

- **`linear_layer`**: an existing `nn.Linear` whose weight and bias are used for initialisation.
- **`rank`**: desired rank; the actual rank used is `min(rank, in_features, out_features)`.

**Parameters**:
- `self.U` (M×r), `self.S` (r), `self.Vh` (r×N) are `nn.Parameter`s and thus can be updated during fine‑tuning.
- `self.bias` is copied or set to `None`.

**`forward(x)`**:
1. `x_v = torch.matmul(x, self.Vh.t())` – multiply by V_r^T.
2. `out = torch.matmul(x_v * self.S.unsqueeze(0), self.U.t())` – scale by singular values, then multiply by U_r^T.
3. Add bias if present.

This implementation uses two matrix multiplications and an element‑wise multiplication. It is clean and does not reconstruct the full weight matrix, thus achieving both memory and computational benefits.

### Integration

The module is not registered as a compressor; it is intended to be used directly in model modification scripts. A simple replacement loop can be:

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        setattr(parent, name, SVDLinearLayer(module, rank=64))
```

---

## Usage and Configuration

To replace a single linear layer:

```python
import torch.nn as nn
from svd_linear_layer import SVDLinearLayer

original_linear = nn.Linear(768, 768)
compressed_linear = SVDLinearLayer(original_linear, rank=64)

# Use as normal
x = torch.randn(32, 768)
y = compressed_linear(x)
```

To compress an entire model, one would iterate over all `nn.Linear` layers and replace them, as shown above. No additional configuration is needed beyond the rank.

**Choosing the rank**:  
- Higher rank retains more accuracy but offers less compression.  
- A good starting point is to analyse the singular value spectrum of the layer and choose a rank that preserves, say, 90–95% of the cumulative energy.  
- For uniform compression across a model, a fixed rank (e.g., 64) is often a reasonable default.

---

## Discussion

### Advantages

- **Optimal approximation**: Truncated SVD gives the best low‑rank approximation in Frobenius norm, guaranteeing minimal distortion for the given rank.
- **Immediate parameter and FLOP reduction**: The factorised forward pass directly uses fewer multiplications; no additional steps needed.
- **No calibration data**: Purely weight‑based; works on any pretrained model out‑of‑the‑box.
- **Fine‑tuning compatible**: The factors are learnable parameters, allowing the model to adapt to the low‑rank constraint if necessary.
- **Simple and interpretable**: The singular values and vectors can be inspected to understand the layer’s structure.

### Limitations

- **Fixed rank per layer**: The same rank is used for all replaced layers unless manually tuned per layer.
- **SVD cost**: Computing the full SVD on CPU for many large layers might be slow for extremely large models (e.g., LLMs with hidden size > 4096). In such cases, randomized SVD or GPU‑based SVD can be used.
- **Bias not compressed**: The bias remains at full size, which is negligible.
- **Loss of accuracy without fine‑tuning**: The approximation discards information; a small accuracy drop is expected. Fine‑tuning can recover most of it.

### Comparison with Other Low‑Rank Modules

| Module           | Factorisation | Forward Cost       | Parameter Count        | Notes |
|------------------|---------------|--------------------|------------------------|-------|
| SVDLinearLayer   | U Σ V^T       | O(r(M+N))          | r(M+N) + r + M         | Explicit diagonal Σ, best rank‑r approx. |
| SoKALinear       | U V           | O(r(M+N))          | r(M+N) + M             | Absorbs Σ into U, slightly fewer params but identical forward cost. |
| A3 (weight overwrite) | Dense low‑rank | O(M·N)         | M·N                    | No inference speed‑up; just overwrites weight. |

`SVDLinearLayer` explicitly retains the singular values as a separate vector, which can be useful if one wishes to prune them later or apply a sparse mask (as in SALR). For pure compression, SoKALinear’s absorption of Σ into U is equivalent in terms of FLOPs and parameters (r parameters less). The choice is stylistic.

---

## Conclusion

`SVDLinearLayer` provides a clean, ready‑to‑use low‑rank replacement for standard linear layers, offering immediate reductions in both memory footprint and inference cost. Its initialisation from a pretrained layer via truncated SVD is optimal and requires no calibration. The module is a solid foundation for further compression pipelines and can be easily integrated into any PyTorch model. For production use, the rank should be chosen based on a trade‑off between accuracy and efficiency, and optionally followed by a few steps of fine‑tuning to recover any lost performance.