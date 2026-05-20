# ORION: Optimized Rectified Input‑aware Orthogonal Network — Hybrid Sparse‑Kernel Attention with SVD‑Factorised Projections

**Abstract**

We present **ORION (Optimized Rectified Input‑aware Orthogonal Network)**, a hybrid attention module for efficient Transformers that combines three design elements: **(1)** energy‑thresholded truncated SVD factorisation of all linear projections for parameter‑efficient zero‑shot transfer from pretrained weights; **(2)** a head‑wise split where a designated subset of heads performs linear‑complexity kernelised attention via random Fourier features, while the remaining heads perform full dot‑product attention with a sparsity‑inducing **Softpick** rectified softmax; and **(3)** a unified masking procedure for padding tokens. By allocating heads to either efficient global linear attention or exact full attention with natural sparsity, ORION achieves a tunable balance between computational cost and representational power, while drastically reducing the number of trainable parameters through SVD. The module is designed as a drop‑in replacement for the self‑attention sub‑layer in DistilBERT, supporting zero‑shot initialisation and full integration into the compression framework via a registry‑based injection function. This paper provides a rigorous mathematical description of each component, the head‑wise hybridisation strategy, and the overall architecture.

---

## 1. Introduction

The self‑attention mechanism in Transformer models is both a key to their success and a bottleneck due to its \(O(N^2)\) time and memory complexity. Several recent works have explored hybrid approaches that combine different attention patterns—local vs. global, exact vs. approximate—to achieve better efficiency. **ORION** adopts a head‑wise hybrid scheme: the total set of attention heads is partitioned into two groups, each employing a different attention computation.

- **Kernel heads** use **linear attention** with orthogonal random Fourier features (FAVOR+), which approximates the softmax kernel and scales linearly with sequence length \(N\).
- **Full heads** use **exact scaled dot‑product attention** but with a **Softpick** activation function in place of softmax. Softpick naturally produces sparse attention maps by zeroing out weak connections, reducing the effective number of attended tokens and potentially improving interpretability.

All projections (query, key, value, output) are compressed via a truncated SVD of the pretrained weights, retaining a configurable fraction of the singular‑value energy. This greatly reduces the parameter count and enables zero‑shot deployment.

The combination is designed so that the kernel heads capture global context efficiently, while the full heads with Softpick provide high‑precision, sparse interactions. The module is fully compatible with the DistilBERT architecture and the surrounding compression framework.

In this paper, we detail the mathematical construction of the factorised linear layers, the kernelised linear attention, the Softpick activation and full attention, and the overall ORION forward pass.

---

## 2. Method

### 2.1 Factorised Linear Projections via Energy‑Thresholded SVD

Let \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) be a pretrained weight matrix and \(\mathbf{b}\) its optional bias. The `FactorisedLinear` class constructs a bottleneck layer:

\[
\mathbf{y} = \mathbf{U}_r (\mathbf{V}_r \mathbf{x}) + \mathbf{b},
\]

where \(\mathbf{V}_r \in \mathbb{R}^{r \times d_{\text{in}}}\) and \(\mathbf{U}_r \in \mathbb{R}^{d_{\text{out}} \times r}\). The rank \(r\) is chosen adaptively by thresholding the cumulative singular‑value energy:

\[
r = \min\left\{ k \;\middle|\; \frac{\sum_{i=1}^k \sigma_i^2}{\sum_i \sigma_i^2} \ge \eta \right\},
\]

clamped between `min_rank` and `max_rank` (default 4 and 64). The factors are initialised as:

\[
\mathbf{V}_r = \mathbf{V}_{\text{full}[:r,:]}, \qquad
\mathbf{U}_r = \mathbf{U}_{\text{full}[:,:r]} \operatorname{diag}(\sigma_1,\dots,\sigma_r).
\]

These replace the standard linear layers for Q, K, V, and output, drastically reducing parameters while preserving the original model’s behaviour at initialisation.

### 2.2 Head‑Wise Hybridisation

The module partitions the \(H\) attention heads into two disjoint sets:
- \(H_k\) kernel heads (default 8)
- \(H_f\) full heads (default 4), where \(H_k + H_f = H\).

After obtaining the multi‑head queries, keys, and values \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times N \times d_h}\) via the factorised projections, we split them along the head dimension:

\[
\mathbf{Q}_{\text{kern}} = \mathbf{Q}_{[:, :H_k, :, :]}, \quad \mathbf{Q}_{\text{full}} = \mathbf{Q}_{[:, H_k:, :, :]},
\]

and similarly for \(\mathbf{K}, \mathbf{V}\).

### 2.3 Kernelised Linear Attention (Kernel Heads)

The kernel heads employ linear attention using positive orthogonal random features (a Performer‑style mechanism). A learnable random projection matrix \(\mathbf{R} \in \mathbb{R}^{d_h \times F}\) is stored per kernel head (shared across sequences). The feature map is:

\[
\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} \exp\!\big( \mathbf{x} \mathbf{R} \big),
\]

where \(F\) is the `rff_features` dimension (default 256). To improve stability, the queries and keys are first L2‑normalised. The feature maps are:

\[
\mathbf{\Phi}_Q = \exp(\hat{\mathbf{Q}}_{\text{kern}} \mathbf{R}) / \sqrt{F},\quad
\mathbf{\Phi}_K = \exp(\hat{\mathbf{K}}_{\text{kern}} \mathbf{R}) / \sqrt{F},
\]

with \(\hat{\mathbf{Q}}, \hat{\mathbf{K}}\) being unit‑norm versions. The linear attention output is computed via the associative property:

\[
\mathbf{KV} = \sum_{j=1}^N \mathbf{\Phi}_K(:,j) \mathbf{V}_{\text{kern}}(:,j)^\top \quad \in \mathbb{R}^{B \times H_k \times F \times d_h},
\]
\[
\mathbf{Z} = \sum_{j=1}^N \mathbf{\Phi}_K(:,j) \quad \in \mathbb{R}^{B \times H_k \times F},
\]
\[
\mathbf{C}_{\text{kern}} = \frac{\mathbf{\Phi}_Q \mathbf{KV}}{\mathbf{\Phi}_Q \mathbf{Z} + \epsilon}.
\]

All operations are linear in \(N\); the dominant cost is \(O(N F d_h)\).

### 2.4 Full Attention with Softpick Sparsity (Full Heads)

The full heads compute standard scaled dot‑product attention:

\[
\mathbf{S}_{\text{full}} = \frac{\mathbf{Q}_{\text{full}} \mathbf{K}_{\text{full}}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H_f \times N \times N}.
\]

Before applying the activation, a padding mask (if present) is applied by setting invalid positions to a large negative value. Then, instead of softmax, the **Softpick** function is applied:

\[
p_{ij} = \frac{\operatorname{ReLU}\!\big( e^{(S_{ij} - m_i)/\tau} - e^{-m_i/\tau} \big)}
            {\sum_{k} \operatorname{ReLU}\!\big( e^{(S_{ik} - m_i)/\tau} - e^{-m_i/\tau} \big) + \epsilon},
\]

where \(m_i = \max_j S_{ij}\) and \(\tau\) is a temperature parameter (default 1.0). Softpick tends to set many attention weights exactly to zero, yielding a sparse attention map without a hard top‑\(k\). The context for full heads is:

\[
\mathbf{C}_{\text{full}} = \operatorname{Softpick}(\mathbf{S}_{\text{full}}) \mathbf{V}_{\text{full}}.
\]

### 2.5 Padding Mask Handling

The module accepts an attention mask that can be 2D, 3D, or 4D. A robust conversion extracts a boolean `token_valid` tensor of shape \((B, N)\). For the full heads, this mask is applied to the scores before Softpick. For the kernel heads, no mask is applied (since the linear attention aggregates all tokens; padding tokens would need to be zeroed out in the key/value accumulation, which is not done in this implementation—a limitation). After obtaining both contexts, the final output is multiplied by the token validity mask to zero out padding positions.

### 2.6 Combination and Output

The two context tensors are concatenated along the head dimension to restore the full set of heads:

\[
\mathbf{C} = \operatorname{cat}(\mathbf{C}_{\text{kern}}, \mathbf{C}_{\text{full}}, \text{dim}=1) \in \mathbb{R}^{B \times H \times N \times d_h}.
\]

After masking, the heads are concatenated, reshaped to \((B, N, H d_h)\), and projected through the factorised output layer.

---

## 3. Implementation Details

- **`ORIONConfig`**: a dataclass holding all hyperparameters (energy, rank bounds, head counts, feature dimension, temperature, dropout).
- **`FactorisedLinear`**: performs SVD once, selects rank, stores `down` and `up` layers.
- **`ORIONAttention`**:
  - Inherits from `nn.Module` (not `BaseAttention`), but registered as `"orion"`.
  - Constructor requires `pretrained_attn` to initialise the four factorised projections.
  - The random Fourier projection matrix `rff_proj` is a learnable parameter of shape \((H_k, d_h, F)\).
  - Forward pass: split heads, compute kernel and full contexts, concatenate, mask, output projection.
- **Injection**: `inject_orion` reads parameters from `comp_config`, constructs `ORIONConfig`, and creates the module, replacing the original attention.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"orion"`. Relevant parameters in `comp_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_energy` | 0.95 | Fraction of singular‑value energy retained in factorised projections. |
| `min_rank` / `max_rank` | 4 / 64 | Rank limits for SVD. |
| `num_kernel_heads` | 8 | Number of heads using kernel linear attention. |
| `num_full_heads` | 4 | Number of heads using full attention with Softpick. |
| `rff_features` | 256 | Feature dimension for random Fourier features. |
| `softpick_temperature` | 1.0 | Temperature for Softpick. |
| `dropout` | 0.1 | Dropout on attention weights. |

Example injection via framework:
```python
comp_config.attention_type = "orion"
comp_config.num_kernel_heads = 10
comp_config.num_full_heads = 2   # must sum to total heads
```

The injection function copies pretrained weights and initialises the factorised layers.

---

## 5. Discussion

ORION presents a well‑rounded hybrid attention design. The head‑wise separation allows each type of attention to specialise: kernel heads provide efficient global context without quadratic cost, while full heads retain the ability to perform exact pairwise comparisons and, via Softpick, naturally learn sparse interactions. The use of SVD‑factorised projections drastically reduces the number of trainable parameters, which can be further beneficial when only fine‑tuning the model on a downstream task—the low‑rank adapters can be trained while the rest remains fixed.

Potential limitations:
- Padding tokens are not explicitly masked in the kernel heads’ accumulation, which could pollute the global context. This should be fixed by multiplying keys and values by a mask before aggregation.
- The random projection matrix `rff_proj` is shared across all kernel heads but is head‑specific in shape (one per head), which is correct.
- The Softpick activation, while inducing sparsity, does not guarantee a fixed sparsity budget, making the computational cost of the full heads still quadratic in \(N\). For very long sequences, one might want to further restrict the full heads with a window or low‑rank approximation.
- The total number of heads must be exactly partitioned; the default (8 kernel + 4 full) assumes a 12‑head model, which is typical for DistilBERT.

Overall, ORION demonstrates a sophisticated combination of recent advances—low‑rank factorisation, kernel linear attention, and sparse attention via rectified softmax—packed into a single, configurable module.

---

## 6. Conclusion

We have described ORION, a hybrid attention module that marries SVD‑compressed projections, kernelised linear attention for a subset of heads, and full attention with Softpick sparsity for the remaining heads. The design achieves parameter efficiency and a flexible trade‑off between computational cost and representational quality. Integrated into the DistilBERT compression framework, ORION provides a powerful drop‑in replacement for standard self‑attention, suitable for tasks requiring both efficiency and accuracy. Future work will extend the masking to the kernel branch and explore dynamic head allocation based on input length.