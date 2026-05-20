# Performer Attention: Fast Attention Via Positive Orthogonal Random Features (Simplified)

**Abstract**

We present a simplified implementation of the **Performer** attention module, based on the FAVOR+ algorithm (Fast Attention Via Positive Orthogonal Random Features) introduced by Choromanski et al. (2021). This module replaces the quadratic softmax self‑attention with a linear‑complexity approximation using random feature maps. Queries and keys are projected using a per‑head orthogonal random matrix, and a positive feature map (ReLU) is applied to produce non‑negative feature vectors. The attention output is then computed via the associative property of linear attention, resulting in \(O(L d^2)\) time and memory. Unlike the original Performer, this implementation uses a simple ReLU feature map instead of an exponential one, does not include the L2 normalisation term in the kernel approximation, and regenerates the random projection matrix at every forward pass when `redraw_projection` is enabled. The module is integrated as a drop‑in replacement for DistilBERT’s self‑attention layer within the compression framework. This paper details the mathematical construction, the orthogonal random feature generation, and the attention pipeline, while also discussing the deviations from the original paper.

---

## 1. Introduction

Multi‑head self‑attention scales quadratically with the sequence length \(L\), posing challenges for long sequences. The **Performer** (Choromanski et al., 2021) introduced a breakthrough linear attention mechanism via **FAVOR+**: a low‑variance approximation of the softmax kernel using **positive orthogonal random features**. By applying an orthogonal random matrix followed by an exponential and a normalisation term, one obtains a feature map that accurately approximates \(\exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})\). The attention can then be re‑written as a recurrent matrix product, avoiding the \(O(L^2)\) bottleneck.

The module described here adapts the Performer idea but with two notable simplifications:
1. The feature map is a plain **ReLU** activation applied after projection and scaling, rather than the exponential‑based map.  
2. The orthogonal random projection matrix is regenerated at each forward pass when `redraw_projection=True` (contrary to the usual fixed projection).  

Despite these differences, the module still performs linear attention with a positive feature map, achieving linear complexity. It is integrated into the compression framework as a baseline for kernel‑based efficient attention.

In this paper, we provide a complete mathematical description of the projection, feature map, linear attention, and the handling of padding masks. We also discuss the impact of the simplifications and the overall architecture.

---

## 2. Method

### 2.1 Multi‑Head Projections

Let the input be \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\). The module projects to queries, keys, and values for all heads simultaneously:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad \mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad \mathbf{V} = \mathbf{X}\mathbf{W}_V,
\]

with \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\). The tensors are reshaped to \((B, H, L, d_h)\) via the `_reshape` method, where \(H\) is the number of heads and \(d_h = D/H\).

### 2.2 Orthogonal Random Projection Matrix

For each head \(h\), a random matrix \(\mathbf{R}_h \in \mathbb{R}^{d_h \times R}\) is constructed, where \(R\) is the `rank` (feature dimension, default 256). The matrix is required to have orthonormal columns, which minimises the variance of the kernel approximation (FAVOR+). The construction uses a QR decomposition of a standard normal random matrix:

\[
\mathbf{M}_h \sim \mathcal{N}(0, 1)^{d_h \times R}, \quad
\mathbf{Q}_h \mathbf{R}_{\text{upper}} = \text{QR}(\mathbf{M}_h),
\]
\[
\mathbf{R}_h = \mathbf{Q}_h \quad (\text{orthonormal columns}).
\]

The resulting matrices \(\mathbf{R}_h\) are stacked into a single tensor \(\mathbf{R} \in \mathbb{R}^{H \times d_h \times R}\). The module provides a flag `redraw_projection`: if `True`, \(\mathbf{R}\) is regenerated at each forward call; otherwise, it is stored as a buffer and reused. In typical Performer usage, the matrix is fixed after initialisation to reduce variance across mini‑batches; the redraw option is unusual.

### 2.3 Positive Feature Map (ReLU)

The feature map \(\phi: \mathbb{R}^{d_h} \to \mathbb{R}^R\) is defined as:

\[
\phi(\mathbf{x}) = \operatorname{ReLU}\!\big( \mathbf{x} \mathbf{R}_h \cdot s \big),
\]

where \(s = 1 / d_h^{1/4}\) is a scaling factor. The ReLU ensures non‑negativity, making the feature map **positive**. In the original Performer, the map is \(\exp(\mathbf{x}\mathbf{R} - \|\mathbf{x}\|^2/2)\), which precisely approximates the Gaussian kernel. Using ReLU alone loses the kernel interpretation but still yields a valid positive feature map for linear attention, often with decent performance.

Thus, for each head, we have:

\[
\mathbf{Q}^\phi = \phi(\mathbf{Q}), \qquad
\mathbf{K}^\phi = \phi(\mathbf{K}),
\]

both of shape \((B, H, L, R)\).

### 2.4 Linear Attention Computation

Linear attention avoids the full \(L \times L\) matrix by computing the key‑value memory:

\[
\mathbf{KV} = \sum_{j=1}^L \mathbf{K}^\phi_{:,j} \otimes \mathbf{V}_{:,j} \;\in \mathbb{R}^{B \times H \times R \times d_h},
\]
\[
\mathbf{Z} = \sum_{j=1}^L \mathbf{K}^\phi_{:,j} \;\in \mathbb{R}^{B \times H \times R}.
\]

The output for each query is then:

\[
\mathbf{C}_{h,i} = \frac{\mathbf{Q}^\phi_{h,i} \mathbf{KV}}{\mathbf{Q}^\phi_{h,i} \mathbf{Z} + \epsilon},
\]

where division is applied after the dot products. In batched form, this is implemented with `einsum`:

- \(\mathbf{KV} = \text{einsum('bhnr,bhnd->bhrd', K\_prime, V)}\)
- \(\mathbf{K\_sum} = \text{K\_prime.sum(dim=2, keepdim=True)} \;\; \text{(shape (B,H,1,R))}\)
- \(\mathbf{num} = \text{einsum('bhnr,bhrd->bhnd', Q\_prime, KV)}\)
- \(\mathbf{den} = \text{einsum('bhnr,bhr->bhn', Q\_prime, K\_sum.squeeze(2)).unsqueeze(-1)} + \epsilon\)

The context is \(\mathbf{C} = \mathbf{num} / \mathbf{den}\).

### 2.5 Padding Mask

A 2D padding mask of shape \((B, L)\) (with 1 for valid tokens) is expanded to \((B, 1, 1, L)\). After the attention output is computed, the mask is used to zero out the context at padding positions:

\[
\mathbf{C} \leftarrow \mathbf{C} \odot \mathbf{mask.unsqueeze(-1)}.
\]

This is a simple masking strategy; importantly, it does **not** exclude padding tokens from the key‑value memory \(\mathbf{KV}\) and normalisation \(\mathbf{Z}\), which could bias the normalisation. A correct implementation would zero out \(\mathbf{K}^\phi\) and \(\mathbf{V}\) for padding positions before the summation. This is a known limitation.

### 2.6 Output Projection

After masking, the context is permuted to \((B, L, H, d_h)\), concatenated to \((B, L, H d_h)\), and projected:

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

The module returns \((\mathbf{Y}, \text{None})\) or \((\mathbf{Y},)\) depending on the `output_attentions` flag.

### 2.7 Complexity

The dominant operations are:
- Projections: \(O(L D^2)\)
- Feature map: \(O(L H d_h R) = O(L D R)\)
- KV accumulation: \(O(L H R d_h) = O(L D R)\)
- Query‑memory product: \(O(L H R d_h) = O(L D R)\)

Overall time is \(O(L D (D + R))\), linear in \(L\). Memory for the state \(\mathbf{KV}\) is \(O(H R d_h) = O(D R)\), independent of \(L\).

---

## 3. Implementation Details

- **Class `PerformerAttention`** inherits from `BaseAttention`, registered as `"performer"`.
- Constructor parameters:
  - `rank`: feature dimension \(R\) (default 256).
  - `redraw_projection`: whether to regenerate the random projection each forward (default True – atypical).
  - `pretrained_attn`: not used; projections are randomly initialised.
- **Projection generation**: `_get_projection_matrix` creates per‑head orthogonal matrices via QR decomposition. The buffer `self.projection` stores them; if `redraw_projection` is True, it is overwritten each call.
- **Feature map**: `_feature_map` computes `torch.einsum('bhnd,hdr->bhnr', x, proj) * scale`, then applies `torch.relu`.
- **Linear attention**: implemented as described.
- **Masking**: Only zeroes output at padding positions; does not mask before accumulation, which can be a source of error.

---

## 4. Usage and Configuration

Activated by setting attention type to `"performer"`. Configurable parameters via `comp_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 256 | Feature dimension \(R\). |
| `redraw_projection` | True | Whether to redraw random features each forward. |
| `dropout` | from config | Dropout on attention weights (applied? actually dropout is defined but never used in forward; it is a mistake). |

Example injection:
```python
comp_config.attention_type = "performer"
comp_config.rank = 128
```

No zero‑shot weight transfer is included; an injection function (not shown) would be needed to copy pretrained weights.

---

## 5. Discussion

This simplified Performer implementation demonstrates the core linear attention paradigm but deviates significantly from the original FAVOR+ method. The use of ReLU as the feature map discards the exponential kernel approximation and is unlikely to mimic softmax attention well, leading to potential performance degradation on tasks that heavily rely on sharp attention distributions. The redrawing of random projections at each forward adds unnecessary noise and computational overhead; a fixed projection is preferable. Moreover, the omission of proper padding masking in the key‑value accumulation can bias the normalisation, especially in batches with many padding tokens. The dropout layer is defined but never applied to attention weights, further reducing the model’s regularisation.

Despite these weaknesses, the module still provides a linear‑complexity attention mechanism and can serve as a simple baseline for experimenting with random feature maps and orthogonal projections. A production‑ready Performer would include the exponential map, the squared norm correction term, fixed projections, and full mask handling.

---

## 6. Conclusion

We have described a simplified Performer attention module that uses orthogonal random projections and a ReLU positive feature map to achieve linear complexity in sequence length. While it captures the spirit of the original FAVOR+ algorithm, several implementation choices—most notably the ReLU feature map and the lack of mask handling in the KV accumulation—limit its fidelity to the full softmax attention. Nonetheless, the module is a useful pedagogical tool and a starting point for more accurate kernel‑based attention methods within the DistilBERT compression framework. Future work should restore the exponential feature map and proper masking to align with the original Performer.