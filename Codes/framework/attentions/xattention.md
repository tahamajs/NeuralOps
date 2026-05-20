# XAttention: Block Sparse Attention with Antidiagonal Block Scoring

**Abstract**  
We present **XAttention**, an efficient sparse self‑attention module that reduces the quadratic complexity of standard attention by partitioning the attention matrix into fixed‑size blocks and retaining only the most important blocks according to an **antidiagonal scoring** criterion. The sequence is first padded to a multiple of the block size and reshaped into a 3D grid of blocks (batch, heads, number of blocks, block length, feature dimension). Each block is represented by its mean query and key vectors, and the block‑level importance matrix is computed via scaled dot‑product. For each query block, the top‑\(k\) key blocks with the highest scores are selected, where \(k\) is controlled by a sparsity parameter. Exact scaled dot‑product attention is then computed only within these selected block pairs, and the outputs for a query block are summed over the chosen key blocks. The method achieves a structured sparsity pattern that is both content‑aware (since block scores come from the actual queries and keys) and training‑free (no additional learnable parameters beyond standard projections). The module is implemented as a drop‑in replacement for the self‑attention layer in DistilBERT, integrating with the compression framework. This paper provides a rigorous mathematical description of the block scoring, selection, and sparse attention, along with implementation details and an analysis of its complexity and limitations.

---

## 1. Introduction

Standard multi‑head self‑attention scales quadratically with the sequence length \(L\), which is prohibitive for long sequences. Block‑sparse attention methods reduce this cost by dividing the attention matrix into blocks and only computing interactions between a subset of blocks. The **XAttention** module introduces a simple yet effective block selection criterion: for each query block, it computes the **mean query and key vectors** over the tokens in that block, and then uses their dot‑product as a proxy for the block’s overall relevance. The top‑\(k\) key blocks with the highest scores are retained, and attention is computed only for those block pairs. This design avoids the need for a separate learned gate (unlike SeerAttention) or fixed patterns (like sliding windows), while still achieving sub‑quadratic complexity.

The method is inspired by the concept of antidiagonal scoring, where the importance of a block pair could be measured by summing attention scores along antidiagonals of the block; here, the block‑average approach provides a simple, efficient approximation. The sparsity level is controlled by a single parameter \(\rho\) (the fraction of blocks to prune), giving direct control over the trade‑off between speed and accuracy. Because the block scores are derived from the input data, the sparsity pattern adapts to each sample.

In this paper, we detail the mathematical formulation of XAttention, including padding, block aggregation, top‑\(k\) selection, and sparse attention. We also discuss the current implementation’s loop‑based computation and the implications for real‑world use.

---

## 2. Method

### 2.1 Multi‑Head Projections and Block Partitioning

Given an input sequence of hidden states \(\mathbf{H} \in \mathbb{R}^{B \times L \times D}\), the module first applies standard linear projections to obtain multi‑head queries, keys, and values:

\[
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V,
\]

with \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\), where \(H\) is the number of heads and \(d_h = D / H\). The tensors are reshaped to \((B, H, L, d_h)\) and transposed to the head‑first layout \((B, H, L, d_h)\).

To facilitate block‑wise operations, the sequence length \(L\) is padded to the next multiple of the block size \(B_s\) (default 64) by appending zeros. Let the padded length be \(L_p = \lceil L / B_s \rceil \times B_s\), and the number of blocks be \(N_b = L_p / B_s\). The padded tensors are then reshaped into blocks:

\[
\mathbf{Q}_{\text{blk}} \in \mathbb{R}^{B \times H \times N_b \times B_s \times d_h}, \qquad
\mathbf{K}_{\text{blk}}, \mathbf{V}_{\text{blk}} \text{ similarly}.
\]

### 2.2 Block Importance Scoring via Mean Pooling

For each block, a representative vector is computed by averaging over the block’s tokens:

\[
\bar{\mathbf{Q}}_{b,i} = \frac{1}{B_s} \sum_{j=1}^{B_s} \mathbf{Q}_{\text{blk}, b, i, j}, \qquad
\bar{\mathbf{K}}_{b,i} = \frac{1}{B_s} \sum_{j=1}^{B_s} \mathbf{K}_{\text{blk}, b, i, j},
\]

yielding tensors of shape \((B, H, N_b, d_h)\). The block‑level relevance scores are then computed as the scaled dot product:

\[
\mathbf{S}^{\text{block}} = \frac{\bar{\mathbf{Q}} \bar{\mathbf{K}}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times N_b \times N_b}.
\]

This matrix measures the average similarity between tokens in query block \(i\) and key block \(j\).

### 2.3 Top‑\(k\) Block Selection

Given a sparsity parameter \(\rho\) (the fraction of blocks to prune), the number of key blocks to keep for each query block is:

\[
k = \max(1, \lfloor N_b \times (1 - \rho) \rfloor).
\]

For each query block \(i\), we select the \(k\) key blocks with the highest scores:

\[
\mathcal{T}_i = \operatorname{TopK}(\mathbf{S}^{\text{block}}_{:, :, i, :}, k).
\]

A binary mask \(\mathbf{M} \in \{0,1\}^{B \times H \times N_b \times N_b}\) is constructed where \(M_{b,h,i,j} = 1\) if block \(j\) is selected for query block \(i\) (under head \(h\) and batch element \(b\)), and 0 otherwise. The top‑\(k\) operation ensures a fixed sparsity level, giving direct control over the computational budget.

### 2.4 Block‑Sparse Attention

The final attention output is computed by iterating over all pairs of query blocks and their selected key blocks. For a given query block \(i\) and selected key block \(j\) (with \(M_{b,h,i,j}=1\)), we compute:

\[
\mathbf{S}_{i,j} = \frac{\mathbf{Q}_{\text{blk}, i} \mathbf{K}_{\text{blk}, j}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times B_s \times B_s},
\]
\[
\mathbf{A}_{i,j} = \operatorname{softmax}(\mathbf{S}_{i,j}),
\]
\[
\mathbf{O}_{i,j} = \mathbf{A}_{i,j} \mathbf{V}_{\text{blk}, j}.
\]

The outputs for query block \(i\) are accumulated over all selected key blocks \(j\):

\[
\mathbf{O}_{\text{blk}, i} = \sum_{j: M_{b,h,i,j}=1} \mathbf{O}_{i,j}.
\]

Finally, the block outputs are reshaped back to a sequence, the padding is removed, and the result is projected via \(\mathbf{W}_O\).

### 2.5 Complexity

The block‑level scoring costs \(O(B H N_b^2 d_h) = O(B H (L/B_s)^2 d_h) = O(B D L^2 / (B_s^2 H?)\) Actually, the dominant term is \(O(B H (L/B_s)^2 d_h)\). The sparse attention step costs \(O(B H N_b \cdot k \cdot B_s^2 d_h) = O(B H (L/B_s) \cdot k \cdot B_s^2 d_h) = O(B H L \cdot k \cdot B_s \cdot d_h)\). For \(B_s = 64\), \(k\) a fraction of \(N_b\), the overall complexity is roughly \(O(B D L \cdot k B_s)\). Since \(k \propto L / B_s\), the sparse attention term becomes \(O(B D L^2 \cdot (1-\rho) / B_s)\)? Wait, if \(k \propto N_b = L/B_s\), then the sparse attention cost is \(O(B H (L/B_s) \cdot (L/B_s) \cdot B_s^2 d_h) = O(B H L^2 d_h)\), which is still quadratic. This is because the number of selected blocks grows linearly with the number of blocks, so the total number of token‑level comparisons stays quadratic. To achieve sub‑quadratic scaling, one would need to make \(k\) a constant independent of \(L\) (e.g., select a fixed number of blocks regardless of sequence length). The current implementation uses \(k = \max(1, \text{int}(N_b * (1 - \rho)))\), which grows with \(L\), so the complexity remains \(O(L^2)\) but with a reduced constant factor. This is an important point to note.

---

## 3. Implementation Details

- **Class `XAttention`** inherits from `nn.Module` and is registered under `"xattention"` in the `ATTENTION_REGISTRY`.
- **Constructor parameters**:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `block_size` (default 64): size of each block \(B_s\).
  - `sparsity` (default 0.5): fraction of blocks to prune \(\rho\).
  - Standard linear layers for Q, K, V, O, and dropout.
- **Forward pass**:
  1. Project Q, K, V, reshape to multi‑head.
  2. Pad sequence to a multiple of `block_size`.
  3. Reshape to blocks: `(B, H, n_blocks, block_size, d)`.
  4. Compute block scores via mean‑pooled Q and K.
  5. Top‑\(k\) selection: keep `int(n_blocks * (1 - sparsity))` blocks per query block.
  6. Build binary mask.
  7. Loop over query blocks \(i\) and key blocks \(j\):
     - If mask is non‑zero for block pair \((i,j)\), compute attention, softmax, and accumulate output.
  8. Reshape back to sequence, remove padding, apply output projection.
  9. Return `(out, None)`.

- **Inefficiency**: The nested Python loops over \(i\) and \(j\) are slow for large \(N_b\). A batched implementation using `gather` and `einsum` would be more efficient. The current loop also recomputes attention for the same key block multiple times if it is selected by many query blocks; caching the key/value blocks could speed up computation.

- **Mask handling**: The module does **not** use the `attention_mask` argument. Padding tokens (including the extra padding to reach a multiple of \(B_s\)) are included in the block averages and attention, which may distort the block scores and final output. A proper implementation should mask out padding tokens during the averaging and attention steps.

- **Causal masking**: Not implemented; the attention is bidirectional.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"xattention"` in the compression configuration. Configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 64 | Size of each block. |
| `sparsity` | 0.5 | Fraction of blocks pruned (0 = keep all, 1 = keep only 1). |
| `dropout` | (from config) | Dropout on attention weights. |

Example injection via framework:
```python
comp_config.attention_type = "xattention"
comp_config.block_size = 128
comp_config.sparsity = 0.7
```

No weight transfer from pretrained attention is performed; the linear layers are randomly initialised. For zero‑shot use, an injection function should copy the original Q, K, V, O weights.

---

## 5. Discussion

XAttention presents a straightforward block‑sparse attention method. Its key advantage is simplicity: the block importance is computed directly from the data without any learned parameters or complex hashing. The top‑\(k\) selection provides a predictable sparsity level, which is useful for resource budgeting.

However, the method has several drawbacks:

- **Quadratic scaling with \(L\)**: As noted, the number of selected blocks grows linearly with the number of blocks, so the overall complexity remains \(O(L^2)\), just with a smaller constant. To achieve sub‑quadratic scaling, one would need to select a constant number of blocks per query block (e.g., top‑\(k\) with fixed \(k\) independent of \(L\)). This could be an easy modification.

- **Block averaging loss**: Using mean vectors for block scoring discards within‑block variation, potentially missing important token‑level interactions. A more accurate method might use the maximum or sum of attention within a block.

- **Padding issues**: The padding tokens are not masked, which can bias the block scores and attention outputs. This should be fixed for variable‑length sequences.

- **Loop inefficiency**: The Python loops over blocks are not suitable for GPU execution at scale. A vectorised implementation using scatter/gather would be necessary for production.

Despite these limitations, XAttention serves as a clear and educational implementation of block‑sparse attention with dynamic, data‑dependent sparsity. It could be extended with causal masking, adaptive \(k\), and better block scoring to become a more practical module.

---

## 6. Conclusion

We have described XAttention, a block‑sparse attention module that selects top‑\(k\) key blocks for each query block based on mean query‑key similarity. The method offers a content‑adaptive sparsity pattern with a tunable sparsity level. While its current form is limited by quadratic scaling in \(k\) and inefficiencies in the implementation, it provides a solid foundation for developing more advanced block‑sparse attention mechanisms. The module is integrated into the DistilBERT compression framework for experimentation and further development.