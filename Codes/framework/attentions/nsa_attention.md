# NSA: Native Sparse Attention — Local Window + Global Top‑k Sparse Attention

**Abstract**

We present **Native Sparse Attention (NSA)**, a self‑attention module that combines a full local window of fixed size with a globally selected set of top‑\(k\) long‑range connections for each query. For every query position, the module includes all tokens within a predefined window (local context) and additionally the \(k\) tokens with the highest attention scores among the remaining non‑local positions. This hard sparsity pattern approximates full attention while reducing the number of attended key‑value pairs from \(O(L)\) to \(O(W + k)\), where \(W\) is the window size and \(k\) is the number of selected global tokens. The current implementation materialises the full \(L \times L\) attention score matrix to determine the top‑\(k\) indices, making it a conceptual baseline rather than an optimised production module. The module is integrated as a drop‑in replacement for the self‑attention layer in DistilBERT, supporting standard padding masks. This paper provides a rigorous mathematical description of the sparsity construction, the attention computation, and discusses the practical limitations of the full‑score materialisation.

---

## 1. Introduction

Standard multi‑head self‑attention computes all pairwise dot‑products, leading to \(O(L^2)\) time and memory complexity. Sparse attention methods aim to restrict each query to a subset of keys, reducing the number of comparisons. A common pattern is to use a **local window** to capture fine‑grained dependencies and a **global sparsity** component to capture long‑range interactions. The **Native Sparse Attention (NSA)** module implements this idea by constructing a sparsity mask that is the **union** of a fixed‑size local window and a **per‑query top‑\(k\) selection** from the complementary set. The top‑\(k\) is chosen based on the actual attention scores, making the sparsity content‑adaptive.

The NSA module described here is a naive implementation that first computes the full attention score matrix \(\mathbf{S} \in \mathbb{R}^{B \times H \times L \times L}\) and then, for each query, identifies the top‑\(k\) scores among non‑local positions. While this is not memory‑efficient, it directly demonstrates the sparsity scheme and can serve as a reference for more optimised versions (e.g., using approximate top‑\(k\) or block‑wise selection). The module supports standard QKV projections, dropout, and padding masks.

In this paper, we detail the mathematical construction of the combined mask, the attention computation, and the implementation specifics. We also discuss the complexity and point out the bottlenecks that prevent true efficiency in the current code.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{H} \in \mathbb{R}^{B \times L \times D}\), the module applies linear projections to obtain multi‑head queries, keys, and values:

\[
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V,
\]

with \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\). These are reshaped to \((B, H, L, d_h)\) and transposed to the head‑first layout.

### 2.2 Local Window Mask

A local symmetric window of size \(W\) (the `window_size` hyperparameter) is defined via a distance matrix:

\[
M^{\text{local}}_{ij} = \begin{cases}
1 & \text{if } |i - j| \le W,\\
0 & \text{otherwise}.
\end{cases}
\]

This mask is broadcast to shape \((1, 1, L, L)\) and is later combined with the global sparse mask.

### 2.3 Full Score Computation and Top‑k Global Selection

First, the full attention scores are computed without any masking:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times L \times L}.
\]

Then, for each query position \(i\) (and for each batch element and head), we construct a per‑query mask by:

1. **Zeroing out local positions** in a temporary copy of the scores \(s_i\) so that the top‑\(k\) selection only considers non‑local tokens:
   \[
   \tilde{s}_i = s_i \text{ with } \tilde{s}_{ij} = -\infty \text{ if } j \in \mathcal{W}(i),
   \]
   where \(\mathcal{W}(i) = \{j : |i-j| \le W\}\).

2. **Selecting top‑\(k\)** from the modified scores \(\tilde{s}_i\):
   \[
   \mathcal{T}_i = \operatorname{TopK}(\tilde{s}_i, k), \quad k = \min(\text{top\_k}, L).
   \]

3. **Updating a global mask** \(\mathbf{M}^{\text{topk}}\) of shape \((B, H, L, L)\): for each selected pair \((i, j)\) with \(j \in \mathcal{T}_i\), set \(\mathbf{M}^{\text{topk}}_{\cdot, i, j} = 1\).

The selection is performed by iterating over the length dimension with a Python loop (inefficient but functional). The final combined mask is the element‑wise logical OR of the local and top‑k masks:

\[
\mathbf{M} = \mathbf{M}^{\text{local}} \lor \mathbf{M}^{\text{topk}}.
\]

Any score where \(\mathbf{M}_{ij}=0\) is set to \(-\infty\), effectively removing it from the softmax.

### 2.4 Padding Mask

After constructing the sparse mask, an optional padding mask (indicating which token positions are valid) is applied. The padding mask is expected as a 2D tensor of shape \((B, L)\) with \(1\) for valid and \(0\) for padding. It is expanded to \((B, 1, 1, L)\) and converted to an additive mask:

\[
\mathbf{S} \gets \mathbf{S} + (1 - \mathbf{M}^{\text{pad}}) \cdot (-\infty).
\]

This ensures padding tokens do not influence the softmax. Note that the padding mask is applied **after** the sparse mask, so if a padding token was incorrectly selected as a global top‑\(k\) (which can happen since its score might be high due to random noise), it will be suppressed by the padding mask.

### 2.5 Softmax and Aggregation

Softmax is applied to the masked scores:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S}) \in \mathbb{R}^{B \times H \times L \times L},
\]

followed by dropout. The context is then computed as:

\[
\mathbf{C} = \mathbf{A} \mathbf{V}.
\]

After reshaping and output projection, the module returns the result.

---

## 3. Implementation Details

- **`NativeSparseAttention`** is registered under the key `"nsa"`.
- Constructor parameters:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `window_size` (default 128): the local window radius.
  - `top_k` (default 32): number of global tokens selected per query.
- **Forward**:
  1. Project Q, K, V.
  2. Compute full scores \(S = QK^T/\sqrt{d}\).
  3. Build local mask.
  4. Initialize `topk_mask` as zeros.
  5. Loop over `i` in `range(L)`:
     - Extract `s_i` (scores for query position `i`).
     - Mask out local positions by setting them to `-inf`.
     - Call `torch.topk` to get indices of top‑\(k\) highest remaining scores.
     - Scatter those indices into `topk_mask` at position `i`.
  6. Combine masks: `final_mask = topk_mask + local_mask.float()` (since masks are 0/1, addition works as OR).
  7. Mask scores where `final_mask == 0` to `-inf`.
  8. Apply attention mask (padding) similarly.
  9. Softmax, dropout, weighted sum, output.

- **Mask combination**: The code uses addition of binary masks to form the union, which is valid because both are 0/1 tensors. However, after masking the scores, the `topk_mask` may have included some local positions (if they were among the top‑\(k\) globally), but those are already covered by the local mask, so the union is correct.

- **Inefficiency**: The full scores tensor `(B, H, L, L)` is materialised, and the loop over `L` computes top‑\(k\) sequentially. For long sequences, this is extremely slow and memory‑intensive. A proper implementation would use approximate nearest‑neighbour search or block‑wise processing to avoid materialising the entire matrix.

- **No gradient flow through mask creation**: The top‑\(k\) operation is non‑differentiable, but since the indices are used only to mask scores, gradients still flow through the unmasked positions. This is a straight‑through estimator.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"nsa"` in the compression configuration. Configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 128 | Width of the local window (distance threshold). |
| `top_k` | 32 | Number of additional global tokens per query. |
| `dropout` | (from config) | Dropout rate on attention weights. |

Example injection:
```python
comp_config.attention_type = "nsa"
comp_config.window_size = 256
comp_config.top_k = 64
```

Manual instantiation:
```python
attn = NativeSparseAttention(config=model_config, window_size=256, top_k=64)
layer.attention = attn
```

The linear projections are randomly initialised; for zero‑shot transfer, the injection function should copy the original `q_lin`, `k_lin`, etc., but this is not included in the snippet.

---

## 5. Discussion

NSA provides a clear and interpretable sparsity pattern: it always includes a dense local window to capture immediate context, and adds a few highly relevant long‑range connections per query. This pattern resembles the sparse attention used in models like Longformer (local + global) but with dynamic selection of the global tokens based on content. The method is highly intuitive and can serve as a strong baseline for sparse attention.

However, the current implementation suffers from critical performance flaws:

- **Full score materialisation**: Computing all \(L^2\) scores contradicts the goal of sparsity. A genuine sparse attention implementation would compute the local window scores without producing the full matrix, and would use efficient top‑\(k\) algorithms (e.g., product quantization, locality‑sensitive hashing) to find global tokens without all‑to‑all comparisons.
- **Python loop over \(L\)**: Even with a full matrix, the per‑query top‑\(k\) selection could be vectorised using batched top‑\(k\) operations along the last dimension without a loop.
- **No causal mask support**: The local mask is symmetric, making the attention bidirectional. For autoregressive tasks, a causal mask would be needed.
- **Padding mask application order**: The padding mask is applied after the sparse mask, which might cause the model to lose the sparsity benefits for padding‑affected rows (no issue in practice).
- **No weight transfer**: The module lacks zero‑shot initialisation from pretrained weights; thus, fine‑tuning from a pretrained model is necessary to recover performance.

Despite these limitations, the conceptual clarity of NSA is valuable. With an optimised kernel (e.g., using `scatter` to build the mask more efficiently), it could become a practical option. The module can serve as a reference implementation for researchers experimenting with sparse attention patterns.

---

## 6. Conclusion

We have described Native Sparse Attention (NSA), a self‑attention module that restricts each query to attend to its local window plus a dynamically selected top‑\(k\) global tokens. The sparsity mask is formed by taking the union of a fixed window and the top‑\(k\) attention scores outside the window. While the current implementation materialises the full attention matrix and loops over positions, making it impractical for large sequences, it effectively illustrates the core idea. NSA can be integrated into DistilBERT via the compression framework as a baseline for sparse attention experiments. Future work will focus on efficient computation of the top‑\(k\) selection without full score materialisation, enabling true linear complexity.