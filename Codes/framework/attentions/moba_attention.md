# MoBA Attention: Block‑Sparse Attention via Top‑k Block Selection

**Abstract**

We present **MoBA Attention (Mixture of Block Attention)**, a block‑sparse self‑attention module that reduces the quadratic complexity of standard attention by partitioning the sequence into fixed‑size blocks and restricting each query to attend only to the top‑\(k\) blocks that are predicted to be most relevant. Block importance is estimated by averaging the query and key representations within each block to form block representatives, and then scoring them via dot products. The query block then attends to all tokens within the selected key‑value blocks. This yields a complexity of \(O(L \cdot k B)\) where \(B\) is the block size and \(k\) is the number of selected blocks, significantly lower than \(O(L^2)\) when \(k B \ll L\). The module is implemented as a drop‑in replacement for standard self‑attention in Transformers, with configurable block size and top‑\(k\). This paper provides a complete mathematical description of the block gating, the gathering mechanism, and the per‑block attention, along with implementation details and discussion of its current limitations, especially regarding padding and masking.

---

## 1. Introduction

Standard multi‑head self‑attention requires computing a full \(L \times L\) attention matrix, which is prohibitive for long sequences. Sparse attention mechanisms reduce this cost by allowing each token to attend to only a subset of tokens. **MoBA (Mixture of Block Attention)** is a block‑based sparse attention approach that partitions the sequence into equal‑sized blocks and selects the most important blocks for each query block based on a simple average‑pooling similarity score. Once the top‑\(k\) key blocks are identified, full attention is computed between the query tokens in that block and the tokens from the selected key blocks, resulting in a block‑sparse pattern.

The MoBA module described here offers a straightforward implementation of this idea, with padding to ensure the sequence length is a multiple of the block size, and a Python loop over query blocks. The method is conceptually similar to the block‑sparse attention used in models like BigBird and Longformer, but with a dynamic selection of blocks rather than a fixed pattern.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given input hidden states \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module applies linear projections to obtain queries, keys, and values for all heads:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\). The tensors are reshaped to \((B, H, L, d_h)\) and transposed to the head‑first layout \((B, H, L, d_h)\).

### 2.2 Block Partitioning and Padding

Let \(B_s\) be the `block_size` (default 64). The sequence length \(L\) may not be a multiple of \(B_s\). To facilitate block operations, the module pads the sequence length to the next multiple of \(B_s\) by appending zeros to \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) along the length dimension. Let the padded length be \(L_p = \lceil L / B_s \rceil \times B_s\). The number of blocks is \(n_b = L_p / B_s\).

### 2.3 Block Gating via Mean Pooling

For each head, the padded queries and keys are reshaped into blocks:

\[
\mathbf{Q}_b \in \mathbb{R}^{B \times H \times n_b \times B_s \times d_h}, \quad
\mathbf{K}_b \in \mathbb{R}^{B \times H \times n_b \times B_s \times d_h}.
\]

We then compute a representative for each block by averaging over the block’s tokens:

\[
\bar{\mathbf{Q}}_{b,i} = \frac{1}{B_s} \sum_{j=1}^{B_s} \mathbf{Q}_{b,i,j}, \qquad
\bar{\mathbf{K}}_{b,i} = \frac{1}{B_s} \sum_{j=1}^{B_s} \mathbf{K}_{b,i,j},
\]

yielding tensors of shape \((B, H, n_b, d_h)\). The block‑level attention scores (or gate scores) are computed as the scaled dot product between these representatives:

\[
\mathbf{S}^{\text{block}} = \frac{\bar{\mathbf{Q}} \bar{\mathbf{K}}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times n_b \times n_b}.
\]

### 2.4 Top‑k Block Selection

For each query block \(i\), we select the \(k\) key blocks with the highest scores, where \(k = \min(\text{top\_k}, n_b)\):

\[
\mathcal{T}_i = \operatorname{TopK}(\mathbf{S}^{\text{block}}_{:,:,i,:}, k).
\]

The indices \(\mathcal{T}_i\) have shape \((B, H, k)\).

### 2.5 Block‑Sparse Attention

For each query block \(i\), we gather the corresponding key and value blocks from \(\mathbf{K}_b\) and \(\mathbf{V}_b\) using the indices \(\mathcal{T}_i\). The gathered tensors have shape \((B, H, k, B_s, d_h)\), and we flatten the selected blocks into a single key set of size \(k B_s\):

\[
\mathbf{K}_{\text{sel},i} = \text{flatten}( \mathbf{K}_b[:, :, \mathcal{T}_i, :, :] ), \qquad
\mathbf{V}_{\text{sel},i} = \text{flatten}( \mathbf{V}_b[:, :, \mathcal{T}_i, :, :] ),
\]

where flattening concatenates the block dimension with the token dimension, resulting in shapes \((B, H, k B_s, d_h)\). The query tokens from block \(i\) (of shape \((B, H, B_s, d_h)\)) then compute attention scores with this combined key set:

\[
\mathbf{S}_i = \frac{\mathbf{Q}_{b,i} \mathbf{K}_{\text{sel},i}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times B_s \times (k B_s)}.
\]

Softmax is applied to obtain attention weights \(\mathbf{A}_i\), dropout is applied, and the output for block \(i\) is:

\[
\mathbf{O}_{b,i} = \mathbf{A}_i \mathbf{V}_{\text{sel},i} \in \mathbb{R}^{B \times H \times B_s \times d_h}.
\]

### 2.6 Reassembling and Output

The per‑block outputs are placed into an output tensor of shape \((B, H, L_p, d_h)\). The padded portion (beyond \(L\)) is then discarded, and the tensor is reshaped to \((B, L, D)\) and projected with \(\mathbf{W}_O\).

### 2.7 Complexity

The block‑level scoring costs \(O(B H n_b^2 d_h) = O(B H (L/B_s)^2 d_h) = O(B D L^2 / B_s^2)\), which is quadratic in the number of blocks but typically much smaller than \(L^2\) for large \(B_s\). The attention within each query block costs \(O(B H B_s \cdot k B_s \cdot d_h) = O(B D L k B_s / B_s?) Actually the total attention cost is \(O(B H n_b \cdot B_s \cdot k B_s \cdot d_h) = O(B D L \cdot k B_s)\). For \(B_s = 64\), \(k=4\), this is \(O(256 B D L)\), a significant reduction over \(O(B D L^2)\). The overall complexity is sub‑quadratic, dominated by the block gating and the sparse attention.

---

## 3. Implementation Details

- **`MoBAAttention`** inherits from `nn.Module` and is registered as `"moba"`.
- Constructor:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `block_size`: size of each block (default 64).
  - `top_k`: number of blocks selected per query block (default 4).
- Layers: standard Q, K, V, O linear projections and dropout.
- **Forward**:
  1. Project and reshape to multi‑head.
  2. Pad to multiple of `block_size`.
  3. Reshape to blocks: \((B, H, n_b, B_s, d)\).
  4. Compute block scores using mean‑pooled Q and K.
  5. Get top‑\(k\) indices per query block.
  6. Loop over blocks (Python `for` loop):
     - Gather the selected K and V blocks using `torch.gather`.
     - Compute attention scores between the query block and gathered keys.
     - Softmax, dropout, weighted sum.
     - Store the result in the output tensor.
  7. Remove padding, reshape, output projection.
  8. Return `(out, None)`.
- **Mask handling**: The code does **not** use `attention_mask`. Padded tokens (the extra zeros added to reach a multiple of \(B_s\)) participate in block averaging and attention, which can skew the block representatives and the attention weights. A proper implementation should mask out these padding tokens during averaging and attention, or avoid padding by using variable‑length block schemes.
- **Inefficiency**: The loop over blocks is implemented in Python, which is slow for large \(n_b\); a batched gather with a scatter operation would be more efficient.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"moba"` in the compression configuration. The relevant hyperparameters are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 64 | Size of each block. |
| `top_k` | 4 | Number of blocks selected per query block. |
| `dropout` | (from config) | Dropout rate on attention weights. |

Example injection:
```python
comp_config.attention_type = "moba"
comp_config.block_size = 128
comp_config.top_k = 8
```

Manual instantiation:
```python
attn = MoBAAttention(config=model_config, block_size=128, top_k=8)
layer.attention = attn
```

No zero‑shot weight transfer is performed; the linear layers are randomly initialised. To preserve pretrained knowledge, the injection function should copy the original projection weights, which is not implemented in this snippet.

---

## 5. Discussion

MoBA provides a simple and effective block‑sparse attention pattern. By averaging within blocks and selecting the most similar blocks, it approximates the full attention matrix while drastically reducing computation. The top‑\(k\) mechanism is dynamic and can adapt to each query block, unlike fixed patterns. The block size and top‑\(k\) parameters offer a direct trade‑off between speed and accuracy.

However, several limitations are present. The padding to multiples of \(B_s\) introduces dummy tokens that are not masked, potentially distorting the attention. The block averaging is naive—outliers could skew the representative. The loop over blocks is not optimised for GPU execution. Also, the module does not support causal masks; it is purely bidirectional. For autoregressive tasks, a causal variant would need to modify block selection to only past blocks.

Despite these, the module demonstrates the core idea of block‑sparse attention with learned (or score‑based) block selection, and can serve as a baseline for more sophisticated sparse attention methods.

---

## 6. Conclusion

We have described MoBA Attention, a block‑sparse self‑attention module that partitions the sequence into blocks and restricts each query to attend to the top‑\(k\) key blocks identified via block‑average similarity. The method achieves sub‑quadratic complexity and is implemented as a drop‑in replacement for standard attention in DistilBERT. Future work will address padding masking, more efficient block gathering, and extension to causal scenarios. MoBA provides a valuable building block for efficient Transformers.