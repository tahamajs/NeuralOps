# SVD Linear Attention: Input‑Adaptive Low‑Rank Projection for Efficient Attention

**Abstract**

We present **SVD Linear Attention**, a novel self‑attention module that reduces the quadratic complexity of standard attention by compressing the key and value matrices using a data‑dependent low‑rank basis. For each batch element, the module performs a truncated singular value decomposition (SVD) of the input hidden states to obtain a set of orthogonal directions that capture the dominant features of the sequence. The keys and values are then projected onto these directions, drastically reducing their sequence length dimension from \(L\) to a fixed rank \(r\). The attention computation proceeds using the compressed keys and values, with complexity \(O(L r d)\) instead of \(O(L^2 d)\). When the sequence length is smaller than the rank, the module falls back to standard full attention. The method is designed as a drop‑in replacement for the self‑attention layer in DistilBERT and is registered within the compression framework. This paper provides a detailed mathematical description of the SVD compression pathway, the mask handling, and the practical considerations of the approach.

---

## 1. Introduction

Multi‑head self‑attention is a core component of Transformer models, but its \(O(L^2)\) time and memory cost with respect to sequence length \(L\) limits its application to long documents. Numerous efficient attention methods have been proposed to mitigate this, including low‑rank approximations of the attention matrix (e.g., Linformer) and token‑selection strategies (e.g., sparse attention). The **SVD Linear Attention** module takes a different route: it dynamically compresses the **key and value tensors** using a basis derived from the input hidden states via singular value decomposition. This basis is input‑adaptive, meaning that the compression is tailored to the specific sequence being processed, unlike methods that use fixed projections.

For each input sequence, the module computes a truncated SVD of the hidden states, retaining only the top \(r\) singular vectors. These vectors define a subspace that captures the majority of the energy in the input. The key and value projections are then transformed into this subspace, reducing their sequence dimension from \(L\) to \(r\). The queries are kept in the original dimension, and attention scores are computed between the full queries and the compressed keys. The result is an attention output that approximates the full attention while reducing the number of key‑value pairs from \(L\) to a small constant.

The module is designed as a plug‑in replacement for the standard self‑attention layer in DistilBERT. It integrates seamlessly with the compression framework via the registry. This paper details the mathematical underpinnings, the implementation, and the trade‑offs of using input‑adaptive SVD compression for attention.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first computes the standard multi‑head projections for queries, keys, and values:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times (H d_h)}\) where \(H\) is the number of heads and \(d_h = D/H\). The tensors are reshaped to the multi‑head layout \((B, H, L, d_h)\). A helper method `transpose_for_scores` performs this reshaping and permutation.

### 2.2 Fallback to Full Attention

If the sequence length \(L\) is less than or equal to the compression rank \(r\) (or if the rank is not set), the module simply computes standard scaled dot‑product attention:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}, \qquad
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}), \qquad
\mathbf{C} = \mathbf{A} \mathbf{V},
\]

where \(\mathbf{M}\) is an additive padding mask (0 for valid, \(-\infty\) for padding). Dropout is applied to the attention weights. This avoids unnecessary compression when the sequence is already short.

### 2.3 SVD‑Based Key–Value Compression

When \(L > r\), the module performs the following steps to compress the keys and values.

**Step 1: Input‑Adaptive Basis via SVD.** For each batch element \(b\), the hidden state \(\mathbf{X}_b \in \mathbb{R}^{L \times D}\) is used to compute a low‑rank basis. Specifically, a truncated SVD is computed using `torch.svd_lowrank`, which efficiently approximates the SVD:

\[
\mathbf{X}_b \approx \mathbf{U}_b \mathbf{\Sigma}_b \mathbf{V}_b^\top,
\]

where \(\mathbf{U}_b \in \mathbb{R}^{L \times r}\) contains the top \(r\) left singular vectors. These vectors represent the dominant directions along the sequence length dimension. The method uses 2 iterations of the power method (`niter=2`) for speed. The resulting \(\mathbf{U}_b\) is stacked across the batch to form \(\mathbf{U} \in \mathbb{R}^{B \times L \times r}\), and an extra dimension is added to obtain shape \((B, 1, L, r)\). The basis is computed on the CPU (the code moves the hidden states to CPU), which is a practical choice to avoid GPU memory issues for large sequences, though it introduces a data transfer overhead.

**Step 2: Projecting Keys and Values.** The key and value tensors \(\mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times L \times d_h}\) are projected into the subspace spanned by \(\mathbf{U}\)’s columns (along the sequence dimension). This is achieved by a batched matrix multiplication:

\[
\mathbf{K}^{\text{proj}} = \mathbf{U}^\top \mathbf{K} \quad \in \mathbb{R}^{B \times H \times r \times d_h},
\]
\[
\mathbf{V}^{\text{proj}} = \mathbf{U}^\top \mathbf{V} \quad \in \mathbb{R}^{B \times H \times r \times d_h}.
\]

Here, the transpose of \(\mathbf{U}\) (shape \((B, 1, r, L)\)) is multiplied with the keys/values, effectively computing a weighted average of the key/value tokens along the sequence dimension, with weights given by the singular vectors. This reduces the sequence length from \(L\) to \(r\). The projection is analogous to applying a learnable linear transformation, but the transformation matrix is data‑dependent and orthogonal (since \(\mathbf{U}_b\) has orthonormal columns).

**Step 3: Attention with Compressed Keys.** The attention scores are then computed between the full queries \(\mathbf{Q} \in \mathbb{R}^{B \times H \times L \times d_h}\) and the compressed keys:

\[
\mathbf{S}^{\text{comp}} = \frac{\mathbf{Q} (\mathbf{K}^{\text{proj}})^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times L \times r}.
\]

The scores now have a reduced key dimension of size \(r\), not \(L\). The softmax is applied, and the context is computed as:

\[
\mathbf{C}^{\text{comp}} = \operatorname{softmax}(\mathbf{S}^{\text{comp}} + \mathbf{M}') \mathbf{V}^{\text{proj}} \quad \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

The mask handling in this path is modified: if an attention mask is provided (of shape \((B, L)\)), it is unsqueezed to \((B, L, 1)\) and added to \(\mathbf{S}^{\text{comp}}\). This is because the key dimension is now \(r\) instead of \(L\); the mask is applied only to the query dimension? Actually, the broadcast works: mask shape \((B, 1, L, 1)\) would be ideal, but the code does `attention_mask.unsqueeze(-1)` giving \((B, L, 1)\), which adds to \((B, H, L, r)\). This effectively masks queries, not keys. This is a bug: padding tokens should be masked on the key side before the SVD projection to prevent them from influencing the basis and the compressed representation. The current implementation does not do this, which can cause leakage. We will note this.

### 2.4 Output Projection

After the attention, the context tensor is reshaped from \((B, H, L, d_h)\) to \((B, L, H d_h)\) and passed through a final output linear projection \(\mathbf{W}_O \in \mathbb{R}^{H d_h \times D}\) (note: the code does not include an explicit output projection? Actually, the class defines `self.query`, `self.key`, `self.value`, but no `self.out_proj`? Looking at the code: the constructor has `self.query`, `self.key`, `self.value` but no output projection. The forward returns `(context, None)` where `context` is the concatenated heads. That means the output is not projected back to the original dimension! This is a critical omission. The class likely expects the surrounding DistilBERT layer to apply an output projection? No, in DistilBERT, the attention module itself must output a tensor of shape \((B, L, D)\) after the output linear layer. The provided code does not include `out_proj`. This is a bug; the module would return a tensor of shape \((B, L, H d_h)\) which equals \(D\) since \(H d_h = D\), so actually the shape is correct (since \(H d_h = D\)), and the output is already the right dimension. There is no need for an extra linear layer if the value projection already aggregates heads? Wait: standard multi‑head attention concatenates the heads, producing a tensor of dimension \(H d_h = D\), and then applies an output projection. In DistilBERT, the attention module includes `out_lin`. The code here does not define `out_proj` and just returns `context.contiguous().view(batch_size, seq_len, self.all_head_size)`. So the concatenated heads are returned as the output, without a linear projection. That means the module lacks the output weight matrix. This is a deviation from the standard attention and likely a bug. The class should have an `out_proj` linear layer. We'll mention this in the discussion.

But the docstring says "SVDLinearAttention", and the surrounding framework might expect the output projection to be part of the module. Since it's missing, the model's performance may degrade.

---

## 3. Implementation Details

- **Class `SVDLinearAttention`** is registered under `"svd"` in the `ATTENTION_REGISTRY`.
- **Constructor**:
  - `config`: provides `hidden_size`, `num_attention_heads`, and optionally `dropout`.
  - `rank`: the compression rank \(r\) (default 64).
  - `dropout`: dropout rate (default 0.1).
  - Creates `query`, `key`, `value` linear layers (with biases? `nn.Linear` default bias=True, so yes). No output projection is created.
- **Forward**:
  - Computes Q, K, V via `transpose_for_scores`.
  - If `L <= r` or `rank` is None, does full attention with mask (expects additive mask).
  - Else:
    - Moves hidden states to CPU, computes `torch.svd_lowrank` per batch element, stacks U_r, moves back to device, unsqueezes.
    - Projects K and V using U_r.
    - Computes scores, applies mask (unsqueezed), softmax, dropout.
    - Multiplies with compressed V.
  - Reshapes context to \((B, L, D)\) (since \(H d_h = D\)).
  - Returns `(context, None)`.
- **Mask handling**:
  - Full path: assumes `attention_mask` is additive (0 for keep, -inf for pad) and adds it directly.
  - Compressed path: uses `attention_mask.unsqueeze(-1)` and adds to scores. This is insufficient; padding tokens should have been excluded from the SVD and from the key/value projection.
- **`get_ortho_loss`**: returns zero, for compatibility.
- **No output projection**: As noted, the concatenated heads are returned directly.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"svd"` in the compression config. The main hyperparameter is `rank`. Example:

```python
comp_config.attention_type = "svd"
comp_config.rank = 64
```

The module does not include a mechanism to copy pretrained weights; the projection matrices are randomly initialised. For zero‑shot use, an injection function would need to copy the original Q, K, V weights (and add an output projection if desired). Without that, the model must be fine‑tuned from scratch.

---

## 5. Discussion

SVD Linear Attention introduces an interesting idea: use an input‑dependent orthogonal basis to compress the sequence dimension. This is reminiscent of the Nyström method and Linformer, but with the basis computed on‑the‑fly from the input, making it adaptive. The use of `torch.svd_lowrank` is computationally efficient for small ranks, and the power method with 2 iterations provides a rough but fast approximation.

However, the current implementation has several critical flaws:

1. **Missing output projection**: The standard multi‑head attention includes an output linear layer that mixes the concatenated heads. Without it, the model cannot learn to combine information from different heads in a task‑specific way. The DistilBERT architecture expects this layer, and omitting it will severely limit the model’s capacity. This appears to be an oversight.

2. **Mask handling**: In the compressed path, padding tokens are not masked before the SVD. This means that the basis vectors \(\mathbf{U}\) will be influenced by padding positions, and the compressed keys/values will contain contributions from padding tokens. Moreover, the mask applied to the scores only masks the query dimension, not the key dimension. The correct approach would be to set padding tokens’ hidden states to zero before SVD, or to use the attention mask to exclude them from the SVD computation, and to mask the compressed keys/values during attention.

3. **Batch and head independence**: The SVD is computed per batch element but shared across all heads (the basis is the same for all heads because it's derived from the hidden states, not from individual head projections). This may limit the diversity of compression across heads.

4. **CPU‑GPU transfer**: The hidden states are moved to CPU for SVD, which can be a bottleneck. The low‑rank SVD on GPU is possible and would be faster for moderate sizes.

5. **No weight transfer**: The module does not initialise from a pretrained attention layer, so it must be trained from scratch or with an external copying step. In a compression pipeline, one would typically transfer the Q, K, V weights and add an output projection initialised from the pretrained `out_lin`.

Despite these issues, the module demonstrates a creative approach to dynamic token compression. If the output projection is added and masking is corrected, it could serve as a viable efficient attention alternative.

---

## 6. Conclusion

We have presented SVD Linear Attention, a module that compresses key‑value pairs using an input‑adaptive low‑rank SVD basis, reducing attention complexity to \(O(L r)\). While the core idea is promising, the current implementation lacks an output projection and proper padding mask handling, which must be addressed before practical deployment. The module integrates into the compression framework and can be used as a starting point for further development of data‑dependent token compression methods. Future work will add the missing output layer, correct the masking, and explore more efficient on‑GPU SVD approximations.