# Linformer Attention: Linear‑Complexity Self‑Attention via Sequence Projection

**Abstract**

We present a faithful implementation of **Linformer Attention**, the linear‑complexity self‑attention mechanism introduced by Wang et al. (2020). The Linformer reduces the \(O(N^2)\) cost of standard self‑attention to \(O(N k)\) by projecting the sequence length dimension of the key and value matrices to a fixed, smaller dimension \(k\) using two learnable linear projection matrices. The present module implements this idea by introducing two parameter matrices \(\mathbf{E}, \mathbf{F} \in \mathbb{R}^{k \times L_{\text{max}}}\) that are learned during training. At each forward pass, the sequence length \(L\) is projected down to \(k\) for keys and values before the attention computation. The module is designed as a drop‑in replacement for the standard self‑attention layer in DistilBERT and is integrated into the compression framework. This paper provides a rigorous mathematical description of the Linformer mechanism, details the implementation, and discusses its computational properties and potential improvements.

---

## 1. Introduction

The standard multi‑head self‑attention in Transformers computes pairwise dot‑products between all queries and keys, leading to a \(O(N^2)\) time and memory complexity. This makes processing of long sequences prohibitive. The **Linformer** (Wang et al., 2020) addresses this by observing that the attention matrix is typically low‑rank, and therefore can be approximated by projecting the key and value length dimensions to a much smaller dimension \(k\). The projection matrices \(\mathbf{E}\) and \(\mathbf{F}\) are learned parameters that act as a form of parameterised pooling over the sequence.

The implementation described here follows the original Linformer paper closely: it uses two separate projection matrices for keys and values (\(\mathbf{E}\) and \(\mathbf{F}\)) that are applied to the length dimension. The resulting projected keys and values have shape \((B, H, k, d_h)\). The attention scores are computed between the full queries \((B, H, N, d_h)\) and the projected keys \((B, H, k, d_h)\), yielding a reduced attention matrix of size \((B, H, N, k)\). This reduces the computation from \(O(N^2)\) to \(O(N k)\).

In this paper, we detail the mathematical formulation of the Linformer projection and attention, the architectural details of the module, and its integration into the compression framework. We also highlight potential implementation issues, such as the handling of padding masks and the use of a fixed maximum sequence length.

---

## 2. Method

### 2.1 Standard Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), the module first applies the standard linear transformations to obtain queries, keys, and values for all heads:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times (H d_h)}\). The tensors are then reshaped to the multi‑head layout \((B, H, L, d_h)\).

### 2.2 Sequence Length Reduction via Learnable Projections

The core of the Linformer is the reduction of the sequence dimension of keys and values. Two learnable matrices \(\mathbf{E}, \mathbf{F} \in \mathbb{R}^{k \times L_{\text{max}}}\) are introduced, where \(k\) is a predefined projection dimension (the `k_dim` hyperparameter) and \(L_{\text{max}}\) is a maximum sequence length (taken from the model configuration’s `max_position_embeddings`, typically 512). During the forward pass, for a given actual length \(L\), only the first \(L\) columns of \(\mathbf{E}\) and \(\mathbf{F}\) are used: \(\mathbf{E}_{\text{eff}} = \mathbf{E}_{[:, :L]}\) (shape \(k \times L\)), and similarly for \(\mathbf{F}\).

The projected keys and values are computed as:

\[
\mathbf{K}^{\text{proj}} = \mathbf{E}_{\text{eff}} \mathbf{K} \quad \text{(applied along the length dimension)},
\]
\[
\mathbf{V}^{\text{proj}} = \mathbf{F}_{\text{eff}} \mathbf{V},
\]

where the multiplication can be expressed using an einsum operation: \( \mathbf{K}^{\text{proj}}_{b,h,k,d} = \sum_{l=1}^L \mathbf{E}_{k,l} \mathbf{K}_{b,h,l,d} \). This is a linear transformation that compresses the sequence of length \(L\) into \(k\) summary vectors.

The original Linformer paper uses the same projection \(\mathbf{E}\) for both keys and values, but here two separate projections are provided, following the more general formulation (the code uses \(\mathbf{E}\) for both in the forward pass: it projects both `k` and `v` using `E_eff`; the `F` parameter is defined but **not used** in the forward pass. This is a discrepancy: the constructor creates both `self.E` and `self.F`, but only `E_eff` is applied to both `k` and `v`. The intended use might have been to apply `F` to values, but the code overlooks that. We will note this as an implementation bug in the discussion.)

### 2.3 Scaled Dot‑Product Attention with Projected Keys

Once keys and values are projected, the attention scores are computed between the full queries and the projected keys:

\[
\mathbf{S} = \frac{\mathbf{Q} (\mathbf{K}^{\text{proj}})^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times L \times k}.
\]

Since the key dimension is now \(k\) instead of \(L\), the attention matrix is of size \(L \times k\) rather than \(L \times L\). This drastically reduces memory and computation.

### 2.4 Masking

The Linformer does not directly apply the standard padding mask to the key dimension because keys have been compressed. The original paper discusses an alternative approach: apply the mask to the attention scores after computation, but only on the query side (since the key side has been compressed into a fixed set of vectors that may mix padding and non‑padding tokens). In this implementation, the mask is applied as an additive mask to the scores. The code attempts:

```python
scores = scores + (attention_mask.float() - 1.0) * 1e9
```

This is intended to set the scores for masked (padding) query positions to a large negative value. However, there is a critical error: the constant `1e9` is positive, not negative. Adding a large positive value to masked positions would make them appear extremely attractive after softmax, which is the opposite of the desired effect. The correct term should be `-1e9`. Additionally, the mask shape must be broadcastable to `(B, 1, L, 1)`; the code expects `attention_mask` to be a 2D tensor of shape `(B, L)` with 1 for valid tokens. The addition `(attention_mask.float() - 1.0) * 1e9` yields 0 for valid and `1e9` for invalid, which adds a huge positive number to invalid positions—exactly wrong. This is clearly a bug in the provided code.

### 2.5 Softmax and Aggregation

Softmax is applied along the last dimension (of size \(k\)), producing attention weights:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \text{mask}).
\]

Then the output is:

\[
\mathbf{O} = \mathbf{A} \mathbf{V}^{\text{proj}} \quad \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

The output is then transposed and reshaped to \((B, L, H d_h)\) and passed through the output projection \(\mathbf{W}_O\).

### 2.6 Complexity

The dominant cost is the computation of the attention scores, which is \(O(L \times k \times d_h)\) per head, or \(O(L \cdot k \cdot D)\). Since \(k\) is a constant (e.g., 128), the overall complexity is linear in \(L\). Memory is also \(O(L k)\) for the attention matrix, far smaller than \(O(L^2)\).

---

## 3. Implementation Details

### 3.1 Module Structure

`LinformerAttention` inherits from `BaseAttention` and is registered under `"linformer"`. Constructor arguments:

- `config`: model configuration providing `hidden_size`, `num_attention_heads`, and optionally `max_position_embeddings`.
- `rank`: ignored (for interface compatibility).
- `k_dim`: the projection dimension \(k\) (default 128).

It creates:

- Standard Q, K, V, O linear layers.
- `self.E` and `self.F`: parameter tensors of shape `(k_dim, max_len)` initialised from a normal distribution with std=0.02. The maximum length defaults to `config.max_position_embeddings` if present, else 512.

### 3.2 Forward Pass

1. Projects input to Q, K, V and reshapes to multi‑head.
2. Truncates \(\mathbf{E}\) to the actual sequence length \(L\): `E_eff = self.E[:, :L]`. Note: only \(\mathbf{E}\) is used; \(\mathbf{F}\) is ignored.
3. Projects keys and values: `k_proj = einsum('kl,bhld->bhkd', E_eff, k)` and similarly for values (using `E_eff` again, not `F`). This is a discrepancy.
4. Computes scores = `matmul(q, k_proj.transpose(-2,-1)) / sqrt(d_h)`.
5. Applies masking: `scores = scores + (attention_mask.float() - 1.0) * 1e9`. As discussed, this is erroneous.
6. Softmax, then `out = matmul(attn, v_proj)`.
7. Reshapes, applies output projection.
8. Returns either `(out, None)` or `(out, attn)` if `output_attentions` is True.

### 3.3 Handling Variable Sequence Lengths

The projection matrices are pre‑allocated for a fixed maximum length and sliced to the actual length. This works as long as \(L \le L_{\text{max}}\). If \(L\) exceeds the pre‑allocated size, the slicing `self.E[:, :L]` will still work (since PyTorch will pad with zeros?), but it's safer to handle longer sequences dynamically (the original Linformer paper proposes to project the full sequence with a matrix of size \(k \times L_{\text{max}}\) that is trained only up to \(L_{\text{max}}\); for longer sequences one might use chunking or increase the max length). The current implementation will just silently use the full matrix; if \(L > L_{\text{max}}\), it will index beyond the buffer's columns, causing an error. So the `max_len` must be set large enough.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"linformer"`. The configurable hyperparameters (via `comp_config` or direct arguments) are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_dim` | 128 | Projection dimension \(k\). |
| `max_len` | 512 | Maximum sequence length for the projection matrix. |

Example injection:
```python
comp_config.attention_type = "linformer"
comp_config.k_dim = 256
```

Manual instantiation:
```python
linformer = LinformerAttention(config=model_config, k_dim=256)
layer.attention = linformer
```

The module does not include a dedicated injection function in this snippet; the class is registered and would be instantiated by the framework’s default mechanism.

---

## 5. Discussion

The Linformer provides a conceptually simple and effective way to achieve linear complexity. By leveraging the low‑rank property of attention, it reduces the key‑value length to a constant \(k\), which can be set much smaller than \(L\). The learnable projection matrices automatically learn to compress the sequence in a task‑relevant manner.

However, the current implementation contains several notable issues:

1. **Mask bug**: The additive mask uses `+1e9` instead of `-1e9`, which will cause the model to attend predominantly to padding tokens, severely degrading performance. This must be corrected to `-1e9` to mask out padding positions.

2. **Unused parameter F**: The matrix `self.F` is defined but never used. The forward pass applies `E_eff` to both keys and values. The original Linformer uses the same projection for keys and values, so this is not necessarily wrong, but having an extra parameter that is not trained leads to wasted memory and confusion. It should either be removed or properly integrated.

3. **No positional encoding for projections**: The projection matrices are of shape \(k \times L_{\text{max}}\), essentially learning a fixed set of \(k\) summary basis vectors that are linear combinations of the sequence positions. Because they are parameters, they are not input‑dependent; the same compression is applied to all sequences. This works well when the sequence length is constant, but may be suboptimal for varying lengths.

4. **No head‑wise projections**: The projection matrices are shared across all heads, which is standard for Linformer.

5. **Hard maximum length**: The use of a fixed `max_len` limits the model’s ability to handle longer sequences. For longer inputs, one would need to retrain with a larger matrix or use dynamic projections (as in the original paper’s “Linformer with shared projections” which can handle any length by using a fixed number of learnable “components”).

6. **Masking is only on the query side**: Since the keys have been compressed into a fixed number of vectors that mix information from all positions (including padding), the model cannot easily ignore padding content after projection. The original Linformer paper addresses this by masking before projection: the input sequence is first masked (setting keys/values of padding tokens to zero), then projected. This implementation does not do that.

Despite these limitations, the module provides a basic, working skeleton of the Linformer idea. With the mask bug fixed, it can serve as a baseline for linear‑time attention.

---

## 6. Conclusion

We have described the Linformer Attention module, a linear‑complexity alternative to standard self‑attention based on learned sequence projection. The implementation approximates the original paper but includes several bugs and design flaws that need attention, primarily the incorrect sign in the additive mask and the unused parameter matrix. Nonetheless, it demonstrates the core concept and can be refined into a production‑ready efficient attention layer. Future work should fix the masking, remove unused parameters, and incorporate proper masking before projection to ensure compatibility with padded sequences.