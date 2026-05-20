# Nyströmformer Attention: Linear‑Time Attention via Nyström Approximation

**Paper Reference**  
Nyströmformer: A Nyström‑Based Algorithm for Approximating Self‑Attention  
Xiong et al., 2021 (arXiv:2102.03902)

---

## 1. Introduction

The quadratic time and memory complexity of self‑attention with respect to sequence length \(L\) limits the application of Transformers to long sequences. The Nyströmformer (Xiong et al., 2021) introduces a principled low‑rank approximation of the softmax attention matrix using the Nyström method for approximating kernel matrices. By selecting a small set of \(m\) landmark tokens, the full \(L \times L\) attention matrix is approximated by leveraging exact similarities between the landmarks and the full set of queries and keys. This reduces the complexity from \(O(L^2)\) to \(O(L m)\), where \(m\) is a constant, typically 64 or 128.

The module presented here implements the Nyströmformer attention mechanism. It selects landmarks via either segmented mean pooling (if the sequence length is divisible by \(m\)) or by uniform sampling along the length dimension. It then computes three sub‑matrices: (i) the similarity between all queries and landmark keys, (ii) the similarity among landmarks, and (iii) the similarity between landmark queries and all keys. The Moore‑Penrose pseudoinverse of the landmark similarity matrix is computed iteratively, and the full approximate attention is assembled as the product of the three sub‑matrices with the pseudoinverse in the middle. An optional residual connection adds the value matrix directly to the output.

The module is integrated as a drop‑in replacement for the standard self‑attention layer in DistilBERT, with support for zero‑shot weight transfer from pretrained weights (though the code does not include the copying step). This paper provides a rigorous mathematical description of the Nyström approximation, the landmark selection strategies, the iterative pseudoinverse, and the attention computation.

---

## 2. Method

### 2.1 Background: Nyström Approximation of the Softmax Kernel

Standard self‑attention computes the matrix \(\mathbf{S} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_h})\). The Nyströmformer approximates the attention matrix before softmax, i.e., the matrix \(\mathbf{A} = \exp(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_h})\) (ignoring the softmax normalisation temporarily). In practice, the method approximates the attention scores directly and then applies softmax. The Nyström method for a symmetric positive semidefinite matrix \(\mathbf{S}\) uses a subset of landmarks to form a low‑rank approximation:

\[
\mathbf{S} \approx \mathbf{S}_Q \mathbf{S}_C^+ \mathbf{S}_K,
\]

where \(\mathbf{S}_Q\) is the sub‑matrix containing similarities between all queries and landmark keys, \(\mathbf{S}_K\) contains similarities between landmark queries and all keys, and \(\mathbf{S}_C\) is the square matrix of similarities among landmarks. The pseudoinverse \(\mathbf{S}_C^+\) allows handling the case where the landmark matrix may be singular.

### 2.2 Landmark Selection

Two strategies are provided:

1. **Segmented mean pooling** (when \(L \bmod m = 0\)): The sequence is partitioned into \(m\) equal segments, and the queries and keys within each segment are averaged to form a single landmark vector. This yields a landmark set that summarises each segment.

2. **Uniform sampling** (when \(L \bmod m \neq 0\)): Landmarks are chosen by taking \(m\) uniformly spaced indices along the sequence dimension. This provides a simple selection without averaging.

Let \(\mathbf{Q}_{\text{land}} \in \mathbb{R}^{B \times H \times m \times d_h}\) and \(\mathbf{K}_{\text{land}} \in \mathbb{R}^{B \times H \times m \times d_h}\) denote the landmark queries and keys.

### 2.3 Sub‑Matrix Computation

Using the full queries \(\mathbf{Q}\) and keys \(\mathbf{K}\) and the landmark sets, we compute three scaled similarity matrices:

\[
\mathbf{F}_1 = \frac{\mathbf{Q} \mathbf{K}_{\text{land}}^\top}{\sqrt{d_h}} \quad (L \times m),
\]
\[
\mathbf{F}_2 = \frac{\mathbf{Q}_{\text{land}} \mathbf{K}_{\text{land}}^\top}{\sqrt{d_h}} \quad (m \times m),
\]
\[
\mathbf{F}_3 = \frac{\mathbf{Q}_{\text{land}} \mathbf{K}^\top}{\sqrt{d_h}} \quad (m \times L).
\]

These are the core building blocks. The original Nyströmformer paper also applies softmax to \(\mathbf{F}_2\) to make it a proper stochastic matrix, but this implementation does not; it directly computes the pseudoinverse. This is a simplification that still yields a reasonable approximation.

### 2.4 Moore‑Penrose Pseudoinverse of \(\mathbf{F}_2\)

Because \(\mathbf{F}_2\) is an \(m \times m\) matrix that may be ill‑conditioned, its pseudoinverse is computed using an iterative Newton‑Schulz method. The algorithm initialises \(\mathbf{X} = \mathbf{F}_2^\top / \|\mathbf{F}_2\|\) and then iterates:

\[
\mathbf{X} \leftarrow \mathbf{X} (2\mathbf{I} - \mathbf{F}_2 \mathbf{X}).
\]

This converges to the Moore‑Penrose pseudoinverse under suitable conditions (typically for a positive definite matrix; here \(\mathbf{F}_2\) is not guaranteed to be positive definite, but in practice the method works). The number of iterations is set by `pinv_iterations` (default 6).

### 2.5 Approximate Attention Matrix

The approximate attention matrix is assembled as:

\[
\tilde{\mathbf{S}} = \mathbf{F}_1 \, \mathbf{F}_2^+ \, \mathbf{F}_3 \quad \in \mathbb{R}^{B \times H \times L \times L}.
\]

Because \(\mathbf{F}_2^+\) has size \(m \times m\) and the other matrices are \(L \times m\) and \(m \times L\), the product yields an \(L \times L\) matrix without materialising an intermediate \(L \times L\) tensor. The computation is performed in the order \(\mathbf{F}_1 (\mathbf{F}_2^+ \mathbf{F}_3)\), keeping the memory footprint \(O(L m)\).

### 2.6 Softmax and Aggregation

The approximate scores \(\tilde{\mathbf{S}}\) are then passed through softmax to obtain attention probabilities:

\[
\mathbf{A} = \operatorname{softmax}(\tilde{\mathbf{S}} + \mathbf{M}_{\text{pad}}),
\]

where the padding mask \(\mathbf{M}_{\text{pad}}\) is applied as an additive bias (if provided). Dropout is applied to \(\mathbf{A}\).

The context is computed as:

\[
\mathbf{C} = \mathbf{A} \mathbf{V} \quad \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

If `residual=True`, the original value matrix \(\mathbf{V}\) is added to the context before the output projection:

\[
\mathbf{C} \leftarrow \mathbf{C} + \mathbf{V}.
\]

This residual connection acts as a kind of gating, preserving the original values alongside the attended context.

### 2.7 Output Projection

After concatenating heads and reshaping to \((B, L, D)\), the output is projected:

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O.
\]

The module returns a tuple; if `output_attentions` is true, it returns the full approximate attention weights (which can be huge), otherwise only the output.

---

## 3. Implementation Details

- **`NystromAttention`** inherits from `BaseAttention` and is registered as `"nystrom"`.
- Constructor parameters:
  - `config`: model configuration.
  - `rank`: number of landmarks \(m\) (default 64).
  - `pinv_iterations`: iterations for Newton‑Schulz pseudoinverse (default 6).
  - `residual`: whether to add \(\mathbf{V}\) to the output context (default True).
  - `pretrained_attn`: not used in the constructor (it only receives `**kwargs` that includes it but ignores it). The linear layers are randomly initialised; an injection function is expected to copy weights from the original attention module, but the code does not include that.
- **`_reshape`**: reshapes to multi‑head layout.
- **Landmark selection**: as described.
- **Pseudoinverse**: `_compute_moore_penrose` function runs the iterative algorithm.
- **Forward**:
  1. Project Q, K, V.
  2. Select landmarks (Q_land, K_land).
  3. Compute F1, F2, F3.
  4. Compute F2_pinv.
  5. Assemble approximate scores: `attn = F1 @ (F2_pinv @ F3)`.
  6. Apply padding mask if provided (the mask is added directly, assumed to be an additive tensor with 0 for valid and -∞ for invalid, shape broadcastable to `(B, 1, L)`? The code adds `attention_mask.unsqueeze(1)`, expecting a 2D mask of shape `(B, L)` with 1/0; but then it adds it to `attn` of shape `(B, H, L, L)` after unsqueeze, which gives `(B, 1, L)` broadcast to `(B, H, L, L)`. That works if mask values are 0 for valid and a large negative for padding. However, the mask is typically a float where 1=valid, 0=padding; the code adds it directly, which would add 1 to valid positions and 0 to padding – not correct. The intended additive mask should be `(1.0 - mask) * -1e9`. This is a bug.)
  7. Softmax, dropout.
  8. Multiply by V.
  9. If `residual`, add V.
  10. Reshape, output projection, return.

- **Mask bug**: The padding mask handling is flawed; it adds the mask directly to the scores, which will not produce the desired masking unless the mask is already in additive form. This should be corrected.

- **Complexity**: Computing F1, F3 costs \(O(L m d_h)\) each, F2 costs \(O(m^2 d_h)\), pseudoinverse costs \(O(m^3)\) iterations, and the final assembly costs \(O(L m^2)\). Overall complexity is \(O(L m d_h + L m^2 + m^3)\). For \(m \ll L\), this is linear in \(L\).

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"nystrom"`. Configuration parameters can be passed via `comp_config` or constructor:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 64 | Number of landmarks \(m\). |
| `pinv_iterations` | 6 | Iterations for pseudoinverse. |
| `residual` | True | Whether to add a residual value connection. |
| `dropout` | (from config) | Dropout rate on attention weights. |

Example injection (requires an injection function to copy pretrained weights, not shown):
```python
comp_config.attention_type = "nystrom"
comp_config.rank = 128
```

Manual instantiation:
```python
attn = NystromAttention(config=model_config, rank=128, residual=True)
layer.attention = attn
```

---

## 5. Discussion

The Nyströmformer is a mathematically elegant method that reduces the complexity of attention to linear time with respect to sequence length. By leveraging a small set of landmarks, it captures the global structure of the attention matrix while avoiding materialising the full \(L \times L\) matrix. The iterative pseudoinverse computation is efficient and stable for small \(m\). The optional residual addition of \(\mathbf{V}\) can help retain the original token information, a trick that sometimes improves performance.

However, this implementation has several issues:

- **Mask handling**: The mask addition is incorrect for standard binary masks. It should be converted to additive form.
- **No softmax on \(\mathbf{F}_2\)**: The original Nyströmformer applies softmax to \(\mathbf{F}_2\) to make it a proper stochastic matrix. Omitting this may cause the approximation to be less accurate.
- **Landmark selection when \(L \bmod m \neq 0\)** uses `torch.linspace` which can cause duplicate indices if the number of landmarks is close to \(L\), but that is unlikely.
- **No weight transfer**: The module does not include a mechanism to initialise its projections from a pretrained attention module. For zero‑shot performance, an injection function must be used to copy the weights.
- **The residual addition of \(\mathbf{V}\)** is not part of the original Nyströmformer and may interfere with the normal attention flow; it should be used with caution.
- **Attention matrix return**: When `output_attentions` is True, the full \(L \times L\) approximate attention matrix is returned, which can be extremely memory‑consuming.

Despite these limitations, the module provides a functional implementation of the Nyström attention idea and can serve as a baseline for further experimentation.

---

## 6. Conclusion

We have presented the Nyströmformer attention module, which approximates full self‑attention using the Nyström method with a configurable number of landmarks. The design achieves linear time complexity by substituting the quadratic attention matrix with a product of three smaller matrices. The implementation includes landmark selection via segment averaging or uniform sampling, iterative pseudoinverse computation, and an optional residual connection. Future work should address the mask handling, incorporate the softmax on the landmark matrix, and provide a robust weight initialisation procedure. The module is a valuable addition to the library of efficient attention mechanisms within the DistilBERT compression framework.