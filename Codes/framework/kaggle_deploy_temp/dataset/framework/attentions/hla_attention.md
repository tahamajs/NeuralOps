# HLA: Hadamard Linear Attention

**Abstract**

We present **Hadamard Linear Attention (HLA)**, an efficient attention mechanism that replaces the quadratic softmax self‑attention with a linear recurrent form whose feature map is induced by a **Hadamard transform**. By multiplying queries and keys with a fixed Hadamard matrix before the outer‑product accumulation, the method aims to improve the approximation quality of the softmax kernel while maintaining \(O(L d^2)\) time and \(O(d^2)\) memory complexity. The current implementation provides a skeleton that performs the recurrent linear attention loop, but the Hadamard projection step is not yet incorporated; thus, it serves as a baseline vanilla linear attention. The module is designed as a drop‑in replacement for DistilBERT’s self‑attention layer and is registered within the compression framework. This paper describes the intended mathematical formulation using Hadamard matrices, the current implementation, and plans for full integration.

---

## 1. Introduction

The quadratic cost of the standard Transformer attention mechanism with respect to sequence length \(L\) motivates the search for scalable alternatives. Linear attention replaces the softmax by a kernel feature map \(\phi(\mathbf{q})^\top \phi(\mathbf{k})\) and rearranges the computation to avoid the \(L \times L\) attention matrix. The quality of the approximation hinges on the choice of feature map. The **Hadamard Linear Attention (HLA)** proposes to use the **Hadamard transform** as the feature map, drawing on the property that the Hadamard matrix is an orthogonal transform that spreads information uniformly, which can act as a random projection to approximate a radial basis function kernel.

In this work, we detail the intended HLA mechanism: before the outer‑product accumulation, queries and keys are multiplied by a fixed Hadamard matrix to improve the kernel approximation. However, the provided code implements only the recurrent linear attention loop without the Hadamard transformation; the projection is left as future work. We therefore describe the full mathematical model and then discuss the current placeholder implementation.

---

## 2. Mathematical Formulation (Intended)

### 2.1 Standard Multi‑Head Projections

Given an input \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), we first compute queries, keys, and values for \(H\) heads, each of dimension \(d_h = d_{\text{model}}/H\):

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

and reshape to \((B, H, L, d_h)\).

### 2.2 Hadamard Feature Map

A Hadamard matrix \(\mathbf{H} \in \{\pm 1\}^{d_h \times d_h}\) (with columns normalised to unit length) satisfies \(\mathbf{H} \mathbf{H}^\top = \mathbf{I}\). The proposed feature map for a vector \(\mathbf{z} \in \mathbb{R}^{d_h}\) is:

\[
\phi_{\text{Had}}(\mathbf{z}) = \frac{1}{\sqrt{d_h}} \mathbf{H} \mathbf{z}.
\]

Applying this to queries and keys, the kernel estimate becomes:

\[
\phi_{\text{Had}}(\mathbf{q})^\top \phi_{\text{Had}}(\mathbf{k}) = \frac{1}{d_h} \mathbf{q}^\top \mathbf{H}^\top \mathbf{H} \mathbf{k} = \frac{1}{d_h} \mathbf{q}^\top \mathbf{k},
\]

which is simply the scaled dot product. However, if we use only a subset of rows of \(\mathbf{H}\) (a truncated Hadamard transform), we obtain a low‑dimensional projection. The intended HLA likely uses a random subset of Hadamard rows (or a Hadamard‑derived random matrix) to create a feature map analogous to random Fourier features but with structured orthogonality.

More generally, the HLA feature map could be defined as:

\[
\phi_{\text{HLA}}(\mathbf{z}) = \frac{1}{\sqrt{m}} \big[ \exp(\mathbf{H}_m \mathbf{z}) \big],
\]

where \(\mathbf{H}_m \in \mathbb{R}^{m \times d_h}\) consists of \(m\) rows of the Hadamard matrix, and the exponential is applied element‑wise. This yields a positive feature map that approximates a Gaussian kernel, with the Hadamard structure ensuring low variance (similar to the FAVOR+ orthogonal random features).

### 2.3 Linear Attention Recurrence

Using a feature map \(\phi\), the linear attention output is computed via the recurrence:

\[
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t) \mathbf{v}_t^\top, \qquad
\mathbf{o}_t = \phi(\mathbf{q}_t)^\top \mathbf{S}_t,
\]

where \(\mathbf{S}_t \in \mathbb{R}^{B \times H \times m \times d_h}\) (or with dimensions swapped) accumulates the key‑value pairs. The final output for each head is \(\mathbf{o}_t\) and the normalisation denominator is implicit (often \(\phi(\mathbf{q}_t)^\top \sum_{i=1}^t \phi(\mathbf{k}_i)\)). In the identity feature map case, the denominator is omitted, leading to an unnormalised linear attention.

### 2.4 Current Implementation: Identity Feature Map

The provided HLA module does not apply any Hadamard transform. It directly uses the identity map \(\phi(\mathbf{z}) = \mathbf{z}\), and the recurrence is:

- State \(\mathbf{H} \in \mathbb{R}^{B \times H \times d_h \times d_h}\) initialised to zero.
- For \(t = 1\) to \(L\):
  \[
  \mathbf{H} \leftarrow \mathbf{H} + \mathbf{k}_t \otimes \mathbf{v}_t,
  \]
  \[
  \mathbf{o}_t = \mathbf{H} \mathbf{q}_t.
  \]
- No denominator normalisation is applied.

This is the simplest form of causal linear attention. The name “Hadamard Linear Attention” is aspirational, indicating the intended use of Hadamard matrices; the current code serves as a placeholder that can be extended by inserting the Hadamard feature map before accumulation.

---

## 3. Implementation Details

### 3.1 Class Definition

`HLAAttention` inherits from `BaseAttention` and is registered under `"hla"`. The constructor creates standard linear projections for Q, K, V, and output, with no additional parameters.

### 3.2 Forward Pass

1. Project input \(\mathbf{x}\) to Q, K, V, and reshape to \((B, H, L, d_h)\).
2. Initialise a state tensor \(\mathbf{h}\) of zeros with shape \((B, H, d_h, d_h)\).
3. Loop over \(t\) from 0 to \(L-1\):
   - Extract query \(\mathbf{q}_t\), key \(\mathbf{k}_t\), value \(\mathbf{v}_t\).
   - Update state: \(\mathbf{h} \leftarrow \mathbf{h} + \text{einsum}(k, v)\) (outer product).
   - Compute output: \(\text{out}_t = \text{einsum}(q, h)\) (matrix‑vector product across the last two dims).
4. Stack outputs along the time dimension, reshape to \((B, L, D)\), and apply output projection.
5. Return \((\text{output}, \text{None})\).

The module ignores the `attention_mask` parameter, meaning padding tokens are not masked and will contribute to the state.

### 3.3 Complexity

Per‑step cost is \(O(H \cdot d_h^2) = O(d_{\text{model}} \cdot d_h)\). Total \(O(L \cdot d_{\text{model}} \cdot d_h) \approx O(L D^2)\). The state memory is \(O(H \cdot d_h^2) = O(d_{\text{model}} \cdot d_h)\) per batch element, independent of sequence length.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"hla"`. No additional configuration parameters are required. For zero‑shot initialisation, the injection function (not shown in this snippet) would typically copy the projection weights from a pretrained attention layer. The current code does not include a dedicated injection function; the registry entry expects a direct class instantiation.

Example manual replacement:
```python
new_attn = HLAAttention(config=model_config)
layer.attention = new_attn
```

The outer Transformer layer still applies a residual connection and LayerNorm.

---

## 5. Discussion

The current HLA module provides a baseline linear attention mechanism without any learned or fixed feature map beyond the identity. While it demonstrates the linear‑complexity recurrence, it does not yet implement the Hadamard transform that gives the method its name. The intended Hadamard‑based feature map could be inserted by modifying the forward pass to include a matrix multiplication with a precomputed Hadamard matrix before the outer‑product update, and potentially a denominator normalisation to stabilise training.

The lack of masking and the sequential Python loop are practical limitations that should be addressed in a production version. Parallelisation via a parallel scan algorithm and support for padding masks would make the module fully competitive with other efficient attention implementations.

Despite these limitations, the skeleton provides a clear template for exploring Hadamard‑based linear attention. The use of a fixed orthogonal transformation (Hadamard) before the outer product is known to reduce the variance of the kernel approximation and can approach the softmax attention quality with lower feature dimension. Future work will integrate the Hadamard transform and evaluate the speed‑accuracy trade‑off on long‑sequence benchmarks.

---

## 6. Conclusion

We have described Hadamard Linear Attention (HLA), an efficient attention method that ultimately aims to employ the Hadamard transform as a feature map to approximate the softmax kernel. The current implementation provides the recurrent linear attention backbone, ready for the addition of the Hadamard projection. Integrated into the DistilBERT compression framework, the module serves as a foundation for future experimentation with orthogonal feature maps for linear attention. Once the Hadamard transform is incorporated, HLA will offer an appealing combination of low variance, linear complexity, and compatibility with existing Transformer architectures.