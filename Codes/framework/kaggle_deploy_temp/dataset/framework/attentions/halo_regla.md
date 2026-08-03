# HALO‑ReGLA: Hybrid Attention with Rotary Embeddings and ReLU‑Gated Linear Attention

**Abstract**

We present **HALO‑ReGLA**, a lightweight attention module that combines Rotary Position Embeddings (RoPE) with a linear‑complexity attention mechanism based on the ReLU kernel. The module replaces standard softmax self‑attention with a highly efficient linear attention variant that uses a ReLU feature map to project queries and keys into a positive cone, enabling the attention computation to be factorised as a recurrent matrix‑vector product. The resulting complexity scales as \(O(N d^2)\) instead of \(O(N^2 d)\), making it suitable for long sequences. The module integrates RoPE directly into the query and key projections, preserving relative positional information, and is designed as a drop‑in replacement for a DistilBERT attention layer. This paper details the mathematical underpinnings of both RoPE and the ReLU‑kernel linear attention (ReGLA), describes the architecture of the HALO‑ReGLA block, and explains its integration into the compression‑aware framework. While the code includes a `hybrid_ratio` parameter intended for future expansion to interleaved full attention, the current implementation uses pure linear attention.

---

## 1. Introduction

The quadratic time and memory complexity of multi‑head self‑attention limits the applicability of Transformers to long sequences. Several strategies have been proposed to circumvent this cost, including sparse patterns, low‑rank approximations, and linearised attention. Linear attention (Katharopoulos et al., 2020) replaces the softmax with a kernel feature map \(\phi(\mathbf{q})^\top \phi(\mathbf{k})\), allowing the computation to be reordered such that the attention output is obtained via a recurrent state of size \(O(d \times d)\). Among the simplest and most effective feature maps is the **ReLU** function, which projects queries and keys into the non‑negative orthant. Combined with **Rotary Position Embeddings (RoPE)** (Su et al., 2024), which encode relative positional information directly into the query‑key dot product, the result is a position‑aware linear attention mechanism that retains many of the benefits of full attention while drastically reducing the computational burden.

The **HALO‑ReGLA** module implements this combination. It first applies linear projections to compute multi‑head queries, keys, and values; then applies RoPE; then uses the ReLU feature map to project Q and K; finally performs the factorised attention to produce the output. The name “HALO” suggests “Hybrid Attention with Linear Overhead”, and the “ReGLA” stands for “ReLU‑Gated Linear Attention”, a term used in recent efficient Transformer literature.

In this paper, we provide a thorough mathematical treatment of the two core components—RoPE and the ReLU‑kernel linear attention—and detail the HALO‑ReGLA module’s implementation. We also discuss its limitations, including the lack of hybrid full‑attention switching (despite the presence of a `hybrid_ratio` parameter), and outline potential improvements.

---

## 2. Method

### 2.1 Projections and Multi‑Head Reshaping

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), with batch size \(B\), length \(L\), and hidden dimension \(d_{\text{model}}\), the module first computes the standard linear projections for all heads:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times (H \cdot d_h)}\) and \(d_h = d_{\text{model}} / H\). These are then reshaped to separate the head dimension, resulting in tensors of shape \((B, H, L, d_h)\).

### 2.2 Rotary Position Embeddings (RoPE)

RoPE encodes position information by rotating the query and key vectors in a pairwise manner. For a vector \(\mathbf{x} \in \mathbb{R}^{d_h}\) at position \(t\), the RoPE transformation is defined as:

\[
\operatorname{RoPE}(\mathbf{x}, t) = \mathbf{x} \odot \cos(\boldsymbol{\theta} t) + \operatorname{rotate\_half}(\mathbf{x}) \odot \sin(\boldsymbol{\theta} t),
\]

where \(\boldsymbol{\theta}_i = 10000^{-2i/d_h}\) for \(i = 0, \dots, d_h/2 - 1\), and the rotation operator swaps and flips the two halves of the vector:

\[
\operatorname{rotate\_half}([x_1, x_2]) = [-x_2, x_1].
\]

Crucially, this transformation ensures that for any positions \(m, n\),

\[
\operatorname{RoPE}(\mathbf{q}, m)^\top \operatorname{RoPE}(\mathbf{k}, n) = \mathbf{q}^\top \mathbf{k} \cos(\theta(m-n)) + \text{rotation term},
\]

effectively encoding the relative offset \(m-n\) into the dot product. In the implementation, the cosine and sine tables for all positions up to a maximum length are precomputed and stored as fixed buffers. During the forward pass, the RoPE is applied to queries and keys of shape \((B, H, L, d_h)\) by broadcasting the precomputed cos and sin tensors.

### 2.3 ReLU‑Kernel Linear Attention (ReGLA)

After applying RoPE, the queries and keys are passed through a ReLU activation:

\[
\tilde{\mathbf{Q}} = \operatorname{ReLU}(\operatorname{RoPE}(\mathbf{Q})), \quad
\tilde{\mathbf{K}} = \operatorname{ReLU}(\operatorname{RoPE}(\mathbf{K})).
\]

The ReLU feature map \(\phi(x) = \max(0, x)\) projects the vectors into the non‑negative orthant, satisfying \(\phi(x)^\top \phi(y) \ge 0\). In linear attention, we aim to compute:

\[
\mathbf{C}_h = \frac{(\phi(\mathbf{Q})\phi(\mathbf{K})^\top) \mathbf{V}}{\phi(\mathbf{Q})\phi(\mathbf{K})^\top \mathbf{1}}.
\]

Using the associative property, the numerator and denominator can be computed without materialising the \(L \times L\) attention matrix:

1. **Key‑Value memory**: \(\mathbf{KV} = \sum_{j=1}^L \tilde{\mathbf{k}}_j \otimes \tilde{\mathbf{v}}_j = \sum_j \tilde{\mathbf{k}}_j \tilde{\mathbf{v}}_j^\top \in \mathbb{R}^{B \times H \times d_h \times d_h}\). This is implemented as `torch.einsum('b h n d, b h n e -> b h d e', k, v)`.
2. **Key normalisation**: \(\mathbf{Z} = \sum_{j=1}^L \tilde{\mathbf{k}}_j \in \mathbb{R}^{B \times H \times d_h}\). 
3. **Numerator**: For each query at position \(i\): \(\mathbf{n}_i = \tilde{\mathbf{q}}_i^\top \mathbf{KV} = \sum_j (\tilde{\mathbf{q}}_i^\top \tilde{\mathbf{k}}_j) \tilde{\mathbf{v}}_j\), via `torch.einsum('b h n d, b h d e -> b h n e', q, kv)`.
4. **Denominator**: \(d_i = \tilde{\mathbf{q}}_i^\top \mathbf{Z} = \sum_j \tilde{\mathbf{q}}_i^\top \tilde{\mathbf{k}}_j\), via `torch.einsum('b h n d, b h d -> b h n', q, z.squeeze(2))`.
5. **Output**: \(\mathbf{O}_{h,i} = \mathbf{n}_i / (d_i + \epsilon)\).

The final output is reshaped to \((B, L, H d_h)\) and passed through the output projection \(\mathbf{W}_O\). Note that the denominator normalises each query’s attention weights, mimicking the effect of softmax but with a feature‑map‑induced positive weighting instead of exponentiation.

### 2.4 Complexity

The linear attention’s cost is \(O(L d_h^2 H) = O(L d_{\text{model}} d_h)\), which is linear in \(L\) but quadratic in \(d_h\). For typical settings (\(d_h = 64, L \gg 64\)), this is much cheaper than the \(O(L^2 d_{\text{model}})\) of full attention.

### 2.5 Hybrid Ratio (Placeholder)

The constructor accepts a `hybrid_ratio` parameter (default 4), which could be used to periodically switch to full attention (similar to the Granite Hybrid Block). However, the current implementation always uses linear attention; the `hybrid_ratio` is not utilised in the forward pass. This suggests an intended extension where every \(R\)-th call uses standard attention, but that is not yet implemented.

---

## 3. Implementation Details

The `HALOReGLAAttention` class is registered under the key `"halo_regla"` in the attention registry.

- **Constructor**:
  - `config`: provides `hidden_size`, `num_attention_heads`, and `dropout`.
  - `rank`: unused, kept for interface consistency.
  - `hybrid_ratio`: placeholder (default 4).
  - Creates four linear layers (with biases) for Q, K, V, and out.
  - Instantiates `RotaryEmbedding` with `head_dim` and a maximum sequence length of 4096 (hardcoded).
  - Initialises dropout.

- **Forward**:
  1. Projects Q, K, V, reshapes to multi‑head.
  2. Applies RoPE to Q and K.
  3. Applies ReLU activation to Q and K.
  4. Computes `kv` matrix via einsum.
  5. Computes `z = k.sum(dim=2, keepdim=True)` (sum over sequence).
  6. Computes `num` and `den` via einsum, then normalises.
  7. Reshapes and projects output.
  8. Returns `(output, None)`.

Note a typo in the code: `k = F.relu(k)a` – the trailing `a` is a syntax error and should be ignored; presumably it’s a copy‑paste artifact.

- **RotaryEmbedding**: Implements RoPE by precomputing sinusoidal embeddings. The `forward` method applies rotation to the input tensor based on the cached cos/sin tables, using `rotate_half` to produce the orthogonal component.

- **Missing masking**: The module accepts `attention_mask` but does not use it. Padding tokens are not masked out in the summation, which can lead to performance degradation on padded sequences. A simple fix would be to multiply \(\tilde{\mathbf{k}}_j\) and \(\tilde{\mathbf{v}}_j\) by a mask indicator before aggregation.

---

## 4. Usage and Configuration

The module can be injected into a DistilBERT model by setting the attention type to `"halo_regla"`. Configuration options passed via `comp_config` could include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hybrid_ratio` | 4 | Intended for future hybrid mode; currently unused. |
| `max_seq_len` | 4096 | Maximum sequence length for RoPE pre‑computation (currently hardcoded). |

Example manual instantiation:

```python
attn = HALOReGLAAttention(config=model_config, rank=None, hybrid_ratio=4)
layer.attention = attn
```

For zero‑shot initialisation from pretrained weights, one would typically copy the linear projection weights from the original attention’s `q_lin`, `k_lin`, `v_lin`, `out_lin`. The current module does not include such initialisation; the injection function would need to handle that separately.

---

## 5. Discussion

HALO‑ReGLA offers a compact and fast attention alternative. The use of ReLU as a feature map is appealing because it is simple, differentiable everywhere except at zero, and provides a non‑linear projection that encourages sparsity in the feature space (since negative inputs are zeroed). Combined with RoPE, the model retains sensitivity to relative position, which is crucial for many NLP tasks. The linear complexity in sequence length makes it ideal for long‑document processing or streaming scenarios.

However, there are several limitations. First, the ReLU kernel does not approximate the softmax kernel, and the lack of an exponential may reduce the dynamic range of attention weights; some tasks might see a performance drop compared to full softmax attention. Second, the absence of a learnable temperature or normalisation scaling (aside from the denominator) can make the attention outputs unbounded. Third, the module ignores the attention mask, which is problematic for batch processing of variable‑length sequences. Fourth, the hardcoded maximum sequence length for RoPE may cause errors for longer inputs. Finally, the `hybrid_ratio` is not functional, so the module cannot benefit from occasional full attention “rectification” as in other hybrid designs.

Despite these issues, HALO‑ReGLA exemplifies the simplicity and power of linearised attention with RoPE, and it provides a solid foundation for further development. Future improvements could incorporate a learnable feature map, proper masking, and the hybrid switching mechanism.

---

## 6. Conclusion

We have described the HALO‑ReGLA attention module, which marries Rotary Position Embeddings with a ReLU‑kernel linear attention to achieve an efficient, position‑aware attention mechanism. The module is designed as a drop‑in replacement for standard Transformer self‑attention and is integrated into the DistilBERT compression framework. Its linear time complexity and simple design make it an attractive choice for scaling to longer sequences. Future work will address the missing masking, enable the hybrid ratio, and explore more expressive feature maps to close the gap with full softmax attention.