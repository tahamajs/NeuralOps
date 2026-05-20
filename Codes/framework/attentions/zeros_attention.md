# ZeroSAttention: Zero‑Mean ReLU Linear Attention

**Abstract**  
We present **ZeroSAttention**, a lightweight linear attention module that replaces the standard softmax self‑attention with a simple, efficient alternative based on the ReLU feature map and a zero‑mean key normalisation. The module projects the input to queries, keys, and values in the usual multi‑head manner. Before computing the attention, the keys are centered by subtracting their mean along the sequence dimension, which helps to stabilise the linear attention and reduce variance. The queries and centered keys are then passed through a ReLU activation (with a small positive offset) to ensure non‑negativity. The attention output is obtained via the associative form of linear attention: a key‑value memory matrix of size \(d \times d\) is accumulated, and the output is normalised by the dot product of the query and the sum of the key features. The entire operation has complexity \(O(N d^2)\) per head, which is linear in sequence length \(N\) and independent of the feature dimension squared. ZeroSAttention is implemented as a drop‑in replacement for the self‑attention layer in DistilBERT and is registered within the compression framework under the key `"zeros"`. This paper provides a rigorous mathematical description of the zero‑mean centering, the ReLU‑based linear attention, and the normalisation strategy, along with implementation details and a discussion of its potential benefits and limitations.

---

## 1. Introduction

Linear attention mechanisms (Katharopoulos et al., 2020) avoid the quadratic complexity of softmax attention by substituting the exponential kernel with a simpler feature map \(\phi(\mathbf{q})^\top \phi(\mathbf{k})\). The resulting computation can be rearranged to yield \(O(N d^2)\) time and memory, making it attractive for long sequences. A common choice for \(\phi\) is \(\operatorname{ELU}(x)+1\), which is positive and preserves gradient flow. However, the quality of the approximation and the training stability can vary. The **ZeroSAttention** module explores a different combination: it uses **ReLU** as the feature map (with a small constant added to avoid division by zero) and applies **zero‑mean centering** to the keys before the activation. This centering reduces the mean of the key vectors, which can stabilise the linear attention by preventing any one direction from dominating the key‑value memory. The module also uses a standard normalisation denominator that ensures the output is a weighted average of the values.

The design is intentionally minimal, with no additional learnable parameters beyond the standard projections. The result is a fast, simple, and differentiable attention alternative that can be easily plugged into existing Transformer architectures. The module is designed as a drop‑in replacement for DistilBERT’s self‑attention layer and is integrated into the compression framework via the `ATTENTION_REGISTRY`.

In this paper, we detail the mathematical formulation of ZeroSAttention, including the zero‑mean operation, the ReLU feature map, the linear attention recurrence, and the normalisation. We also discuss the implementation, the computational complexity, and the potential trade‑offs compared to full softmax attention.

---

## 2. Method

### 2.1 Multi‑Head Projections

Let \(\mathbf{X} \in \mathbb{R}^{B \times N \times D}\) be the input hidden states, where \(B\) is the batch size, \(N\) the sequence length, and \(D\) the model dimension. The module applies three learned linear projections to obtain multi‑head queries, keys, and values:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D}\) (since the module uses a full‑size projection, splitting into heads afterwards). The tensors are reshaped to separate \(H\) heads, each of dimension \(d_h = D/H\):

\[
\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times N \times d_h}.
\]

### 2.2 Zero‑Mean Key Centering

Before applying the feature map, the keys are centered by subtracting their mean over the sequence length:

\[
\tilde{\mathbf{K}} = \mathbf{K} - \frac{1}{N} \sum_{j=1}^{N} \mathbf{K}_{:, :, j, :},
\]

where the mean is computed per head and per batch element, and broadcast back over the sequence dimension. This operation ensures that the sum of the key vectors along the sequence has zero mean, which can improve the conditioning of the key‑value memory and prevent drift in the state.

### 2.3 ReLU Feature Map

To obtain a non‑negative feature representation, the ReLU activation is applied to both queries and the centered keys, with a small epsilon added to avoid strictly zero vectors (which would cause division‑by‑zero in the normalisation):

\[
\phi_Q(\mathbf{q}) = \operatorname{ReLU}(\mathbf{q}) + \varepsilon, \qquad
\phi_K(\tilde{\mathbf{k}}) = \operatorname{ReLU}(\tilde{\mathbf{k}}) + \varepsilon,
\]

where \(\varepsilon = 10^{-8}\). The ReLU ensures that the feature vectors lie in the non‑negative orthant, making the subsequent linear attention well‑defined.

### 2.4 Linear Attention via Key‑Value Memory

Standard softmax attention computes \(\mathbf{C} = \operatorname{softmax}(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}) \mathbf{V}\). Linear attention replaces the softmax with the product of feature maps:

\[
\mathbf{C}_i \approx \frac{\phi(\mathbf{q}_i)^\top \sum_{j} \phi(\tilde{\mathbf{k}}_j) \mathbf{v}_j^\top}{\phi(\mathbf{q}_i)^\top \sum_{j} \phi(\tilde{\mathbf{k}}_j)}.
\]

In ZeroSAttention, the numerator and denominator are computed via the associative property. First, the key‑value memory matrix \(\mathbf{KV} \in \mathbb{R}^{B \times H \times d_h \times d_h}\) is formed:

\[
\mathbf{KV} = \sum_{j=1}^{N} \phi_K(\tilde{\mathbf{k}}_j) \otimes \mathbf{v}_j.
\]

This is implemented as a batch operation: `torch.einsum('bhnk,bhnd->bhkd', k_nonneg, v)` where the index \(k\) runs over the feature dimension and \(d\) over the value dimension. The resulting tensor is a per‑head matrix of size \(d_h \times d_h\). Next, the key sum \(\mathbf{Z} \in \mathbb{R}^{B \times H \times 1 \times d_h}\) is computed:

\[
\mathbf{Z} = \sum_{j=1}^{N} \phi_K(\tilde{\mathbf{k}}_j).
\]

Then, for each query position, the output is obtained by:

\[
\mathbf{C} = \frac{\mathbf{Q} \,\mathbf{KV}}{\mathbf{Q} \,\mathbf{Z}^\top + \varepsilon}.
\]

In practice, \(\mathbf{Q} \,\mathbf{KV}\) is a batched matrix‑matrix multiplication giving shape \((B, H, N, d_h)\), and the denominator is computed as \(\mathbf{Q} \mathbf{Z}^\top\) (where \(\mathbf{Z}^\top\) has shape \((B, H, d_h, 1)\)) resulting in shape \((B, H, N, 1)\). The division is broadcast over the feature dimension.

### 2.5 Output Projection

After the context \(\mathbf{C}\) is obtained, the heads are concatenated back to shape \((B, N, D)\) and passed through a final output linear layer \(\mathbf{W}_O\):

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

The module returns a tuple \((\mathbf{Y}, \text{None})\) for compatibility.

### 2.6 Complexity

The memory state \(\mathbf{KV}\) has size \(O(H \cdot d_h^2) = O(D \cdot d_h)\), which is independent of sequence length. The per‑head operations cost \(O(N d_h^2)\), leading to total time \(O(N D d_h)\). For \(D = 768\), \(d_h = 64\), this is approximately 49k operations per token, which is linear in \(N\) and much cheaper than the \(O(N^2 d_h)\) of full attention for long sequences. The zero‑mean centering and ReLU add negligible overhead.

---

## 3. Implementation Details

- **Class `ZeroSAttention`** inherits from `BaseAttention` and is registered as `"zeros"` in the `ATTENTION_REGISTRY`.
- **Constructor**:
  - `config`: provides `hidden_size` and `num_attention_heads`.
  - `rank`: ignored (kept for interface compatibility).
  - Creates `W_q`, `W_k`, `W_v`, `out` as `nn.Linear` layers with bias (default).
- **Forward**:
  1. Project inputs to Q, K, V, reshape to multi‑head `(B, H, N, d_h)`.
  2. Center keys: `k_centered = k - k.mean(dim=2, keepdim=True)`.
  3. Apply ReLU with epsilon: `q_nonneg = relu(q) + 1e-8`, `k_nonneg = relu(k_centered) + 1e-8`.
  4. Compute `kv = einsum('bhnk,bhnd->bhkd', k_nonneg, v)` (note: indices here treat `k` as feature dim, producing `(B, H, d_h, d_h)`).
  5. Compute `k_sum = k_nonneg.sum(dim=2, keepdim=True)` → `(B, H, 1, d_h)`.
  6. `num = torch.matmul(q_nonneg, kv)` → `(B, H, N, d_h)`.
  7. `den = torch.matmul(q_nonneg, k_sum.transpose(-2,-1)) + 1e-8` → `(B, H, N, 1)`.
  8. `out = num / den`.
  9. Reshape, output projection.
  10. Return `(out, None)`.

- **Mask handling**: The forward method accepts an `attention_mask` argument but does **not** use it. Padding tokens are not masked from the key‑value memory or the normalisation, which can be problematic for variable‑length sequences. A correct implementation should zero out the keys and values of padding positions before aggregation, or at least mask them in the sum. This is a known limitation.

- **No dropout on attention weights**: Unlike standard attention, there is no explicit attention weight matrix; dropout could be applied to the values or context, but it is not included here.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"zeros"` in the compression configuration. No additional hyperparameters are required beyond the model config. Example injection via framework:

```python
comp_config.attention_type = "zeros"
```

Manual instantiation:
```python
attn = ZeroSAttention(config=model_config)
layer.attention = attn
```

The linear layers are randomly initialised; for zero‑shot performance, an injection function should copy the pretrained Q, K, V, O weights (e.g., from `old_attn`). The module can be fine‑tuned to adapt the linear projections and the output layer.

---

## 5. Discussion

ZeroSAttention embodies a minimalistic approach to linear attention. By using ReLU and zero‑mean centering, it avoids the complexity of exponential maps or random features while still achieving linear complexity. The zero‑mean operation helps to distribute the key magnitudes more evenly, potentially preventing the memory from being dominated by a few large keys. The ReLU + epsilon ensures positivity without sacrificing simplicity.

However, the method has several drawbacks compared to full softmax attention or more sophisticated kernel approximations:

- **Lack of exponential sharpening**: Softmax attention provides a highly peaked distribution, which can be crucial for focusing on specific tokens. ReLU‑based linear attention tends to produce broader attention patterns, which may reduce precision in tasks requiring sharp focus.
- **No masking**: The absence of mask handling means padded sequences will contaminate the memory and normalisation, degrading performance. Adding mask support is essential for practical use.
- **Unbounded memory growth**: The key‑value memory \(\mathbf{KV}\) grows with the sum of key‑value outer products, which can become large for long sequences. The normalisation denominator compensates, but the numerator may still produce large outputs. The ReLU + epsilon limits the minimal feature value to \(\varepsilon\), but large values can still accumulate.
- **Potential for rank collapse**: The \(d_h \times d_h\) memory matrix may become low‑rank if the keys lie in a low‑dimensional subspace, leading to loss of information. Centering may help by diversifying the key directions.

Despite these issues, ZeroSAttention is a useful baseline for linear attention. It is simple to implement, computationally efficient, and can serve as a starting point for further enhancements (e.g., adding a gating mechanism, using ELU instead of ReLU, or incorporating relative position encodings).

In the context of the compression framework, it provides an alternative that drastically reduces memory and compute for long sequences while preserving a familiar projection structure.

---

## 6. Conclusion

We have presented ZeroSAttention, a zero‑mean ReLU linear attention module. The method centers the keys, applies ReLU, and performs linear attention via a \(d \times d\) key‑value memory with proper normalisation. It achieves linear time and constant memory in sequence length, making it suitable for efficient sequence processing. The module is integrated into the DistilBERT compression framework as a drop‑in replacement for standard self‑attention. Future work should incorporate proper padding mask support and explore the combination of the zero‑mean trick with other feature maps to further improve accuracy and stability. ZeroSAttention demonstrates that even the simplest non‑linearity, when paired with a careful normalisation, can yield a functional efficient attention mechanism.