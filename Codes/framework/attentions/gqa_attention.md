# Grouped Query Attention: Efficient Multi‑Head Attention with Shared Key‑Value Heads

**Abstract**

We present a faithful implementation of **Grouped Query Attention (GQA)**, an efficient variant of multi‑head self‑attention introduced by Ainslie et al. (2023). In GQA, the standard set of query heads is partitioned into groups, and each group shares a single key‑value head. This reduces the memory and computational cost of the key‑value projections by a factor equal to the group size, with minimal impact on quality. The module described in this paper provides a drop‑in replacement for the full self‑attention layer in Transformer models. It accepts standard projection weights for queries, a reduced number of key‑value heads, and performs the necessary expansion via repetition before computing the attention scores and output. The implementation supports optional padding masks and returns a tuple compatible with the surrounding framework. We detail the mathematical formulation, the implementation specifics, and the integration into the compression‑aware registry used for DistilBERT.

---

## 1. Introduction

Multi‑head attention (MHA) is a core component of Transformer models. In the standard formulation, each of the \(H\) heads independently computes an attention function using its own query, key, and value projections. While powerful, this requires storing and processing \(H\) key and \(H\) value tensors, which can dominate the memory footprint during inference, especially for large models. **Grouped Query Attention (GQA)** mitigates this by allowing multiple query heads to share the same key and value heads. The original MHA is the extreme case where each query head has its own key‑value head (\(H\) groups of size 1), while Multi‑Query Attention (MQA) is the opposite extreme where all heads share a single key‑value head (\(1\) group of size \(H\)). GQA interpolates between these, offering a tunable trade‑off between speed and accuracy.

In this paper, we describe a concise, self‑contained implementation of GQA that can be injected into any Transformer model. The module reduces the number of key and value projections from \(H\) to \(K\) (where \(K \le H\)), stores the smaller projection matrices, and expands the computed keys and values to match the full number of heads via repetition. The attention computation then proceeds as in standard multi‑head attention. This design allows existing pretrained models to be adapted to GQA with minimal code changes and a straightforward reduction in parameters and compute.

---

## 2. Method

### 2.1 Motivation and Parameter Sharing

Let the total number of query heads be \(H\) and the number of key‑value heads be \(K\). We assume \(H\) is divisible by \(K\); the group size is then \(G = H / K\). Every group of \(G\) query heads is assigned a single key‑value head. Concretely, for query head index \(h\), the corresponding key‑value head index is \(\lfloor h / G \rfloor\).

### 2.2 Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), the query projection remains unchanged from standard MHA:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{W}_Q \in \mathbb{R}^{d_{\text{model}} \times (H \cdot d_h)},
\]

where \(d_h = d_{\text{model}} / H\) is the head dimension. The key and value projections, however, output only \(K \cdot d_h\) features:

\[
\mathbf{K}_{\text{small}} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times (K \cdot d_h)},
\]
\[
\mathbf{V}_{\text{small}} = \mathbf{X} \mathbf{W}_V, \quad \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times (K \cdot d_h)}.
\]

The shapes of the projected tensors are \((B, L, H \cdot d_h)\) for \(\mathbf{Q}\) and \((B, L, K \cdot d_h)\) for \(\mathbf{K}\) and \(\mathbf{V}\).

### 2.3 Reshaping and Repetition

All three tensors are first reshaped to separate the head dimension:

\[
\mathbf{Q} \to (B, L, H, d_h),
\]
\[
\mathbf{K}_{\text{small}} \to (B, L, K, d_h), \quad
\mathbf{V}_{\text{small}} \to (B, L, K, d_h).
\]

To align the key and value heads with the queries, each key‑value head is replicated \(G\) times along the head dimension:

\[
\mathbf{K} = \text{repeat\_interleave}(\mathbf{K}_{\text{small}}, G, \text{dim}=2) \quad \to (B, L, H, d_h),
\]
\[
\mathbf{V} = \text{repeat\_interleave}(\mathbf{V}_{\text{small}}, G, \text{dim}=2) \quad \to (B, L, H, d_h).
\]

The repetition is exactly the tensor‑level equivalent of assigning the same key‑value head to multiple query heads.

### 2.4 Scaled Dot‑Product Attention

After transposing to the conventional head‑first layout \((B, H, L, d_h)\), the attention computation is identical to standard MHA:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}},
\]
\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}),
\]
\[
\mathbf{C} = \mathbf{A} \mathbf{V}.
\]

Here \(\mathbf{M}\) is an additive mask derived from the `attention_mask`. If a 2D mask of shape \((B, L)\) with 1 for valid tokens and 0 for padding is provided, it is expanded to \((B, 1, 1, L)\) and turned into \((1 - \text{mask}) \cdot (-\infty)\). Dropout is applied to the attention weights \(\mathbf{A}\).

### 2.5 Output Projection

The context tensor \(\mathbf{C}\) of shape \((B, H, L, d_h)\) is transposed back to \((B, L, H \cdot d_h)\) (i.e., `(B, L, -1)`) and then passed through the output projection:

\[
\mathbf{Y} = \mathbf{C} \mathbf{W}_O, \quad \mathbf{W}_O \in \mathbb{R}^{(H \cdot d_h) \times d_{\text{model}}}.
\]

The final output is \(\mathbf{Y} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), and a `None` for the attention weights is returned for interface compatibility.

---

## 3. Implementation Details

The `GroupedQueryAttention` class is implemented as a PyTorch `nn.Module` and is registered under the key `"gqa"` in the attention registry. The constructor takes two arguments:

- `config`: a configuration object providing `num_attention_heads`, `hidden_size`, and `dropout`.
- `num_kv_heads` (default 4): the number of key‑value heads \(K\).

The key implementation steps are:

1. **Linear layers**:
   - `q_proj`: input `hidden_size`, output `num_heads * head_dim`.
   - `k_proj`: input `hidden_size`, output `num_kv_heads * head_dim`.
   - `v_proj`: input `hidden_size`, output `num_kv_heads * head_dim`.
   - `out_proj`: input `num_heads * head_dim`, output `hidden_size`.
   - Biases are included (default `True` in `nn.Linear`).

2. **Forward pass**:
   - Compute Q, K, V.
   - Reshape: Q to \((B, L, H, d_h)\); K, V to \((B, L, K, d_h)\).
   - Replicate K and V using `torch.repeat_interleave(repeat, dim=2)` where `repeat = H // K`.
   - Transpose all to \((B, H, L, d_h)\).
   - Compute scores, scale, apply mask, softmax, dropout.
   - Multiply with V, transpose back, flatten to \((B, L, H d_h)\).
   - Apply `out_proj` and return `(output, None)`.

3. **Mask handling**: Supports both 2D masks (which are expanded) and already expanded 4D masks. An additive mask of `-1e9` is used.

---

## 4. Usage and Configuration

The module is injected into a Transformer layer by specifying the attention type `"gqa"` in the compression configuration. The relevant hyperparameters that can be passed through `comp_config` or directly:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_kv_heads` | 4 | Number of key‑value heads \(K\). Must divide `num_heads`. |
| `dropout` | from config | Dropout rate on attention weights. |

Example manual instantiation:

```python
attn = GroupedQueryAttention(config=model_config, num_kv_heads=4)
layer.attention = attn
```

To replicate the original MHA, set `num_kv_heads = num_heads`. For Multi‑Query Attention (MQA), set `num_kv_heads = 1`. The code automatically computes the repeat factor.

---

## 5. Discussion

Grouped Query Attention offers a simple yet effective way to compress the attention module. By reducing the key‑value heads, the memory required for the key and value caches during autoregressive decoding drops by a factor of \(H/K\). This is particularly beneficial for large language models deployed in memory‑constrained environments. The implementation described here is exact: it computes the same full attention scores as MHA, just with shared keys and values. As a result, there is no loss of quality due to approximation; the trade‑off is entirely in the capacity of the key‑value projections.

One limitation of this implementation is that the repetition of key‑value heads is performed in materialised form, which temporarily creates tensors of size \((B, L, H, d_h)\). For very large \(H\), this can still be memory‑intensive. An optimised implementation would avoid the explicit repetition by using grouped matrix operations or by restructuring the attention computation to directly use the smaller K, V tensors. However, for typical Transformer sizes and sequence lengths, the current approach is simple and efficient enough.

The module does not copy pretrained weights; in a typical usage, the injection function would initialise the projections from the original layer’s `q_lin`, `k_lin`, `v_lin`, `out_lin`, or train from scratch. For zero‑shot compression, one would need to reduce the rank of the key‑value weight matrices appropriately, which is not handled here. The module assumes that the provided `config` matches the model dimensions.

---

## 6. Conclusion

We have presented a clean implementation of Grouped Query Attention, a parameter‑ and memory‑efficient variant of multi‑head attention. The module interleaves multiple query heads with shared key‑value heads, reducing the number of key‑value projections without altering the attention computation. Its plug‑and‑play nature makes it suitable for rapid experimentation with efficient Transformer architectures. Future work could integrate this module with pretrained weight initialisation for seamless compression, and extend it to support causal masking and FlashAttention for further acceleration.