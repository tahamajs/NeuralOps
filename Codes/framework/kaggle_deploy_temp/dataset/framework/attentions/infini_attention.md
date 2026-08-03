# Infini‑Attention: Simplified Compressive Memory with Sliding Window

**Abstract**

We present a simplified implementation of **Infini‑Attention**, an efficient attention module that combines a local sliding window with a **compressive memory** to extend the receptive field beyond the window without quadratic cost. The module maintains a learnable memory matrix of shape \((H, C, d_h)\) that represents a compressed summary of the global context. For each query, the module attends to both the local window tokens (via standard dot‑product attention) and the compressive memory (via a separate softmax). The results are summed and projected. In the full Infini‑Attention (Munkhdalai et al., 2024), the memory is dynamically updated with new key‑value pairs; here, for simplicity, the memory is a fixed, trainable parameter. The module serves as a drop‑in replacement for self‑attention in Transformer models and is integrated into the DistilBERT compression framework. This paper details the mathematical formulation of the local attention and the static compressive memory, describes the implementation, and discusses the relationship to the full Infini‑Attention.

---

## 1. Introduction

Self‑attention in Transformers scales quadratically with the sequence length \(L\), making it impractical for very long contexts. To overcome this, various approaches have been proposed, including sparse attention patterns, linear attention, and memory‑based mechanisms. The **Infini‑Attention** (Munkhdalai et al., 2024) introduces a novel compressive memory that accumulates and compresses past key‑value information, allowing the model to attend to an unbounded history with constant memory footprint. The memory is updated recurrently and participates in the attention computation as a set of compressed keys and values.

The implementation provided in this module focuses on the attention‑side of Infini‑Attention: it implements a **static compressive memory**—a learnable parameter representing a compressed view of the entire context—combined with a **sliding window local attention**. While the full Infini‑Attention would dynamically update the memory with each token using an associative update rule, this simplified version treats the memory as a trainable tensor that is learned during fine‑tuning. This design still captures the spirit of using a compact memory to augment local attention, providing a global context without storing all past tokens.

In this paper, we present the mathematical details of the local window attention and the static memory attention, the implementation specifics, and discuss the limitations of this simplified version relative to the complete Infini‑Attention.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input hidden state \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), the module first applies linear projections to obtain queries, keys, and values for all heads:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V,
\]

with \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times (H d_h)}\), where \(H\) is the number of heads and \(d_h = d_{\text{model}} / H\). The tensors are reshaped to \((B, H, L, d_h)\).

### 2.2 Sliding Window Local Attention

The local attention restricts each query to attend only to a fixed window of previous tokens (causal) and a few future tokens if not causal. Here, a causal mask is enforced via a triangular matrix: using `torch.tril` with diagonal offset equal to `local_window - 1`. Concretely, a boolean mask \(\mathbf{W} \in \{0,1\}^{L \times L}\) is defined as:

\[
W_{ij} = \begin{cases}
1 & \text{if } j \le i \text{ and } i - j < \text{local\_window},\\
0 & \text{otherwise}.
\end{cases}
\]

Then local attention scores are computed as:

\[
\mathbf{S}^{\text{local}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}.
\]

Positions where \(W_{ij}=0\) are set to \(-\infty\). Additionally, a padding mask (if provided) is applied to mask out invalid key positions. The padding mask is assumed to be a float tensor where 1 indicates valid and 0 indicates padding; it is converted to an additive mask via `(mask - 1) * -1e9` and added to scores. The local attention weights are then:

\[
\mathbf{A}^{\text{local}} = \operatorname{softmax}(\mathbf{S}^{\text{local}} + \mathbf{M}_{\text{pad}}),
\]

where \(\mathbf{M}_{\text{pad}}\) incorporates both the window and padding masks. Dropout is applied to the attention weights (with a fixed probability of 0.1, but in the code it uses `F.dropout` with `p=0.1` hardcoded, not the configurable dropout from `config`). The local output is:

\[
\mathbf{O}^{\text{local}} = \mathbf{A}^{\text{local}} \mathbf{V}.
\]

### 2.3 Compressive Memory Attention

The compressive memory is represented by a learnable parameter \(\mathbf{M} \in \mathbb{R}^{1 \times H \times C \times d_h}\), where \(C\) is the `compress_dim` (default 128). This tensor can be seen as a set of \(C\) compressed key‑value pairs per head. For each head, the memory attention scores are:

\[
\mathbf{S}^{\text{mem}} = \frac{\mathbf{Q} \mathbf{M}^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times L \times C}.
\]

Applying softmax yields:

\[
\mathbf{A}^{\text{mem}} = \operatorname{softmax}(\mathbf{S}^{\text{mem}}),
\]

and the memory output is:

\[
\mathbf{O}^{\text{mem}} = \mathbf{A}^{\text{mem}} \mathbf{M} \quad \in \mathbb{R}^{B \times H \times L \times d_h}.
\]

Here, note that \(\mathbf{M}\) serves as both the keys and values; the attention weights are computed between queries and the memory “keys”, and the weighted average is taken over the memory “values” (which are the same tensor). This is a standard key‑value retrieval from a static memory.

### 2.4 Combination and Output

The final output of the attention module is the element‑wise sum of the local and memory outputs:

\[
\mathbf{O} = \mathbf{O}^{\text{local}} + \mathbf{O}^{\text{mem}}.
\]

This is then transposed to \((B, L, H, d_h)\), flattened to \((B, L, H d_h)\), and projected:

\[
\mathbf{Y} = \mathbf{W}_O \mathbf{O}_{\text{flat}} + \mathbf{b}_O.
\]

The module returns \((\mathbf{Y}, \text{None})\) (or the local attention weights if `output_attentions=True`).

---

## 3. Implementation Details

### 3.1 Module Structure

- `InfiniAttention` inherits from `BaseAttention` and is registered as `"infini"`.
- Constructor parameters:
  - `config`: provides `num_attention_heads`, `hidden_size`, and `dropout` (though dropout is hardcoded to 0.1 in the forward pass, not using config).
  - `rank`: ignored (for interface compatibility).
  - `compress_dim` (default 128): size of the compressive memory \(C\).
  - `local_window` (default 128): size of the local causal window.
- `self.memory` is an `nn.Parameter` of shape `(1, num_heads, compress_dim, head_dim)` initialised with a normal distribution (std=0.02). It is expanded along batch dimension during forward.

### 3.2 Forward Pass

1. Project hidden states to Q, K, V, reshape to \((B, H, L, d_h)\).
2. **Memory attention**: 
   - Expand memory to batch size.
   - Compute `mem_scores = Q @ mem^T / sqrt(d_h)`.
   - Apply softmax to get `mem_attn`.
   - Compute `mem_out = mem_attn @ mem` (weighted sum over compressed dimension).
3. **Local attention**:
   - Build a lower‑triangular mask with diagonal offset `local_window - 1` (so that each token attends to up to `local_window` previous tokens, including itself? Actually `tril(diagonal=local_window - 1)` gives a band of width `local_window` where indices are ≤ current index and ≥ (current index - local_window + 1). For `i` and `j`, condition `i - j < local_window` and `j ≤ i`. That's a causal window of size `local_window`.)
   - Compute scores `Q @ K^T / sqrt(d_h)`, mask out positions outside window with `-1e9`.
   - Apply padding mask if provided (adding `(mask - 1) * -1e9`).
   - Softmax, dropout (hardcoded `p=0.1`), multiply by V.
4. Sum `local_out + mem_out`.
5. Reshape, project, return.

### 3.3 Relationship to Full Infini‑Attention

In the full Infini‑Attention (arXiv:2404.07143), the compressive memory is dynamically updated using a delta rule (or similar) after each layer, aggregating new key‑value pairs into the memory while maintaining a fixed capacity. This allows the model to handle arbitrarily long contexts without storing all activations. The memory update involves an attention‑driven gating mechanism and a linear projection to compress. The current implementation omits this update step and uses a **static, trainable memory**. This simplifies the module to a kind of “global memory” that is shared across all sequences in a batch and learned during training, much like a set of global tokens. The advantage is simplicity and ease of integration; the disadvantage is that the memory does not adapt to the specific input sequence (beyond the queries), making it more like a parameter‑based memory than a true context‑compressive memory.

The module could be extended to include the memory update mechanism by adding a recurrent pass over the sequence that updates `self.memory` with information from past tokens, as described in the original paper.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"infini"` in the compression configuration. The relevant configurable parameters (via `comp_config`) include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compress_dim` | 128 | Number of compressed memory slots \(C\). |
| `local_window` | 128 | Width of the sliding local window. |
| `dropout` | (from config, but hardcoded to 0.1 in this code) | Dropout rate for local attention weights. |

Example injection:
```python
comp_config.attention_type = "infini"
comp_config.compress_dim = 256
comp_config.local_window = 64
```

Manual instantiation:
```python
attn = InfiniAttention(config=model_config, compress_dim=256, local_window=64)
layer.attention = attn
```

The outer Transformer layer’s residual and LayerNorm remain in place.

---

## 5. Discussion

This simplified Infini‑Attention provides a straightforward way to incorporate a global, compressed memory into a Transformer. By attending to a fixed set of learned memory vectors, the model can access a high‑level summary of the entire context, which may help with tasks requiring long‑range coherence. The local window preserves detailed local interactions. The combination is efficient: local attention costs \(O(L \cdot W \cdot d)\), while memory attention costs \(O(L \cdot C \cdot d)\), both linear in \(L\).

However, the static memory has limited capacity to capture sequence‑specific information, as it is the same for all inputs. This could lead to a performance gap compared to the full Infini‑Attention, where the memory dynamically updates. Moreover, the memory attention uses the same tensor for both keys and values, which limits the expressiveness of the retrieval (compared to having separate projected keys and values). The hardcoded dropout value of 0.1 ignores the config’s `dropout` parameter, which could lead to inconsistencies. The padding mask handling assumes a float mask with 1 for valid and 0 for padding, which is standard; if a 2D bool mask is passed, it would need to be converted beforehand (the code does not do that conversion, so it expects a float mask of correct shape).

Future improvements could implement the dynamic memory update from the original Infini‑Attention paper, use separate key and value projections for memory, and properly integrate the config’s dropout. Despite these simplifications, the module serves as a useful baseline for exploring memory‑augmented attention with linear complexity.

---

## 6. Conclusion

We have presented a simplified Infini‑Attention module that combines a static compressive memory with a sliding window local attention. The design reduces the quadratic cost of global attention while providing a learned, global memory to supplement local interactions. Integrated into the DistilBERT compression framework, it offers a plug‑and‑play alternative to standard self‑attention. Future work will incorporate the recurrent memory update to realise the full potential of Infini‑Attention for infinite‑context processing.