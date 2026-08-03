# MLA: Multi‑Head Latent Attention — DeepSeek‑V2’s Low‑Rank Key‑Value Cache Compression

**Abstract**

We present **Multi‑Head Latent Attention (MLA)**, a highly efficient attention module introduced in DeepSeek‑V2 (Liu et al., 2024). MLA reduces the memory footprint of the key‑value (KV) cache during autoregressive inference by compressing the keys and values of all heads into a single, low‑rank latent vector per token. Instead of storing a separate KV pair for each attention head, the model projects the hidden state down to a small latent dimension \(r\) (e.g., 128), and then expands the latent vector to reconstruct the keys and values for all heads. The query projection remains standard. This design allows the entire KV cache to be stored as a single compact tensor of size \(B \times L \times r\), drastically reducing the per‑token memory from \(2 \cdot H \cdot d_h\) to just \(r\). The attention computation itself remains the full scaled dot‑product, so the representational power is preserved. The module is implemented as a drop‑in replacement for the self‑attention layer in DistilBERT, and is integrated into the compression framework via the registry. This paper provides a complete mathematical description of the latent compression mechanism, the architecture, and the implementation details.

---

## 1. Introduction

Large language models typically employ multi‑head attention, where each of the \(H\) heads maintains its own key and value tensors. During autoregressive generation, all previous keys and values must be cached to avoid recomputation, leading to a KV cache of size \(2 \times B \times L \times H \times d_h\), which becomes prohibitive for long sequences or large batch sizes. **Multi‑Head Latent Attention (MLA)**, proposed in the DeepSeek‑V2 architecture, addresses this by applying a simple but effective form of **cross‑head compression**: the keys and values of all heads are generated from a common, compact latent representation. This reduces the cache size by a factor of \((2 \cdot H \cdot d_h) / r\), where \(r\) is a tunable rank (typically 128).

MLA achieves this by inserting a bottleneck linear layer that projects the hidden state down to a latent vector of dimension \(r\), and then using two separate up‑projection matrices to generate the keys and values for all heads simultaneously. The latent vector replaces the per‑head KV cache, dramatically cutting memory usage with minimal impact on model quality. The query projection is left unchanged, ensuring that the attention mechanism can still freely attend to the full‑dimensional keys.

The module described here is a self‑contained MLA implementation that can be injected into a DistilBERT encoder layer. It follows the original design but is adapted for the full‑sequence (non‑autoregressive) encoder case, where the benefits are primarily in parameter reduction rather than cache size (since the encoder processes the entire sequence at once and does not maintain a persistent KV cache). However, it still exhibits a lower parameter count for key and value projections.

---

## 2. Method

### 2.1 Latent Key‑Value Compression

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) be the input hidden states, where \(D\) is the model dimension, \(B\) the batch size, and \(L\) the sequence length. The module first computes a compressed latent representation \(\mathbf{Z} \in \mathbb{R}^{B \times L \times r}\) via a down‑projection:

\[
\mathbf{Z} = \mathbf{X} \mathbf{W}_{\text{down}}^\top,
\]

where \(\mathbf{W}_{\text{down}} \in \mathbb{R}^{r \times D}\) (using `nn.Linear` with `bias=False`). The rank \(r\) is a hyperparameter (default 128).

From this shared latent vector, we then reconstruct the full keys and values for all heads simultaneously. The up‑projections are defined as:

\[
\mathbf{K} = \mathbf{Z} \mathbf{W}_{\text{K}}^\top, \qquad
\mathbf{V} = \mathbf{Z} \mathbf{W}_{\text{V}}^\top,
\]

with \(\mathbf{W}_{\text{K}}, \mathbf{W}_{\text{V}} \in \mathbb{R}^{(H \cdot d_h) \times r}\). The outputs \(\mathbf{K}\) and \(\mathbf{V}\) have shape \((B, L, H d_h)\). These are then reshaped into the multi‑head format \((B, L, H, d_h)\) and transposed to \((B, H, L, d_h)\).

Note that the same latent vector is used to produce both keys and values, which enforces a shared bottleneck and reduces the effective number of parameters. The queries are projected independently via a standard linear layer:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q^\top, \qquad \mathbf{W}_Q \in \mathbb{R}^{(H d_h) \times D},
\]

again reshaped to \((B, H, L, d_h)\).

### 2.2 Scaled Dot‑Product Attention

With \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) in the standard multi‑head layout, attention proceeds as usual:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}},
\]
\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}), \qquad \text{(mask optional)}
\]
\[
\mathbf{C} = \mathbf{A} \mathbf{V}.
\]

The mask \(\mathbf{M}\) is expected to be an additive tensor broadcastable to \((B, H, L, L)\). In the provided implementation, the mask handling is simplified: if an `attention_mask` is provided, it is added directly after unsqueezing the second dimension (yielding shape \((B, 1, L)\)). This corresponds to a key‑padding mask with 0 for valid positions and \(-\infty\) for padding tokens. The code assumes the mask is already in the appropriate format; it does not perform conversion from a 2D boolean mask.

Dropout is applied to the attention weights \(\mathbf{A}\).

### 2.3 Output Projection

The context tensor \(\mathbf{C}\) of shape \((B, H, L, d_h)\) is concatenated along the head dimension, resulting in \((B, L, H d_h)\), and projected back to \(D\) via a standard linear layer \(\mathbf{W}_O \in \mathbb{R}^{D \times (H d_h)}\) (with bias).

### 2.4 Complexity and Memory

The primary advantage of MLA is the reduction in KV cache size. In a standard Transformer, each token stores \(2 \times H \times d_h = 2D\) values. With MLA, the latent vector requires only \(r\) numbers per token (plus the cached projection matrices, which are shared). Typically \(r \ll D\) (e.g., 128 vs 768), giving over 5× memory savings for the KV cache. During training and non‑autoregressive inference, the intermediate tensors \(\mathbf{K}\) and \(\mathbf{V}\) are still materialised at full size, so the peak memory is unchanged; the benefit is felt during incremental generation.

Parameter count: The total number of parameters for the key‑value projections becomes \(D \times r + 2 \times r \times D = 3Dr\), compared to \(2 \times D \times D = 2D^2\) for a standard setup. With \(r = D/6\), this is about 4× fewer parameters.

---

## 3. Implementation Details

- **`MLAAttention`** inherits from `nn.Module` and is registered under the key `"mla"`.
- Constructor arguments:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `rank`: the latent dimension \(r\) (default 128).
- Layers:
  - `q_proj`: Linear(\(D, H d_h\), bias=False).
  - `kv_down`: Linear(\(D, r\), bias=False) – the bottleneck.
  - `k_up`: Linear(\(r, H d_h\), bias=False) – generates keys.
  - `v_up`: Linear(\(r, H d_h\), bias=False) – generates values.
  - `out_proj`: Linear(\(H d_h, D\)) – includes bias by default.
  - `dropout`: standard dropout on attention weights.
- Forward pass:
  1. Compute \(\mathbf{Q}\) and reshape to multi‑head.
  2. Compute \(\mathbf{Z} = \text{kv\_down}(\mathbf{X})\).
  3. Compute \(\mathbf{K} = \text{k\_up}(\mathbf{Z})\) and \(\mathbf{V} = \text{v\_up}(\mathbf{Z})\), reshape to multi‑head.
  4. Transpose Q,K,V to \((B, H, L, d_h)\).
  5. Compute scaled dot‑product scores.
  6. If mask is not None, unsqueeze to \((B, 1, L)\) and add to scores.
  7. Softmax, dropout.
  8. Weighted sum with V, transpose, flatten, project.
  9. Return `(out, attn_probs)` – note that attention weights are always returned, ignoring the typical `output_attentions` flag. This may deviate from the expected interface; a wrapper could be added to conditionally return `None`.

- **Mask format**: The code assumes an additive mask with 0 for valid tokens and negative values for padding, already shaped as \((B, L)\). It unsqueezes to \((B, 1, L)\), which broadcasts correctly along the query dimension. This is a simple but functional approach for bidirectional attention.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"mla"` in the compression configuration. The primary hyperparameter is the latent rank \(r\), which can be passed via `comp_config` or directly:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 128 | Dimension of the shared latent vector \(r\). |
| `dropout` | (from config) | Dropout rate on attention weights. |

Example injection via framework (assuming an injection function exists or is added):
```python
comp_config.attention_type = "mla"
comp_config.rank = 64
```

Manual instantiation:
```python
mla = MLAAttention(config=model_config, rank=64)
layer.attention = mla
```

No pretrained weight transfer is performed in this snippet; for zero‑shot initialisation, one would need to initialise the projection matrices appropriately (e.g., by splitting or factorising the original weight). In practice, MLA is often trained from scratch, or the weights are initialised via a similar SVD‑based method. The injection function could be extended to handle this.

---

## 5. Discussion

MLA offers a simple and highly effective way to compress the key‑value cache. By forcing all heads to share a single latent representation, it not only saves memory but also acts as a regulariser that may improve generalisation by encouraging the model to extract reusable, head‑agnostic features. The design is particularly appealing for deployment of large conversational models where the KV cache is a critical bottleneck.

However, in the context of an encoder‑only model like DistilBERT, the benefits are less dramatic because the full key and value tensors are still materialised during the forward pass. The primary advantage in this setting is a reduction in the number of trainable parameters, which can speed up fine‑tuning and lower storage requirements. The module also does not modify the attention pattern, so it remains compatible with all downstream tasks that rely on standard attention.

The code’s handling of the attention mask is minimal but sufficient for most use cases. A potential improvement would be to properly convert a 2D boolean mask (as typically produced by the tokeniser) into the additive form. Additionally, the module always returns attention weights, which may cause memory overhead if not needed; a conditional return based on `output_attentions` would align better with the framework.

---

## 6. Conclusion

We have presented MLA, a multi‑head latent attention module that compresses the key‑value projections into a low‑rank bottleneck, dramatically reducing KV cache size and parameter count. The implementation is a clean, drop‑in replacement for DistilBERT’s self‑attention, and integrates seamlessly with the compression framework. MLA is a proven technique from the DeepSeek‑V2 architecture, and this module makes it accessible for experimentation and deployment in encoder‑based models. Future enhancements could include learnable latent‑dimension adaptation and improved mask handling.