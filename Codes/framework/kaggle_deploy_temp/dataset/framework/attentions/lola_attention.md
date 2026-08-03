# LoLA Attention: Low‑Rank Linear Attention with Sparse Caching

**Paper Reference**  
LoLA: Low‑Rank Linear Attention With Sparse Caching (2025)  
https://arxiv.org/abs/2509.58281

---

## 1. Introduction

Standard self‑attention in Transformers scales quadratically with the sequence length, making it impractical for long documents. Several efficient attention variants have been proposed, including low‑rank approximations and sparse memory mechanisms. **LoLA (Low‑Rank Linear Attention)** combines two key ideas: (1) **low‑rank factorisation of the query, key, value, and output projections** to drastically reduce parameters while retaining most of the pretrained model’s knowledge, and (2) a **three‑memory caching system**—recent, sparse, and low‑rank—that allows the model to attend to an effectively infinite context with constant memory footprint. The cache is updated on‑the‑fly during inference (or training) and provides additional keys and values for the attention computation, enabling training‑free retrieval augmentation.

The LoLA attention module implemented here is a drop‑in replacement for a DistilBERT attention layer. It uses zero‑shot initialisation of the low‑rank projection matrices from the pretrained weights via truncated SVD. This paper details the architecture, the mathematical formulation of the low‑rank projections and caching system, and the integration into the compression framework.

---

## 2. Method

### 2.1 Low‑Rank Factorised Projections

Each of the standard attention projections—query, key, value, and output—is replaced by a low‑rank factorised linear layer `LowRankLinear`. Given a pretrained weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), we approximate it as the product of two much smaller matrices \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}\) and \(\mathbf{V} \in \mathbb{R}^{r \times d_{\text{in}}}\), where \(r \ll \min(d_{\text{out}}, d_{\text{in}})\). The forward computation is:

\[
\mathbf{y} = \mathbf{U} (\mathbf{V} \mathbf{x}) + \mathbf{b},
\]

thus avoiding the storage and multiplication of the full \(d_{\text{out}} \times d_{\text{in}}\) matrix. The bias is retained if present in the original layer.

**Initialisation via SVD** – For a pretrained linear layer, we perform the full SVD \(\mathbf{W} = \mathbf{U}_{\text{full}} \mathbf{S} \mathbf{V}_{\text{full}}^\top\) and take the top \(r\) singular values and vectors, storing:

\[
\mathbf{U} = \mathbf{U}_{\text{full}}[:,:r] \operatorname{diag}(\sqrt{\mathbf{s}_r})?, \quad
\mathbf{V} = \operatorname{diag}(\sqrt{\mathbf{s}_r}) \mathbf{V}_{\text{full}}[:r, :].
\]

In the provided code, the factor matrices are initialised as:

- `self.U` = \(\mathbf{U}_{\text{full}}[:,:r] \cdot \mathbf{S}_{[:r]}\) (absorbing the full singular values into U)
- `self.V` = \(\mathbf{V}_{\text{full}}[:r, :]\)

This yields \(\mathbf{U}\mathbf{V} \approx \mathbf{W}_{\text{full}}\), the optimal rank‑\(r\) approximation. Biases are directly copied.

### 2.2 Three‑Memory Caching System

The `LowRankCache` maintains three distinct key‑value stores that are updated after each forward pass (or token step, in autoregressive mode). The current implementation simplifies by using only the first batch element and first head, accumulating the three caches as fixed‑size tensors. The caches are:

- **Recent cache** – a sliding window of the most recent `max_len` key‑value pairs.
- **Sparse cache** – a subsampled version (e.g., every 4th token) of the recent history, also capped to half of `max_len`.
- **Low‑rank cache** – obtained by applying truncated SVD to the recent keys (and similarly to values), retaining only the top `low_rank_k` components. This acts as a compressed, global summary of the recent context.

During inference, these caches provide a fixed‑size set of auxiliary keys and values that supplement the current sequence’s attention.

### 2.3 Low‑Rank Global Attention

For the current input sequence of length \(N\), we compute the standard scaled dot‑product attention scores using the low‑rank projected queries, keys, and values:

\[
\mathbf{S}^{\text{global}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}} \cdot \alpha_{\text{LR}},
\]

where \(\alpha_{\text{LR}} = \text{self.LR\_scale}\) is a learnable scalar that controls the contribution of the current sequence’s attention. This is then augmented by the cached content.

### 2.4 Sparse Cache Retrieval

If the cache is non‑empty, we expand the cached keys and values (concatenated from recent, sparse, and low‑rank stores) to match the batch and head dimensions, yielding \(\mathbf{K}^{\text{cache}}, \mathbf{V}^{\text{cache}} \in \mathbb{R}^{B \times H \times C \times d_h}\), where \(C\) is the total number of cached tokens. The scores against the cached keys are computed as:

\[
\mathbf{S}^{\text{cache}} = \frac{\mathbf{Q} (\mathbf{K}^{\text{cache}})^\top}{\sqrt{d_h}} \cdot \beta,
\]

where \(\beta = \text{self.sparse\_gate}\) is another learnable scalar weighting the cached contribution. The full attention scores are obtained by concatenating the global and cached scores along the last dimension:

\[
\mathbf{S} = \left[ \mathbf{S}^{\text{global}} \;\|\; \mathbf{S}^{\text{cache}} \right] \in \mathbb{R}^{B \times H \times N \times (N + C)}.
\]

Similarly, the value tensor is concatenated:

\[
\mathbf{V}^{\text{combined}} = \left[ \mathbf{V} \;\|\; \mathbf{V}^{\text{cache}} \right] \in \mathbb{R}^{B \times H \times (N + C) \times d_h}.
\]

Softmax is applied to the concatenated scores, and the output is the weighted sum of the combined values. The learnable scales \(\alpha_{\text{LR}}\) and \(\beta\) allow the model to learn the relative importance of the current sequence and the cached memory.

### 2.5 Masking and Padding

The standard attention mask is extended to cover the cached keys (which are always valid). This ensures that the padded positions in the current sequence do not influence the output, while the cache is fully unmasked.

### 2.6 Output Projection

The resulting context is reshaped and passed through the low‑rank output projection \(\mathbf{W}_O\). The module returns the output and optionally the attention weights.

---

## 3. Implementation Details

### 3.1 `LowRankLinear`
- Inherits from `nn.Module`.
- Stores `U` and `V` parameters with orthogonal initialisation, and optional bias.
- Forward: `x → Vx → U(Vx) + bias`.
- A method `set_low_rank_factors` allows explicit assignment for weight transfer.

### 3.2 `LowRankCache`
- Maintains three tensors: `recent_k`, `recent_v`, `sparse_k`, `sparse_v`, `low_rank_k`, `low_rank_v` as buffers.
- The `update` method currently operates only on the first batch element and first head (a simplification that would need to be extended for full multi‑head, multi‑batch operation).
- The `get_cached` method concatenates all three stores into a single pair of tensors.

### 3.3 `LoLAAttention`
- Constructor accepts `model_config`, `rank`, `dropout`, `pretrained_attn`, and `comp_config`.
- Initialises four `LowRankLinear` projections and copies pretrained weights using `_init_from_linear`, which performs the SVD truncation.
- Creates the `LowRankCache` with configurable `max_seq_len` and `low_rank_cache_size`.
- Defines learnable scalars `LR_scale` (initial value 0.5) and `sparse_gate` (0.5).
- The forward pass:
  1. Projects Q, K, V.
  2. Computes global scores, multiplies by `LR_scale`.
  3. Retrieves cached K,V; if present, computes cache scores, concatenates scores and values.
  4. Extends mask (if any) to cover the cached tokens.
  5. Softmax, dropout, weighted sum.
  6. Reshapes, applies output projection.
  7. Updates the cache with the current K,V (for future steps).
- The cache update occurs at every forward pass; for incremental inference, it should be called per token.

### 3.4 Integration
Registered under `"lola_attn"` via the injection function `inject_lola_attn`, which reads parameters from `comp_config` and initialises the module using the pretrained attention layer.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"lola_attn"`. Relevant configuration keys (in `comp_config`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 32 | Rank for the low‑rank projections. |
| `dropout` | 0.1 | Dropout rate. |
| `max_seq_len` | 128 | Maximum size of the recent cache window. |
| `low_rank_cache_size` | 16 | Rank of the low‑rank SVD cache. |

The injection function copies pretrained weights from the original attention layer.

---

## 5. Discussion

LoLA Attention presents a powerful framework for efficient inference on long sequences. The low‑rank projections drastically reduce the parameter count (by a factor of about \(2r / d_{\text{model}}\)) and can be initialised from pretrained weights with minimal loss. The three‑memory cache provides a dynamic, training‑free augmentation of the context window: the recent cache captures fine‑grained local details, the sparse cache retains temporally diverse information, and the low‑rank cache gives a compressed global view. The learnable scales allow the model to adaptively blend these sources.

However, the current implementation has several simplifications:

- The cache update operates on a single batch element and head, which limits parallelism and is not suitable for batched inference. A production version would manage per‑batch, per‑head caches.
- The cache size is fixed; for very long sequences, the low‑rank SVD and sliding window may discard useful older information.
- The sparse cache uses a fixed stride (every 4th token) – a more sophisticated importance sampling could be beneficial.
- The attention mechanism concatenates the cached keys and values to the current ones, making the total attended length \(N + C\). While \(C\) is constant, the memory and computation per token still scale linearly with \(N\), not constant; however, the current sequence length \(N\) is often kept manageable (e.g., truncated to a window) while the cache provides long‑range history, achieving near‑constant memory overall.

Despite these, LoLA Attention represents a promising direction for efficient Transformers and aligns well with the compression framework’s goal of reducing resource usage while retaining performance.

---

## 6. Conclusion

We have presented LoLA Attention, a low‑rank linear attention module augmented with a three‑memory sparse caching system. The combination of SVD‑initialised factorised projections and a dynamic cache enables efficient, high‑quality attention with a fixed‑size memory footprint. The module is seamlessly integrated into DistilBERT via the compression registry and can be used as a drop‑in replacement for standard self‑attention. Future refinements will address batched caching, adaptive sparse sampling, and integration with incremental decoding for truly infinite context.