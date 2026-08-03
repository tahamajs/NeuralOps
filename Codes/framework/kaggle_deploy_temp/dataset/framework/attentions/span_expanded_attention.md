# Span‑Expanded Attention with Hybrid Low‑Rank Adaptation (HyLoRA)

**Abstract**

We present **Span‑Expanded Attention**, a lightweight attention module designed for efficient inference with an extended context window. The module accepts an optional, pre‑computed **key‑value cache** that represents a large, fixed external memory (up to 1024 tokens or more). During the forward pass, the queries attend not only to the current sequence but also to this external cache, effectively expanding the attention span without storing all intermediate activations. When no cache is provided, the module reduces to standard multi‑head self‑attention. Additionally, a low‑rank adaptation (LoRA) pathway can be enabled, adding a small, trainable residual to the attention output, thereby enhancing capacity with minimal parameter overhead. The design is intended as a drop‑in replacement for the self‑attention sub‑layer in DistilBERT, with a particular focus on scenarios where a fixed, long‑term memory is available (e.g., retrieval‑augmented models or context‑carryover). This paper provides a rigorous mathematical description of the attention mechanism, the cache integration, the LoRA branch, and the overall architecture, along with a discussion of the missing mask handling and other limitations.

---

## 1. Introduction

Standard multi‑head self‑attention scales quadratically with the input length, and its effective context window is limited by the sequence length processed by the model. In many applications, it is desirable to attend to a very long context without recomputing all activations. The **Span‑Expanded Attention** module addresses this by introducing a **fixed external key‑value cache**. This cache can be computed offline (e.g., from a large document or a set of retrieved passages) and reused across many forward passes. During inference, the queries attend to both the current sequence’s keys and values and those from the cache, effectively expanding the receptive field. This approach is reminiscent of encoder‑decoder attention, where the decoder attends to encoder outputs, but here both the cache and the current tokens come from the same model (self‑attention with memory).

In addition, the module incorporates a **Hybrid Low‑Rank Adaptation (HyLoRA)** option: a separate LoRA branch processes the original hidden states and adds a low‑rank correction to the attention output. This allows fine‑tuning of the model with a tiny number of additional parameters while keeping the main attention weights frozen (if desired). The module is designed as a plug‑in replacement for the self‑attention layer in DistilBERT, and is registered within the compression framework.

In this paper, we detail the mathematical formulation of the span‑expanded attention, the LoRA adaptation, and the forward pass. We also discuss the absence of attention masking and its implications, as well as the intended usage scenarios.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module applies standard linear projections for all heads:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D}\) (bias‑free). The query tensor is reshaped to \((B, H, L, d_h)\) where \(H\) is the number of heads and \(d_h = D/H\). The key and value tensors are reshaped similarly when no cache is provided.

### 2.2 Span‑Expanded Attention with External Key‑Value Cache

The core innovation is the ability to incorporate an external **key‑value cache** \(\mathbf{K}^{\text{cache}}, \mathbf{V}^{\text{cache}} \in \mathbb{R}^{B \times C \times D}\) (or \(C \times D\) per batch, with the head dimension not yet split). The cache represents a fixed memory of \(C\) token representations, precomputed from a separate context. When a cache is provided, the module expands the cache across all heads:

\[
\mathbf{K}^{\text{full}} = \mathbf{K}^{\text{cache}} \text{ with shape } (B, H, C, d_h), \quad
\mathbf{V}^{\text{full}} = \mathbf{V}^{\text{cache}} \text{ similarly},
\]

by unsqueezing and expanding the batch and head dimensions (the code does `k_cache.unsqueeze(1).expand(-1, H, -1, -1)`). The queries are then attended to the full set of keys (cache keys) rather than the current sequence’s keys:

\[
\mathbf{S} = \frac{\mathbf{Q} (\mathbf{K}^{\text{full}})^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times L \times C}.
\]

If no cache is provided, the standard self‑attention scores are computed using the current sequence’s keys, yielding shape \((B, H, L, L)\).

After softmax, the attention weights are applied to the full values \(\mathbf{V}^{\text{full}}\) (cache or current), giving context \(\mathbf{C} \in \mathbb{R}^{B \times H \times L \times d_h}\). The context is then reshaped back to \((B, L, D)\).

This mechanism allows the model to attend to a long external context without recomputing its keys and values at every step. It is particularly useful for retrieval‑augmented generation or when the same fixed document needs to be queried repeatedly.

### 2.3 Low‑Rank Adaptation (HyLoRA)

When `use_lora=True`, an additional **LoRA branch** is applied to the original hidden states. A low‑rank decomposition is used:

\[
\mathbf{Y}_{\text{lora}} = \mathbf{B} \, \mathbf{A} \, \mathbf{X},
\]

where \(\mathbf{A} \in \mathbb{R}^{r \times D}\) and \(\mathbf{B} \in \mathbb{R}^{D \times r}\) are the learnable LoRA matrices (implemented as linear layers `lora_A` and `lora_B` with no bias). The output \(\mathbf{Y}_{\text{lora}} \in \mathbb{R}^{B \times L \times D}\) is added to the attention output **before** the final output projection:

\[
\tilde{\mathbf{C}} = \mathbf{C} + \mathbf{Y}_{\text{lora}}.
\]

This injection of a trainable low‑rank residual allows task‑specific adaptation while keeping the main projection weights frozen (they can be frozen if desired). The LoRA rank \(r\) (e.g., 8) is a hyperparameter. Note that the LoRA branch is applied to the original hidden states, not to the attended context, which is a design choice; it could alternatively be applied to the queries or keys.

### 2.4 Output Projection

Finally, the combined representation (attention output + optional LoRA) is projected to the original dimension using a linear layer \(\mathbf{W}_O \in \mathbb{R}^{D \times D}\):

\[
\mathbf{Y} = \tilde{\mathbf{C}} \mathbf{W}_O^\top + \mathbf{b}_O.
\]

The module returns a tuple \((\mathbf{Y}, \text{None})\) for compatibility.

---

## 3. Implementation Details

- **`SpanExpandedAttention`** inherits from `BaseAttention` and is registered under `"span_expanded"` in the `ATTENTION_REGISTRY`.
- **Constructor parameters**:
  - `config`: provides `hidden_size`, `num_attention_heads`.
  - `rank`: ignored (for interface consistency).
  - `expansion_span`: intended maximum cache length (not directly used in the code, only a descriptor).
  - `use_lora` (default True): whether to add the LoRA branch.
  - `lora_rank` (default 8): rank \(r\) for the LoRA matrices.
  - Creates `W_q`, `W_k`, `W_v` (no bias), `out_proj` (with bias). If `use_lora`, creates `lora_A` (Linear, no bias) and `lora_B` (Linear, no bias).
- **Forward**:
  - Accepts `hidden_states`, optional `kv_cache` (tuple of keys and values), and `**kwargs` (ignores `attention_mask`).
  - Projects Q, reshapes.
  - If cache provided, expands cache to all heads and uses it as keys and values; otherwise uses current K, V.
  - Computes attention scores, softmax (no mask).
  - Aggregates values.
  - Reshapes, optionally adds LoRA output.
  - Applies `out_proj`.
  - Returns `(out, None)`.

- **No mask handling**: The attention mask parameter is completely ignored. This is a significant limitation; the module assumes sequences are not padded, or that padding is handled elsewhere. In a practical setting, one would need to mask padding positions in the current sequence (and potentially in the cache if the cache has padding). This omission should be addressed before deployment.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"span_expanded"`. Configuration parameters (via `comp_config` or direct) include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expansion_span` | 1024 | Maximum cache length (for reference, not used in code). |
| `use_lora` | True | Whether to enable the LoRA branch. |
| `lora_rank` | 8 | Rank of the LoRA matrices. |
| `dropout` | (from config, but not applied in this code) | Not applied inside module. |

Example manual instantiation:
```python
attn = SpanExpandedAttention(config=model_config, use_lora=True, lora_rank=16)
```

When using the module, the caller is responsible for providing the `kv_cache` argument if span expansion is desired. The cache should be a tuple `(K, V)` of shape `(B, C, D)` each (pre‑projected and possibly detached). If no cache is given, the module works as standard self‑attention.

No zero‑shot weight transfer is performed in this snippet; the linear layers are randomly initialised. To preserve pretrained knowledge, the injection function should copy the original `q_lin`, `k_lin`, `v_lin`, `out_lin` weights, but this is not included. In practice, one would freeze the original projections and only train the LoRA parameters.

---

## 5. Discussion

Span‑Expanded Attention presents a simple mechanism for extending the context window via an external cache. It is akin to a memory‑augmented Transformer and can be particularly useful for tasks such as question answering over a fixed document, where the document can be pre‑encoded into a cache and queried multiple times. The LoRA branch adds a parameter‑efficient adaptation capability, allowing the model to adjust to new tasks without altering the pretrained weights.

However, the module is incomplete in several respects:

- **No padding mask**: For any practical use with batched variable‑length sequences, the attention scores must be masked for padding positions. The module currently ignores the `attention_mask` argument entirely, which would cause padded tokens to influence the softmax and waste computation. The cache might also contain padding, requiring mask handling there as well.

- **Cache format**: The cache is expected to be already projected and split into heads? Actually, the code receives `k_cache, v_cache` of shape `(B, C, D)` (without heads) and expands to `(B, H, C, d_h)`. That implies the cache is in the full hidden dimension, not per‑head. However, the original keys and values from the sequence are also projected to `(B, L, D)` and then reshaped to multi‑head. The expansion of the cache with `unsqueeze(1).expand(-1, H, -1, -1)` assumes that the full‑dimension cache can be split into heads by simply reshaping, which is correct only if the projection was done with the same head‑wise splitting convention (i.e., the cache should have been computed using the same projection, then reshaped). If the cache is provided in a per‑head format, the code would be wrong. The intended usage likely expects the cache to be precomputed by the same model, so shape compatibility holds.

- **LoRA branch placement**: Adding the LoRA correction to the attention output before the output projection is a valid choice, but the LoRA could also be applied to the queries/keys/values or inside the attention to modify the attention scores. The current placement acts as a residual to the attention context, which may be less effective for certain tasks.

- **No attention weights return**: The module always returns `(out, None)`, ignoring the `output_attentions` flag. If attention weights are needed for analysis, this would need to be extended.

Despite these limitations, the module offers a clear framework for memory‑augmented attention and can be a useful baseline for systems that rely on cached contexts. It also demonstrates how a simple LoRA adapter can be integrated into a custom attention block.

---

## 6. Conclusion

We have described Span‑Expanded Attention, a hybrid module that combines a fixed external key‑value cache for extending the attention span with an optional low‑rank adaptation branch. The design enables efficient processing of long‑range contexts and task‑specific fine‑tuning with minimal parameters. While the current implementation lacks crucial masking and assumes a specific cache format, it provides a solid foundation for building memory‑augmented Transformers within the DistilBERT compression framework. Future work will add proper mask support, clarify the cache interface, and explore more effective placements of the LoRA adapter.The **Span‑Expanded Attention** module extends standard self‑attention with an optional external key‑value cache (for long‑range memory) and a low‑rank adaptation (LoRA) branch for efficient fine‑tuning. It serves as a drop‑in replacement for DistilBERT’s attention layer, though it currently ignores padding masks and lacks pretrained weight transfer.