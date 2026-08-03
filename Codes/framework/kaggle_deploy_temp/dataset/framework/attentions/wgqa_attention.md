# Weighted Grouped‑Query Attention (WGQA): Learnable Per‑Head Importance Weights

**Abstract**  
We present **Weighted Grouped‑Query Attention (WGQA)**, an efficient self‑attention module that extends Grouped‑Query Attention (GQA) by introducing **learnable per‑head importance weights** on the key and value heads. In standard GQA, the number of key‑value heads is smaller than the number of query heads, and each query head is assigned to a fixed key‑value head via repetition. WGQA retains this grouped structure but equips each key‑value head with a trainable scalar multiplier—one for keys and one for values—that is broadcast across the batch and sequence dimensions. These weights modulate the contribution of each key‑value head before it is replicated to the query heads, allowing the model to learn to emphasise or suppress specific heads. The rest of the attention mechanism remains unchanged. This simple modification adds only \(2 \times \text{num\_kv\_heads}\) parameters and can be trained end‑to‑end. The module serves as a drop‑in replacement for the self‑attention layer in DistilBERT and is registered within the compression framework via the key `"wgqa"`. This paper provides a rigorous mathematical description of the weighted grouping, the attention computation, and the potential benefits and limitations of the approach.

---

## 1. Introduction

Multi‑head self‑attention in Transformers uses separate query, key, and value projections for each head, leading to a large memory footprint for the key‑value (KV) cache during inference. Grouped‑Query Attention (GQA) addresses this by sharing a smaller number of key‑value heads among multiple query heads, reducing the size of the KV cache. However, the sharing pattern is rigid: all query heads in a group use exactly the same key‑value head without any means to adapt the importance of that head. The **Weighted Grouped‑Query Attention (WGQA)** module enhances GQA by attaching a learnable scalar weight to each key head and each value head. These weights are independent per head and are multiplied onto the key and value tensors before the repetition to match the query heads. This simple mechanism allows the model to learn:

- Which key‑value heads are more informative for the task, and
- How to balance the contribution of keys versus values across heads.

Because the weights are scalar and shared across all tokens and batch elements, they act as a global head‑importance gating. They are trained jointly with the rest of the model, requiring no additional loss or regularisation. WGQA retains all the memory benefits of GQA (since the KV cache size is still determined by the number of KV heads), while introducing a tiny number of additional parameters: \(2 \times K\), where \(K\) is the number of key‑value heads. For typical settings (e.g., \(K=4\)), this adds only 8 parameters.

The module is implemented as a drop‑in replacement for DistilBERT’s attention layer and is integrated into the compression framework. In this paper, we formalise the weighted grouping, describe the architecture, and discuss the practical implications.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) (batch size \(B\), length \(L\), hidden size \(D\)), the module projects it to queries, keys, and values. Let \(H\) be the total number of query heads, \(K\) the number of key‑value heads, and \(d_h = D / H\) the head dimension. The projections are:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{B \times L \times H d_h}, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{B \times L \times K d_h}, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{B \times L \times K d_h}.
\]

These are reshaped to separate the heads: \(\mathbf{Q} \in \mathbb{R}^{B \times L \times H \times d_h}\), \(\mathbf{K} \in \mathbb{R}^{B \times L \times K \times d_h}\), \(\mathbf{V} \in \mathbb{R}^{B \times L \times K \times d_h}\).

### 2.2 Per‑Head Importance Weights

Two learnable parameter vectors are introduced:

\[
\boldsymbol{\omega}_K \in \mathbb{R}^{K}, \qquad \boldsymbol{\omega}_V \in \mathbb{R}^{K},
\]

initialised to ones. They are reshaped to \((1, 1, K, 1)\) and multiplied element‑wise with the key and value tensors:

\[
\tilde{\mathbf{K}} = \mathbf{K} \odot \boldsymbol{\omega}_K, \qquad
\tilde{\mathbf{V}} = \mathbf{V} \odot \boldsymbol{\omega}_V.
\]

This operation independently scales each key head and each value head. Since the weights are positive (they can become negative during training, but are initialised to one and can evolve freely), they can amplify or suppress entire heads. A negative weight would flip the sign of the key or value features, which might be beneficial but could also destabilise training. In practice, one might constrain them to be non‑negative (e.g., via a softplus or sigmoid), but the current implementation uses unconstrained scalars.

### 2.3 Grouping by Repetition

The group size is \(G = H / K\) (assumed to be an integer). To match the number of query heads, the weighted key and value heads are repeated \(G\) times along the head dimension:

\[
\mathbf{K}_{\text{full}} = \operatorname{repeat\_interleave}(\tilde{\mathbf{K}}, G, \text{dim}=2) \in \mathbb{R}^{B \times L \times H \times d_h},
\]
\[
\mathbf{V}_{\text{full}} = \operatorname{repeat\_interleave}(\tilde{\mathbf{V}}, G, \text{dim}=2) \in \mathbb{R}^{B \times L \times H \times d_h}.
\]

After this, the queries, keys, and values all have \(H\) heads and are transposed to the standard layout \((B, H, L, d_h)\) for attention.

### 2.4 Scaled Dot‑Product Attention

The rest of the attention is identical to standard multi‑head attention:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}_{\text{full}}^\top}{\sqrt{d_h}}, \qquad
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}), \qquad
\mathbf{C} = \mathbf{A} \mathbf{V}_{\text{full}},
\]

where \(\mathbf{M}\) is an optional padding mask (the code expects an additive mask of shape broadcastable to \((B, H, L, L)\)). Dropout is applied to the attention weights \(\mathbf{A}\). The output context \(\mathbf{C}\) is then reshaped to \((B, L, D)\) and projected via \(\mathbf{W}_O\).

### 2.5 Computational Complexity and Memory

The KV cache during inference stores only the \(K\) weighted key‑value heads per layer, each of dimension \(d_h\). The per‑head weights do not increase the cache size. The attention computation remains \(O(B H L^2 d_h)\), but with the same FLOPs as GQA. The additional memory for the weights is negligible.

---

## 3. Implementation Details

- **Class `WeightedGroupedQueryAttention`** is registered under `"wgqa"` in the `ATTENTION_REGISTRY`.
- **Constructor**:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `num_kv_heads`: number of KV heads \(K\) (default 4).
  - Initialises `k_weights` and `v_weights` as parameters of shape \((K, 1)\), filled with ones.
  - Creates standard linear projections for Q, K, V, O.
- **Forward**:
  1. Project Q, K, V and reshape to multi‑head.
  2. Apply per‑head weights: multiply K and V by their respective weight vectors (broadcast).
  3. Repeat K and V to match H heads.
  4. Transpose to \((B, H, L, d_h)\).
  5. Compute attention scores, apply mask (if provided; expects additive mask), softmax, dropout.
  6. Aggregate values, concatenate heads, output projection.
  7. Return `(output, attn_probs)`.

- **Mask handling**: The code unsqueezes the mask to \((B, 1, L)\) (or \((B, 1, L, L)\)? It does `attention_mask.unsqueeze(1)`, which for a 2D mask gives \((B, 1, L)\), then it is added to scores of shape \((B, H, L, L)\) via broadcasting. This works if the mask values are 0 for valid and \(-\infty\) for padding. The module does not convert a binary mask to additive; it expects the mask to be pre‑processed.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"wgqa"` in the compression configuration. Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_kv_heads` | 4 | Number of KV heads \(K\). |
| `dropout` | (from config) | Dropout rate on attention weights. |

Example:
```python
comp_config.attention_type = "wgqa"
comp_config.num_kv_heads = 8
```

No pretrained weight transfer is performed in the snippet; an injection function would need to copy the original Q, K, V, O weights and possibly initialise the importance weights appropriately (e.g., from the original grouped pattern if available). The module can be trained from scratch or after copying.

---

## 5. Discussion

WGQA adds a minimal yet potentially powerful degree of freedom to the grouped‑query attention. By learning head‑wise importance, the model can:

- **Suppress redundant heads**: If a key‑value head is found to be uninformative, its weight can be driven toward zero, effectively pruning it.
- **Emphasise salient heads**: Important heads can be amplified, giving them more influence in the attention scores.
- **Dynamically adjust during training**: The weights can learn to adapt to the target task, potentially outperforming standard GQA where all heads are treated equally.

The design is extremely lightweight and introduces no overhead during inference beyond two scalar multiplications per head. The weights could also be interpreted as a form of soft head pruning, which may improve generalisation by regularising the importance of heads.

However, a few points should be considered:

- **Unconstrained parameters**: Without a non‑negativity constraint, weights can become negative, which might lead to unstable training. A sigmoid or softplus activation could be added to keep weights positive, though the current initialisation at 1 and typical gradient updates may keep them in a reasonable range.
- **Fixed grouping**: The grouping is still fixed (regular repetition). More flexible dynamic grouping, as in QCQA (Quality‑Capacity‑Aware GQA), could be combined with the weighting idea for even greater adaptivity.
- **Mask format dependency**: As with many modules, the mask must be provided in additive form, which may require a pre‑processing step if the raw masks are binary.

Overall, WGQA is a simple yet effective extension to GQA that can serve as a baseline for learning head importance in efficient attention mechanisms. It integrates seamlessly into the compression framework and can be used as a drop‑in replacement for standard attention in DistilBERT.

---

## 6. Conclusion

We have presented Weighted Grouped‑Query Attention (WGQA), which augments grouped‑query attention with learnable per‑head importance weights for keys and values. The module maintains the memory efficiency of GQA while allowing the model to adaptively scale the contribution of each key‑value head. Its simplicity and negligible parameter overhead make it a practical addition to the family of efficient attention methods. Future work could explore applying similar weighting to query heads, incorporating dynamic gating, or combining with other attention compression techniques. WGQA is ready for use within the DistilBERT compression framework.