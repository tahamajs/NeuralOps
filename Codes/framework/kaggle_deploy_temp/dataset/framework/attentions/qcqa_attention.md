


# QCQA: Quality & Capacity‑Aware Grouped Query Attention

**Abstract**

We introduce **Quality & Capacity‑Aware Grouped Query Attention (QCQA)**, a novel attention mechanism that generalises both Multi‑Query and Grouped Query Attention by allowing each query head to **dynamically soft‑select** which key‑value (KV) heads to attend to, rather than being permanently assigned to a fixed group. A lightweight router network, conditioned solely on the query vector, produces a softmax distribution over the available KV heads. The key and value tensors for each query head are then computed as a weighted mixture of the original KV heads. The rest of the attention pipeline remains the standard scaled dot‑product. This design retains the memory savings of using fewer KV heads while offering more flexibility than fixed groupings. The module is implemented as a drop‑in replacement for the self‑attention layer in DistilBERT and integrates with the compression framework via the registry.

---

## 1. Introduction

Multi‑head attention (MHA) stores and processes an independent key and value tensor for each of the \(H\) heads, which is costly during autoregressive decoding where KV caches dominate memory. Multi‑Query Attention (MQA) reduces this to a single shared KV head, while Grouped Query Attention (GQA) partitions the heads into groups that share a KV head. Both enforce a rigid assignment: a query head can only attend to the KV head in its group.

**QCQA** lifts this restriction by introducing a **learnable, per‑query routing mechanism**. For each query head, a small linear router takes the query vector and outputs a softmax score over the \(K\) KV heads. The effective key and value for that query head are then a weighted sum of the original KV heads. This allows different query heads to blend information from multiple KV heads in a content‑dependent manner, potentially overcoming the limitations of fixed grouping. At the same time, because only \(K\) KV heads are stored (with \(K \ll H\)), the KV cache size is reduced by a factor of \(H/K\), similar to GQA.

The router is extremely lightweight (a single linear projection per query head) and adds negligible cost. The mixing operation is batched and efficient. The result is a flexible, capacity‑aware attention variant that can be trained end‑to‑end.

---

## 2. Method

### 2.1 Projections

Given an input \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first projects to queries, keys, and values. Let \(H\) be the total number of query heads, \(K\) the number of KV heads (\(K \le H\)), and \(d_h = D/H\) the head dimension.

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{B \times L \times (H d_h)}, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{B \times L \times (K d_h)}, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{B \times L \times (K d_h)}.
\]

These are reshaped to separate heads: \(\mathbf{Q} \in \mathbb{R}^{B \times L \times H \times d_h}\), \(\mathbf{K} \in \mathbb{R}^{B \times L \times K \times d_h}\), \(\mathbf{V} \in \mathbb{R}^{B \times L \times K \times d_h}\).

### 2.2 Quality‑Aware Routing

A router network, which is a linear transformation \(\mathbf{W}_R \in \mathbb{R}^{d_h \times K}\) (without bias), maps each query vector \(\mathbf{q}_{b,l,h} \in \mathbb{R}^{d_h}\) to logits over the \(K\) KV heads:

\[
\mathbf{r}_{b,l,h} = \frac{(\mathbf{q}_{b,l,h} \mathbf{W}_R)}{\tau},
\]

where \(\tau\) is a temperature hyperparameter (default 1.0) that controls the sharpness of the routing. The routing weights are obtained via softmax:

\[
\boldsymbol{\alpha}_{b,l,h} = \operatorname{softmax}(\mathbf{r}_{b,l,h}) \in \mathbb{R}^{K}, \quad \sum_{j=1}^K \alpha_{b,l,h,j} = 1.
\]

These weights reflect the **quality** (relevance) of each KV head for that particular query head and token position.

### 2.3 Capacity‑Aware Mixing

For each query head \(h\), the effective key and value vectors are computed as the weighted sum of the \(K\) KV heads:

\[
\tilde{\mathbf{K}}_{b,l,h} = \sum_{j=1}^{K} \alpha_{b,l,h,j} \, \mathbf{K}_{b,l,j}, \qquad
\tilde{\mathbf{V}}_{b,l,h} = \sum_{j=1}^{K} \alpha_{b,l,h,j} \, \mathbf{V}_{b,l,j}.
\]

This mixing is performed for every token position and batch element. In practice, we flatten the batch and length dimensions, perform a batch matrix multiplication, and reshape:

\[
\mathbf{K}_{\text{flat}} \in \mathbb{R}^{(BL) \times K \times d_h},\quad
\boldsymbol{\alpha}_{\text{flat}} \in \mathbb{R}^{(BL) \times H \times K},
\]
\[
\tilde{\mathbf{K}}_{\text{flat}} = \boldsymbol{\alpha}_{\text{flat}} \mathbf{K}_{\text{flat}} \in \mathbb{R}^{(BL) \times H \times d_h},
\]

and similarly for \(\tilde{\mathbf{V}}\). The result is a mixed key/value per query head, of shape \((B, L, H, d_h)\).

### 2.4 Attention and Output

The mixed keys and values replace the original per‑head KV tensors in the standard attention computation. After transposing to the head‑first layout \((B, H, L, d_h)\),

\[
\mathbf{S} = \frac{\mathbf{Q} \tilde{\mathbf{K}}^\top}{\sqrt{d_h}}, \qquad
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}), \qquad
\mathbf{C} = \mathbf{A} \tilde{\mathbf{V}},
\]

where \(\mathbf{M}\) is an optional padding mask. Dropout is applied to the attention weights. The output is then concatenated and projected as usual.

### 2.5 Complexity and Memory

The KV cache during generation stores only \(K \times d_h\) features per token instead of \(H \times d_h\), giving a compression ratio of \(H/K\). The router adds a small per‑token overhead of \(O(H \cdot K \cdot d_h)\) for the softmax mixing, which is negligible for typical head counts. The attention complexity remains \(O(L^2 d_h H)\) as in standard MHA, but the method primarily targets KV cache reduction and flexible head assignment rather than attention matrix compression.

---

## 3. Implementation Details

- **`QCQAAttention`** is registered as `"qcqa"` in the `ATTENTION_REGISTRY`.
- **Constructor parameters**:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `num_kv_heads`: number of KV heads \(K\) (default 4).
  - `router_temp`: temperature \(\tau\) for the routing softmax (default 1.0).
- **Layers**:
  - `q_proj`: Linear(\(D, H d_h\)).
  - `k_proj`: Linear(\(D, K d_h\)).
  - `v_proj`: Linear(\(D, K d_h\)).
  - `router`: Linear(\(d_h, K\), bias=False) – applied independently per query head.
  - `out_proj`: Linear(\(H d_h, D\)).
  - `dropout`: on attention weights.
- **Forward**:
  1. Project Q (full heads), K and V (compressed heads).
  2. Reshape to separate heads.
  3. Compute route logits from Q using `self.router`, scale by temperature, softmax → routing weights.
  4. Flatten, mix K and V via batched matrix multiply (using `torch.bmm`).
  5. Transpose to \((B, H, L, d_h)\).
  6. Scaled dot‑product attention with mixed keys and values.
  7. Apply mask (assumes additive mask already, e.g., `0` for valid, `-inf` for padding; the code does `attention_mask.unsqueeze(1)` and adds to scores).
  8. Softmax, dropout, context computation.
  9. Reshape and output projection.
  10. Return `(output, attn_probs)`.

- **Mask handling**: The code expects `attention_mask` to be in an additive format (e.g., a float tensor with `0` for valid tokens and `-1e9` for padding) and adds it directly after unsqueezing to `(B, 1, L)` which broadcasts to `(B, H, L, L)`. This is a common convention but may not match the 2D binary masks often used in DistilBERT. A conversion might be needed in an injection function.

- **Router initialisation**: The router weight is default‑initialised (Kaiming uniform), so initially the mixing is roughly uniform. Temperature can be adjusted to make the routing sharper.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"qcqa"`. Configurable parameters via `comp_config` or constructor:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_kv_heads` | 4 | Number of KV heads \(K\). |
| `router_temp` | 1.0 | Temperature for the routing softmax. |

Example injection:
```python
comp_config.attention_type = "qcqa"
comp_config.num_kv_heads = 8
comp_config.router_temp = 0.5
```

No zero‑shot weight transfer is performed in this snippet; the projection weights are randomly initialised. For practical use, an injection function should copy the pretrained weights from the original attention (e.g., initialising the Q projection from the original `q_lin` and splitting/aggregating for K and V, or using a technique to initialise the router). As with many modules in this framework, that step is left to the surrounding pipeline.

---

## 5. Discussion

QCQA offers an elegant continuum between MHA, GQA, and MQA. With \(K=H\) and a hard routing (one‑hot), it reduces to MHA; with \(K=1\) and equal weights, it becomes MQA; with intermediate \(K\) and uniform weights, it approximates GQA with fixed groups. The learned, soft routing allows the model to discover an optimal sharing pattern for each task and even for each input token.

The mixing of KV heads via a weighted sum can be interpreted as creating a new virtual KV head that combines the representational strengths of several base heads. This is similar in spirit to dynamic convolution or mixture‑of‑experts in the key‑value space. The router is conditioned only on the query, making it lightweight and easy to compute.

Potential limitations:
- **No capacity constraints**: The softmax allows all KV heads to contribute, which could dilute the information. However, the temperature can be tuned, or entropy regularisation could be added to encourage sparser routing.
- **No shared KV cache across layers**: Each layer has its own set of KV heads; QCQA does not attempt cross‑layer sharing, though the learned routing could implicitly discover reusable patterns.
- **Mask handling**: As noted, the mask expectation may not align with the standard DistilBERT mask format, requiring an adapter in the injection function.
- **Initialisation**: Starting with a uniform routing may be suboptimal; pre‑training the router or initialising it to mimic a fixed GQA pattern could accelerate convergence.

Overall, QCQA is a compelling and simple extension of existing grouped attention schemes, providing dynamic, content‑aware head allocation while preserving the memory benefits of KV head reduction.

---

## 6. Conclusion

We have presented QCQA, a quality & capacity‑aware grouped query attention module that replaces fixed KV head grouping with a learned, per‑query soft routing mechanism. The design offers the same KV cache compression as GQA while enabling flexible, input‑dependent blending of key‑value information. The module is implemented as a drop‑in replacement for DistilBERT’s self‑attention and is integrated into the compression framework. Future work will address proper pretrained weight initialisation, mask conversion, and exploration of sparsity‑inducing routing regularisation. QCQA represents a step toward more adaptive and efficient attention mechanisms for large‑scale Transformers.



