# ClusterFormer Attention: Token Clustering for Efficient Global Attention

**Abstract**

We introduce **ClusterFormer Attention**, an efficient self‑attention module that reduces the quadratic complexity of standard attention by partitioning tokens into a fixed number of clusters via differentiable *k‑means* clustering, and then performing full attention exclusively within each cluster. By grouping semantically similar tokens, the module retains the ability to model long‑range dependencies—tokens from distant positions can be clustered together—while sharply limiting the number of pairwise comparisons. The clustering is soft and computed per layer, using the hidden states themselves as features. The module is designed as a drop‑in replacement for standard self‑attention in Transformer models, and is well‑suited for long‑sequence tasks where global context is important but full attention is computationally prohibitive.

---

## 1. Introduction

Standard multi‑head self‑attention has \(O(N^2)\) time and memory complexity for sequences of length \(N\). Numerous approaches have sought to alleviate this bottleneck, including local windowing, kernelised linear attention, and token‑pruning methods. However, many of these methods sacrifice direct global interaction. **ClusterFormer Attention** takes a different route: it dynamically groups tokens into a small number of clusters (e.g., 64) using a fast, iterative *k‑means* algorithm, and then lets each token attend only to other tokens within the same cluster. Because clustering is based on learned hidden‑state representations, tokens that are semantically related but spatially distant can end up in the same cluster, thereby enabling global information exchange without computing all \(O(N^2)\) pairwise scores.

The cluster assignments are computed per forward pass directly from the hidden states, making the mechanism adaptive to the input. The attention computation becomes \(O(N C^2 / C) = O(N C)\) where \(C\) is the average cluster size, but because the clusters are balanced, the total cost is approximately \(O(N^2 / K)\), where \(K\) is the number of clusters. For a fixed \(K\), this yields linear complexity in \(N\).

In this paper, we describe the architecture and mathematical underpinnings of the ClusterFormer Attention module, discuss its implementation details, and analyse its computational properties.

---

## 2. Method

### 2.1 Token Clustering via *k‑means*

Given an input sequence of hidden states \(\mathbf{H} \in \mathbb{R}^{B \times L \times D}\), the first step is to partition the \(L\) tokens into \(K\) clusters. We use a simple batched *k‑means* algorithm with \(T\) iterations (default \(T = 3\)). 

Initialisation: For each batch element, \(K\) initial centroids \(\mathbf{C}^{(0)} \in \mathbb{R}^{B \times K \times D}\) are selected by randomly sampling \(K\) token positions without replacement.

At each iteration \(t = 1, \dots, T\):
1. **Assignment**: For each token \(i\), its cluster \(a_i\) is the index of the nearest centroid in Euclidean distance:

\[
a_i = \arg\min_{k \in \{1,\dots,K\}} \|\mathbf{h}_i - \mathbf{c}_k\|_2^2.
\]

This is done efficiently using `torch.cdist`.

2. **Update**: Each centroid is updated to the mean of the hidden states assigned to it:

\[
\mathbf{c}_k = \frac{\sum_{i: a_i = k} \mathbf{h}_i}{\max(1, \sum_{i: a_i = k} 1)}.
\]

After \(T\) iterations, we obtain cluster assignments \(\mathbf{A} \in \{1,\dots,K\}^{B \times L}\) and final centroids \(\mathbf{C} \in \mathbb{R}^{B \times K \times D}\).

### 2.2 Multi‑Head Projections

In parallel, the hidden states are linearly projected to queries, keys, and values for each attention head:

\[
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q^\top, \quad
\mathbf{K} = \mathbf{H} \mathbf{W}_K^\top, \quad
\mathbf{V} = \mathbf{H} \mathbf{W}_V^\top,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times (H \cdot d_h)}\), and then reshaped to \((B, L, H, d_h)\). All heads share the same cluster assignments, which are computed from \(\mathbf{H}\) (not from the per‑head projections).

### 2.3 Intra‑Cluster Attention

For each cluster \(k = 1, \dots, K\):

- Let \(\mathcal{S}_k = \{ i \mid a_i = k \}\) be the set of tokens assigned to cluster \(k\).
- Extract the corresponding query, key, and value slices: \(\mathbf{Q}_k, \mathbf{K}_k, \mathbf{V}_k \in \mathbb{R}^{B \times H \times |\mathcal{S}_k| \times d_h}\). Tokens not in the cluster are zeroed out for simplicity in implementation.
- Compute scaled dot‑product attention **only among the tokens in \(\mathcal{S}_k\)**:

\[
\mathbf{S}_k = \frac{\mathbf{Q}_k \mathbf{K}_k^\top}{\sqrt{d_h}} + \mathbf{M}_k,
\]

where \(\mathbf{M}_k\) is an additive mask that sets positions outside the cluster to \(-\infty\) (i.e., for token pairs where either the query or key is not in cluster \(k\)). In practice, this is done by constructing a boolean mask from the cluster assignments.

- Attention weights: \(\mathbf{A}_k = \operatorname{softmax}(\mathbf{S}_k)\).
- Output for cluster \(k\): \(\mathbf{O}_k = \mathbf{A}_k \mathbf{V}_k\).

The final output for the whole sequence is the sum of the outputs from all clusters:

\[
\mathbf{O} = \sum_{k=1}^{K} \mathbf{O}_k.
\]

Because each token belongs to exactly one cluster, only one term in the sum is non‑zero per token. This sum is equivalent to concatenating per‑cluster outputs and scattering them back to their original positions.

### 2.4 Output Projection

The aggregated output \(\mathbf{O} \in \mathbb{R}^{B \times H \times L \times d_h}\) is rearranged to \((B, L, H d_h)\) and multiplied by the output weight \(\mathbf{W}_O \in \mathbb{R}^{H d_h \times D}\) to produce the final attention output.

---

## 3. Complexity Analysis

Let \(K\) be the number of clusters. With *k‑means*, the assignment step costs \(O(L K D)\) per iteration, and the update step costs \(O(L D)\). For \(T\) iterations, clustering is \(O(T L K D)\). In the attention step, within a cluster of size \(C_k \approx L / K\), the attention cost is \(O(C_k^2 D) = O((L^2 / K^2) D)\). Summing over all \(K\) clusters gives \(O(L^2 D / K)\). The total per‑layer complexity is \(O(T L K D + L^2 D / K)\). For very long sequences where \(L\) dominates, the second term is the main cost, which is reduced by a factor of \(K\) compared to full \(O(L^2 D)\) attention. With \(K\) in the hundreds, significant speedups are achieved while still allowing any token to interact with any other token, as long as they are assigned to the same cluster.

---

## 4. Implementation Details

The `ClusterAttention` module inherits from `nn.Module` and is registered under the name `"cluster"` in the `ATTENTION_REGISTRY`.

- **Constructor**: Takes `config` (model configuration), `num_clusters` (default 64), and `n_iters` (default 3). Creates four linear projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and dropout.
- **`_kmeans(x, iters)`**: Implements the batched *k‑means* described above. Returns centroids and cluster IDs. Important: the centroid update loop uses a Python `for` loop over clusters, which can be slow for large \(K\). In practice, a fully vectorized version using scatter operations would be faster.
- **Forward pass**:
  1. Project hidden states to Q, K, V.
  2. Run *k‑means* on the hidden states to obtain `cluster_ids` of shape \((B, L)\).
  3. Iterate over \(K\) clusters:
     - Build a mask for tokens belonging to cluster \(k\).
     - Zero out Q, K, V for non‑members (or use masking in attention scores).
     - Compute attention scores, add additive mask where `cluster_ids != k`.
     - Apply softmax, dropout, and multiply with V.
     - Add to output.
  4. Reshape and apply output projection.

Note: The current implementation loops over clusters in Python, which is feasible for moderate \(K\) (e.g., 64) but could become a bottleneck. The per‑cluster attention uses full \(O(|S_k|^2)\) attention, which is acceptable because cluster sizes are roughly \(L/K\). No causal masking is applied; for autoregressive decoding, a causal variant would need to be added.

Masking of padding tokens is not explicitly handled; the input `attention_mask` is accepted but not used. For padded sequences, padding tokens would be clustered and could introduce noise. A proper integration would mask them out in the *k‑means* and attention steps.

---

## 5. Usage and Configuration

The module is injected through the framework with:

```python
comp_config.num_clusters = 64
comp_config.n_iters = 3
```

or manually:

```python
attn = ClusterAttention(
    config=model_config,
    num_clusters=64,
    n_iters=3
)
layer.attention = attn
```

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clusters` | 64 | Number of clusters for partitioning tokens. |
| `n_iters` | 3 | Number of k‑means iterations. |

---

## 6. Discussion

ClusterFormer Attention offers a fresh perspective on efficient attention. By clustering tokens before attention, it retains a fully dense attention pattern within each cluster, which can capture rich interactions, while dramatically reducing the total number of comparisons. The method is inherently content‑adaptive: clusters are formed based on the current hidden states, allowing the model to dynamically route information. This is in contrast to fixed‑pattern sparsity (e.g., local windows, strided patterns) which cannot adapt to the content.

However, the module has limitations. The *k‑means* algorithm, despite being fast in practice, adds overhead per layer. The use of hard cluster assignments means that tokens in different clusters cannot interact directly, which may limit global information flow compared to methods that use soft attention over a global set of representatives (e.g., set Transformers, Perceiver). The quality of clustering may degrade if the hidden states are not well‑suited for Euclidean distance; using a learned distance metric could improve performance. Finally, the current implementation does not handle padding or causal masking, limiting its immediate applicability to certain tasks.

Despite these caveats, ClusterFormer provides a simple and interpretable building block for scaling Transformers to longer sequences, and its plug‑and‑play nature facilitates rapid experimentation.

---

## 7. Conclusion

We have presented ClusterFormer Attention, an efficient self‑attention module that uses *k‑means* clustering to group tokens and restricts attention to within‑cluster interactions. This reduces the complexity from \(O(N^2)\) to \(O(N^2 / K + NK)\), enabling processing of longer sequences while preserving the ability to form global token relationships. The module is easy to integrate into existing Transformer architectures and offers a tunable trade‑off between speed and accuracy via the number of clusters. Future work will extend the method with soft clustering, causal masking, and padding support, as well as explore learned clustering strategies.