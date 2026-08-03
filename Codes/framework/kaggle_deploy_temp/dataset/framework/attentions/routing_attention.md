# Routing Attention: Dynamic Clustering for Sparse Self‑Attention

**Abstract**

We present **Routing Attention**, a sparse self‑attention module that partitions the query tokens into a fixed number of clusters via a lightweight, iterative k‑means assignment, and then performs exact dot‑product attention **within each cluster**. The clustering is based on the query vectors themselves, ensuring that tokens with similar query patterns attend to the same set of keys. This content‑adaptive sparsity reduces the effective number of pairwise comparisons from \(O(L^2)\) to \(O(L K)\), where \(K\) is the number of clusters (e.g., 32). The cluster centroids are initialised from the first \(K\) query vectors and refined for a few iterations, making the method fast and differentiable in the forward pass (with a straight‑through approximation for the argmin). The module is designed as a drop‑in replacement for the self‑attention layer in DistilBERT, integrating seamlessly with the compression framework via the registry. This paper provides a rigorous mathematical description of the clustering algorithm, the attention mechanism, and the overall architecture.

---

## 1. Introduction

Standard multi‑head self‑attention incurs a quadratic cost in the sequence length \(L\) because every query attends to every key. To reduce this burden, one can restrict attention to a subset of keys, either via fixed patterns (local windows) or via content‑based sparsity. **Routing Attention** adopts a **content‑based clustering approach**: the query vectors are grouped into \(K\) clusters, and each token attends only to the keys and values belonging to the same cluster. The underlying hypothesis is that queries that are similar (in the sense of being close in Euclidean distance) are likely to attend to similar sets of keys; therefore, clustering queries and performing intra‑cluster attention is a good approximation to full attention.

The clustering is performed by a simplified k‑means algorithm executed directly on the query tensors. After a few iterations (default 3), each query is assigned to the nearest centroid. The attention computation is then carried out independently for each cluster, and the outputs are scattered back to the original positions. This approach is reminiscent of the **Reformer**’s LSH bucketing but uses a soft, differentiable clustering (though the argmin is not differentiable, gradients still flow to the queries via the cluster centroids and the within‑cluster attention). The module also shares the cluster structure across all heads (the centroids are the same for all heads? Actually, the clustering is performed separately per head, which is more flexible). The design is simple, avoids the need for a separate hashing function, and can adapt to the geometry of the queries.

In this paper, we detail the mathematical formulation of the k‑means clustering within the attention forward pass, the intra‑cluster attention, and the integration into the compression framework.

---

## 2. Method

### 2.1 Multi‑Head Projections

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) be the input hidden states. As usual, we project to queries, keys, and values for all heads:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with \(\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V \in \mathbb{R}^{D \times (H d_h)}\). The tensors are reshaped to \((B, H, L, d_h)\) and transposed to the head‑first layout \((B, H, L, d_h)\).

### 2.2 Centroid Initialisation and Iterative Refinement

The clustering is performed **independently for each head**. For a given head, let its query tensor be \(\mathbf{Q}_h \in \mathbb{R}^{B \times L \times d_h}\) (we omit the head index for clarity but keep the batch). The number of clusters is \(K = \min(K_{\text{max}}, L)\), where \(K_{\text{max}}\) is the `num_clusters` hyperparameter (default 32). The initial centroids are taken as the first \(K\) query tokens along the sequence dimension:

\[
\mathbf{c}_k^{(0)} = \mathbf{Q}_{[:, k, :]} \quad \text{for } k = 0, \dots, K-1.
\]

Thus, the centroids have shape \((B, K, d_h)\).

For each iteration \(t = 1, \dots, T\) (where \(T\) is `n_iters`, default 3):

1. **Assignment**: Each query is assigned to the nearest centroid based on Euclidean distance:
   \[
   a_{i}^{(t)} = \arg\min_{k \in \{1,\dots,K\}} \|\mathbf{q}_i - \mathbf{c}_k^{(t-1)}\|_2.
   \]
   This is implemented via `torch.cdist` and `torch.argmin`.

2. **Update**: Each centroid is updated to the mean of the queries assigned to it:
   \[
   \mathbf{c}_k^{(t)} = \frac{\sum_{i: a_i = k} \mathbf{q}_i}{\max(1, \sum_{i: a_i = k} 1)}.
   \]

The final assignment after \(T\) iterations is used for the attention mask.

### 2.3 Intra‑Cluster Attention

After the final cluster assignments \(\mathbf{A} \in \{0,\dots,K-1\}^{B \times H \times L}\), the output is computed cluster by cluster. For each cluster \(k\):

- A binary mask \(\mathbf{M}_k \in \{0,1\}^{B \times H \times L \times 1}\) is formed, where \(M_{k,b,h,i} = 1\) iff token \(i\) belongs to cluster \(k\).
- The queries, keys, and values are masked by multiplying element‑wise with \(\mathbf{M}_k\). This zeroes out tokens not in the cluster, though they remain in the tensors. (A more efficient implementation would gather only the relevant indices.)
- Scaled dot‑product attention is computed:
  \[
  \mathbf{S}_k = \frac{(\mathbf{Q} \odot \mathbf{M}_k) (\mathbf{K} \odot \mathbf{M}_k)^\top}{\sqrt{d_h}}.
  \]
  Because the non‑cluster tokens have zero queries/keys, their scores are zero, and after softmax they will be ignored.
- Softmax is applied, dropout is applied, and the context is obtained:
  \[
  \mathbf{C}_k = \operatorname{softmax}(\mathbf{S}_k) (\mathbf{V} \odot \mathbf{M}_k).
  \]
- The per‑cluster outputs are accumulated: \(\mathbf{C} = \sum_{k=1}^K \mathbf{C}_k\).

Since a token belongs to exactly one cluster, its contribution to the final output comes from exactly one term in the sum.

### 2.4 Output Projection

The aggregated context is reshaped to \((B, L, H d_h)\) and projected:

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

The module returns \((\mathbf{Y}, \text{None})\).

### 2.5 Complexity

The cost of k‑means is \(O(B H L K d_h \cdot T)\). The intra‑cluster attention cost is roughly \(O(B H \sum_{k} L_k^2 d_h)\), where \(L_k\) is the size of cluster \(k\). In the worst case, one cluster may contain all tokens, leading to \(O(L^2)\); but in practice, clusters tend to be balanced, giving approximately \(O(B H (L^2/K) d_h)\). For \(K \propto L\), this becomes \(O(L)\), but with a constant \(K=32\) it is quadratic with a reduced factor. This is similar to other clustering‑based sparse attention schemes.

---

## 3. Implementation Details

- **`RoutingAttention`** inherits from `nn.Module` and is registered as `"routing"` in the `ATTENTION_REGISTRY`.
- Constructor parameters:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `num_clusters` (default 32): the number of clusters \(K\).
  - `n_iters` (default 3): number of k‑means iterations.
  - Standard linear layers for Q, K, V, O, and dropout.
- **`_kmeans_assign`**: Given queries \(\mathbf{Q}\) of shape \((B, H, L, d_h)\) and centroids \((B, H, K, d_h)\), computes pairwise Euclidean distance via `torch.cdist` and returns the index of the closest centroid. This function is used both during the iterative update and for the final assignment.
- **Forward**:
  1. Project Q, K, V.
  2. Initialise centroids as first \(K\) query tokens along the length dimension.
  3. For `n_iters` iterations:
     a. Assign each query to nearest centroid.
     b. Update centroids by averaging queries in each cluster (using a loop over clusters).
  4. Compute final cluster assignments.
  5. Initialise output tensor to zeros.
  6. For each cluster \(k\):
     - Build boolean mask via `(cluster_ids == k).unsqueeze(-1)`.
     - If no tokens in this cluster, skip.
     - Multiply Q, K, V by mask (zeroing non‑members).
     - Compute attention scores, softmax, dropout, weighted sum of V.
     - Add to output.
  7. Reshape, output projection.
- The loop over clusters is implemented in Python, which is acceptable for \(K=32\) but could be slow for larger \(K\). The method would benefit from batched scatter/gather operations.

- **Mask handling**: The `attention_mask` parameter is accepted but **not used** in this implementation. Padding tokens could be placed in any cluster and would influence both the centroid computation and the attention. A production version would need to exclude padding tokens from k‑means and attention.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"routing"` in the compression config. The configurable hyperparameters are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clusters` | 32 | Number of clusters \(K\). |
| `n_iters` | 3 | Number of k‑means iterations. |
| `dropout` | (from config) | Dropout on attention weights. |

Example injection:
```python
comp_config.attention_type = "routing"
comp_config.num_clusters = 64
comp_config.n_iters = 5
```

Manual instantiation:
```python
attn = RoutingAttention(config=model_config, num_clusters=64, n_iters=5)
layer.attention = attn
```

No zero‑shot weight transfer is performed; linear layers are randomly initialised. For best results, fine‑tuning is required.

---

## 5. Discussion

Routing Attention offers an intriguing data‑driven sparsity pattern: by clustering queries, it dynamically adapts the attention pattern to the content. Tokens that are “looking for” similar information (as represented by their query vectors) will attend to the same subset of keys. This is more flexible than fixed‑window or strided patterns and does not require any additional learned parameters for the routing (the k‑means uses only the query vectors themselves).

However, there are several practical issues. The k‑means algorithm with `argmin` is not differentiable with respect to the queries; the centroids are updated but the assignment is discrete. While the attention within a cluster provides gradient signal to the queries (via the softmax), the clustering itself is fixed during the backward pass (straight‑through). This may hinder learning optimal query representations for clustering. A fully differentiable soft clustering (e.g., using Gumbel‑Softmax or EM‑like soft assignments) could be explored.

The current implementation processes clusters sequentially, which is inefficient for large \(K\). A batched version using scatter operations would be more suited for GPU. The lack of padding mask handling is a significant limitation; padding tokens can corrupt the centroids and cluster assignments. A fix would be to mask the queries before k‑means and only cluster valid tokens, which requires variable‑length handling.

Nevertheless, Routing Attention provides a clean example of combining classical clustering algorithms with deep learning attention mechanisms. It serves as a baseline for more advanced routing‑based sparse attention methods.

---

## 6. Conclusion

We have described Routing Attention, a self‑attention module that uses k‑means clustering of query vectors to partition the attention computation into intra‑cluster chunks. This content‑adaptive sparsity reduces the effective cost of attention while maintaining the ability to capture global dependencies across tokens with similar query patterns. Integrated into the DistilBERT compression framework, the module offers a simple yet conceptually interesting alternative to standard attention. Future work will focus on differentiable clustering, efficient cluster‑wise parallelisation, and proper padding mask integration.