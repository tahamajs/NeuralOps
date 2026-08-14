# ClusterFormer: Token Clustering Attention

## File Reference
`./cluster_attention.py`

## Overview
**ClusterFormer** (`ClusterAttention`) partitions token sequence representations into $K$ semantic clusters using batch-wise iterative $K$-means clustering. Dense scaled dot-product attention is subsequently executed strictly within tokens sharing the same cluster assignment, reducing attention matrix evaluations and enabling sparse structural token interactions.

---

## Mathematical Formulation

### 1. Batch Iterative $K$-Means Clustering
Given sequence embeddings $X \in \mathbb{R}^{B \times L \times D}$, initial centroids $\mu_k^{(0)}$ are sampled from input sequence positions. For $I$ Lloyd's algorithm iterations:
$$\text{dist}(X_i, \mu_k) = \|X_{b,i,:} - \mu_{b,k,:}\|_2$$
$$c_i = \arg\min_{k \in \{1, \dots, K\}} \text{dist}(X_i, \mu_k)$$
$$\mu_k^{(t+1)} = \frac{1}{|C_k|} \sum_{i \in C_k} X_{b,i,:}$$

### 2. Intra-Cluster Partitioned Attention
For each partition $C_k = \{i \in \{1, \dots, L\} : c_i = k\}$:
$$S_{ij}^{(k)} = \frac{Q_i K_j^\top}{\sqrt{d_h}} + M_{ij}^{(k)}$$
$$M_{ij}^{(k)} = \begin{cases} 0 & \text{if } c_j = k \\ -\infty & \text{otherwise} \end{cases}$$
$$A^{(k)} = \text{Softmax}(S^{(k)})$$
$$O_i = \sum_{j=1}^L A_{ij}^{(k)} V_j$$

---

## Computational & Space Complexity
- **Clustering Overhead:** $\mathcal{O}(I \cdot B \cdot L \cdot K \cdot D)$ where $I$ is the number of $K$-means iterations.
- **Attention Matrix Evaluation:** $\mathcal{O}\left(B \cdot H \cdot \frac{L^2}{K} \cdot d_h\right)$ per forward pass, providing an effective $K\times$ speedup over standard $O(L^2)$ attention.

---

## Usage Example

```python
import torch
from attentions.cluster_attention import ClusterAttention


class Config:

  def __init__(self):
    self.hidden_size = 768
    self.num_attention_heads = 12
    self.dropout = 0.1


config = Config()
attn = ClusterAttention(config=config, num_clusters=16, n_iters=3)

x = torch.randn(2, 128, 768)
out, attn_map = attn(x)
print(out.shape)  # Output: torch.Size([2, 128, 768])
```

---

## Technical Features & Performance Trade-offs
- **Dynamic Structural Sparsity:** Tokens dynamically group by semantic affinity rather than static positional windows.
- **Scalable Long-Contexts:** Reduces quadratic memory walls for long sequence lengths $L \gg 1024$.
