# 1. Introduction

## 1.1 Motivation

The softmax attention operation computes a pairwise score between every 
query and every key:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

This requires $\mathcal{O}(L^2 d)$ time and $\mathcal{O}(L^2)$ memory for 
sequence length $L$. For $L = 4096$ (a common long-context setting), the 
attention matrix alone requires 64 MB per head, per layer, in FP16.

**Linear attention** reduces this to $\mathcal{O}(L d)$ or 
$\mathcal{O}(L \log L)$ by either restricting attention (sparse, local 
windows) or approximating the softmax kernel. Both approaches come with 
trade-offs:

- **Sparse attention** (Longformer, BigBird) captures local structure well 
  but loses global context.
- **Kernel attention** (Performer, Linear Transformer) preserves global 
  context but approximates softmax, which is problematic when the true 
  attention is peaked.

**Cluster-based attention** offers a third path. Instead of comparing 
individual tokens, group them into $K$ clusters, compute attention at the 
**cluster level**, and redistribute the results. If $K \ll L$, this is much 
cheaper than dense attention while preserving **global reach** (every 
cluster attends to every key).

## 1.2 Prior Work on Clustering Attention

**Routing Transformer** (Roy et al., 2021) clusters tokens and restricts 
attention within each cluster. Its complexity is $\mathcal{O}(L \cdot W)$ 
but requires a large cluster size $W$ to maintain context, negating the 
savings.

**Reformer** (Kitaev et al., 2020) uses locality-sensitive hashing (LSH) 
to approximate attention. Its complexity depends on hash collisions and is 
not guaranteed to be linear.

**Cluster-Former** (Wang et al., 2021) uses clustering for document-level 
tasks but applies it at a coarse granularity (sentence-level).

**ClusterFormer** differs by computing **centroid-to-key** attention 
directly, giving exact global context at $\mathcal{O}(K L d)$ complexity.

## 1.3 Problem Statement

Given a pretrained attention module with projections $W_Q, W_K, W_V, W_O$, 
we seek a modified attention function $\mathcal{A}'$ such that:

1. **Subquadratic complexity.** $\mathcal{A}'$ runs in $\mathcal{O}(K L d)$ 
   with $K \ll L$.
2. **Global context.** Every query attends to a representation of the whole 
   sequence, not just a window.
3. **Local structure.** Nearby syntactic dependencies are preserved.
4. **Step-0 preservation.** With zero-init LoRA adapters, the initial forward 
   pass matches the pretrained model.
5. **Multi-architecture support.** Works on flat, nested, and causal 
   transformers.

## 1.4 Our Approach

ClusterFormer computes attention in three steps:

**Step 1: Cluster queries.** Apply fast differentiable K-means to the query 
representations $Q \in \mathbb{R}^{B \times H \times L \times d_h}$, 
producing $K$ centroids $C \in \mathbb{R}^{B \times H \times K \times d_h}$.

**Step 2: Centroid-to-key attention.** Compute attention from each centroid 
to all keys:

$$
V_{\text{cent}} = \text{softmax}\!\left(\frac{C K^\top}{\sqrt{d_h}}\right) V \in \mathbb{R}^{B \times H \times K \times d_h}
\tag{1.2}
$$

This costs $\mathcal{O}(K L d_h)$ — linear in $L$.

**Step 3: Redistribute.** Assign each query to its nearest centroid and 
collect the corresponding context:

$$
\text{Global}(Q) = \text{softmax}\!\left(\frac{Q C^\top}{\sqrt{d_h} \cdot \tau}\right) V_{\text{cent}}
\tag{1.3}
$$

where $\tau$ is a learnable temperature.

To capture local structure missed by clustering, we add a sliding-window 
attention branch and blend via a learnable gate $\alpha$:

$$
\text{Output} = (1 - \alpha) \cdot \text{Global}(Q) + \alpha \cdot \text{Local}(Q)
\tag{1.4}
$$

## 1.5 Contributions

- **Centroid-to-key cross-attention.** A novel formulation of cluster 
  attention that computes exact global context at linear cost. Section 4.3.
- **Fast differentiable K-means.** A vectorized Lloyd algorithm with 
  padding-aware initialization and 3 iterations, sufficient for attention 
  clustering. Section 4.2.
- **Learnable temperature and blend.** A temperature parameter $\tau$ for 
  assignment sharpening and a blend scalar $\alpha$ for global-local fusion, 
  both learned during recovery. Section 4.4.
- **Broad architecture support.** Multi-architecture weight extraction plus 
  GQA, RoPE, nested encoder bypass, and causal masking. Section 5.
- **Systematic evaluation.** Three models, SST-2, ablations over $K$, 
  window size, and blend initialization. Section 7.
- **Honest disclosure.** Parameter count increases due to LoRA and the 
  learnable scalars; we report this transparently. Section 7.2.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts 
  released under MIT.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention, 
K-means, and linear attention. Section 4 presents ClusterFormer. Section 5 
describes architecture integration. Section 6 details experimental setup. 
Section 7 reports results. Section 8 analyzes complexity, clustering quality, 
and ablation contributions. Section 9 discusses usage. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $K$ | Number of clusters |
| $W$ | Sliding window radius |
| $\tau$ | Learnable temperature |
| $\alpha$ | Learnable blend scalar |
| $C$ | Cluster centroids |
| $V_\text{cent}$ | Centroid contexts |
| $r_L$ | LoRA rank |

---