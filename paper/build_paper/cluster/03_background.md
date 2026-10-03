# 3. Background

## 3.1 Multi-Head Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V \in \mathbb{R}^{B \times L \times d}
\tag{3.1}
$$

Standard attention:

$$
\text{Attn}(X) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

## 3.2 K-Means Clustering

Given $L$ points $x_1, \dots, x_L \in \mathbb{R}^d$, K-means minimizes:

$$
\mathcal{L}(C) = \sum_{i=1}^{L} \min_{k \in [K]} \|x_i - c_k\|_2^2
\tag{3.3}
$$

Lloyd's algorithm alternates:

1. **Assignment:** $a_i = \arg\min_k \|x_i - c_k\|_2^2$
2. **Update:** $c_k = \frac{\sum_{i: a_i = k} x_i}{|a_i = k|}$

**Complexity.** $\mathcal{O}(L K d)$ per iteration.

**Vectorization.** Both steps parallelize over $B, H$ in attention context.

## 3.3 Differentiable K-Means

Standard K-means uses `argmin`, which is non-differentiable. For 
end-to-end training:

- **Forward:** Use `argmin` to get hard assignments (for the update step).
- **Backward:** Use softmax for gradient flow through assignments.

ClusterFormer is fully differentiable:
- Centroid updates use one-hot (non-differentiable) — this is fine since 
  centroids are deterministic functions of $Q$.
- Assignment uses `softmax` with learnable temperature, giving gradients.

## 3.4 Linear Attention

A generic linear attention reformulation:

$$
\text{Attn}(Q, K, V) = \frac{\phi(Q) \sum_j \phi(K_j) V_j^\top}{\phi(Q) \sum_j \phi(K_j)}
\tag{3.4}
$$

for some feature map $\phi$. If $\phi$ has dimension $m$, complexity is 
$\mathcal{O}(L m d)$.

ClusterFormer does not use this formulation. Instead, it clusters queries 
and computes attention from $K$ centroids to keys.

## 3.5 LoRA Adapters

LoRA (Hu et al., 2021) parameterizes:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A, \quad B = 0 \text{ at init}
\tag{3.5}
$$

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $C$ | Centroid matrix $[B, H, K, d_h]$ |
| $V_\text{cent}$ | Centroid context $[B, H, K, d_h]$ |
| $a_i$ | Cluster assignment for query $i$ |
| $\tau$ | Temperature |
| $\alpha$ | Blend |
| $W$ | Window radius |

---