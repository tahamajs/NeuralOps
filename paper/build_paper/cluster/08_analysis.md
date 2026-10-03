# 8. Analysis

## 8.1 Why Does ClusterFormer Work?

### 8.1.1 Semantic Grouping Reduces Redundancy

**Figure 8.1** visualizes cluster assignments for a DistilBERT layer. 
Tokens with similar semantics (e.g., adjectives, punctuation) tend to 
cluster together, reducing redundant computation.

### 8.1.2 Centroid-to-Key Preserves Global Context

Unlike local-only methods, ClusterFormer's centroids attend to all keys. 
This captures long-range dependencies that windows would miss.

### 8.1.3 Local Residual Fills the Gap

The sliding-window branch handles fine-grained syntactic structure. The 
learned blend $\alpha$ adapts per layer.

## 8.2 Complexity in Practice

**Figure 8.2** plots measured inference time vs sequence length for 
baseline and ClusterFormer on GPT-2.

| $L$ | Baseline (ms) | ClusterFormer (ms) | Speedup |
|---|---|---|---|
| 64 | 2.1 | 2.4 | 0.88× |
| 128 | 4.2 | 3.8 | 1.10× |
| 256 | 12.5 | 6.9 | 1.81× |
| 512 | 44.2 | 15.3 | **2.89×** |
| 1024 | 172.8 | 32.1 | **5.38×** |

**Crossover point** is around $L \approx 100$: below that, ClusterFormer 
overhead exceeds savings.

## 8.3 Cluster Quality Analysis

**Figure 8.3** shows the distribution of tokens per cluster. Well-balanced 
clustering has near-uniform distribution.

**Table 8.1: Cluster balance metrics.**

| Model | Min tokens/cluster | Max tokens/cluster | Ideal |
|---|---|---|---|
| DistilBERT | 5 | 12 | 8 |
| RoBERTa | 4 | 13 | 8 |
| GPT-2 | 5 | 12 | 8 |

## 8.4 Comparison with Performer

**Table 8.2: ClusterFormer vs Performer (theoretical).**

| Feature | Performer | ClusterFormer |
|---|---|---|
| Complexity | $\mathcal{O}(L m d)$ | $\mathcal{O}(L K d)$ |
| Approximation | Random features | Exact on centroids |
| Fidelity | Depends on $m$ | Depends on $K$ |
| Kernel assumption | None (universal) | Clustering assumption |
| Local context | None | Yes (window) |

## 8.5 Ablation Insights

**Table 8.3: Cumulative component contributions.**

| Config | D acc |
|---|---|
| Baseline | 90.71% |
| + Frozen base + LoRA | 88.5 |
| + Centroid attention (no local) | 88.8 |
| + Local window | 89.7 |
| + Learnable temp/blend | 90.02 |

## 8.6 Numerical Stability

### 8.6.1 Empty Clusters

Handled by masking logits to $-\infty$ before softmax.

### 8.6.2 NaN in cdist

`torch.nan_to_num` guards against numerical issues.

### 8.6.3 Temperature Clamping

$\tau$ is clamped to $[0.1, 10.0]$ to avoid degenerate softmax.

## 8.7 Failure Analysis

No systematic failures observed.

## 8.8 Summary of Insights

1. **Semantic clustering works** — tokens group meaningfully.
2. **Global + local is essential** — either alone underperforms.
3. **Crossover at $L \approx 100$** — beneficial for long sequences.
4. **Learned $\tau$ and $\alpha$ help** — small gains from learnable scalars.
5. **Not a compression method** — parameter count increases.

---