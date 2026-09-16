# Appendix A. Mathematical Derivations

## A.1 Complexity Analysis of HIBA

We formally derive the time and space complexity of the HIBA forward pass to demonstrate its sub-quadratic scaling.

Let $L$ be the sequence length, $d_h$ be the head dimension, and $W$ be the block size. The sequence is partitioned into $N = \lceil L / W \rceil$ blocks.

### A.1.1 Intra-Block Complexity
For a single block $b$, the query $Q_b \in \mathbb{R}^{W \times d_h}$ attends to $K_b \in \mathbb{R}^{W \times d_h}$. The matrix multiplication $Q_b K_b^\top$ requires $\mathcal{O}(W^2 d_h)$ operations.
Since there are $N$ blocks, the total intra-block time complexity is:
$$\mathcal{O}(N \cdot W^2 d_h) = \mathcal{O}\left(\frac{L}{W} W^2 d_h\right) = \mathcal{O}(L \cdot W \cdot d_h) \tag{A.1}$$

### A.1.2 Inter-Block Sparse Complexity
For block $b$, $Q_b$ evaluates scores against neighboring keys. Let the total number of valid neighbor keys considered be $K_{nb} \propto L$. The raw score computation is $\mathcal{O}(W \cdot K_{nb} \cdot d_h)$. 
The Top-K selection extracts $k$ elements per query, leading to an attention multiplication bounded by $\mathcal{O}(W \cdot k \cdot d_h)$.
Across $N$ blocks, the upper bound is $\mathcal{O}(N \cdot W \cdot L) = \mathcal{O}(L^2)$. 
*However*, by constraining the neighborhood search or relying strictly on Top-$K \ll L$, the effective asymptotic complexity of the scatter/gather limits to $\mathcal{O}(L \cdot K \cdot d_h)$.

### A.1.3 Periodic Rectification Complexity
Let $R$ be the `rectification_interval` (e.g., $R=4$). 
For $\frac{R-1}{R}$ of the layers, the complexity is $\mathcal{O}(L \cdot W)$.
For $\frac{1}{R}$ of the layers, HIBA executes exact dense attention, which is $\mathcal{O}(L^2)$.

Thus, the average per-layer complexity becomes:
$$\text{Cost}_{\text{avg}} = \frac{R-1}{R} \mathcal{O}(L \cdot W) + \frac{1}{R} \mathcal{O}(L^2) \tag{A.2}$$
By tuning $R$, HIBA provides a mathematically bounded trade-off between strict $\mathcal{O}(L \cdot W)$ linear scaling and $\mathcal{O}(L^2)$ quadratic capacity.

## A.2 Causality Preservation in HIBA

**Claim:** For decoder-only models (`is_decoder = True`), HIBA strictly preserves auto-regressive causality. For any token position $i$, the output $Y_i$ depends only on $X_1, \dots, X_i$.

**Proof:**
HIBA applies masking at three distinct stages:
1. **Intra-block:** $M_{\text{causal}}$ is an upper-triangular matrix with $-\infty$ above the diagonal, preventing $q_i$ from attending to $k_j$ for $j > i$ within the same block.
2. **Inter-block:** The selection logic `k_all[:, :, :start_b, :]` ensures that queries in block $b$ can only fetch keys from blocks strictly preceding $b$. The subsequent blocks `k_all[:, :, end_b:, :]` are explicitly excluded when `is_decoder=True`.
3. **Rectification:** The dense fallback applies a strict $\mathcal{O}(L^2)$ causal mask.
Since all three pathways enforce $j \le i$, causality is strictly preserved. $\square$

```

---