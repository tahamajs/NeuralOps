# Factorised Matrix State Attention

## 1. Executive Abstract
**Factorised Matrix State Attention** replaces dense quadratic attention with an associative multi-head matrix memory updated via bounded decay recurrence and transformed by low-rank factorized bilinear output projection matrices ($W = L R^\top$).

---

## 2. Mathematical Equations & Recurrence

### 2.1 Multi-Head Projections & Hyperspherical Normalization
$$Q = X W_q, \quad K = X W_k, \quad V = X W_v$$
$$\tilde{q} = \text{RMSNorm}(Q), \quad \tilde{k} = \text{RMSNorm}(K) / \sqrt{d_k}$$

### 2.2 Bidirectional Associative Recurrence
With learnable decay $\lambda = \sigma(\beta) \in (0, 1)$:
$$\begin{aligned}
\vec{S}_t &= \lambda \vec{S}_{t-1} + \tilde{k}_t \otimes v_t, \quad \vec{z}_t = \lambda \vec{z}_{t-1} + \tilde{k}_t \\
\overleftarrow{S}_t &= \lambda \overleftarrow{S}_{t+1} + \tilde{k}_t \otimes v_t, \quad \overleftarrow{z}_t = \lambda \overleftarrow{z}_{t+1} + \tilde{k}_t \\
\vec{Y}_t &= \frac{\vec{S}_t \tilde{q}_t}{\vec{z}_t^\top \tilde{q}_t + \epsilon}, \quad \overleftarrow{Y}_t = \frac{\overleftarrow{S}_t \tilde{q}_t}{\overleftarrow{z}_t^\top \tilde{q}_t + \epsilon}
\end{aligned}$$

### 2.3 Low-Rank Factorized Bilinear Projection
$$Y = \text{Concat}\left(\frac{1}{2}(\vec{Y} + \overleftarrow{Y})\right)$$
$$\text{Output} = (Y L) R^\top + Y W_{\text{res}} + b$$
where $L \in \mathbb{R}^{D \times r}$ and $R \in \mathbb{R}^{D \times r}$ are factorized rank-$r$ parameter matrices.

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v + B \cdot L \cdot D \cdot r)$ linear in $L$.
- **SST-2 Performance**: Unbounded $x_t x_t^\top$ outer products without decay originally caused activation explosion. Multi-head decomposition with denominator tracking and bidirectional scanning stabilizes all representations.
