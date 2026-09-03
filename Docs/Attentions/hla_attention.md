# Hadamard Linear Attention (HLA)

## 1. Executive Abstract
**Hadamard Linear Attention (HLA)** replaces the $\mathcal{O}(L^2)$ softmax attention matrix with an associative multi-head linear recurrent state equipped with ELU+1 non-negative feature maps, learned decay gating $\lambda \in (0, 1)$, denominator normalization, and bidirectional sequence scanning.

---

## 2. Mathematical Equations & Recurrence

### 2.1 ELU+1 Feature Mapping
$$\phi(Q) = \text{ELU}(X W_q) + 1, \quad \phi(K) = \text{ELU}(X W_k) + 1, \quad V = X W_v$$

### 2.2 Bidirectional Recurrence with Denominator Tracking
For decay $\lambda = \sigma(\beta) \in (0, 1)$:
$$\begin{aligned}
\vec{S}_t &= \lambda \vec{S}_{t-1} + \phi(K_t) \otimes V_t, \quad \vec{z}_t = \lambda \vec{z}_{t-1} + \phi(K_t) \\
\overleftarrow{S}_t &= \lambda \overleftarrow{S}_{t+1} + \phi(K_t) \otimes V_t, \quad \overleftarrow{z}_t = \lambda \overleftarrow{z}_{t+1} + \phi(K_t) \\
\vec{Y}_t &= \frac{\vec{S}_t \phi(Q_t)}{\vec{z}_t^\top \phi(Q_t) + \epsilon}, \quad \overleftarrow{Y}_t = \frac{\overleftarrow{S}_t \phi(Q_t)}{\overleftarrow{z}_t^\top \phi(Q_t) + \epsilon}
\end{aligned}$$

### 2.3 Output Projection
$$\text{Output} = \text{Concat}\left(\frac{1}{2}(\vec{Y} + \overleftarrow{Y})\right) W_{\text{out}} + b_{\text{out}}$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot L \cdot d_k^2)$ linear in sequence length $L$.
- **SST-2 Performance**: Denominator normalization prevents numerical divergence while bidirectional scanning provides full bidirectional representation.
