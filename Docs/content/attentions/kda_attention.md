# Kernelized Decay Attention (KDA)

## 1. Executive Abstract
**Kernelized Decay Attention (KDA)** establishes a multi-head bidirectional linear recurrence mechanism with learnable gating decay $\lambda = \sigma(\beta)$, spherical $\ell_2$ normalization of keys/queries, and denominator normalizer tracking.

---

## 2. Mathematical Equations & Recurrence

### 2.1 Spherical Projections
$$Q = \text{normalize}(X W_q, \text{dim}=-1), \quad K = \frac{\text{normalize}(X W_k, \text{dim}=-1)}{\sqrt{d_k}}, \quad V = X W_v$$

### 2.2 Bidirectional Recurrence
$$\begin{aligned}
\vec{S}_t &= \sigma(\beta) \vec{S}_{t-1} + K_t \otimes V_t, \quad \vec{z}_t = \sigma(\beta) \vec{z}_{t-1} + K_t \\
\overleftarrow{S}_t &= \sigma(\beta) \overleftarrow{S}_{t+1} + K_t \otimes V_t, \quad \overleftarrow{z}_t = \sigma(\beta) \overleftarrow{z}_{t+1} + K_t \\
\vec{o}_t &= \frac{\vec{S}_t Q_t}{\vec{z}_t^\top Q_t + \epsilon}, \quad \overleftarrow{o}_t = \frac{\overleftarrow{S}_t Q_t}{\overleftarrow{z}_t^\top Q_t + \epsilon} \\
\text{Output} &= \text{Concat}\left(\frac{1}{2}(\vec{o} + \overleftarrow{o})\right) W_{\text{out}} + b_{\text{out}}
\end{aligned}$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v)$ linear $\mathcal{O}(L)$.
- **Empirical Fix**: Initializing $\beta = 2.0 \implies \sigma(\beta) \approx 0.88$ provides bounded fading memory across the 128-token sequence.
