# HGRN-2 Outer-Product Recurrent Linear Attention

## 1. Executive Abstract
`HGRN2OuterBlock` implements multi-head bidirectional outer-product associative matrix memory updates with log-space bounded decay gating, denominator normalization, and pretrained weight transfer.

---

## 2. Mathematical Equations & Recurrence

### 2.1 Multi-Head Projections & Bounded Forget Gating
$$Q = X W_q, \quad K = X W_k, \quad V = X W_v$$
$$\tilde{q} = \text{RMSNorm}(Q), \quad \tilde{k} = \text{RMSNorm}(K) / \sqrt{d_k}$$
$$f_t = \exp(-\text{softplus}(W_f x_t + b_f)) \in (0, 1)^{B \times H}$$

### 2.2 Bidirectional Associative Matrix Recurrence
For multi-head states $S \in \mathbb{R}^{B \times H \times d_k \times d_v}$ and normalizers $z \in \mathbb{R}^{B \times H \times d_k \times 1}$:
$$\begin{aligned}
\vec{S}_t &= f_t \odot \vec{S}_{t-1} + (1 - f_t) \odot (\tilde{k}_t \otimes v_t) \\
\vec{z}_t &= f_t \odot \vec{z}_{t-1} + (1 - f_t) \odot \tilde{k}_t \\
\overleftarrow{S}_t &= f_t \odot \overleftarrow{S}_{t+1} + (1 - f_t) \odot (\tilde{k}_t \otimes v_t) \\
\overleftarrow{z}_t &= f_t \odot \overleftarrow{z}_{t+1} + (1 - f_t) \odot \tilde{k}_t
\end{aligned}$$

### 2.3 Normalized Readout & Projection
$$\vec{y}_t = \frac{\vec{S}_t \tilde{q}_t}{\vec{z}_t^\top \tilde{q}_t + \epsilon}, \quad \overleftarrow{y}_t = \frac{\overleftarrow{S}_t \tilde{q}_t}{\overleftarrow{z}_t^\top \tilde{q}_t + \epsilon}$$
$$Y = \text{Concat}\left(\frac{1}{2}(\vec{Y} + \overleftarrow{Y})\right)$$
$$\text{Output} = Y W_{\text{out}} + b_{\text{out}}$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v)$ linear in $L$.
- **Why Causal Failed**: Without normalization and bidirectional scanning, the un-normalized outer product accumulated large eigenvalues ($S_t \to \infty$), destabilizing gradients. Normalizing $S_t$ with denominator $z_t$ and adding bidirectional scanning guarantees numerical stability.
