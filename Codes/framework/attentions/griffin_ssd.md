# Griffin SSD: Hybrid Sliding-Window Attention & State Space Duality (SSD)

## 1. Executive Abstract
**Griffin SSD** combines local bidirectional sliding-window attention with global multi-head State Space Duality (SSD) linear recurrence and automated weight transfer.

---

## 2. Mathematical Equations & Recurrence

### 2.1 Local Sliding-Window Path
$$\begin{aligned}
S_{\text{local}} &= \frac{Q_{\text{loc}} K_{\text{loc}}^\top}{\sqrt{d_k}} + M_{\text{window}} + M_{\text{pad}} \\
Y_{\text{local}} &= \text{Softmax}(S_{\text{local}}) V_{\text{loc}}
\end{aligned}$$
where $M_{\text{window}, i, j} = -\infty$ for $|i - j| > W$.

### 2.2 Global State Space Duality (SSD) Recurrence
$$\begin{aligned}
\vec{H}_t &= \sigma(\beta) \vec{H}_{t-1} + \tilde{k}_t \otimes v_t, \quad \vec{Y}_t = \tilde{q}_t \vec{H}_t \\
\overleftarrow{H}_t &= \sigma(\beta) \overleftarrow{H}_{t+1} + \tilde{k}_t \otimes v_t, \quad \overleftarrow{Y}_t = \tilde{q}_t \overleftarrow{H}_t \\
Y_{\text{global}} &= \frac{1}{2}(\vec{Y} + \overleftarrow{Y})
\end{aligned}$$

### 2.3 Residual Output Projection
$$\text{Output} = (Y_{\text{local}} + Y_{\text{global}}) W_{\text{out}} + b_{\text{out}}$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Complexity**: $\mathcal{O}(B \cdot H \cdot L \cdot (W \cdot d_k + d_k \cdot d_v))$ linear in sequence length $L$.
- **SST-2 Analysis**: Bidirectional sliding windows capture local syntactic context while global SSD aggregates long-range context without quadratic scaling.
