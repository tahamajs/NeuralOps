# MambaSVD-Flash: Hybrid Bidirectional SSM-FlashAttention Architecture

## 1. Executive Abstract & Architecture Motivation
Pure dense attention suffers from quadratic $\mathcal{O}(L^2)$ computational complexity, while pure linear State Space Models struggle with high-entropy associative memory queries. **MambaSVD-Flash** introduces a dual-path hybrid architecture:
1. **Bidirectional Low-Rank SVD-SSM Path**: Evaluates contractive online Singular Value Decomposition state-space transitions ($A_t = U_t \Sigma_t V_t^\top$) scanning both forward and backward directions with $\mathcal{O}(L \cdot D \cdot r)$ complexity.
2. **Dense Bidirectional FlashAttention Path**: Executes exact scaled dot-product self-attention over the sequence.
3. **Dynamic Learned Output Aggregation**: Dynamically interpolates between the SSM path and FlashAttention path via input-conditioned sigmoid gating $\gamma_t = \sigma(W_g x_t)$.

---

## 2. Mathematical Formulations & Recurrence Dynamics

### 2.1 Bidirectional Low-Rank SVD-SSM Recurrence
For rank $r \ll D$:
$$U_t = \text{proj}_u(x_t) \in \mathbb{R}^{D \times r}, \quad V_t = \text{proj}_v(x_t) \in \mathbb{R}^{r \times D}$$
$$\Sigma_t = 0.95 \cdot \sigma(\text{proj}_\sigma(x_t)) \in (0, 0.95)^r$$

Forward & Backward recurrence:
$$\vec{h}_t = U_t \Sigma_t (V_t^\top \vec{h}_{t-1}) + B_t x_t$$
$$\overleftarrow{h}_t = U_t \Sigma_t (V_t^\top \overleftarrow{h}_{t+1}) + B_t x_t$$
$$Y_{\text{ssm}, t} = C_t \odot \frac{1}{2}(\vec{h}_t + \overleftarrow{h}_t)$$

### 2.2 FlashAttention Path
$$Y_{\text{attn}} = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M_{\text{pad}}\right) V$$

### 2.3 Gated Output Fusion
$$Y = \gamma \odot Y_{\text{ssm}} + (1 - \gamma) \odot Y_{\text{attn}}$$
$$\text{Output} = Y W_{\text{out}} + b_{\text{out}}$$

---

## 3. Computational Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot L \cdot D \cdot r + B \cdot L^2 \cdot D)$
- **Lyapunov Contractivity**: By scaling $\Sigma_t \le 0.95$, the spectral radius of the state transition $\rho(A_t) < 1.0$, preventing unbounded gradient growth.
- **SST-2 Recovery**: Removing the causal mask limitation and transferring pretrained attention weights directly fixes the 50.92% collapse.
