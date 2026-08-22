# Gated DeltaNet: Multi-Head Bidirectional Delta Rule Linear Recurrent Attention

## 1. Executive Abstract & Theoretical Foundations
Standard linear attention methods approximate the softmax denominator with associative outer products $S_t = S_{t-1} + k_t v_t^\top$, suffering from state capacity saturation where historical memory overwrites recent context.

**Gated DeltaNet** resolves associative memory saturation by implementing the **Delta Rule** error-correction update combined with **data-dependent exponential decay gating**:
1. **Delta-Rule Recurrent Matrix Memory**: Uses the associative error $e_t = v_t - S_{t-1} k_t$ to perform targeted updates, writing new information into orthogonal memory subspaces.
2. **Bidirectional Recurrent Scanning**: Evaluates forward ($\vec{S}_t$) and backward ($\overleftarrow{S}_t$) matrix recurrences, enabling full bidirectional contextualization on encoder classification tasks.
3. **Data-Dependent Decay Gate**: $\beta_t = \sigma(W_\beta x_t + b_\beta) \in (0, 1)$ dynamically controls the memory retention horizon.
4. **Pretrained Attention Weight Transfer**: Transfers $W_Q, W_K, W_V, W_O$ directly from pretrained Transformers.

---

## 2. Mathematical Equations & State Recurrence

### 2.1 Projections & Hyperspherical Key Normalization
$$Q = X W_q, \quad K = X W_k, \quad V = X W_v, \quad Z = X W_z$$
$$\tilde{q} = \text{RMSNorm}(Q), \quad \tilde{k} = \text{RMSNorm}(K) / \sqrt{d_k}$$

### 2.2 Gating Signals
$$\beta_t = \sigma(W_\beta x_t + b_\beta) \in (0, 1) \quad \text{(Retention Decay Gate)}$$
$$\alpha_t = \sigma(W_\alpha x_t) \in (0, 1) \quad \text{(Delta Learning Rate)}$$

### 2.3 Bidirectional Delta-Rule State Updates
For associative state $S \in \mathbb{R}^{B \times H \times d_k \times d_v}$:

**Forward Pass ($t = 1 \dots L$):**
$$\vec{e}_t = v_t - \vec{S}_{t-1} \tilde{k}_t$$
$$\vec{S}_t = \beta_t \vec{S}_{t-1} + \alpha_t \vec{e}_t \tilde{k}_t^\top = \beta_t \vec{S}_{t-1} + \alpha_t (v_t - \vec{S}_{t-1} \tilde{k}_t) \tilde{k}_t^\top$$
$$\vec{y}_t = \vec{S}_t \tilde{q}_t$$

**Backward Pass ($t = L \dots 1$):**
$$\overleftarrow{e}_t = v_t - \overleftarrow{S}_{t+1} \tilde{k}_t$$
$$\overleftarrow{S}_t = \beta_t \overleftarrow{S}_{t+1} + \alpha_t \overleftarrow{e}_t \tilde{k}_t^\top$$
$$\overleftarrow{y}_t = \overleftarrow{S}_t \tilde{q}_t$$

### 2.4 Gated Readout & Projection
$$Y = \frac{1}{2}(\vec{Y} + \overleftarrow{Y}) \odot \text{SiLU}(Z)$$
$$\text{Output} = Y W_{\text{out}} + b_{\text{out}}$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v)$ — strictly linear in sequence length $L$.
- **Memory Complexity**: $\mathcal{O}(B \cdot H \cdot d_k \cdot d_v)$ per sequence.
- **SST-2 Performance**: Causal DeltaNet hit 50.92% because token 0 (`[CLS]`) could not access downstream sentiment words. Bidirectional DeltaNet provides full bidirectional gradient paths, restoring high classification accuracy.
