# UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Mamba SSM

## 1. Executive Abstract
**UPI-Stuffed Mamba** extends the bidirectional Mamba selective state-space architecture with an input-dependent uncertainty penalty $\gamma_t$ that dynamically interpolates between persistent associative memory and fading memory, enabling calibrated forgetting and robust fine-tuning.

---

## 2. Mathematical Equations & Recurrence

### 2.1 Input Projection & Symmetric Convolution
$$[X_{\text{ssm}}, Z] = X W_{\text{in}} \in \mathbb{R}^{B \times L \times 2 D}$$
$$\widetilde{X}_{\text{ssm}} = \text{SiLU}(\text{Conv1D}(X_{\text{ssm}}^\top)^\top) \in \mathbb{R}^{B \times L \times D}$$

### 2.2 Uncertainty-Penalized Discretization
$$\Delta t = \text{softplus}(\widetilde{X}_{\text{ssm}} W_{\Delta t} + b_{\Delta t}) \in \mathbb{R}^{B \times L \times D}$$
$$\gamma_t = \sigma(X W_\gamma) \in (0, 1)^{B \times L \times D} \quad \text{(Uncertainty Penalty)}$$
$$\bar{A}_t = \exp(-\exp(A_{\text{log}}) \odot \Delta t_t) \odot (1.0 - 0.5 \gamma_t) \in (0, 1)$$

### 2.3 Bidirectional State Recurrence
$$\vec{h}_t = \bar{A}_t \odot \vec{h}_{t-1} + (\Delta t_t \odot \widetilde{x}_t) \otimes \mathbf{1}_{d_{\text{state}}}$$
$$\overleftarrow{h}_t = \bar{A}_t \odot \overleftarrow{h}_{t+1} + (\Delta t_t \odot \widetilde{x}_t) \otimes \mathbf{1}_{d_{\text{state}}}$$
$$y_t = \sum_{s=1}^{d_{\text{state}}} \left( \frac{1}{2}(\vec{h}_{t, :, s} + \overleftarrow{h}_{t, :, s}) \cdot D_s \right)$$

### 2.4 Gated Output Projection
$$\text{Output} = (y \odot \text{SiLU}(Z)) W_{\text{out}} + b_{\text{out}}$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ linear in sequence length $L$.
- **SST-2 Analysis**: The uncertainty penalty $\gamma_t$ downweights irrelevant tokens in the bidirectional sequence, while pretrained weight copying ensures rapid convergence.
