# Simple State Space Model (SimpleSSM)

## 1. Executive Abstract
**SimpleSSM** is a lightweight selective state-space model that integrates symmetric 1D depthwise convolution, input-dependent discretization timescales ($\Delta t$), stable diagonal transition matrices ($A$), and bidirectional state recurrence.

---

## 2. Mathematical Equations & Recurrence

### 2.1 Input Projection & Symmetric 1D Convolution
$$[X_{\text{ssm}}, Z] = X W_{\text{in}} \in \mathbb{R}^{B \times L \times 2 D}$$
$$\widetilde{X}_{\text{ssm}} = \text{SiLU}(\text{Conv1D}(X_{\text{ssm}}^\top)^\top) \in \mathbb{R}^{B \times L \times D}$$

### 2.2 Discretization & Bidirectional Diagonal State Recurrence
$$\Delta t = \text{softplus}(\widetilde{X}_{\text{ssm}} W_{\Delta t} + b_{\Delta t}) \in \mathbb{R}^{B \times L \times D}$$
$$\bar{A} = \exp(-\exp(A_{\text{log}}) \odot \Delta t) \in (0, 1)^{B \times L \times D \times d_{\text{state}}}$$
$$\vec{h}_t = \bar{A}_t \odot \vec{h}_{t-1} + (\Delta t_t \odot \widetilde{x}_t) \otimes \mathbf{1}_{d_{\text{state}}}$$
$$\overleftarrow{h}_t = \bar{A}_t \odot \overleftarrow{h}_{t+1} + (\Delta t_t \odot \widetilde{x}_t) \otimes \mathbf{1}_{d_{\text{state}}}$$
$$y_t = \sum_{s=1}^{d_{\text{state}}} \left( \frac{1}{2}(\vec{h}_{t, :, s} + \overleftarrow{h}_{t, :, s}) \cdot D_s \right)$$

### 2.3 SiLU Gating & Output
$$\text{Output} = (y \odot \text{SiLU}(Z)) W_{\text{out}} + b_{\text{out}}$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ linear in sequence length $L$.
- **SST-2 Analysis**: Bidirectional diagonal recurrence with pretrained weight initialization solves the unidirectional vanishing gradient problem.
