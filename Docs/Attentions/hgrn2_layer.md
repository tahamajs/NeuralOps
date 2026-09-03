# Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer

## 1. Executive Abstract
**Hierarchically Gated Recurrent Network v2 (HGRN-2)** implements element-wise linear recurrence modulated by dynamic input-dependent forget and input gates. It replaces multi-head attention with an $\mathcal{O}(L \cdot D)$ recurrent mechanism that features bounded log-space forget gating and bidirectional sequence scanning.

---

## 2. Mathematical Formulations & State Recurrence

### 2.1 Dynamic Gates & Value Projections
Given token activations $x_t \in \mathbb{R}^D$:
$$f_t = \exp(-\text{softplus}(W_f x_t + b_f)) \in (0, 1)^D \quad \text{(Log-space Bounded Retention Gate)}$$
$$i_t = \sigma(W_i x_t) \in (0, 1)^D \quad \text{(Input Modulation Gate)}$$
$$v_t = W_v x_t \in \mathbb{R}^D, \quad z_t = W_g x_t \in \mathbb{R}^D$$

With $b_f = -3.0$, the baseline forget factor initializes to $f_t \approx 0.95$, preventing premature memory loss.

### 2.2 Bidirectional Element-Wise Recurrence
$$\vec{h}_t = \text{clamp}(f_t \odot \vec{h}_{t-1} + i_t \odot v_t, -C, C)$$
$$\overleftarrow{h}_t = \text{clamp}(f_t \odot \overleftarrow{h}_{t+1} + i_t \odot v_t, -C, C)$$
where $C = 5.0$ is the residual clamping threshold.

### 2.3 Output Aggregation
$$Y = \frac{1}{2}(\vec{h} + \overleftarrow{h}) \odot \text{SiLU}(z)$$
$$\text{Output} = Y W_o + b_o$$

---

## 3. Complexity & Empirical SST-2 Analysis
- **Time Complexity**: $\mathcal{O}(B \cdot L \cdot D)$ — strictly linear in sequence length $L$.
- **Memory Complexity**: $\mathcal{O}(B \cdot D)$ recurrent footprint.
- **SST-2 Performance**: Pretrained weight transfer ($W_k \to W_i, W_v \to W_v, W_q \to W_g, W_o \to W_o$) combined with bidirectional scanning restores fine-tuning performance on text classification.
