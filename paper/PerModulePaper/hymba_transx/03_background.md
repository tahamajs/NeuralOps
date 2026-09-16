# 3. Background

## 3.1 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

**Noise.** Empirically, softmax attention spreads mass across many tokens, 
including irrelevant ones. This is a form of **attention noise** — the 
model's output contains contributions from tokens that carry no signal.

## 3.2 Differential Attention

Ye et al. (2024) propose **differential attention**: use two independent 
Q-K pairs and subtract the resulting softmax maps:

$$
A_\text{diff} = \text{softmax}(Q_1 K_1^\top) - \lambda \cdot \text{softmax}(Q_2 K_2^\top)
\tag{3.3}
$$

with $\lambda \in [0, 1]$. Ye et al. use $\lambda = 0.8$; we use $0.2$ to 
retain more signal.

**Why it works.** If both attention maps share a "noise" component (e.g., 
attention to padding, or a diffuse mass over the sequence), the subtraction 
cancels it. The differential map $A_\text{diff}$ contains only the tokens 
that the two maps disagree on — which correlates with **signal**.

**ReLU post-activation.** We apply ReLU to prevent negative attention 
weights:

$$
A_\text{diff}^+ = \text{ReLU}(A_\text{diff}), \quad
A_\text{diff}^\text{norm} = \frac{A_\text{diff}^+}{\sum A_\text{diff}^+ + \epsilon}
\tag{3.4}
$$

## 3.3 State Space Models

An SSM defines a linear recurrence:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad
y_t = \bar{C} h_t
\tag{3.5}
$$

where $\bar{A}, \bar{B}, \bar{C}$ are the discretized parameters. Selective 
SSMs (Mamba) make $\bar{A}, \bar{B}$ input-dependent.

**SimpleSSM.** Our SSM branch is a lightweight variant with element-wise 
recurrence and diagonal $\bar{A}$.

## 3.4 Rotary Position Embeddings

RoPE rotates Q, K by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.6}
$$

**HeadRoPE.** When the parent model does not provide external position 
embeddings, we compute them internally from a learned frequency schedule:

$$
\theta_{k, i} = i \cdot \omega_k, \quad
\omega_k = \text{base}^{-2k / d_h}
\tag{3.7}
$$

## 3.5 Gated Fusion

A scalar gate $\alpha \in (0, 1)$ blends two branches:

$$
\text{out} = \alpha \odot A + (1 - \alpha) \odot B
\tag{3.8}
$$

We learn $\alpha$ as a single scalar parameter, initialized via 
$\sigma(-1) \approx 0.27$.

## 3.6 Notation

| Symbol | Meaning |
|---|---|
| $Q_i, K_i$ | Query, key for branch $i$ |
| $A_i$ | Attention map |
| $\lambda$ | Differential damping |
| $h_t$ | SSM state |
| $\alpha$ | Fusion scalar |
| $\theta$ | RoPE angle |

---