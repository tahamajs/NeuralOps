# 3. Background

## 3.1 Truncated SVD

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

The rank-$r$ truncation:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.2}
$$

has Frobenius error $\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

## 3.2 Symmetric $\sqrt{\Sigma}$ Split

Two common factorizations:

**Asymmetric:**
$$
U_r = U_{:, :r}, \quad V_r = \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.3}
$$

**Symmetric:**
$$
U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{3.4}
$$

We use the symmetric split for numerical balance: both factors have similar 
magnitudes, reducing FP16 underflow risk.

## 3.3 Energy Thresholding

Given singular values $\sigma_1 \geq \sigma_2 \geq \dots$, the cumulative 
energy fraction at rank $r$ is:

$$
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(m, n)} \sigma_j^2}
\tag{3.5}
$$

The **energy threshold** $\tau \in (0, 1]$ defines the target:

$$
r^* = \min\{r : E(r) \geq \tau\}
\tag{3.6}
$$

This directly bounds the relative Frobenius error:

$$
\frac{\|W - W_{r^*}\|_F}{\|W\|_F} = \sqrt{1 - E(r^*)} \leq \sqrt{1 - \tau}
\tag{3.7}
$$

For $\tau = 0.92$: relative error $\leq \sqrt{0.08} \approx 0.283$.

## 3.4 Multi-Head Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.8}
$$

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.9}
$$

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.10}
$$

## 3.5 T5 Relative Attention Bias

T5 (Raffel et al., 2020) adds a learned bias to attention scores:

$$
\text{scores}_{ij} \leftarrow \text{scores}_{ij} + b_{\text{bucket}(i, j)}
\tag{3.11}
$$

where $\text{bucket}(i, j)$ maps the relative position to a discrete bin.

Our factorization preserves the relative bias embedding because we only 
replace Q/K/V/O projections, not the bias.

## 3.6 Cross-Attention

Encoder-decoder attention uses **different** K, V inputs:

$$
K = H_\text{enc} W_K, \quad V = H_\text{enc} W_V
\tag{3.12}
$$

where $H_\text{enc}$ is the encoder's output. Our factorized K, V projections 
apply to any input, so cross-attention works unchanged.

## 3.7 Residual Adapters (LoRA)

We add a trainable low-rank residual:

$$
\Delta W = B A
\tag{3.13}
$$

with $A \in \mathbb{R}^{r_a \times d}$, $B \in \mathbb{R}^{d \times r_a}$, and 
$B = 0$ at initialization.

## 3.8 Notation

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, \Sigma, V$ | SVD components |
| $\sigma_i$ | $i$-th singular value |
| $r$ | Selected rank |
| $\tau$ | Energy threshold |
| $r_a$ | Residual rank |
| $A, B$ | Adapter matrices |
| $H$ | Number of heads |
| $d_h$ | Head dim |
```

---