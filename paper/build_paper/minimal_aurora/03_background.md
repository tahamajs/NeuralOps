# 3. Background

## 3.1 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

with $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD:** $W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$, with 
$\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2$.

**Factorization form:** define $U_r = U_{:, :r} \Sigma_{:r}^{1/2}$ and 
$V_r = \Sigma_{:r}^{1/2} V_{:r, :}^\top$, then $W_r = U_r V_r$.

## 3.2 Energy Thresholding

Define cumulative energy:

$$
E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2}
\tag{3.2}
$$

**Energy-selected rank:**

$$
r(\varepsilon) = \min\{r : E(r) \geq \varepsilon\}
\tag{3.3}
$$

**Reconstruction bound:**

$$
\|W - W_{r(\varepsilon)}\|_F^2 \leq (1 - \varepsilon) \|W\|_F^2
\tag{3.4}
$$

## 3.3 LoRA

LoRA (Hu et al., 2021):

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.5}
$$

with $A \in \mathbb{R}^{r_L \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r_L}$, $B = 0$ at init.

## 3.4 The FactorisedLinear Module

MinimalAURORA uses:

```python
self.down = nn.Linear(in_features, rank, bias=False)
self.up = nn.Linear(rank, out_features, bias=True)
```

Forward:

$$
y = \text{up}(\text{down}(x)) + \text{LoRA}(x)
\tag{3.6}
$$

**Parameter count:** $r(d_\text{in} + d_\text{out})$ vs $d_\text{in} d_\text{out}$.

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, \Sigma, V^\top$ | SVD |
| $r$ | Rank |
| $\varepsilon$ | Energy threshold |
| $E(r)$ | Cumulative energy |
| $\sigma_i$ | Singular value |
| $B, A$ | LoRA factors |
| $r_L$ | LoRA rank |

---