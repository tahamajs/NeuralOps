# 3. Background

## 3.1 Truncated SVD

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top \approx U_r \Sigma_r V_r^\top
\tag{3.1}
$$

with error $\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

**Factorization.** We use:

$$
U_r = U_{:, :r} \Sigma_r^{1/2}, \quad V_r = \Sigma_r^{1/2} V_{:r, :}^\top
\tag{3.2}
$$

for balanced magnitudes.

## 3.2 Low-Rank Adapters (LoRA)

**LoRA** (Hu et al., 2021) parameterizes a rank-$r_L$ update:

$$
\Delta W = B A, \quad A \in \mathbb{R}^{r_L \times d}, \quad B \in \mathbb{R}^{d \times r_L}
\tag{3.3}
$$

with $B = 0$ at init.

## 3.3 Variational Inference

**Variational Inference** approximates an intractable posterior $p(z | x)$ 
with a tractable $q_\phi(z)$. The ELBO is:

$$
\mathcal{L} = \mathbb{E}_{q}[\log p(x | z)] - \text{KL}(q(z) \| p(z))
\tag{3.4}
$$

**Reparameterization.** For Gaussian $q(z) = \mathcal{N}(\mu, \sigma^2)$:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\tag{3.5}
$$

allows gradient flow to $\mu, \sigma$.

## 3.4 Multi-Head Attention

Standard:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) V
\tag{3.6}
$$

**GQA.** $H_{kv} < H$ heads share K, V; `repeat_kv` expands.

## 3.5 Token Routing

Given scores $s_i$, select top-$k$:

$$
\mathcal{S} = \text{TopK}(s, k), \quad k = \lceil \rho L \rceil
\tag{3.7}
$$

with $s_0 = \infty$ (retain `[CLS]`) and $s_i = -\infty$ for padding.

## 3.6 Notation

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, S, V$ | SVD factors |
| $r$ | SVD rank |
| $r_L$ | LoRA rank |
| $\mu, \sigma$ | Variational parameters |
| $g$ | Spectral gate |
| $h$ | Head gate |
| $s$ | Token salience |
| $\rho$ | Keep ratio |

---