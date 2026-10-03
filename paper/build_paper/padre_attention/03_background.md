# 3. Background

## 3.1 Softmax Attention

For input $X \in \mathbb{R}^{B \times N \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

The softmax kernel $e^{q \cdot k}$ is positive-definite and gives the 
attention matrix a probabilistic interpretation.

## 3.2 Why Softmax Is Hard to Kernelize

For the softmax:

$$
\text{softmax}(s_{ij}) = \frac{e^{s_{ij}}}{\sum_{j'} e^{s_{ij'}}}
\tag{3.3}
$$

The exponential $e^{s_{ij}} = \exp(q_i \cdot k_j)$ has a positive-definite 
kernel, but the explicit feature map is **infinite-dimensional**:

$$
\exp(x \cdot y) = \sum_{n=0}^\infty \frac{(x \cdot y)^n}{n!}
\tag{3.4}
$$

Truncating at order $d$ gives a finite feature map but with exponentially 
many features ($\binom{d + d_h}{d}$). This motivates **random** feature 
approximations (RFF, PORF).

## 3.3 Polynomial Kernel

A simple alternative is the **polynomial kernel**:

$$
K(q, k) = (1 + q \cdot k)^d
\tag{3.5}
$$

Expansion:

$$
(1 + q \cdot k)^d = \sum_{n=0}^d \binom{d}{n} (q \cdot k)^n
\tag{3.6}
$$

**Feature map.** $\phi(x) = \left( \sqrt{\binom{d}{n}} x^{\otimes n} \right)_{n=0}^d$ with dimension 
$\binom{d + d_h}{d}$.

For small $d$ (e.g., $d = 2$), the feature map is tractable:
- $d = 1$: dimension $1 + d_h$
- $d = 2$: dimension $1 + d_h + d_h^2$

## 3.4 PADRe Kernel

PADRe uses the normalized version:

$$
K_\text{PADRe}(q, k) = \left(1 + \frac{s}{\text{order}}\right)^\text{order}
\tag{3.7}
$$

where $s = (q \cdot k)/\sqrt{d_h}$ (with centering).

**Why the $1/\text{order}$ scaling?** As $\text{order} \to \infty$:

$$
\lim_{d \to \infty} \left(1 + \frac{s}{d}\right)^d = e^s
\tag{3.8}
$$

So PADRe is a **Taylor approximation of softmax** with a specific 
scaling that keeps the base close to 1.

## 3.5 Max-Centering for Stability

Without centering, $s_{ij}$ can be large positive values, and 
$(1 + s/\text{order})^\text{order}$ overflows for large $s$.

**Solution.** Center over the max:

$$
s_{ij} \leftarrow s_{ij} - \max_j s_{ij}
\tag{3.9}
$$

This ensures $s_{ij} \leq 0$, so $1 + s_{ij}/\text{order} \leq 1$, and 
the result is bounded.

## 3.6 LoRA Recovery

LoRA (Hu et al., 2021) parameterizes weight updates:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.10}
$$

with $B = 0$ at initialization. Used for post-substitution recovery.

## 3.7 Grouped-Query Attention (GQA)

For $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times N \times H_{kv} \times d_h}
\tag{3.11}
$$

Expanded to $H$ heads via `repeat_kv`.

## 3.8 Rotary Position Embeddings (RoPE)

$$
\tilde{q}_i = q_i \cos(\theta_i) + \text{rot}(q_i) \sin(\theta_i)
\tag{3.12}
$$

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $s_{ij}$ | Centered scaled score |
| $m_i$ | Max score |
| $d$ | Polynomial order |
| $r_L$ | LoRA rank |
| $\phi$ | Feature map |

---