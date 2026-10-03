<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Linear Attention

Standard softmax attention:

$$
\text{Attn}(q_i) = \frac{\sum_j e^{q_i^\top k_j} v_j}{\sum_j e^{q_i^\top k_j}}
\tag{3.1}
$$

Linear attention replaces $e^{q^\top k}$ with $\phi(q)^\top \phi(k)$:

$$
\text{LinAttn}(q_i) = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.2}
$$

The sums $S = \sum_j \phi(k_j) v_j^\top$ and $z = \sum_j \phi(k_j)$ are 
computed once for the whole sequence, then reused for each query.

**Complexity.** Computing $S$ is $\mathcal{O}(L m d)$ where $m$ is the 
feature dimension and $d$ the head dimension. Computing the output is 
$\mathcal{O}(L m d)$. Total: $\mathcal{O}(L m d)$, linear in $L$.

## 3.2 Causal Linear Attention

For autoregressive models, the sums must be *prefix* sums:

$$
S_i = \sum_{j \le i} \phi(k_j) v_j^\top, \quad
z_i = \sum_{j \le i} \phi(k_j)
\tag{3.3}
$$

The output is:

$$
y_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
\tag{3.4}
$$

**Efficient computation.** Prefix sums can be computed with a parallel 
scan (Blelloch, 1990), giving $\mathcal{O}(\log L)$ depth. Our implementation 
uses a naive **sequential scan** in Python; this is a known performance 
bottleneck.

## 3.3 Softplus Kernel

The softplus function is:

$$
\text{softplus}(x) = \log(1 + e^x)
\tag{3.5}
$$

**Properties.**
- Smooth ($C^\infty$)
- Strictly positive: $\text{softplus}(x) > 0$ for all $x$
- Approximately linear for large $x$: $\text{softplus}(x) \approx x$
- Approximately exponential for small $x$: $\text{softplus}(x) \approx e^x$

**Comparison to elu+1.** The function $\mathrm{elu}(x) + 1$ has a kink at 
$x = 0$, where the derivative jumps from 1 to $e$. Softplus is smooth. This 
may improve gradient flow.

## 3.4 Rank-$r$ Approximation

Given a matrix $M \in \mathbb{R}^{d \times d}$, its best rank-$r$ 
approximation is:

$$
M_r = U_r \Sigma_r V_r^\top
\tag{3.6}
$$

where $U_r, \Sigma_r, V_r$ are from the SVD. If $M$ is already low-rank by 
construction ($M = U V^\top$ with $U, V \in \mathbb{R}^{d \times r}$), then 
$M_r = M$.

In RALA, we parameterize $M_h = U_h V_h^\top$ directly with $U_h, V_h \in 
\mathbb{R}^{d \times r}$.

## 3.5 LoRA

LoRA (Hu et al., 2021) adds a rank-$r_{\text{L}}$ update to a frozen 
weight:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha_{\text{L}}}{r_{\text{L}}} B A
\tag{3.7}
$$

with $B = 0$ at initialization.

In RALA, each projection is a `LoRALinear` with base weights transferred 
from the pretrained attention and LoRA adapters initialized to zero.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $m$ | Feature map dimension |
| $r$ | Augmentation rank |
| $U_h, V_h$ | Augmentation factors |
| $\alpha$ | Scalar gate |
| $M_h$ | Augmentation matrix |
| $\phi$ | Kernel feature map |
| $S_i, z_i$ | Prefix sums |

---