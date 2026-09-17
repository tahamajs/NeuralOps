# Appendix A. Mathematical Derivations

## A.1 Yat-Kernel Derivation

The Yat-Kernel feature map is:

$$
\phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega) \,\|\, \sin(x \Omega)]
\tag{A.1}
$$

where $\Omega \in \mathbb{R}^{d_h \times F}$ is a random projection.

**Inner product:**

$$
\phi(x)^\top \phi(y) = \frac{1}{\sqrt{F}} [\cos(x\Omega)^\top \cos(y\Omega) 
+ \sin(x\Omega)^\top \sin(y\Omega)]
\tag{A.2}
$$

Using $\cos(a)\cos(b) + \sin(a)\sin(b) = \cos(a - b)$:

$$
\phi(x)^\top \phi(y) = \frac{1}{\sqrt{F}} \sum_{i=1}^{F} \cos(\omega_i^\top (x - y))
\tag{A.3}
$$

This is a scaled cosine-similarity kernel.

## A.2 Spherical Normalization

$$
\hat{x} = \frac{x}{\|x\|_2 + \epsilon}
\tag{A.4}
$$

**Property.** $\|\hat{x}\|_2 = 1$ for all $x$.

**Effect on kernel.** $\phi(\hat{x})^\top \phi(\hat{y}) = \frac{1}{\sqrt{F}} 
\sum_i \cos(\omega_i^\top (\hat{x} - \hat{y}))$, which depends only on the 
**angle** between $x, y$.

## A.3 Linear Attention

$$
O_i^\text{linear} = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.5}
$$

**Complexity.** Computing $KV = \sum_j \phi(k_j) v_j^\top$ is 
$\mathcal{O}(L \cdot 2F \cdot d_h)$. Evaluating (A.5) is $\mathcal{O}(L \cdot 2F \cdot d_h)$. Total: $\mathcal{O}(L \cdot 2F \cdot d_h)$.

## A.4 Causal Linear Attention via Cumsum

For decoders, we require:

$$
O_t^\text{linear} = \frac{\phi(q_t)^\top \sum_{j \leq t} \phi(k_j) v_j^\top}{\phi(q_t)^\top \sum_{j \leq t} \phi(k_j)}
\tag{A.6}
$$

Define:

$$
S_t = \sum_{j \leq t} \phi(k_j) v_j^\top, \quad
Z_t = \sum_{j \leq t} \phi(k_j)
\tag{A.7}
$$

Both are cumulative sums, computed via `torch.cumsum`.

**Memory.** $S_t$ has shape $(B, H, L, 2F, d_h)$, which is 
$\mathcal{O}(L \cdot 2F \cdot d_h)$ memory.

## A.5 Gate Fusion

$$
g = \sigma(W_g X + b_g)
\tag{A.8}
$$

$$
O = g \odot O^\text{dense} + (1 - g) \odot O^\text{linear}
\tag{A.9}
$$

**Step-0 preservation.** With $W_g = 0$ and $b_g = 5.0$:

$$
g = \sigma(5.0) \approx 0.993
\tag{A.10}
$$

So $O \approx 0.993 \cdot O^\text{dense} + 0.007 \cdot O^\text{linear}$, 
dominated by dense.

## A.6 Complexity Summary

| Operation | Time | Memory |
|---|---|---|
| Q, K, V proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Dense scores | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |
| Dense softmax | $\mathcal{O}(L^2)$ | $\mathcal{O}(L^2)$ |
| Dense @ V | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |
| Spherical norm | $\mathcal{O}(L d_h)$ | $\mathcal{O}(L d_h)$ |
| RFF | $\mathcal{O}(L \cdot d_h \cdot F)$ | $\mathcal{O}(L \cdot 2F)$ |
| Linear bidirectional | $\mathcal{O}(L \cdot 2F \cdot d_h)$ | $\mathcal{O}(2F \cdot d_h)$ |
| Linear causal | $\mathcal{O}(L \cdot 2F \cdot d_h)$ | $\mathcal{O}(L \cdot 2F \cdot d_h)$ |
| Gate | $\mathcal{O}(L d H)$ | $\mathcal{O}(L H)$ |
| **Total** | **$\mathcal{O}(L^2 d_h + L \cdot 2F \cdot d_h)$** | **Causal: $\mathcal{O}(L \cdot 2F \cdot d_h)$** |

## A.7 Breakeven Analysis

When does the linear stream pay off?

**Dense time:** $T_d = c_1 L^2 d_h$.
**Linear time:** $T_l = c_2 L \cdot 2F \cdot d_h + c_3$.
**Hybrid:** $T_d + T_l$.

Breakeven when $T_d = T_l$:

$$
L^* = \frac{c_2 \cdot 2F}{c_1}
\tag{A.11}
$$

For typical constants, $L^* \approx 2F \cdot \kappa$ where $\kappa$ is a 
hardware-dependent constant.

---