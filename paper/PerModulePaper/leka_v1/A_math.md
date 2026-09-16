# Appendix A. Mathematical Derivations

## A.1 Kernel Attention Derivation

Starting from softmax attention:

$$
A_i = \frac{\sum_j \kappa(q_i, k_j) v_j}{\sum_j \kappa(q_i, k_j)}
\tag{A.1}
$$

If $\kappa(q, k) = \phi(q)^\top \phi(k)$:

$$
A_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.2}
$$

The complexity is $\mathcal{O}(L F d_h)$.

## A.2 Bochner's Theorem

For a shift-invariant kernel $\kappa(x - y)$:

$$
\kappa(x - y) = \int p(\omega) e^{i \omega^\top (x - y)} d\omega
\tag{A.3}
$$

For the Gaussian kernel:

$$
\kappa(\Delta) = \exp\!\left(-\frac{\|\Delta\|^2}{2}\right) \implies p(\omega) = \mathcal{N}(0, I)
\tag{A.4}
$$

## A.3 RFF Estimator

$$
\kappa(x - y) = \mathbb{E}_\omega[\cos(\omega^\top (x - y))] = \mathbb{E}_\omega[\cos(\omega^\top x) \cos(\omega^\top y) + \sin(\omega^\top x) \sin(\omega^\top y)]
\tag{A.5}
$$

Sampling $F$ frequencies $\omega_1, \dots, \omega_F$:

$$
\hat{\kappa}(x - y) = \frac{1}{F} \sum_i [\cos(\omega_i^\top x) \cos(\omega_i^\top y) + \sin(\omega_i^\top x) \sin(\omega_i^\top y)]
\tag{A.6}
$$

Define:

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\omega_1^\top x), \dots, \cos(\omega_F^\top x), \sin(\omega_1^\top x), \dots, \sin(\omega_F^\top x)]
\tag{A.7}
$$

Then $\hat{\kappa}(x - y) = \phi(x)^\top \phi(y)$.

## A.4 Variance of the Estimator

For iid sampling:

$$
\text{Var}[\hat{\kappa}] = \frac{1}{F} \text{Var}[\cos(\omega^\top \Delta)] = \mathcal{O}(1/F)
\tag{A.8}
$$

For orthogonalized sampling:

$$
\text{Var}[\hat{\kappa}] = \mathcal{O}(1/F^2)
\tag{A.9}
$$

## A.5 Causal Cumsum

For decoders, we need cumulative sums:

$$
\text{KV}_i = \sum_{j \leq i} \phi(k_j) v_j^\top
\tag{A.10}
$$

This is computed in $\mathcal{O}(L)$ via `torch.cumsum` on the outer product.

**Memory.** The outer product has shape $(B, H, L, 2F, d_h)$. For 
$L = 128$, this is ~1.6×10⁸ elements at FP32 = 640 MB.

## A.6 Preservation Gate

$$
\alpha = \sigma(w), \quad \text{out} = \alpha A + (1 - \alpha) x
\tag{A.11}
$$

Gradient w.r.t. $w$:

$$
\frac{\partial \text{out}}{\partial w} = \alpha(1 - \alpha)(A - x)
\tag{A.12}
$$

The gate learns to increase α when $A$ has larger magnitude than $x$, and 
decrease when $x$ is more important.

## A.7 Parameter Count

| Component | Params |
|---|---|
| Q/K/V/O (frozen) | $4 d^2$ |
| LoRA (×4) | $4 \cdot 2 r_L d$ |
| Random projection (per head) | $H d_h F$ |
| LayerNorm | $2 d_h$ |
| Gates | 2 |

For $d = 768$, $H = 12$, $d_h = 64$, $F = 256$, $r_L = 16$:

| Component | Params |
|---|---|
| Q/K/V/O | 2,359,296 |
| LoRA | 98,304 |
| Projection | 196,608 |
| LN | 128 |
| Gates | 2 |
| **Total** | 2,654,338 |
| **vs baseline** | 2,359,296 |
| **Increase** | 12.5% |

---