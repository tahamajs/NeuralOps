# Appendix A. Mathematical Derivations

## A.1 Differential Attention

Two softmax maps:

$$
A_1 = \text{softmax}\!\left(\frac{Q_1 K_1^\top}{\sqrt{d_h}}\right), \quad
A_2 = \text{softmax}\!\left(\frac{Q_2 K_2^\top}{\sqrt{d_h}}\right)
\tag{A.1}
$$

Differential map:

$$
A_\text{diff} = \text{ReLU}(A_1 - \lambda A_2), \quad \lambda = 0.2
\tag{A.2}
$$

Normalized:

$$
\tilde{A}_\text{diff} = \frac{A_\text{diff}}{\sum_j A_{\text{diff}, ij} + \epsilon}
\tag{A.3}
$$

## A.2 Noise Cancellation Model

Suppose $A_1 = S + N_1$ and $A_2 = S + N_2$, where $S$ is the shared 
signal and $N_i$ is branch-specific noise. Then:

$$
A_1 - \lambda A_2 = (1 - \lambda) S + N_1 - \lambda N_2
\tag{A.4}
$$

If $N_1, N_2$ are independent with mean 0 and variance $\sigma^2$:

$$
\mathbb{E}[(N_1 - \lambda N_2)^2] = (1 + \lambda^2) \sigma^2
\tag{A.5}
$$

For $\lambda = 0.2$: noise variance scales as $1.04 \sigma^2$ — barely 
amplified. The signal is scaled by $0.8$.

If $N_1, N_2$ are positively correlated (common-mode noise), the variance 
is smaller:

$$
\mathbb{E}[(N_1 - \lambda N_2)^2] = (1 - \lambda)^2 \sigma^2 + 2\lambda(1 - \rho) \sigma^2
\tag{A.6}
$$

For $\rho \to 1$: variance $\to (1 - \lambda)^2 \sigma^2$ — heavily 
suppressed.

## A.3 SSM Recurrence

SimpleSSM:

$$
h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t
\tag{A.7}
$$

Unrolled:

$$
h_t = \sum_{j=1}^{t} \left( \prod_{i=j+1}^{t} \bar{A}_i \right) \bar{B}_j x_j
\tag{A.8}
$$

Effective decay: $\prod_{i=j+1}^t \bar{A}_i$ — geometric if $\bar{A}$ is 
constant.

## A.4 Fusion

$$
\text{combined} = \alpha A_\text{out} + (1 - \alpha) M_\text{ssm}
\tag{A.9}
$$

Gradient w.r.t. $\alpha$:

$$
\frac{\partial \text{combined}}{\partial \alpha} = A_\text{out} - M_\text{ssm}
\tag{A.10}
$$

The gate learns to prefer the branch with higher utility.

## A.5 Complexity

Per block:

| Component | Time |
|---|---|
| Q1, K1 projection | $O(L d^2)$ |
| Q2, K2 projection | $O(L d^2)$ |
| V projection | $O(L d^2)$ |
| Diff attention | $O(L^2 d)$ |
| SSM (fwd) | $O(L d)$ |
| SSM (bwd, encoder) | $O(L d)$ |
| Output projection | $O(L d^2)$ |

Total: $O(L^2 d)$ — dominated by attention.

## A.6 RoPE Frequencies

$$
\omega_k = \text{base}^{-2k/d_h}, \quad k = 0, 1, \dots, d_h/2 - 1
\tag{A.11}
$$

$$
\theta_{k, i} = i \cdot \omega_k
\tag{A.12}
$$

For base = $10^4$ and $d_h = 64$:

$$
\omega_0 = 1, \quad \omega_{31} = 10^{-4}
\tag{A.13}
$$

RoPE rotates the first half of the head dimension with $\omega_0, \dots$ and 
the second half with the same frequencies (concatenated).

---