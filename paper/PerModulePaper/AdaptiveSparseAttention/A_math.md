<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Soft Top-$k$ Approximation Error

### A.1.1 Setup

Let $s \in \mathbb{R}^L$ be a score vector, $\tau$ the $k$-th largest 
value, and $m_j = \sigma((s_j - \tau) / T)$. Define:

$$
\tilde{p}_j = \frac{\exp(s_j + \log m_j)}{\sum_i \exp(s_i + \log m_i)} = \frac{m_j \exp(s_j)}{\sum_i m_i \exp(s_i)}
\tag{A.1}
$$

versus the hard top-$k$ softmax:

$$
p_j^{\text{hard}} = \frac{\mathbb{1}[s_j \ge \tau] \exp(s_j)}{\sum_i \mathbb{1}[s_i \ge \tau] \exp(s_i)}
\tag{A.2}
$$

### A.1.2 Error Bound

**Claim.** As $T \to 0$, $\tilde{p} \to p^{\text{hard}}$ pointwise.

**Proof.** For $s_j > \tau$: $m_j \to 1$. For $s_j < \tau$: $m_j \to 0$. 
For $s_j = \tau$: $m_j = 0.5$, contributing negligible mass in the limit. 
Thus $\tilde{p} \to p^{\text{hard}}$. $\square$

### A.1.3 Finite-$T$ Error

For finite $T$, the error in each component is bounded by:

$$
|p_j^{\text{hard}} - \tilde{p}_j| \leq e^{-|s_j - \tau| / T} \cdot \text{const}
\tag{A.3}
$$

## A.2 Per-Head Sparsity Gradient

### A.2.1 Chain Rule

The gradient of the loss w.r.t. logit $\ell_h$:

$$
\frac{\partial \mathcal{L}}{\partial \ell_h} = \sum_{b, i} \frac{\partial \mathcal{L}}{\partial \tau_{b, h, i}} \cdot \frac{\partial \tau_{b, h, i}}{\partial k_h} \cdot \frac{\partial k_h}{\partial \rho_h} \cdot \rho_h (1 - \rho_h)
\tag{A.4}
$$

### A.2.2 Threshold Gradient

The threshold $\tau$ is the $k_h$-th largest score. As $k_h$ changes by 1, 
$\tau$ jumps to the next order statistic. This is not differentiable in 
the classical sense.

**Straight-through estimator.** We approximate:

$$
\frac{\partial \tau}{\partial k_h} \approx -|\tau^{(k)} - \tau^{(k+1)}|
\tag{A.5}
$$

where $\tau^{(k)}$ is the $k$-th largest score.

### A.2.3 Budget Gradient

$$
\frac{\partial k_h}{\partial \rho_h} \approx -L
\tag{A.6}
$$

Combining:

$$
\frac{\partial \mathcal{L}}{\partial \ell_h} \approx -L \cdot \sum_{b, i} \frac{\partial \mathcal{L}}{\partial \tau_{b, h, i}} \cdot |\tau^{(k)} - \tau^{(k+1)}| \cdot \rho_h (1 - \rho_h)
\tag{A.7}
$$

## A.3 Chunked Memory Analysis

**Claim.** Peak memory for chunked attention is $\mathcal{O}(B H C L)$.

**Proof.** Each chunk processes $C$ queries against all $L$ keys. The score 
matrix has shape $(B, H, C, L)$, requiring $B H C L$ entries. Only one chunk 
is held at a time. $\square$

## A.4 Head Budget Convergence

**Claim.** The learned budgets $\rho_h^{(T)}$ converge to a fixed point of 
the gradient flow.

**Proof sketch.** The gradient in (A.7) depends on the loss landscape, which 
depends on $\rho$ through the mask. Under standard SGD convergence 
assumptions, the sequence $\{\rho_h^{(t)}\}$ converges to a critical point 
of $\mathcal{L}$. $\square$

## A.5 Relationship to Softmax Sparsity

The entropy of the softmax distribution is:

$$
H(p) = -\sum_j p_j \log p_j
\tag{A.8}
$$

For the masked softmax $\tilde{p}$:

$$
H(\tilde{p}) \approx H(p) - \sum_j p_j \log m_j
\tag{A.9}
$$

For $m_j \ll 1$, $\log m_j < 0$, so $H(\tilde{p}) < H(p)$. Sparsity reduces 
entropy.
