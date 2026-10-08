<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background needed for the method
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages (600 - 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Singular Value Decomposition

For any matrix $W \in \mathbb{R}^{m \times n}$, the SVD is:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are 
orthogonal, and $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with 
non-negative entries $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD.** Keeping the top-$r$ singular values gives the best rank-$r$ 
approximation in Frobenius norm:

$$
W_r = U_{:, :r} \Sigma_{:, :r} V_{:r, :}^\top
\tag{3.2}
$$

with error $\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

**Symmetric factorization.** For numerical stability, it is common to split 
$\Sigma^{1/2}$ between the two factors:

$$
U_r = U_{:, :r} \Sigma_{:, :r}^{1/2}, \qquad
V_r = \Sigma_{:r, :}^{1/2} V_{:r, :}^\top
\tag{3.3}
$$

so that $W_r = U_r V_r$.

## 3.2 The Fisher Information Matrix

For a probabilistic model $p_\theta(y | x)$, the Fisher information is:

$$
\mathcal{F}_{ij} = \mathbb{E}_{x, y}\!\left[\frac{\partial \log p_\theta(y | x)}{\partial \theta_i} \cdot \frac{\partial \log p_\theta(y | x)}{\partial \theta_j}\right]
\tag{3.4}
$$

For a linear layer $y = Wx + b$, the Fisher with respect to $x$ is:

$$
\mathcal{F}_x = \mathbb{E}_{x}\!\left[\frac{\partial \mathcal{L}}{\partial x} \frac{\partial \mathcal{L}}{\partial x}^\top\right]
\tag{3.5}
$$

The **diagonal** of this Fisher measures the importance of each input 
dimension. We use a cheap proxy: since $\partial \mathcal{L} / \partial x$ is 
proportional to $W^\top \cdot \partial \mathcal{L} / \partial y$, and since 
we assume the downstream loss has roughly uniform sensitivity to $y$, the 
diagonal Fisher is approximately:

$$
\mathcal{F}_{x, i} \approx \mathbb{E}_{x}\!\left[x_i^2\right] = F_i
\tag{3.6}
$$

This **activation energy** is cheap to compute and correlates well with the 
true Fisher in practice.

## 3.3 Streaming Covariance Accumulation

Given a stream of activation vectors $x_1, x_2, \dots, x_N \in \mathbb{R}^d$, 
we want to compute:

$$
C = \frac{1}{N} \sum_{t=1}^N x_t x_t^\top, \qquad
F = \frac{1}{N} \sum_{t=1}^N x_t^2
\tag{3.7}
$$

Naively this requires storing all $x_t$, which costs $\mathcal{O}(Nd)$ memory. 
The **streaming** formulation accumulates only the running sums:

$$
C^{(k)} = C^{(k-1)} + \sum_{t \in \text{batch}_k} x_t x_t^\top, \qquad
N^{(k)} = N^{(k-1)} + |\text{batch}_k|
\tag{3.8}
$$

with $\mathcal{O}(d^2)$ memory. We use this in A3.

**Padding handling.** Padding tokens (all-zeros or mask-tagged) are filtered 
out before accumulation.

## 3.4 LayerNorm

LayerNorm (Ba et al., 2016) normalizes activations across the feature 
dimension:

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta, \qquad
\sigma = \sqrt{\mathbb{E}[(x - \mu)^2] + \epsilon}
\tag{3.9}
$$

where $\gamma, \beta \in \mathbb{R}^d$ are learnable parameters.

**Distributional shift.** When a preceding layer is replaced by a factorized 
approximation, its output distribution changes. The LayerNorm that follows 
sees a different input distribution and produces different normalized 
activations. Recalibrating $\gamma$ can partially absorb this shift.

## 3.5 LoRA Recovery

LoRA (Hu et al., 2021) parameterizes weight updates as:

$$
\Delta W = \frac{\alpha}{r} B A
\tag{3.10}
$$

with $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, 
$B = 0$ at initialization. Only $A, B$ are trainable.

**Interaction with A3.** We add LoRA adapters to the factorized layer, 
yielding:

$$
y = (x V^\top) U^\top + (x A^\top) B^\top \cdot s + b
\tag{3.11}
$$

where $s = \alpha / r$ is the scaling factor. The frozen $U, V$ provide the 
bulk of the computation; LoRA provides the recovery signal.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix (out × in) |
| $U, V$ | Low-rank factors |
| $r$ | Rank |
| $F$ | Fisher diagonal (length $d_\text{in}$) |
| $C_X$ | Activation covariance ($d_\text{in} \times d_\text{in}$) |
| $\gamma, \beta$ | LayerNorm parameters |
| $\sigma$ | Standard deviation |
| $N$ | Number of calibration tokens |
| $\tau$ | Reconstruction error threshold |
| $\epsilon$ | Numerical stability constant |
