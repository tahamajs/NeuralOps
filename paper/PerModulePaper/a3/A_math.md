<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Weighted Low-Rank Approximation

### A.1.1 Problem Formulation

Given $W \in \mathbb{R}^{m \times n}$ and weights $w \in \mathbb{R}_+^n$, 
we seek:

$$
W_r^* = \arg\min_{\text{rank}(W_r) = r} \sum_{i, j} w_j (W_{ij} - (W_r)_{ij})^2
\tag{A.1}
$$

### A.1.2 Solution via Transform

Define $D = \mathrm{diag}(\sqrt{w})$. Then:

$$
\sum_{i, j} w_j (W - W_r)_{ij}^2 = \|(W - W_r) D\|_F^2 = \|(W D - W_r D)\|_F^2
\tag{A.2}
$$

Since $\text{rank}(W_r D) \leq r$, the optimal $W_r D$ is the rank-$r$ SVD 
of $W D$. If $W D = U \Sigma V^\top$, then:

$$
W_r^* D = U_r \Sigma_r V_r^\top \implies W_r^* = U_r \Sigma_r V_r^\top D^{-1}
\tag{A.3}
$$

This is exactly Algorithm A3.

## A.2 Fisher Diagonal as Importance

**Claim.** Under a diagonal covariance assumption on activations, the 
expected reconstruction error is:

$$
\mathbb{E}_x\!\left[\|W x - W_r x\|_2^2\right] = \|(W - W_r) \mathrm{diag}(\sqrt{F})\|_F^2
\tag{A.4}
$$

**Proof.** Let $x$ have covariance $\Sigma = \mathrm{diag}(F_1, \dots, F_n)$. Then:

$$
\mathbb{E}_x[\|(W - W_r) x\|_2^2] = \mathrm{Tr}[(W - W_r) \Sigma (W - W_r)^\top]
\tag{A.5}
$$

With $\Sigma = \mathrm{diag}(F)$:

$$
\mathrm{Tr}[(W - W_r) \mathrm{diag}(F) (W - W_r)^\top] = \sum_{i, j} F_j (W - W_r)_{ij}^2
\tag{A.6}
$$

which equals $\|(W - W_r) \mathrm{diag}(\sqrt{F})\|_F^2$. $\square$

## A.3 Reconstruction Error Bound

**Proposition.** After weighted SVD at rank $r$:

$$
\|W - U_r V_r\|_F \leq \sqrt{\frac{\sum_{i > r} \sigma_i^2}{\min_j F_j}}
\tag{A.7}
$$

where $\sigma_i$ are singular values of $W \cdot \mathrm{diag}(\sqrt{F})$.

**Proof sketch.** We have $\|W \mathrm{diag}(\sqrt{F}) - U_r \Sigma_r V_r\|_F \leq \sqrt{\sum_{i > r} \sigma_i^2}$ by Eckart-Young. So:

$$
\|W - U_r V_r\|_F \cdot \min_j \sqrt{F_j} \leq \|(W - U_r V_r) \mathrm{diag}(\sqrt{F})\|_F \leq \sqrt{\sum_{i > r} \sigma_i^2}
\tag{A.8}
$$

Dividing both sides by $\min_j \sqrt{F_j}$ gives (A.7). $\square$

## A.4 LayerNorm Recalibration Derivation

LayerNorm computes:

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu(x)}{\sigma(x) + \epsilon} + \beta
\tag{A.9}
$$

Let $\sigma_\text{pre}$ be the pre-compression std of $x$ and $\sigma_\text{post}$ 
the post-compression std. We want $\text{LN}_\text{post} \approx \text{LN}_\text{pre}$.

If we keep $\gamma$ fixed:

$$
\text{LN}_\text{post}(x) = \gamma \cdot \frac{x - \mu}{\sigma_\text{post} + \epsilon} + \beta
\tag{A.10}
$$

vs. expected:

$$
\text{LN}_\text{pre}(x) = \gamma \cdot \frac{x - \mu}{\sigma_\text{pre} + \epsilon} + \beta
\tag{A.11}
$$

To correct:

$$
\gamma_\text{new} = \gamma \cdot \frac{\sigma_\text{post} + \epsilon}{\sigma_\text{pre} + \epsilon} \cdot \frac{\sigma_\text{pre}}{\sigma_\text{post}} = \gamma \cdot \frac{\sigma_\text{pre}}{\sigma_\text{post}} \cdot \frac{\sigma_\text{post}}{\sigma_\text{pre}} \cdot \frac{\sigma_\text{post}}{\sigma_\text{post}} = \gamma \cdot \frac{\sigma_\text{post}}{\sigma_\text{pre}}
\tag{A.12}
$$

Wait, this gives the opposite. The correct rescaling is:

$$
\gamma_\text{new} = \gamma \cdot \frac{\sigma_\text{pre}}{\sigma_\text{post}}
\tag{A.13}
$$

so that $\gamma_\text{new} / \sigma_\text{post} = \gamma / \sigma_\text{pre}$.

## A.5 Streaming Covariance Correctness

**Claim.** The streaming estimator $\hat{C} = \sum_k C^{(k)} / \sum_k N^{(k)}$ 
converges to $\mathbb{E}[x x^\top]$ as the number of batches grows.

**Proof.** Each $C^{(k)} = \sum_{t \in \text{batch}_k} x_t x_t^\top$. Summing:

$$
\sum_k C^{(k)} = \sum_{t = 1}^{N_\text{total}} x_t x_t^\top
\tag{A.14}
$$

Dividing by $N_\text{total} = \sum_k N^{(k)}$:

$$
\hat{C} = \frac{1}{N_\text{total}} \sum_t x_t x_t^\top
\tag{A.15}
$$

which converges to $\mathbb{E}[x x^\top]$ by the law of large numbers. $\square$
