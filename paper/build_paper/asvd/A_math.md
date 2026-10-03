<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Eckart-Young Theorem

**Claim.** The best rank-$r$ approximation of $W$ in Frobenius norm is the 
truncated SVD $W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$.

**Proof sketch.** For any rank-$r$ matrix $X$:

$$
\|W - X\|_F^2 = \|W\|_F^2 - 2 \langle W, X \rangle + \|X\|_F^2
$$

The optimal $X$ aligns with the top-$r$ singular directions. Full proof in 
Eckart-Young (1936). $\square$

## A.2 Energy Lower Bound

**Claim.** For energy threshold $\varepsilon$:

$$
\|W - U_r \Sigma_r V_r^\top\|_F \leq (1 - \varepsilon) \|W\|_F
$$

**Proof.**

$$
\|W\|_F^2 = \sum_{i=1}^{\min(m, n)} \sigma_i^2
$$

$$
\|W - U_r \Sigma_r V_r^\top\|_F^2 = \sum_{i > r} \sigma_i^2 = (1 - E(r)) \|W\|_F^2
$$

If $E(r) \geq \varepsilon$, then $(1 - E(r)) \leq (1 - \varepsilon)$, giving 
the bound. Taking square roots gives the claim. $\square$

## A.3 Activation-Aware Objective

**Claim.** Under diagonal covariance $C_X = \text{diag}(F)$, minimizing 
$\mathbb{E}_x[\|(W - W_r) x\|_2^2]$ is equivalent to minimizing 
$\|(W - W_r) \text{diag}(\sqrt{F})\|_F^2$.

**Proof.**

$$
\mathbb{E}_x[\|(W - W_r) x\|_2^2] = \text{Tr}[(W - W_r) C_X (W - W_r)^\top]
= \sum_{i, j} F_j (W - W_r)_{ij}^2
= \|(W - W_r) \text{diag}(\sqrt{F})\|_F^2 \qquad \square
$$

## A.4 Complexity Analysis

**Claim.** The full compression of an $N_\ell$-layer model is 
$\mathcal{O}(N_\ell d^3)$ in time.

**Proof.** For each layer, SVD of a $d \times d$ matrix takes 
$\mathcal{O}(d^3)$. There are $N_\ell$ layers, giving $\mathcal{O}(N_\ell d^3)$. 
For DistilBERT ($N_\ell = 6$, $d = 768$): $\sim 2 \times 10^9$ operations, 
~10 seconds on RTX 4070. $\square$

## A.5 Parameter Count

**Claim.** ASVD reduces parameters from $d_\text{in} d_\text{out}$ to 
$r(d_\text{in} + d_\text{out} + 1)$.

**Proof.** $U$ has $d_\text{out} \times r$ parameters. $\Sigma$ has $r$. 
$V$ has $r \times d_\text{in}$. Total: $r(d_\text{out} + d_\text{in} + 1)$. $\square$

Reduction factor: $\frac{d_\text{in} d_\text{out}}{r(d_\text{in} + d_\text{out} + 1)}$.

For $d_\text{in} = d_\text{out} = 768$, $r = 64$:

$$
\frac{589,824}{98,368} \approx 6.0
$$

## A.6 Causality

**Claim.** ASVD preserves causality.

**Proof.** Each projection is position-wise. The attention mask enforces 
$j \leq i$. Therefore output at position $i$ depends only on inputs 
$1..i$. $\square$

---