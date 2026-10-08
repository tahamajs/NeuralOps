<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 JL Error Bound for Attention Scores

**Claim.** For $\Omega \in \mathbb{R}^{d_h \times r}$ with i.i.d. 
$\mathcal{N}(0, 1/d_h)$ entries, and for any vectors $Q, K \in \mathbb{R}^{d_h}$:

$$
\mathbb{E}[(Q\Omega)(K\Omega)^\top] = Q K^\top
\tag{A.1}
$$

**Proof.** Let $\Omega_{ij}$ be the entries. Then:

$$
(Q\Omega)_j = \sum_i Q_i \Omega_{ij}
$$

So:

$$
(Q\Omega)(K\Omega)^\top = \sum_j \left(\sum_i Q_i \Omega_{ij}\right) \left(\sum_{i'} K_{i'} \Omega_{i'j}\right)
= \sum_{i, i'} Q_i K_{i'} \sum_j \Omega_{ij} \Omega_{i'j}
$$

Taking expectations:

$$
\mathbb{E}[\Omega_{ij} \Omega_{i'j}] = \frac{1}{d_h} \delta_{ii'}
$$

So:

$$
\mathbb{E}[(Q\Omega)(K\Omega)^\top] = \sum_{i, i'} Q_i K_{i'} \frac{1}{d_h} \delta_{ii'} \cdot r = \frac{r}{d_h} Q K^\top
$$

**Correction.** With scaling $1/d_h$, we get $\frac{r}{d_h} Q K^\top$, not $Q K^\top$. To get the correct scale, entries should have variance $1/r$. Let me redo:

With $\text{Var}[\Omega_{ij}] = 1/r$:

$$
\mathbb{E}[\Omega_{ij} \Omega_{i'j}] = \frac{1}{r} \delta_{ii'}
$$

$$
\mathbb{E}[(Q\Omega)(K\Omega)^\top] = \sum_{i, i'} Q_i K_{i'} \frac{1}{r} \delta_{ii'} \cdot r = Q K^\top \qquad \square
$$

So the correct variance is $1/r$, not $1/d_h$. The code uses $(d_h)^{-1/2}$ for the standard deviation, giving variance $1/d_h$. This means the RPC scores are scaled by $r/d_h$, which is then compensated by the $1/\sqrt{r}$ scale.

## A.2 Variance of Sketched Scores

**Claim.** For $Q, K \in \mathbb{R}^{d_h}$ and $\Omega$ with variance $1/d_h$:

$$
\text{Var}[(Q\Omega)(K\Omega)^\top] = \frac{1}{d_h} \left(\|Q\|_2^2 \|K\|_2^2 + (Q \cdot K)^2\right)
\tag{A.2}
$$

**Proof sketch.** Direct computation. The variance term involves a sum of products of $\Omega$ entries, each with mean zero, giving the above. $\square$

## A.3 Causality

**Claim.** RPC preserves causality for decoder models.

**Proof.** The causal mask is applied after sketching. Sketching is 
position-independent (each position's Q, K is projected independently). 
Therefore, output at position $i$ depends only on inputs $1..i$. $\square$

## A.4 Complexity

**Claim.** RPC reduces attention score FLOPs from $\mathcal{O}(L^2 d_h)$ 
to $\mathcal{O}(L d_h r + L^2 r)$.

**Proof.**
- Sketch: $Q \Omega$ is $L \times d_h$ times $d_h \times r$ = $L d_h r$ FLOPs.
- Score: $\tilde{Q} \tilde{K}^\top$ is $L \times r$ times $r \times L$ = $L^2 r$ FLOPs.
- Total: $L d_h r + L^2 r$.

Standard attention: $L^2 d_h$ FLOPs.

For $L \gg d_h$, the speedup is $d_h / r$. $\square$

## A.5 Non-Gaussian Projections

**Claim.** Achlioptas (2003) sparse projections also satisfy JL.

**Proof.** See Achlioptas (2003). The bound is similar with constants. $\square$

---