<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Nyström Approximation Error

**Theorem.** For a kernel matrix $K \in \mathbb{R}^{N \times N}$ with 
eigenvalues $\lambda_1 \geq \lambda_2 \geq \dots$, the Nyström 
approximation $\tilde{K} = K_{:, m} K_{m, m}^+ K_{m, :}$ satisfies:

$$
\|K - \tilde{K}\|_F^2 \leq \|K\|_F^2 - \sum_{i=1}^{m} \lambda_i^2
\tag{A.1}
$$

**Proof.** Standard result (see Golub & Van Loan). $\square$

## A.2 Nyström Attention as Reconstruction

**Claim.** If the attention matrix $A = \text{softmax}(Q K^\top)$ has rank $\leq m$, 
then:

$$
A = A_{:, m} A_{m, m}^{-1} A_{m, :}
\tag{A.2}
$$

where $A_{:, m}$ is columns at landmarks, etc.

**Proof.** Rank-$m$ matrix reconstruction. $\square$

## A.3 Newton-Schulz Convergence

**Proposition.** For $A \in \mathbb{R}^{m \times m}$ with $\|A\|_2 < 1$, 
the iteration $V_{k+1} = V_k (2I - A V_k)$ with $V_0 = A^\top$ satisfies:

$$
\|I - A V_{k+1}\|_2 \leq \|I - A V_k\|_2^2
\tag{A.3}
$$

**Proof.** Let $E_k = I - A V_k$. Then:

$$
E_{k+1} = I - A V_{k+1} = I - A V_k (2I - A V_k) = (I - A V_k)^2 = E_k^2
\tag{A.4}
$$

Taking norms: $\|E_{k+1}\| \leq \|E_k\|^2$. $\square$

**Convergence.** Quadratic. If $\|E_0\| < 1$, then $\|E_k\| \leq \|E_0\|^{2^k}$.

## A.4 Preconditioning

For general $A$ with $\|A\|_2 \geq 1$, precondition:

$$
\alpha = \frac{1}{\|A\|_1 \|A\|_\infty + \epsilon}
\tag{A.5}
$$

Then $\|\alpha A\|_2 \leq 1$ by Gelfand's formula. Iterate on $\alpha A$:

$$
V_0 = (\alpha A)^\top, \quad V_{k+1} = V_k (2I - \alpha A V_k)
\tag{A.6}
$$

The approximate pinv is $\alpha V_k \to A^+$.

## A.5 Complexity of Associative Multiplication

**Claim.** Computing $F_1 \cdot \text{pinv}(F_2) \cdot F_3 V$ in the order 
$F_1 (\text{pinv}(F_2) (F_3 V))$ has complexity:

$$
\mathcal{O}(N m d_h + m^2 d_h + N m d_h) = \mathcal{O}(N m d_h + m^2 d_h)
\tag{A.7}
$$

**Proof.** Each matrix multiplication is:
- $F_3 V$: $(m \times N) (N \times d_h) \to \mathcal{O}(N m d_h)$
- $\text{pinv}(F_2) (\cdot)$: $(m \times m) (m \times d_h) \to \mathcal{O}(m^2 d_h)$
- $F_1 (\cdot)$: $(N \times m) (m \times d_h) \to \mathcal{O}(N m d_h)$

For $m \ll N$: dominated by $N m d_h$. $\square$

## A.6 Dense Fallback Threshold

We use dense attention when $N \leq \max(128, 2m)$.

**Justification.** For $N \leq 2m$, the cost of Nyström $\mathcal{O}(N m d_h)$ 
is comparable to dense $\mathcal{O}(N^2 d_h)$, while Nyström introduces 
approximation error.

## A.7 Causal Non-Applicability

**Proposition.** Nyström approximation of a causal attention matrix 
requires landmark tokens that respect causality.

**Proof.** Adaptive pooling averages across the entire sequence. A landmark 
at position $i$ would include future tokens $j > i$, violating causality. $\square$

## A.8 Parameter Count Unchanged

**Claim.** Nyströmformer does not change parameter count.

**Proof.** The projections $W_Q, W_K, W_V, W_O$ are unchanged. The 
Nyström approximation is a **compute** transformation, not a parameter 
transformation. $\square$

## A.9 Memory Complexity

| Path | Memory |
|---|---|
| Dense | $\mathcal{O}(N^2 + N d_h)$ |
| Nyström | $\mathcal{O}(N m + m^2 + N d_h)$ |

For $m \ll N$: Nyström memory is $\mathcal{O}(N m)$ instead of 
$\mathcal{O}(N^2)$.

---