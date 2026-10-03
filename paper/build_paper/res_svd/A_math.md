# Appendix A. Mathematical Derivations

## A.1 Equivalence to Rank-$(r_b + r_r)$ SVD

**Claim.** Res-SVD with $(r_b, r_r)$ is mathematically equivalent to 
truncated SVD of rank $r_b + r_r$.

**Proof.** Let $W = U \Sigma V^\top$ be the full SVD. Then:

$$
W_b = U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top
\tag{A.1}
$$

$$
R = W - W_b = U_{:, r_b:} \Sigma_{r_b:} V_{r_b:, :}^\top
\tag{A.2}
$$

The residual has the same left and right singular vectors as $W$, only 
shifted by $r_b$. Its top-$r_r$ SVD:

$$
R_{r_r} = U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
\tag{A.3}
$$

Sum:

$$
W_b + R_{r_r} = U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top + U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top = U_{:, :r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
\tag{A.4}
$$

This is exactly the rank-$(r_b + r_r)$ truncated SVD of $W$. $\square$

## A.2 Reconstruction Error

**Claim.** At initialization:

$$
\|W - W_b - R_{r_r}\|_F = \sqrt{\sum_{i > r_b + r_r} \sigma_i^2}
\tag{A.5}
$$

**Proof.** Direct from A.1. $\square$

## A.3 Parameter Count

**Claim.** Res-SVD per projection uses $(r_b + r_r)(d_\text{in} + d_\text{out}) + r_b + d_\text{out}$ 
parameters.

**Proof.**
- Base $U_b$: $d_\text{out} \times r_b$
- Base $V_b$: $r_b \times d_\text{in}$
- Residual $B$: $d_\text{out} \times r_r$
- Residual $A$: $r_r \times d_\text{in}$
- Bias: $d_\text{out}$

Total: $r_b(d_\text{in} + d_\text{out}) + r_r(d_\text{in} + d_\text{out}) + d_\text{out} = (r_b + r_r)(d_\text{in} + d_\text{out}) + d_\text{out}$. $\square$

## A.4 Comparison with PiSSA

**PiSSA.** Base is $W - BA$ where $BA = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$. 
LoRA takes the **principal** directions.

**Res-SVD.** Base is $U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top$. LoRA 
takes the **residual** directions.

At equal total rank ($r_b + r_r = r$), both give the same approximation 
at initialization. They differ in:
- Which part is frozen (Res-SVD freezes principal, PiSSA freezes residual).
- Gradient flow (Res-SVD trains residual, PiSSA trains principal).

## A.5 Causality

**Proof.** Position-wise projections + causal mask. $\square$

---