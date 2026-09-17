# Appendix A. Mathematical Proofs

## A.1 Equivalence of Covariance-Weighted SVD
We minimize $\|(W - \hat{W}) X^	op\|_F^2$. Let $C_X = rac{1}{N} X^	op X = C_X^{1/2} C_X^{1/2}$.
Then $\|(W - \hat{W}) X^	op\|_F^2 = N \|(W - \hat{W}) C_X^{1/2}\|_F^2$.
Letting $	ilde{W} = W C_X^{1/2}$ and $	ilde{W}_r = \hat{W} C_X^{1/2}$, the Eckart--Young theorem guarantees that the optimal rank-$r$ approximation $	ilde{W}_r$ is obtained via standard truncated SVD of $	ilde{W}$. Multiplying by $C_X^{-1/2}$ on the right yields the optimal unweighted factor $V_r$.
