# 3. Mathematical Background
Let $W = U \Sigma V^\top$ be the singular value decomposition of projection matrix $W$, with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_d \ge 0$. The total Frobenius energy of $W$ equals:
$$\|W\|_F^2 = \sum_{i=1}^d \sigma_i^2$$
The cumulative energy captured by the top $r$ components is $\sum_{i=1}^r \sigma_i^2$. Setting an energy retention target $\tau \in (0, 1]$ allows rank $r$ to adapt directly to the spectral profile of the matrix.
