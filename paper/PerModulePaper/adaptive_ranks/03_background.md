# 3. Mathematical Background
For weight matrix $W$ with singular values $\sigma_1 \ge \dots \ge \sigma_d \ge 0$, cumulative energy thresholding selects rank $r$ such that $\sum_{i=1}^r \sigma_i^2 \ge 	au \sum_{j=1}^d \sigma_j^2$. Incorporating input covariance $C_X = rac{1}{N} X^	op X$ minimizes output error $\| (W - \hat{W}) C_X^{1/2} \|_F^2$.
