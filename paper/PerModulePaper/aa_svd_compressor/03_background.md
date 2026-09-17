# 3. Mathematical Background

## 3.1 Truncated SVD
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Full Singular Value Decomposition yields:
$$W = U \Sigma V^\top$$
where $U \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ and $V \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ are orthogonal matrices, and $\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_k)$ contains singular values in descending order $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.

Truncation to rank $r < k$ selects the top $r$ singular components:
$$\hat{W} = U_r \Sigma_r V_r^\top$$

## 3.2 Output Error Minimization
Given input representations $X \in \mathbb{R}^{B \times d_{\text{in}}}$, the layer output error is:
$$\mathcal{L} = \| W X^\top - \hat{W} X^\top \|_F^2 = \operatorname{Tr}\left( (W - \hat{W}) X^\top X (W - \hat{W})^\top \right) = \| (W - \hat{W}) C_X^{1/2} \|_F^2$$
where $C_X = \frac{1}{B} X^\top X \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is the uncentered empirical activation covariance matrix.
