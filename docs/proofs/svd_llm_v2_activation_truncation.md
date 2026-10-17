# Mathematical Proof: SVD-LLM v2 & HNC Activation-Aware Truncation Optimality

## 1. Problem Formulation
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote the original weight matrix of a linear projection, and let $X \in \mathbb{R}^{N \times d_{\text{in}}}$ be calibration activation features sampled from real data.

We seek a low-rank approximation $W_r \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with $\operatorname{rank}(W_r) \le r < \min(d_{\text{out}}, d_{\text{in}})$ minimizing the activation-weighted Frobenius error:
$$\min_{\operatorname{rank}(W_r) \le r} \|X(W - W_r)^T\|_F^2 = \min_{\operatorname{rank}(W_r) \le r} \|(W - W_r) X^T\|_F^2$$

## 2. Unweighted vs. Whitened Reformulation
Define the empirical input second-moment (covariance) matrix:
$$H_X = \frac{1}{N} X^T X \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$
Adding Tikhonov regularization $\epsilon I$ ($\epsilon > 0$) ensures $H_X + \epsilon I \succ 0$, admitting a unique lower-triangular Cholesky factor:
$$L L^T = H_X + \epsilon I, \quad L \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

Then the weighted objective can be expressed as:
$$\frac{1}{N} \|(W - W_r)X^T\|_F^2 = \operatorname{Tr}\left((W - W_r) H_X (W - W_r)^T\right) \approx \|(W - W_r) L\|_F^2$$

Let $\widetilde{W} = W L \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and $\widetilde{W}_r = W_r L$. Since $L$ is invertible:
$$\operatorname{rank}(\widetilde{W}_r) = \operatorname{rank}(W_r L) = \operatorname{rank}(W_r) \le r$$

Therefore, the problem is isomorphic to:
$$\min_{\operatorname{rank}(\widetilde{W}_r) \le r} \|\widetilde{W} - \widetilde{W}_r\|_F^2$$

## 3. Eckart-Young-Mirsky Theorem in Whitened Coordinate Space
By the Eckart-Young-Mirsky Theorem, the unique global minimizer of $\|\widetilde{W} - \widetilde{W}_r\|_F^2$ is given by the truncated SVD of $\widetilde{W}$:
$$\widetilde{W} = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i u_i v_i^T, \quad \widetilde{W}_r = \sum_{i=1}^r \sigma_i u_i v_i^T$$
The minimal error in the whitened space is:
$$\min \|\widetilde{W} - \widetilde{W}_r\|_F^2 = \sum_{i=r+1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i^2(\widetilde{W})$$

## 4. Inversion and Suboptimality of Naive SVD
To recover $W_r$, invert the Cholesky factor:
$$W_r^* = \widetilde{W}_r L^{-1} = U_r S_r V_r^T L^{-1}$$
computed stably via triangular back-substitution:
$$V_{\text{unwhitened}} = (L^T)^{-1} (S_r^{1/2} V_r^T)^T$$

Conversely, standard unweighted (naive) SVD directly truncates $W$:
$$W_{\text{naive}} = \sum_{i=1}^r s_i(W) \bar{u}_i \bar{v}_i^T$$
Since $W_{\text{naive}}$ minimizes $\|W - \widehat{W}\|_F^2$ but ignores $H_X$, its weighted error satisfies:
$$\|(W - W_{\text{naive}})L\|_F^2 \ge \|\widetilde{W} - \widetilde{W}_r^*\|_F^2 = \|(W - W_r^*)L\|_F^2$$
Equality holds if and only if $H_X = \alpha I$, which never occurs in practical transformer activations with anisotropic outlier channels.

Hence:
$$\|X(W - W_r^*)^T\|_F^2 \le \|X(W - W_{\text{naive}})^T\|_F^2 \quad \text{with strict inequality for anisotropic features.}$$
$\blacksquare$
