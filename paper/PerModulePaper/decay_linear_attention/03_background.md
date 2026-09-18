# 3. Mathematical Background and Formal Notation

## 3.1 Notation and Tensor Spaces
Let $\mathcal{B}$ denote batch size, $N$ denote sequence length, and $d_{\text{in}}, d_{\text{out}}$ denote layer input and output channel dimensions. For multi-head attention modules, $H$ denotes the number of attention heads, $d$ denotes the head dimension ($d = d_{\text{in}} / H$), and $H_{\text{kv}}$ denotes the number of key/value heads in Grouped-Query Attention (GQA) settings.

The forward mapping of an uncompressed dense linear projection is formalized as:
$$Y = X W^\top + b$$
where $X \in \mathbb{R}^{\mathcal{B} \times N \times d_{\text{in}}}$, $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, $b \in \mathbb{R}^{d_{\text{out}}}$, and $Y \in \mathbb{R}^{\mathcal{B} \times N \times d_{\text{out}}}$.

## 3.2 Spectral Analysis and Singular Value Decomposition
For any linear transformation matrix $W \in \mathbb{R}^{m \times n}$ with rank $\rho = \min(m, n)$, its Singular Value Decomposition (SVD) is defined as:
$$W = U \Sigma V^\top = \sum_{i=1}^{\rho} \sigma_i u_i v_i^\top$$
where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ form orthonormal bases of left and right singular vectors, and $\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_\rho)$ contains singular values ordered monotonically:
$$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_\rho \ge 0$$

## 3.3 Eckart-Young-Mirsky Theorem
According to the Eckart-Young-Mirsky theorem, the optimal rank-$r$ approximation ($r < \rho$) under both the Frobenius norm and spectral operator norm is obtained by truncating the summation at index $r$:
$$W_r = U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^\top = \sum_{i=1}^{r} \sigma_i u_i v_i^\top$$
The approximation error is governed strictly by the discarded singular values:
$$\|W - W_r\|_F^2 = \sum_{i=r+1}^{\rho} \sigma_i^2, \quad \|W - W_r\|_2 = \sigma_{r+1}$$

## 3.4 Balanced Symmetric Square-Root Splitting
In finite-precision deep learning architectures (e.g., IEEE FP16 or BF16), assigning singular values asymmetrically to either $U$ or $V$ causes numerical dynamic range imbalances. `decay_linear_attention` adopts balanced symmetric square-root distribution:
$$U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad V_r = \Sigma_{:r, :r}^{1/2} V_{:, :r}^\top$$
which guarantees equalized Frobenius norms across both factorized projection matrices:
$$\|U_r\|_F = \|V_r\|_F = \left( \sum_{i=1}^{r} \sigma_i \right)^{1/2}$$

## 3.5 Regularized Optimization and Saliency Pruning
Under structural sparsity constraints, the optimization objective balances parameter fidelity against regularization penalties:
$$\min_{W} \mathcal{L}_{\text{task}}(W) + \lambda \mathcal{R}(W) + \mathcal{I}_{\mathcal{S}}(W)$$
where $\mathcal{R}(W)$ denotes a sparsity-inducing norm (such as $\ell_1$ or group Lasso) and $\mathcal{I}_{\mathcal{S}}(W)$ denotes the indicator function over the admissible hardware-aligned sparsity pattern $\mathcal{S}$:
$$\mathcal{I}_{\mathcal{S}}(W) = \begin{cases} 0 & \text{if } W \in \mathcal{S} \\ \infty & \text{otherwise} \end{cases}$$

## 3.6 Taylor Sensitivity Estimation
When calibration activations $X$ are available, parameter saliency is approximated via first-order Taylor expansion of the reconstruction loss:
$$S_{ij} = \left| \frac{\partial \mathcal{L}}{\partial W_{ij}} W_{ij} \right| \approx \left| W_{ij} \right| \cdot \mathbb{E}_{X} \left[ \left| X_j \right| \right]$$
Second-order Fisher information incorporates empirical activation covariances:
$$H = \frac{2}{N} X^\top X + \lambda I$$
with parameter importance quantified via inverse Hessian diagonals:
$$S_{ij}^{(2)} = \frac{W_{ij}^2}{[H^{-1}]_{jj}}$$
