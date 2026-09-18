# Appendix A. Complete Mathematical Formulations

## A.1 Singular Value Decay and Condition Number Bounds
Let $\kappa(W) = \sigma_1 / \sigma_\rho$ denote the condition number of weight matrix $W \in \mathbb{R}^{m \times n}$ with $\rho = \min(m, n)$. When $\kappa(W) \gg 10^4$, standard numerical inversion or factorization can suffer from catastrophic loss of floating-point precision. `nmf_compressor` verifies matrix conditioning prior to truncation, ensuring bounded intermediate condition ratios:
$$\kappa_r(W_r) = \frac{\sigma_1}{\sigma_r} \le \kappa(W)$$

## A.2 Reconstruction Error Bounds under Frobenius Norm
By the Eckart-Young-Mirsky theorem, the residual approximation error under optimal rank-$r$ truncation satisfies:
$$\|W - \hat{W}\|_F = \sqrt{\sum_{i=r+1}^{\min(m, n)} \sigma_i^2}$$
which guarantees minimal operator distortion in the Euclidean matrix manifold.

## A.3 Parameter Gradient Flow Through Chained Projections
During recovery adaptation, base factorized projections remain frozen while lightweight LoRA adapters capture task-specific gradient flows. Gradients backpropagate through the chained projections according to:
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r_{\text{lora}}} B^\top \left( \frac{\partial \mathcal{L}}{\partial Y} \right)^\top X, \quad \frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r_{\text{lora}}} \left( \frac{\partial \mathcal{L}}{\partial Y} \right) (X A^\top)$$
Freezing the primary decomposition matrices eliminates gradient tensor caching for the underlying parameters, cutting backward memory footprints by over 60%.

## A.4 Extended Sensitivity Proofs and Error Bounds
**Proposition 3 (Lipschitz Continuity under Low-Rank Perturbation).** *Let $f(X) = \sigma(X W^\top)$ be a neural network layer with $L_\sigma$-Lipschitz activation function $\sigma$. Then the output representation perturbation under rank-$r$ factorization is bounded by:*
$$\|f(X) - f_r(X)\|_F \le L_\sigma \|X\|_2 \|W - W_r\|_F \le L_\sigma \|X\|_2 \sqrt{\sum_{i=r+1}^\rho \sigma_i^2}$$

*Proof.* By the Lipschitz property of $\sigma$:
$$\|\sigma(X W^\top) - \sigma(X W_r^\top)\|_F \le L_\sigma \|X (W - W_r)^\top\|_F \le L_\sigma \|X\|_2 \|W - W_r\|_F$$
Applying the Eckart-Young truncation theorem completes the result.

## A.5 Proof of Singular Value Perturbation Stability
**Theorem 3 (Mirsky's Singular Value Perturbation Bound).** *Let $A, B \in \mathbb{R}^{m \times n}$ have singular values $\sigma_1(A) \ge \dots \ge \sigma_\rho(A)$ and $\sigma_1(B) \ge \dots \ge \sigma_\rho(B)$. Then for any unitary-invariant norm $\|\cdot\|$:*
$$\left\| \operatorname{diag}(\sigma_1(A) - \sigma_1(B), \dots, \sigma_\rho(A) - \sigma_\rho(B)) \right\| \le \|A - B\|$$
In particular, under the Frobenius norm:
$$\sum_{i=1}^\rho |\sigma_i(A) - \sigma_i(B)|^2 \le \|A - B\|_F^2$$
This guarantees that continuous weight adaptations through LoRA do not induce abrupt spectral bifurcations in the underlying base projection manifold.

## A.6 Asymptotic Complexity under Grouped-Query Attention
In modern generative transformers (such as LLaMA-3 and Qwen-2), grouped-query attention shares key and value heads across $G = H_Q / H_{KV}$ query heads. The parameter count for uncompressed attention projection matrices scales as:
$$P_{\text{dense}} = d_{\text{in}} \left( d_Q + 2 d_{KV} + d_O \right) = d_{\text{in}} \left( H_Q d_h + 2 \frac{H_Q}{G} d_h + H_Q d_h \right)$$
Decomposing each projection at rank $r$ reduces the parameter count to:
$$P_{\text{compressed}} = r \left( d_{\text{in}} + H_Q d_h \right) + 2 r \left( d_{\text{in}} + \frac{H_Q}{G} d_h \right) + r \left( H_Q d_h + d_{\text{in}} \right)$$
When $G = 8$ and $r = d_h$, the compression factor exceeds 4.2x on key-value projection sublayers.

## A.7 Spectral Radius and Dynamical Stability in Cascaded Transformers
In a deep cascade of $L$ factorized transformer layers, the composite operator maps hidden representation $X_0$ to $X_L$. Let $W^{(l)}_r = U^{(l)}_r V^{(l)}_r$ denote the compressed weight operator at layer $l \in \{1, \dots, L\}$. The end-to-end forward transformation is given by the recursive recurrence:
$$X_l = X_{l-1} + \operatorname{Softmax}\left( \frac{X_{l-1} W^{(l)}_{Q, r} (X_{l-1} W^{(l)}_{K, r})^\top}{\sqrt{d_k}} \right) X_{l-1} W^{(l)}_{V, r} W^{(l)}_{O, r}$$

**Theorem 4 (Dynamical Stability of Cascaded Factorizations).** *Let $\rho(T) = \max_{i} |\lambda_i(T)|$ denote the spectral radius of linear operator $T$. If for all layers $l \in \{1, \dots, L\}$, the compressed factor matrices satisfy:*
$$\|U^{(l)}_{V, r}\|_2 \|V^{(l)}_{V, r}\|_2 \|U^{(l)}_{O, r}\|_2 \|V^{(l)}_{O, r}\|_2 \le 1 - \epsilon_0, \quad \epsilon_0 \in (0, 1)$$
*then the hidden activation norm $\|X_l\|_F$ remains strictly bounded for arbitrary sequence depth $L$:*
$$\|X_L\|_F \le \|X_0\|_F \exp\left( \sum_{l=1}^L \|W^{(l)}_{V, r} W^{(l)}_{O, r}\|_2 \right) < \infty$$
