# Appendix A. Complete Mathematical Formulations

## A.1 Singular Value Decay and Condition Number Bounds
Let $\kappa(W) = \sigma_1 / \sigma_\rho$ denote the condition number of weight matrix $W$. When $\kappa(W) \gg 10^4$, standard numerical inversion or factorization can suffer from loss of precision. `decay_linear_attention` checks matrix conditioning prior to truncation:
$$\kappa_r(W_r) = \frac{\sigma_1}{\sigma_r}$$

## A.2 Reconstruction Error Bounds under Frobenius Norm
The residual approximation error satisfies:
$$\|W - \hat{W}\|_F = \sqrt{\sum_{i=r+1}^{\min(m, n)} \sigma_i^2}$$

## A.3 Parameter Gradient Flow Through Chained Projections
During LoRA adaptation, gradients backpropagate through the chained projections:
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r_{\text{lora}}} B^\top \left( \frac{\partial \mathcal{L}}{\partial Y} \right)^\top X, \quad \frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r_{\text{lora}}} \left( \frac{\partial \mathcal{L}}{\partial Y} \right) (X A^\top)$$
