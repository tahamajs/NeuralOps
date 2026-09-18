# 4. Methodology

## 4.1 Algorithmic Formulation
The `proxsparse` engine optimizes the layer parameterization via calibrated transformation routines.

```
ProxSparse: Proximal Gradient Regularized Structured & Semi-Structured Sparsity.
Registered in COMPRESSOR_REGISTRY as 'proxsparse', 'proxsparse_pruner', 'proxsparse_compressor',
'proxsparse_factory', 'proxsparse_pruner_factory', and 'proxsparse_compressor_factory'.

Mathematical Formulation:
77850\min_W \frac{1}{2} \|W - W_0\|_F^2 + \lambda \|W\|_1 + \mathcal{I}_{\mathcal{S}}(W)77850

Proximal Operators:
1. Soft-Thresholding:
   77850\mathcal{S}_{\lambda}(W) = \operatorname{sign}(W) \odot \max(|W| - \lambda, 0)77850
2. Sparsity Projection (2:4 Semi-Structured or Block/Percentile):
   77850\Pi_{\mathcal{S}}(W) = \operatorname{Top-}(1-s)(W) \quad \text{or} \quad \Pi_{2:4}(W)77850
3. Iterative Proximal Gradient Descent:
   77850W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta (W^{(t)} - \ma
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
