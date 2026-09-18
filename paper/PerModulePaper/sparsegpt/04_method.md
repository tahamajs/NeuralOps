# 4. Methodology

## 4.1 Algorithmic Formulation
The `sparsegpt` engine optimizes the layer parameterization via calibrated transformation routines.

```
SparseGPT: One-Shot Layer-Wise Weight Pruning with Second-Order Error Compensation.
Registered in COMPRESSOR_REGISTRY as 'sparsegpt'.

Mathematical Formulation:
1. Empirical Hessian Formulation:
   $$H = 2 X X^\\top + \\lambda I \\in \\mathbb{R}^{d_{in} \\times d_{in}}$$
2. Cholesky Factorization & Inversion:
   $$H = L L^\\top \\implies H^{-1} = (L^{-1})^\\top L^{-1}$$
3. OBS Saliency Metric for Column $j$:
   $$S_{ij} = \\frac{|W_{ij}|^2}{2 [H^{-1}]_{jj}}$$
4. 2:4 Structured or Unstructured Mask $\\mathcal{M}$:
   $$W_{pruned} = W \\odot \\mathcal{M}$$
5. Optimal Brain Surgeon Column Update:
   $$W_{:, j+1:d_{in}} \\leftarrow W_{:, j+1:d_{in}} - (W_{:, j} - W_{pruned, :, j}) \\frac{[H^{-1}]_{j, j+1:d_{in}}}{[H^{-1}]_{jj}}$$
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
