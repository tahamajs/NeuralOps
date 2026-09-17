# 4. Methodology

## 4.1 Algorithmic Formulation
The `dynamic_rank` engine optimizes the layer parameterization via calibrated transformation routines.

```
Dynamic Entropy Singular Value Decomposition (Dynamic Entropy SVD) Compressor.

Implements singular entropy computation over empirical layer activation covariances:
$$H(\\Sigma_l) = -\\sum_{i=1}^{d} p_i \\ln(p_i), \\quad p_i = \\frac{\\sigma_i}{\\sum_j \\sigma_j}$$
Dynamically budgets layer ranks $r_l \\in [r_{\\min}, r_{\\max}]$ proportional to singular entropy,
subject to global rank or parameter budgets, and replaces dense linear layers with low-rank
factorized `DynamicSVDLinear` modules.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
