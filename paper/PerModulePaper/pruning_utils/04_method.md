# 4. Methodology

## 4.1 Algorithmic Formulation
The `pruning_utils` engine optimizes the layer parameterization via calibrated transformation routines.

```
Pruning Utilities and Saliency Estimation for Structural Pruning and Layer Shedding.

Provides shared mathematical routines:
1. First-order Taylor expansion saliency: S_ij = |W_ij * g_ij| or data-driven S_ij = |W_ij| * E[|x_j|]
2. Second-order Empirical Hessian / Inverse Hessian saliency: S_ij = |W_ij|^2 / [H^-1]_jj
3. Semi-structured 2:4 sparsity projection: Pi_2:4(W, S)
4. Block-structured sparsity projection: Pi_BxB(W, S)
5. Layer shedding sensitivity / residual magnitude metrics
6. Fallback calibration data generation
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
