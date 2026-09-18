# 4. Methodology

## 4.1 Algorithmic Formulation
The `slicegpt` engine optimizes the layer parameterization via calibrated transformation routines.

```
SliceGPT: Compress Large Language Models by Deleting Rows and Columns.
Registered in COMPRESSOR_REGISTRY as 'slicegpt'.

Mathematical Formulation:
1. Orthogonal Coordinate Invariance:
   $$W_{\\ell} X_{\\ell} = (W_{\\ell} Q_{\\ell}) (Q_{\\ell}^\\top X_{\\ell})$$
2. Eigen-decomposition of Activation Covariance:
   $$C = X X^\\top = Q \\Lambda Q^\\top$$
3. Coordinate Slicing Transformation:
   $$W_{\\text{sliced}} = W Q P_r, \\quad P_r = [I_r, 0]^\\top$$

Production Hardening:
- Fix 1: Default fallback_to_original changed to True.
- Fix 2: Reliable Conv1D detection by class name and nf attribute without shape heuristics.
- Fix 3: Always log warning when reconstruction error exceeds max_error regardless of fallback.
- Fix 4: Unified critical-layer protection delegated solely to should_compres
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
