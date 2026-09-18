# 4. Methodology

## 4.1 Algorithmic Formulation
The `saes_svd_v2` engine optimizes the layer parameterization via calibrated transformation routines.

```
SAES-SVD v2: Spectrum-Aware Error-Suppressed Singular Value Decomposition.
Registered in COMPRESSOR_REGISTRY as 'saes_svd_v2' and 'saes'.

Mathematical Formulation:
1. Truncated SVD Baseline:
   $$W_0 \\approx U_r \\Sigma_r V_r^\\top = A_0 B_0$$
2. Spectral Energy Distribution:
   $$\\rho_k = \\frac{\\sigma_k^2}{\\sum_{j=1}^r \\sigma_j^2}$$
3. Error-Suppression Iterations:
   $$E_t = W_0 - A_t B_t$$
   $$A_{t+1} = A_t + E_t B_t^\\top (B_t B_t^\\top + \\epsilon I)^{-1}$$
   $$B_{t+1} = B_t + (A_{t+1}^\\top A_{t+1} + \\epsilon I)^{-1} A_{t+1}^\\top E_t$$

Production Hardening:
- Fix 1: Default fallback_to_original changed to True for safety.
- Fix 2: Always log warning when reconstruction error exceeds max_error regardless of fallback.
- Fix 3: Reliable Conv1D detection using class name or n
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
