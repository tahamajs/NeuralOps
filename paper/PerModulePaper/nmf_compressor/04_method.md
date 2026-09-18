# 4. Methodology

## 4.1 Algorithmic Formulation
The `nmf_compressor` engine optimizes the layer parameterization via calibrated transformation routines.

```
NMF: Non-negative Matrix Factorization Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'nmf'.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
