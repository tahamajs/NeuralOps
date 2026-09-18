# 4. Methodology

## 4.1 Algorithmic Formulation
The `swift_svd_act` engine optimizes the layer parameterization via calibrated transformation routines.

```
Swift-SVD Activation-Aware Compressor via cross-covariance / activation subspace projection.
Registered as 'swift_svd_act'.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
