# 4. Methodology

## 4.1 Algorithmic Formulation
The `hashed_net_compressor` engine optimizes the layer parameterization via calibrated transformation routines.

```
HashedNet Compressor: replaces nn.Linear and Conv1D layers with HashedLinear.
Registered in COMPRESSOR_REGISTRY as 'hashed_net', 'hashednets', and 'hashednets_compressor'.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
