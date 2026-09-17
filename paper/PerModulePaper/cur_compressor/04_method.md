# 4. Methodology

## 4.1 Algorithmic Formulation
The `cur_compressor` engine optimizes the layer parameterization via calibrated transformation routines.

```
CUR Compressor: replaces Linear and Conv1D layers with CUR decomposition.
Uses leverage score sampling or energy sampling for column/row selection.
Registered in COMPRESSOR_REGISTRY as 'cur' and 'cur_compressor'.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
