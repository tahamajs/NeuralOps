# 4. Methodology

## 4.1 Algorithmic Formulation
The `cur_linear` engine optimizes the layer parameterization via calibrated transformation routines.

```
CUR Linear Layer: low-rank cross-approximation using column and row skeleton selection.
W ≈ C @ U @ R
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
