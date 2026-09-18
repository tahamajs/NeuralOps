# 4. Methodology

## 4.1 Algorithmic Formulation
The `uniql_compressor` engine optimizes the layer parameterization via calibrated transformation routines.

```
UniQL: Joint low‑rank + quantisation using SVD and uniform quant. Registered as 'uniql'.
Production-hardened with:
- Relative Frobenius reconstruction error validation (err <= max_error)
- Adaptive bit and rank retries
- Conv1D support (GPT-2)
- Critical layer protection (classifier, score, lm_head, pooler, embedding)
- Safe fallback preserving original weights
- Detailed per-layer error stats logging
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
