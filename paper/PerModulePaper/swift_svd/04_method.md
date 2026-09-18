# 4. Methodology

## 4.1 Algorithmic Formulation
The `swift_svd` engine optimizes the layer parameterization via calibrated transformation routines.

```
Swift-SVD: fast closed-form truncated SVD with symmetric square-root singular value distribution.
Registered as 'swift_svd'.
Production-hardened with:
- Relative Frobenius reconstruction error validation (err <= max_error)
- Adaptive rank retries stepping up rank if error threshold is exceeded
- Safe fallback preserving original layer on failure (fallback_to_original default True)
- Accurate Conv1D (GPT-2) and linear layer detection by class name and nf attribute
- Correct raw_shape reporting as (out_features, in_features)
- Unified critical classification/embedding/head protection via should_compress_linear
- Safe stats access and optional validation calibration
- Strict dimensional assertions before module replacement
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
