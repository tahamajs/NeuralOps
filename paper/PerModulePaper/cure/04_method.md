# 4. Methodology

## 4.1 Algorithmic Formulation
The `cure` engine optimizes the layer parameterization via calibrated transformation routines.

```
CURE: CUR-MoE Compressor alias and factory.
Compresses Mixture of Experts (MoE) layers and dense FFN layers using
proper factorized CURLinear (C @ U @ R parameter reduction).
Registered in COMPRESSOR_REGISTRY as 'cure' and 'cure_compressor'.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
