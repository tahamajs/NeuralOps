# 4. Methodology

## 4.1 Algorithmic Formulation
The `zerokv_neo` engine optimizes the layer parameterization via calibrated transformation routines.

```
ZeroKV-Neo: L2-norm based KV cache eviction with attention sink and heavy-hitter preservation.
Re-exports ZeroKVNeoCompressor and ZeroKVNeoCache.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
