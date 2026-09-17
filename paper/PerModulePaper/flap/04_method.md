# 4. Methodology

## 4.1 Algorithmic Formulation
The `flap` engine optimizes the layer parameterization via calibrated transformation routines.

```
FLAP: Fluctuation-Based Adaptive Structured Pruning.
Registered in COMPRESSOR_REGISTRY as 'flap' and 'flap_compressor'. Production hardened
for multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with
reconstruction validation, Conv1D support, and safe fallback.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
