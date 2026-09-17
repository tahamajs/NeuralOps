# 4. Methodology

## 4.1 Algorithmic Formulation
The `erc_svd` engine optimizes the layer parameterization via calibrated transformation routines.

```
ERC-SVD: Error-Controlled SVD with Bounded Frobenius Truncation Error.
Registered as 'erc' and 'erc_svd'. Production hardened for multi-architecture
compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support,
parameter bounds, and safe fallback.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
