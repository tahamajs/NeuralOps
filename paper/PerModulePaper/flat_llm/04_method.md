# 4. Methodology

## 4.1 Algorithmic Formulation
The `flat_llm` engine optimizes the layer parameterization via calibrated transformation routines.

```
FLAT-LLM: Activation-Aware Structured Dimension & Neuron Pruning via Eigen-Decomposition.
Registered in COMPRESSOR_REGISTRY as 'flat_llm' and 'flat_llm_compressor'. Production hardened
for multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with reconstruction
validation, Conv1D support, and safe fallback.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
