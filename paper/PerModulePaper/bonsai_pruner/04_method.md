# 4. Methodology

## 4.1 Algorithmic Formulation
The `bonsai_pruner` engine optimizes the layer parameterization via calibrated transformation routines.

```
Bonsai: Iterative Magnitude and Taylor Pruning (IMP) with Gradient Descent Recovery & [CLS] Routing Protection.
Registered in COMPRESSOR_REGISTRY as 'bonsai' and 'bonsai_pruner'.

Production-hardened with:
- Multi-architecture support: DistilBERT, RoBERTa, GPT-2 (Conv1D), Qwen3.5 (GQA)
- Robust activation collection & calibration forward handling (never crashes on malformed inputs)
- Safe LayerNorm recalibration with try/except guards
- GQA-aware handling (configurable skip_attention_qkv to preserve KV-cache alignment)
- Trainable LoRA adapter integration post-pruning (BonsaiPrunedLoRALinear)
- Guard against tiny layers (< 16 parameters)
- Clamped step size scaling in recovery loop
- Per-layer adaptive sparsity option based on activation sensitivity
- Post-compression forward sanity check

```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
