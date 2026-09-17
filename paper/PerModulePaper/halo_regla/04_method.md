# 4. Methodology

## 4.1 Algorithmic Formulation
The `halo_regla` engine optimizes the layer parameterization via calibrated transformation routines.

```
HALO-ReGLA: Hardware-Aware Linear Optics & Rectified Gated Linear Attention – Production-Hardened.

Combines multi-head linear projection with Rotary Position Embeddings (RoPE),
rectified kernel representations, and adaptive gating for zero-shot distillation from pretrained transformers.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
