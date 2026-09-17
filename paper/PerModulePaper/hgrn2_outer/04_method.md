# 4. Methodology

## 4.1 Algorithmic Formulation
The `hgrn2_outer` engine optimizes the layer parameterization via calibrated transformation routines.

```
HGRN-2 Outer-Product Recurrent Linear Attention Module.

Implements multi-head bidirectional/causal outer-product associative matrix memory updates
with log-space bounded decay gating, denominator normalization, and pretrained weight transfer.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
