# 4. Methodology

## 4.1 Algorithmic Formulation
The `wgqa` engine optimizes the layer parameterization via calibrated transformation routines.

```
Weighted Grouped-Query Attention (WGQA).

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Features learnable per-KV-head importance weights, robust GQA pooling, LoRA adapters,
RoPE support with fallback, dynamic dimension inference, and causal masking.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
