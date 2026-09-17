# 4. Methodology

## 4.1 Algorithmic Formulation
The `galerkin_attention` engine optimizes the layer parameterization via calibrated transformation routines.

```
Galerkin Attention: Petrov-Galerkin Projection Attention for Transformer Architectures.

Formulates transformer self-attention as a continuous non-local integral operator
discretized via Petrov-Galerkin projection onto an orthonormal basis (e.g., discrete cosine basis).
Supports fallback to standard causal attention for decoder models.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking fallback for decoder models 
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
