# 4. Methodology

## 4.1 Algorithmic Formulation
The `factorised_matrix_state` engine optimizes the layer parameterization via calibrated transformation routines.

```
Factorised Matrix State Attention Module.

Implements multi-head bidirectional/causal associative matrix memory updates combined with
learnable decay gating, denominator normalization, low-rank factorized output projection,
and pretrained weight transfer.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models (forward-only recurrence)
- Nested encoder detection to bypass output projection in inner s
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
