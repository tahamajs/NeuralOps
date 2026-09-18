# 4. Methodology

## 4.1 Algorithmic Formulation
The `zeros_attention` engine optimizes the layer parameterization via calibrated transformation routines.

```
Zero-Mean Linear Attention (ZeroS Attention).

Removes the mean from key representations and applies normalized linear attention with ReLU-based non-negative
feature maps, with multi-architecture weight transfer, trainable LoRA residual adapters, GQA, RoPE, and causal decoding.

Hardened for production across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
target_attention_only: Replaces self-attention modules only, preserving all FFN/MLP blocks.
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
