# 4. Methodology

## 4.1 Algorithmic Formulation
The `spectraformer_attention` engine optimizes the layer parameterization via calibrated transformation routines.

```
Spectraformer: Random Feature Linear Attention with Flexible Kernel Approximation.
Replaces softmax attention with random orthogonal feature projections,
enabling true linear O(N·r·d) time and memory complexity.

Production-Hardened with:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Frobenius norm validation (>= 1e-4)
- Grouped-Query Attention (GQA) with repeat_kv
- Rotary Position Embedding (RoPE) support
- Persistent orthogonal random feature buffer R (deterministic across forward passes)
- O(N·r·d) linear recurrent running state causal decoding (no N·d² materialization)
- Nested encoder detection and output bypass
- Trainable LoRA adapters with frozen base weights
- Safe 4D head mask handling
- Framework registration and single-layer injection
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
