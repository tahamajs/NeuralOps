# 4. Methodology

## 4.1 Algorithmic Formulation
The `samba_m3` engine optimizes the layer parameterization via calibrated transformation routines.

```
SAMBA-M3: Hybrid Sliding-Window Attention and Spectral Mamba Block (Hardened).
Supports:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
- True chunked local sliding-window attention (O(L · W))
- No double-residual or double LayerNorm
- Safe injection returning None on failure
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
