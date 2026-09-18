# 4. Methodology

## 4.1 Algorithmic Formulation
The `tlinformer_attention` engine optimizes the layer parameterization via calibrated transformation routines.

```
TLinFormer: Exact linear‑time attention via low‑rank sequence projections.
Reference: LinFormer (Wang et al., 2020) - adapted for DistilBERT, RoBERTa, GPT-2, and Qwen3.5.

Production-Hardened with:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Frobenius norm validation (>= 1e-4)
- Grouped-Query Attention (GQA) with repeat_kv
- Rotary Position Embedding (RoPE) support
- Causal masking for decoder models
- Nested encoder detection and output bypass
- Trainable LoRA adapters with frozen base weights
- Mathematically correct sequence contraction: K_proj = proj_k @ K
- Safe 4D head mask handling
- Framework registration and safe single-layer injection
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
