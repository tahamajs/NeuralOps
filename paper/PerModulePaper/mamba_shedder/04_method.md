# 4. Methodology

## 4.1 Algorithmic Formulation
The `mamba_shedder` engine optimizes the layer parameterization via calibrated transformation routines.

```
Mamba-Shedder: Sensitivity-Based Structured Pruning and Layer Shedding Compression Engine.
Registered in COMPRESSOR_REGISTRY as 'mamba_shedder' and 'mamba_shedder_compressor'.

Mathematical Formulation:
1. Empirical Fisher Information / Sensitivity Metric:
   $$\\mathcal{I}(X_j) = \\sqrt{\\frac{1}{N} \\sum_{n=1}^N X_{n, j}^2}$$
2. Layer Residual Sensitivity:
   $$\\Delta_l = \\frac{\\|X_{l+1} - X_l\\|_F}{\\|X_l\\|_F + \\epsilon}$$
3. Channel & Layer Masking:
   $$W_{pruned} = W \\odot \\mathcal{M}_{channel}^\\top$$

Production Hardening:
- Fix 1: Reconstruction error validation with configurable max_error (default 0.20) and safe fallback.
- Fix 2: Support for Conv1D (GPT-2) with transposition and fused QKV projections (Qwen/LLaMA).
- Fix 3: Saliency decisions in FP32 with magnitude-based f
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
