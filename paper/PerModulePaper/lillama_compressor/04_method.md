# 4. Methodology

## 4.1 Algorithmic Formulation
The `lillama_compressor` engine optimizes the layer parameterization via calibrated transformation routines.

```
LiLLaMA: Local Distillation via Activation Cross-Covariance Singular Value Decomposition.
Registered in COMPRESSOR_REGISTRY as 'lillama' and 'lillama_compressor'.

Mathematical Formulation:
1. Input and Output Activations:
   $$X \\in \\mathbb{R}^{N \\times d_{in}}, \\quad Y \\in \\mathbb{R}^{N \\times d_{out}}$$
2. Empirical Cross-Covariance Matrix:
   $$C_{YX} = \\frac{1}{N} Y^\\top X \\in \\mathbb{R}^{d_{out} \\times d_{in}}$$
3. Truncated Singular Value Decomposition:
   $$C_{YX} = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top$$
4. Optimal Distilled Weight Reconstruction:
   $$W_{distill} = U_r \\Sigma_r V_r^\\top$$

Production Hardening:
- Fix 1: Reconstruction error validation with configurable max_error (default 0.20) and safe fallback.
- Fix 2: Support for Conv1D (GPT-2) and f
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
