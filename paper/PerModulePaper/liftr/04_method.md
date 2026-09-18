# 4. Methodology

## 4.1 Algorithmic Formulation
The `liftr` engine optimizes the layer parameterization via calibrated transformation routines.

```
LIFTR – Linearized Function Trick Compressor.
Registered in COMPRESSOR_REGISTRY as 'liftr' and 'liftr_compressor'.

Mathematical Formulation:
1. Activation Function Linearization:
   - $\\operatorname{ReLU}(x) \\to \\operatorname{Identity}(x)$
   - $\\operatorname{GELU}(x) \\approx \\alpha \\cdot x$ (default $\\alpha = 0.8$)
   - $\\operatorname{SiLU}(x) \\approx \\alpha \\cdot x$ (default $\\alpha = 0.8$)
2. Quadratic Softmax Attention Linearization:
   $$\\operatorname{Attn}(Q, K, V) = \\operatorname{Softmax}\\left(\\frac{Q K^\\top}{\\sqrt{d_k}}\\right) V \\to \\frac{1}{N} \\sum_{i=1}^N V_i$$

Production Hardening:
- Fix 1: Optional output-level validation (validation_calibration) comparing logits before and after, with rollback on degradation.
- Fix 2: Strict protection of critical head
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
