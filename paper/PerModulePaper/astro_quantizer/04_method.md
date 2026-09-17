# 4. Methodology

## 4.1 Algorithmic Formulation
The `astro_quantizer` engine optimizes the layer parameterization via calibrated transformation routines.

```
AstroQuantizer: Dynamic Uniform Range Quantization Engine with Straight-Through Estimator (STE)
and Fisher-Adaptive Mixed-Precision Bit Allocation. Registered as 'astro' and 'astro_quantizer'.

Scientific Novelty:
- Mixed-precision dynamic bit allocation driven by Empirical Fisher Information / Output Activation Energy.
- Optimal integer budget allocation: Channels with higher Fisher sensitivity receive higher bit-widths (up to max_bits),
  while less sensitive channels are quantized with lower bit-widths (down to min_bits), satisfying an exact average bit budget.
- Straight-Through Estimator (STE) gradients flow seamlessly across channels with heterogeneous bit allocations.
- Full robustness across LoRALinear wrappers, Conv1D layers (GPT-2), and multi-architecture models (DistilBERT, RoBE
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
