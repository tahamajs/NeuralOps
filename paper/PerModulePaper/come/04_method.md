# 4. Methodology

## 4.1 Algorithmic Formulation
The `come` engine optimizes the layer parameterization via calibrated transformation routines.

```
CoMe: Compress LLMs through Layer Concatenation and Shedding with Learnable Inter-Layer Interpolation Adapters.
Registered in COMPRESSOR_REGISTRY as 'come' and 'come_compressor'.

Mathematical Formulation:
1. Layer Shedding & Uniform Strided Selection:
   $$\\mathcal{I}_{\\text{keep}} = \\{i_0, i_1, \\dots, i_{K-1}\\}, \\quad i_k = \\left\\lfloor k \\cdot \\frac{L-1}{K-1} \\right\\rceil$$
2. Learnable Linear Interpolation Bridge between Non-Consecutive Layers:
   $$\\forall k \\text{ s.t. } i_{k+1} > i_k + 1: \\quad h_{k}^{\\text{bridged}} = h_k + \\alpha_k \\left( W_{\\text{bridge}}^{(k)} h_k - h_k \\right)$$
   $$W_{\\text{bridge}} \\approx I, \\quad \\alpha_k \\in \\mathbb{R} \\text{ (learnable mixing coefficient)}$$
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
