# 4. Methodology

## 4.1 Algorithmic Formulation
The `coopq_quantizer` engine optimizes the layer parameterization via calibrated transformation routines.

```
CoopQ: Cooperative Game-Theoretic Bit Allocation Quantizer with STE and Per-Channel Asymmetric Calibration.
Registered in COMPRESSOR_REGISTRY as 'coopq'.

Mathematical Formulation:
1. Min Bit-Width Floor Enforcement:
   $$b_l \\ge b_{\\min} = 4, \\quad \\forall l \\in \\{1, \\dots, L\\}$$
2. Per-Channel Asymmetric Scale & Zero-Point Calibration:
   $$\\text{scale}_i = \\max\\left( \\frac{\\max_j W_{ij} - \\min_j W_{ij}}{2^b - 1}, 10^{-8} \\right), \\quad z_i = \\operatorname{clamp}\\left( \\left\\lfloor -\\frac{\\min_j W_{ij}}{\\text{scale}_i} \\right\\rceil, 0, 2^b - 1 \\right)$$
   $$\\widehat{W}_{ij} = \\left( \\operatorname{clamp}\\left( \\left\\lfloor \\frac{W_{ij}}{\\text{scale}_i} \\right\\rceil + z_i, 0, 2^b - 1 \\right) - z_i \\right) \\cdot \\text{scale}_i$$
3. Straight-Through E
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
