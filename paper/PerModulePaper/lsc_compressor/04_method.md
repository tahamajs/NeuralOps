# 4. Methodology

## 4.1 Algorithmic Formulation
The `lsc_compressor` engine optimizes the layer parameterization via calibrated transformation routines.

```
LSC: Low-Rank Spectral & Fourier-Domain Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'lsc' and 'lsc_compressor'.

Mathematical Formulation:
1. 2D Discrete Fourier Transform:
   $$\\widehat{W} = \\mathcal{F}_{2D}(W) \\in \\mathbb{C}^{d_{out} \\times d_{in}}$$
2. Spectral Magnitude Spectrum:
   $$M_{ij} = |\\widehat{W}_{ij}| = \\sqrt{\\operatorname{Re}(\\widehat{W}_{ij})^2 + \\operatorname{Im}(\\widehat{W}_{ij})^2}$$
3. Spectral Coefficient Selection:
   $$\\mathcal{M}_{spectral} = \\operatorname{Top-}k(M), \\quad k = \\min(r \\cdot r, d_{out} \\cdot d_{in})$$
4. Inverse 2D Fourier Reconstruction:
   $$W_{lsc} = \\operatorname{Re}\\left( \\mathcal{F}_{2D}^{-1}(\\widehat{W} \\odot \\mathcal{M}_{spectral}) \\right)$$
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
