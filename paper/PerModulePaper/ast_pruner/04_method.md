# 4. Methodology

## 4.1 Algorithmic Formulation
The `ast_pruner` engine optimizes the layer parameterization via calibrated transformation routines.

```
AST: Adaptive Semi-Structured 2:4 and N:M Pruning with Empirical Fisher Calibration and PCA Coordinate Rotation.
Registered in COMPRESSOR_REGISTRY as 'ast_pruner' and 'ast'.

Mathematical Formulation:
1. Activation Covariance & PCA Coordinate Rotation:
   $$\\Sigma_X = \\frac{1}{N - 1} (X - \\mu)^\\top (X - \\mu) = Q \\Lambda Q^\\top, \\quad Q^\\top Q = I$$
   $$\\widetilde{W} = W Q, \\quad \\widetilde{X} = Q^\\top X$$
2. Empirical Fisher Information Calibration:
   $$\\mathcal{L}_{\\text{calib}} = \\frac{1}{2} \\sum_{i} \\|y_i\\|_2^2, \\quad g = \\frac{\\partial \\mathcal{L}}{\\partial W}$$
   $$F = \\mathbb{E}[g \\odot g], \\quad \\widetilde{F} = F Q^2$$
3. PCA-Rotated Fisher Saliency:
   $$S_{ij} = |\\widetilde{W}_{ij}| \\cdot \\sqrt{\\widetilde{F}_{ij} + \\epsilon}$$
4. 2:4 Structured 
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
