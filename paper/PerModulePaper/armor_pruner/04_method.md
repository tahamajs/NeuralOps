# 4. Methodology

## 4.1 Algorithmic Formulation
The `armor_pruner` engine optimizes the layer parameterization via calibrated transformation routines.

```
ARMOR: Adaptive Resilient Matrix Orthogonal Rank Pruning (2:4 Sparse Core + Low-Rank Correction).
Registered in COMPRESSOR_REGISTRY as 'armor_pruner', 'armor', and 'armor_compressor'.

Mathematical Formulation:
1. Resilient First-Order Taylor Saliency (handling zero-grad, NaN/Inf, and absent activations):
   $$S_{ij} = \\begin{cases} |W_{ij} \\odot g_{ij}| & \\text{if } g \\text{ is valid and } \\|g\\| > 0 \\\\ |W_{ij}| \\cdot \\mathbb{E}[|x_j|] & \\text{if activations available} \\\\ |W_{ij}| \\cdot (\\sigma(W) + 1) & \\text{fallback} \\end{cases}$$
2. 2:4 Semi-Structured Pruning Mask:
   $$\\mathcal{M} = \\Pi_{2:4}(S), \\quad W_{\\text{sparse}} = W \\odot \\mathcal{M}$$
3. Residual Low-Rank SVD Correction:
   $$R = W - W_{\\text{sparse}} \\approx B A, \\quad A \\in \\mathbb{R}^{r \\times
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
