# 4. Methodology

## 4.1 Algorithmic Formulation
The `slicegpt_pca` engine optimizes the layer parameterization via calibrated transformation routines.

```
SliceGPT PCA Compressor: Principal Component Analysis based Dimensionality Slicing.
Registered in COMPRESSOR_REGISTRY as 'slicegpt_pca' and 'slicegpt_pca_factory'.

Mathematical Formulation:
1. Activation Sample Matrix $X \in \mathbb{R}^{N 	imes d_{in}}$
2. Empirical Covariance: $\Sigma = rac{1}{N - 1} (X - \mu)^	op (X - \mu)$
3. Eigendecomposition: $\Sigma = V \Lambda V^	op$
4. Retain principal coordinates covering $(1 - 	ext{sparsity})$ cumulative variance:
   $$k = \min \left\{ r \;\middle|\; rac{\sum_{i=1}^r \lambda_i}{\sum_{i=1}^d \lambda_i} \ge 1 - 	ext{sparsity} 
ight\}$$
5. Transform weights: $W \leftarrow (W V) P_k P_k^	op V^	op$
6. Optional LoRA Adapter Integration:
   $$y = (W_{	ext{sliced}} x + b) + rac{lpha}{r} (B A x)$$
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
