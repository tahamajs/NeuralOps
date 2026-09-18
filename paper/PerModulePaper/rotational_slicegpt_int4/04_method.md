# 4. Methodology

## 4.1 Algorithmic Formulation
The `rotational_slicegpt_int4` engine optimizes the layer parameterization via calibrated transformation routines.

```
Rotational SliceGPT INT4 Compressor: Rotational Invariance, Subspace Slicing, and Second-Order INT4 Quantization.
Registered in COMPRESSOR_REGISTRY as 'rotational_slicegpt_int4' and 'rotational_slicegpt'.

Mathematical Formulation:
1. Orthogonal Coordinate Rotation:
   $$\\widetilde{W} = W Q, \\quad \\widetilde{X} = Q^\\top X, \\quad Q^\\top Q = I$$
2. Variance-Based Subspace Slicing:
   $$\\mathcal{I} = \\operatorname{Top-}r(\\operatorname{Var}(\\widetilde{X})), \\quad P_r = [e_i]_{i \\in \\mathcal{I}}$$
   $$W_{sliced} = \\widetilde{W} P_r, \\quad X_{sliced} = P_r^\\top \\widetilde{X}$$
3. Second-Order Optimal Brain Surgeon INT4 Non-Uniform Quantization:
   $$H = 2 X_{sliced} X_{sliced}^\\top + \\lambda I$$
   $$W_{:, j+1:r} \\leftarrow W_{:, j+1:r} - (w_j - q(w_j)) \\frac{[H^{-1}]_{j, j
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
