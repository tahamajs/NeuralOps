# 4. Methodological Formulation and Execution Pipeline

The core methodology of `svd_llm_compressor` consists of three tightly coupled execution stages:

### Stage 1: Representation Alignment
Input tensors are projected into an aligned coordinate frame where variance is concentrated along dominant principal directions.

### Stage 2: Hardware-Efficient Transformation
Depending on the architectural configuration, computations are executed via associative prefix-sums or factorized low-rank matrix products:
$$
Y = \mathcal{T}(Q, K, V) \quad \text{or} \quad Y = X \widetilde{W}^\top
$$

### Stage 3: Error Compensation and Recovery
To eliminate bias from structural approximations, residual correction vectors or calibrated scale factors are dynamically injected into output activations.

## SOTA Enhancement: SVD-LLM v2 Activation Truncation
To suppress outlier-driven reconstruction error without inflating parameter budgets, SVD-LLM v2 incorporates activation-aware singular value thresholding (activation truncation). Let  \in \mathbb{R}^{B 	imes N 	imes D}$ denote calibration activations with empirical covariance $\Sigma_X = rac{1}{BN} X^	op X$. Rather than computing standard unweighted SVD  = U \Sigma V^	op$, we apply scaled decomposition on  \Sigma_X^{1/2} = 	ilde{U} 	ilde{\Sigma} 	ilde{V}^	op$, truncating singular components whose cumulative energy falls below $\eta = 0.95$ after clipping activations beyond the 99.9th percentile. This isolates outlier dimensions and guarantees bounded post-truncation Frobenius distortion $\|W - \hat{W}\|_{F} \le \epsilon_{	ext{outlier}}$.
