# Findings and Fixes

## 1. Kernel Feature Map Variance
**Issue:** Performer and Galerkin rely on Random Orthogonal Features. If the feature mapping uses $\exp(\cdot)$ but centers it dynamically using $max\_val$, it breaks the outer product approximation if unnormalized, which led to high performance variance and occasionally negative approximations.
**Mathematical Fix:**
We transitioned the mapping to a strictly positive ReLU-based kernel projection:
$$\phi(x) = \frac{\operatorname{ReLU}(x \Omega^\top) + \epsilon}{\sqrt{m} + \epsilon}$$
This prevents variance issues driven by exponential growth and maintains the necessary non-negativity constraint for positive semidefinite attention approximations.

## 2. Cluster & LSH Bucketing Empty Clusters
**Issue:** Cluster attention used Lloyd's $K$-means algorithm. When clusters became empty, the naive sum clamped the counts to 1, effectively mapping empty centroids to 0. This caused clusters to "die". Furthermore, local context was aggressively mixed via a static 0.5 ratio, breaking local relationships when clusters missed tokens.
**Mathematical Fix:**
1. **Empty Cluster Preservation:** Added a condition to retain the prior centroid vector if the cluster count drops to zero during K-Means assignment:
   $\mu^{(t+1)}_k = \frac{\sum x_i}{\max(1, N_k)}$ if $N_k > 0$, else $\mu^{(t)}_k$
2. **Soft Residual for Local Context:** Introduced a learnable `local_residual` parameter $\alpha$:
   $\alpha = \sigma(\theta_{res})$
   $O = (1 - \alpha) \cdot O_{global} + \alpha \cdot O_{local}$

## 3. Quantization STE Zero-Gradient Trap
**Issue:** ASTRO, FAQ, and CoopQ quantization algorithms clamped gradients aggressively using strict range masks: $\mathbb{I}(q_{min} \le x / s \le q_{max})$. If weights slipped outside the quantization range, they received 0 gradient, permanently freezing them outside.
**Mathematical Fix:**
Implemented vanilla Straight-Through Estimator (STE) gradient bypass: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial q}$. This allows the optimization algorithm to organically push weights back into the unclipped grid.

## Test Results
A comprehensive test suite was executed covering Performer, Galerkin, Hamming, Cluster, and AdaSplash attention mechanisms on dynamic input tensors `(2, 128, 768)`. After applying the mathematical fixes and injecting mock configuration parameters, all 5 module variants passed the rigorous backward-pass gradient sanity checks with no `NaN` or shape mismatches detected.
