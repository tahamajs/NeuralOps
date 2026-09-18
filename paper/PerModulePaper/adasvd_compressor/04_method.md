# 4. Method Formulation
1. **Spectral Importance**: Layer budget is proportional to $\|W\|_* = \sum_i \sigma_i$. <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 47 -->
2. **Symmetric Factorization**: $A = U_{:, :r} \sqrt{\Sigma_r}, \quad B = \sqrt{\Sigma_r} V_{:r, :}^	op$. <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 48 -->
3. **Alternating Least Squares**: Refine $A$ and $B$ to minimize $\|W - A B\|_F^2$; revert to SVD factors if ALS error increases. <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 54 -->
4. **SVDLinear Substitution**: Replace dense projections with factorized modules. <!-- source: Codes/framework/compression_engine/adasvd_compressor.py line 50 -->


## 2. Mathematical Formulations & Matrix Decompositions

<!-- source: Codes/frontend/dist/Docs/content/compressions/adasvd_compressor.md -->
### 2.1 Layer Importance via Nuclear Norm Estimation
For a model containing $L$ linear layers with weight matrices $W^{(l)} \in \mathbb{R}^{M_l \times N_l}$ (where $M_l = d_{\text{out}}^{(l)}$ and $N_l = d_{\text{in}}^{(l)}$ for $l \in \{1, \dots, L\}$), the exact SVD of layer $l$ is $W^{(l)} = U^{(l)} \Sigma^{(l)} (V^{(l)})^T$. The nuclear norm $\|W^{(l)}\|_*$, representing the total spectral energy, defines the raw importance metric $\mathcal{I}_l$:

\[
\mathcal{I}_l = \|W^{(l)}\|_* = \sum_{i=1}^{\min(M_l, N_l)} \sigma_i^{(l)}
\]

The normalized global importance fraction $\alpha_l$ for layer $l$ is expressed as:

\[
\alpha_l = \frac{\mathcal{I}_l}{\sum_{j=1}^{L} \mathcal{I}_j} = \frac{\|W^{(l)}\|_*}{\sum_{j=1}^{L} \|W^{(j)}\|_*}
\]

### 2.2 Adaptive Global Rank Allocation Formula
Given a global parameter removal ratio $\rho \in (0, 1)$ (`target_param_ratio`), the engine computes layer $l$'s allocated rank $r_l$ using the information-weighted allocation rule:

\[
r_l = \min\left( \max\left( 1, \left\lfloor R_{\max}^{(l)} \cdot (1 - \rho) \cdot \alpha_l \cdot L \right\rfloor \right), R_{\max}^{(l)} \right)
\]

where $R_{\max}^{(l)} = \min(M_l, N_l)$ is the maximum possible rank for layer $l$, and $L$ represents the total number of compressible linear layers.

### 2.3 Initial Factor Truncation
For target rank $r_l$, the initial left-singular factor $U_{r_l} \in \mathbb{R}^{M_l \times r_l}$, singular values $S_{r_l} \in \mathbb{R}^{r_l}$, and right-singular factor $V_{r_l}^T \in \mathbb{R}^{r_l \times N_l}$ are extracted from the truncated SVD:

\[
W^{(l)} \approx U_{r_l} \operatorname{diag}(S_{r_l}) V_{r_l}^T
\]

### 2.4 Alternating Iterative Least-Squares Refinement
To minimize reconstruction error $\|W^{(l)} - U_{r_l} \operatorname{diag}(S_{r_l}) V_{r_l}^T\|_F^2$ under the allocated rank constraint, `AdaSVDCompressor` performs $T$ iterations (`num_iters`) of alternating ridge-regularized updates:

1. Left-Factor ($U_{r_l}$) Update Step:
Holding $V_{r_l}^T$ and $\Sigma_{r_l} = \operatorname{diag}(S_{r_l})$ fixed, solve for updated $U_{r_l}$:

\[
U_{r_l}^{(t+1)} = W^{(l)} V_{r_l}^{(t)} \left( \Sigma_{r_l} + \epsilon I_{r_l} \right)^{-1}
\]

where $\epsilon = 10^{-6}$ is a Tikhonov stabilization factor.

2. Right-Factor ($V_{r_l}^T$) Update Step:
Holding updated $U_{r_l}^{(t+1)}$ fixed, solve for updated $V_{r_l}^T$:

\[
V_{r_l}^{(t+1)T} = \left( (U_{r_l}^{(t+1)})^T U_{r_l}^{(t+1)} + \epsilon I_{r_l} \right)^{-1} (U_{r_l}^{(t+1)})^T W^{(l)}
\]

3. Diagonal Singular Value Matrix Re-projection:
Re-estimate the updated diagonal singular values $S_{r_l}^{(t+1)}$ by taking the diagonal elements of the projected weight matrix:

\[
S_{r_l}^{(t+1)} = \operatorname{diag}\left( (U_{r_l}^{(t+1)})^T W^{(l)} V_{r_l}^{(t+1)} \right)
\]

The refined weight matrix stored in the layer is synthesized as:

\[
\hat{W}^{(l)} = U_{r_l}^{(T)} \operatorname{diag}(S_{r_l}^{(T)}) V_{r_l}^{(T)T}
\]

---
