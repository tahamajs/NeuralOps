# 4. Method Formulation

## 4.1 Streaming Covariance Accumulation
During calibration over batches $t = 1, \dots, T$, `aa_svd_compressor` hooks target linear layers and maintains a running Exponential Moving Average (EMA) of activation covariance:
$$C_X^{(t)} = \beta C_X^{(t-1)} + (1 - \beta) \frac{1}{B} \sum_{i=1}^B X_i^\top X_i
<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->
$$where default decay factor $\beta = 0.9$ (configurable via `ema_beta`).

## 4.2 Symmetric Whitening and Factorization
To condition the weight matrix without numerical instability, the symmetric covariance $C_X$ is regularized and decomposed via eigendecomposition in FP32 precision:
$$C_{\text{sym}} = \frac{1}{2}(C_X + C_X^\top) = Q \Lambda Q^\top
<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->
$$$$\Lambda_{\text{reg}} = \operatorname{clamp}(\Lambda, \epsilon, \infty), \quad \epsilon = 10^{-6} \cdot \max(\Lambda) + 10^{-8}
<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 529 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 529 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 529 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 529 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 529 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 529 -->
$$$$C_X^{1/2} = Q \Lambda_{\text{reg}}^{1/2} Q^\top, \quad C_X^{-1/2} = Q \Lambda_{\text{reg}}^{-1/2} Q^\top
<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->
$$
The preconditioned weight is formed as:
$$\tilde{W} = W C_X^{1/2}
<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->
$$Performing standard SVD on $\tilde{W}$:
$$\tilde{W} = \tilde{U} \tilde{\Sigma} \tilde{V}^\top
<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->
$$The low-rank factors are constructed with symmetric singular value distribution:
$$U_r = \tilde{U}_{:, :r} \tilde{\Sigma}_r^{1/2}, \quad V_r = (\tilde{\Sigma}_r^{1/2} \tilde{V}_{:r, :}^\top) C_X^{-1/2}
<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->
$$
## 4.3 Fallback and Relative Error Guard
Before layer replacement, relative Frobenius reconstruction error is evaluated:
$$\text{err} = \frac{\|W - U_r V_r\|_F}{\max(\|W\|_F, 10^{-12})}
<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->

<!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 183 -->
$$If $\text{err} > \text{max\_error}$ (default 0.50) or if the parameter constraint $r(d_{\text{in}} + d_{\text{out}}) \ge d_{\text{in}} d_{\text{out}}$ is violated, the layer is safely retained in its original uncompressed form.


## 2. Mathematical Formulations & Matrix Decompositions

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 2.1 Empirical Activation Subspace Extraction
Given a calibration text dataset passed through model $M$, forward hooks intercept the input activation tensor $X^{(l)} \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ for each linear layer $l$. Flattening across batch $B$ and sequence length $N$ yields the calibration matrix $\bar{X}^{(l)} \in \mathbb{R}^{K \times d_{\text{in}}}$ where $K = B \cdot N$.

Applying SVD to the activation matrix $\bar{X}^{(l)}$:

\[
\bar{X}^{(l)} = U_X \Sigma_X V_X^T
\]

where $V_X^T \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ represents the right-singular vectors (principal components) of the input activation distribution.

### 2.2 Activation Anchor Space Construction
To isolate the top $r$ dominant input activation directions, the engine extracts the top $r$ rows of $V_X^T$, forming the projection basis matrix $A_{\text{anchor}} \in \mathbb{R}^{d_{\text{in}} \times r}$:

\[
A_{\text{anchor}} = (V_{X}[:r, :])^T = V_{X}[:, :r]
\]

The corresponding orthogonal projection operator $P_{\text{anchor}} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ onto the rank-$r$ activation subspace is defined as:

\[
P_{\text{anchor}} = A_{\text{anchor}} A_{\text{anchor}}^T
\]

### 2.3 Weight Anchoring & Low-Rank SVD
The dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is projected onto the activation subspace to produce the anchored weight matrix $W_{\text{anchored}}$:

\[
W_{\text{anchored}} = W P_{\text{anchor}} = W A_{\text{anchor}} A_{\text{anchor}}^T \quad \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
\]

Next, SVD is performed on the anchored weight matrix:

\[
W_{\text{anchored}} = U_{\text{anc}} \Sigma_{\text{anc}} V_{\text{anc}}^T
\]

Truncating to rank $r = \min(\text{rank}, d_{\text{out}}, d_{\text{in}})$, the final low-rank approximation $\hat{W}_{\text{AASVD}}$ is synthesized as:

\[
\hat{W}_{\text{AASVD}} = U_{\text{anc}}[:, :r] \operatorname{diag}(\Sigma_{\text{anc}}[:r]) (V_{\text{anc}}[:r, :])^T
\]

Because $W_{\text{anchored}}$ is constrained to the range of $P_{\text{anchor}}$, its dominant singular components align naturally with the principal variance directions of the input activations.

---


## 2. Mathematical Formulations & Matrix Decompositions

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 2.1 Empirical Activation Subspace Extraction
Given a calibration text dataset passed through model $M$, forward hooks intercept the input activation tensor $X^{(l)} \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ for each linear layer $l$. Flattening across batch $B$ and sequence length $N$ yields the calibration matrix $\bar{X}^{(l)} \in \mathbb{R}^{K \times d_{\text{in}}}$ where $K = B \cdot N$.

Applying SVD to the activation matrix $\bar{X}^{(l)}$:

\[
\bar{X}^{(l)} = U_X \Sigma_X V_X^T
\]

where $V_X^T \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ represents the right-singular vectors (principal components) of the input activation distribution.

### 2.2 Activation Anchor Space Construction
To isolate the top $r$ dominant input activation directions, the engine extracts the top $r$ rows of $V_X^T$, forming the projection basis matrix $A_{\text{anchor}} \in \mathbb{R}^{d_{\text{in}} \times r}$:

\[
A_{\text{anchor}} = (V_{X}[:r, :])^T = V_{X}[:, :r]
\]

The corresponding orthogonal projection operator $P_{\text{anchor}} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ onto the rank-$r$ activation subspace is defined as:

\[
P_{\text{anchor}} = A_{\text{anchor}} A_{\text{anchor}}^T
\]

### 2.3 Weight Anchoring & Low-Rank SVD
The dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is projected onto the activation subspace to produce the anchored weight matrix $W_{\text{anchored}}$:

\[
W_{\text{anchored}} = W P_{\text{anchor}} = W A_{\text{anchor}} A_{\text{anchor}}^T \quad \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
\]

Next, SVD is performed on the anchored weight matrix:

\[
W_{\text{anchored}} = U_{\text{anc}} \Sigma_{\text{anc}} V_{\text{anc}}^T
\]

Truncating to rank $r = \min(\text{rank}, d_{\text{out}}, d_{\text{in}})$, the final low-rank approximation $\hat{W}_{\text{AASVD}}$ is synthesized as:

\[
\hat{W}_{\text{AASVD}} = U_{\text{anc}}[:, :r] \operatorname{diag}(\Sigma_{\text{anc}}[:r]) (V_{\text{anc}}[:r, :])^T
\]

Because $W_{\text{anchored}}$ is constrained to the range of $P_{\text{anchor}}$, its dominant singular components align naturally with the principal variance directions of the input activations.

---


## 2. Mathematical Formulations & Matrix Decompositions

<!-- source: Codes/frontend/dist/Docs/content/compressions/aa_svd_compressor.md -->
### 2.1 Empirical Activation Subspace Extraction
Given a calibration text dataset passed through model $M$, forward hooks intercept the input activation tensor $X^{(l)} \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ for each linear layer $l$. Flattening across batch $B$ and sequence length $N$ yields the calibration matrix $\bar{X}^{(l)} \in \mathbb{R}^{K \times d_{\text{in}}}$ where $K = B \cdot N$.

Applying SVD to the activation matrix $\bar{X}^{(l)}$:

\[
\bar{X}^{(l)} = U_X \Sigma_X V_X^T
\]

where $V_X^T \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ represents the right-singular vectors (principal components) of the input activation distribution.

### 2.2 Activation Anchor Space Construction
To isolate the top $r$ dominant input activation directions, the engine extracts the top $r$ rows of $V_X^T$, forming the projection basis matrix $A_{\text{anchor}} \in \mathbb{R}^{d_{\text{in}} \times r}$:

\[
A_{\text{anchor}} = (V_{X}[:r, :])^T = V_{X}[:, :r]
\]

The corresponding orthogonal projection operator $P_{\text{anchor}} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ onto the rank-$r$ activation subspace is defined as:

\[
P_{\text{anchor}} = A_{\text{anchor}} A_{\text{anchor}}^T
\]

### 2.3 Weight Anchoring & Low-Rank SVD
The dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is projected onto the activation subspace to produce the anchored weight matrix $W_{\text{anchored}}$:

\[
W_{\text{anchored}} = W P_{\text{anchor}} = W A_{\text{anchor}} A_{\text{anchor}}^T \quad \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
\]

Next, SVD is performed on the anchored weight matrix:

\[
W_{\text{anchored}} = U_{\text{anc}} \Sigma_{\text{anc}} V_{\text{anc}}^T
\]

Truncating to rank $r = \min(\text{rank}, d_{\text{out}}, d_{\text{in}})$, the final low-rank approximation $\hat{W}_{\text{AASVD}}$ is synthesized as:

\[
\hat{W}_{\text{AASVD}} = U_{\text{anc}}[:, :r] \operatorname{diag}(\Sigma_{\text{anc}}[:r]) (V_{\text{anc}}[:r, :])^T
\]

Because $W_{\text{anchored}}$ is constrained to the range of $P_{\text{anchor}}$, its dominant singular components align naturally with the principal variance directions of the input activations.

---
