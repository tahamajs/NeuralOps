# SVD Projection Attention: Explicit Low-Rank Matrix Reconstruction Multi-Head Attention

## 1. Abstract & Theoretical Motivation
SVD Projection Attention (`SVDProjectionAttention`) introduces a low-rank multi-head self-attention module that compresses query, key, value, and output projection operators ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$) via explicit low-rank matrix reconstruction $W = U V$. Rather than evaluating two sequential linear layer projections ($x V^\top U^\top$), `SVDProjectionLinear` computes explicit weight reconstruction $W = U V \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ on-the-fly where $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and $V \in \mathbb{R}^{r \times d_{\text{in}}}$. Weight initialization absorbs singular value magnitudes $\Sigma$ directly into orthogonal left-singular vectors $U$, yielding zero-shot model compatibility under the Eckart–Young–Mirsky theorem. By retaining full multi-head quadratic softmax self-attention over sequence length $N$, SVD Projection Attention achieves up to $12\times$ parameter compression (at $r=32, D=768$) while preserving downstream task performance.

## 2. Mathematical Derivations & Formulations
Let $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote an uncompressed pretrained projection weight matrix. Its thin SVD factor decomposition is:
$$W_0 = \bar{U} \Sigma \bar{V}^\top = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i \bar{u}_i \bar{v}_i^\top$$
where $\bar{U} \in \mathbb{R}^{d_{\text{out}} \times k}$, $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_k)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$, and $\bar{V} \in \mathbb{R}^{d_{\text{in}} \times k}$.

Under low-rank constraint $r \le k$, singular value matrix $\Sigma_{:r}$ is absorbed into factor $U \in \mathbb{R}^{d_{\text{out}} \times r}$:
$$U = \bar{U}_{:, :r} \text{diag}(\sigma_1, \dots, \sigma_r), \qquad V = \bar{V}_{:r, :}^\top \in \mathbb{R}^{r \times d_{\text{in}}}$$

Reconstruction of the low-rank projection matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ proceeds dynamically:
$$W = U V = \bar{U}_{:, :r} \text{diag}(\sigma_{1:r}) \bar{V}_{:r, :}^\top$$

For token activation input $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$, the reconstructed linear operation evaluates as:
$$\text{SVDProjLinear}(X) = X W^\top + b = X (U V)^\top + b$$

The complete multi-head attention module computes projections for $Q, K, V, O$:
$$Q = X (U_Q V_Q)^\top + b_Q, \quad K = X (U_K V_K)^\top + b_K, \quad V = X (U_V V_V)^\top + b_V$$
$$S = \frac{Q K^\top}{\sqrt{d_h}} + M_{\text{additive}}, \qquad A = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1))$$
$$C = A V \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$O = \text{Reshape}(C) (U_O V_O)^\top + b_O \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step execution path and shape transformations are traced below:

1. **Input Tensor**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Dynamic Weight Matrix Reconstruction ($Q, K, V$)**:
   - $W_Q = U_Q V_Q \in \mathbb{R}^{D \times D}$ (where $U_Q \in \mathbb{R}^{D \times r}$, $V_Q \in \mathbb{R}^{r \times D}$)
   - $W_K = U_K V_K \in \mathbb{R}^{D \times D}$, $W_V = U_V V_V \in \mathbb{R}^{D \times D}$
3. **Linear Operations & Head Separation**:
   - $Q_{\text{flat}} = X W_Q^\top + b_Q \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{view}} [B, N, H, d_h] \xrightarrow{\text{transpose}} [B, H, N, d_h]$
   - $K_{\text{flat}} = X W_K^\top + b_K \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{transpose}} [B, H, N, d_h]$
   - $V_{\text{flat}} = X W_V^\top + b_V \in \mathbb{R}^{B \times N \times D} \xrightarrow{\text{transpose}} [B, H, N, d_h]$
4. **Attention Score & Masking**:
   - $S = Q K^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
   - Mask processing: $S = S + (1.0 - M) \cdot (-10^9)$
5. **Softmax & Value Weighted Aggregation**:
   - $P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$
   - $C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$
6. **Head Concatenation & Output Weight Reconstruction**:
   - $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
   - $W_O = U_O V_O \in \mathbb{R}^{D \times D}$
   - $O = \tilde{C} W_O^\top + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Weight Reconstruction ($U V$ per projection): $4 \times 2 D r D = 8 r D^2$ FLOPs
  - Reconstructed Linear Matmul ($X W^\top$): $4 \times 2 B N D^2 = 8 B N D^2$ FLOPs
  - Dot-product attention ($Q K^\top$): $2 B N^2 D$ FLOPs
  - Context weighting ($P V$): $2 B N^2 D$ FLOPs
  - Total Module FLOPs: $8 r D^2 + 8 B N D^2 + 4 B N^2 D$
- **MAC Operations**:
  - Matrix Reconstruction: $4 r D^2$ MACs
  - Projection Matmuls: $4 B N D^2$ MACs
  - Multi-Head Attention: $2 B N^2 D$ MACs
- **Memory Footprint & Parameter Reduction**:
  - Uncompressed Projection Parameters: $4 D^2$
  - SVD Projection Parameters: $4 \times (D r + r D) = 8 r D$
  - Parameter Reduction Ratio: $\frac{4 D^2}{8 r D} = \frac{D}{2r}$. For $D=768, r=32$, ratio is $12.0\times$.

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Materialization Memory Overhead**:
  Reconstructing $W = U V \in \mathbb{R}^{D \times D}$ during forward passes materializes dense $D \times D$ matrices in GPU RAM. This negates matrix GEMM speedups compared to factored execution ($X V^\top U^\top$), adding memory bandwidth overhead.
- **Asymmetric Gradient Conditioning**:
  Absorbing singular values $\Sigma$ entirely into $U$ ($U = \bar{U} \Sigma$) leaves $V$ as a pure orthogonal matrix ($V V^\top = I_r$). Gradient updates to $U$ exhibit much larger magnitudes than updates to $V$, causing instability during AdamW optimization.
- **Uniform Rank Bottleneck**:
  Setting static rank $r$ across all projections causes truncation errors in high-entropy layers where singular spectra decay slowly.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Low-Rank Subspace GEMM Kernel**:
  Replace $W = U V$ reconstruction with fused low-rank Triton GEMMs: $Y = (X V^\top) U^\top + b$. This avoids instantiating $D \times D$ intermediate weight matrices in global HBM.
- **Symmetric Singular Value Balance**:
  Initialize factor matrices symmetrically: $U \leftarrow \bar{U}_{:, :r} \Sigma_{:r}^{1/2}$ and $V \leftarrow \Sigma_{:r}^{1/2} \bar{V}_{:r, :}^\top$. This balances condition numbers $\kappa(U) = \kappa(V) = \sqrt{\kappa(W_0)}$, preventing vanishing gradients in FP16 training.
- **FlashAttention Integration**:
  Stream intermediate factorized projections $Q, K, V$ into FlashAttention-2 Triton kernels to execute fused online softmax attention without allocating $N \times N$ score matrices in GPU memory.