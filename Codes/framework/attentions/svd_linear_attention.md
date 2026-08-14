# SVD Linear Attention: Semi-Orthogonal Low-Rank Factorized Multi-Head Projections

## 1. Abstract & Theoretical Motivation
SVD Linear Attention (`SVDLinearAttention`) provides a parameter-efficient, low-rank factorized attention sub-layer designed for zero-shot weight initialization and rank-constrained Multi-Head Attention (MHA). Traditional self-attention parameter requirements are driven by four dense linear transformations ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$). SVD Linear Attention substitutes each dense transformation with a pair of factorized low-rank matrix parameters $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and $V \in \mathbb{R}^{r \times d_{\text{in}}}$, where $r \ll \min(d_{\text{in}}, d_{\text{out}})$. The initial weights of $U$ and $V$ are computed directly from the singular value spectrum of pretrained checkpoints via truncated Singular Value Decomposition (SVD), satisfying the Eckart–Young–Mirsky theorem for optimal rank-$r$ Frobenius norm approximation. This enables zero-shot deployment without task retraining while reducing linear projection parameters by up to $12\times$ (e.g., $r=32$ vs $D=768$).

## 2. Mathematical Derivations & Formulations
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ represent an uncompressed linear projection matrix with optional bias $b \in \mathbb{R}^{d_{\text{out}}}$. Its full-rank Singular Value Decomposition is defined as:
$$W = \bar{U} \Sigma \bar{V}^\top = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i \bar{u}_i \bar{v}_i^\top$$
where $\bar{U} \in \mathbb{R}^{d_{\text{out}} \times k}$, $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_k)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$, and $\bar{V} \in \mathbb{R}^{d_{\text{in}} \times k}$.

Under rank restriction $r \le k$, the low-rank factors $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and $V \in \mathbb{R}^{r \times d_{\text{in}}}$ are initialized as:
$$U = \bar{U}_{:, :r} \text{diag}(\sigma_1, \dots, \sigma_r), \qquad V = \bar{V}_{:r, :}^\top$$
yielding the low-rank approximation:
$$W \approx U V$$

For input feature sequence $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$, the factorized forward operation evaluates as:
$$\text{SVDLinear}(X) = (X V^\top) U^\top + b$$

The multi-head query, key, and value representations are derived via factorized projections:
$$Q = \text{SVDLinear}_Q(X) \in \mathbb{R}^{B \times N \times D}, \quad K = \text{SVDLinear}_K(X) \in \mathbb{R}^{B \times N \times D}, \quad V = \text{SVDLinear}_V(X) \in \mathbb{R}^{B \times N \times D}$$

Reshaping into multi-head layout $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ ($d_h = D / H$), scaled dot-product attention scores and output context matrices are given by:
$$S = \frac{Q K^\top}{\sqrt{d_h}} + M_{\text{attn}}$$
$$P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1))$$
$$C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$O = \text{SVDLinear}_O(\text{Reshape}(C_{\text{head}})) \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step tensor shapes and execution path are traced as follows:

1. **Input Tensor**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Factorized Query Projection**:
   - Intermediate low-rank embedding: $Q_{\text{low}} = X V_Q^\top \in \mathbb{R}^{B \times N \times r}$
   - Up-projection with bias: $Q_{\text{full}} = Q_{\text{low}} U_Q^\top + b_Q \in \mathbb{R}^{B \times N \times D}$
3. **Factorized Key & Value Projections**:
   - $K_{\text{low}} = X V_K^\top \in \mathbb{R}^{B \times N \times r} \implies K_{\text{full}} = K_{\text{low}} U_K^\top + b_K \in \mathbb{R}^{B \times N \times D}$
   - $V_{\text{low}} = X V_V^\top \in \mathbb{R}^{B \times N \times r} \implies V_{\text{full}} = V_{\text{low}} U_V^\top + b_V \in \mathbb{R}^{B \times N \times D}$
4. **Multi-Head Reshape & Transpose**:
   - $Q, K, V \to [B, N, H, d_h] \xrightarrow{\text{transpose}(1, 2)} [B, H, N, d_h]$
5. **Attention Matrix Score Computation**:
   - $S = Q K^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
6. **Masking & Softmax Activation**:
   - $S \leftarrow \text{apply\_attention\_mask}(S, \text{mask})$
   - $P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$
7. **Value Weighting & Head Re-assembly**:
   - $C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$
   - $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
8. **Factorized Output Projection**:
   - $O_{\text{low}} = \tilde{C} V_O^\top \in \mathbb{R}^{B \times N \times r}$
   - $O = O_{\text{low}} U_O^\top + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Down-Projections ($Q, K, V, O$): $4 \times 2 B N D r$ FLOPs
  - Up-Projections ($Q, K, V, O$): $4 \times 2 B N r D$ FLOPs
  - Total Projection FLOPs: $16 B N D r$ FLOPs (compared to uncompressed $8 B N D^2$)
  - Dot-product attention ($Q K^\top$): $2 B N^2 D$ FLOPs
  - Weighted sum ($P V$): $2 B N^2 D$ FLOPs
  - Total Layer FLOPs: $16 B N D r + 4 B N^2 D$
- **MAC Operations**:
  - Projections: $8 B N D r$ MACs
  - Attention Matmuls: $2 B N^2 D$ MACs
- **Memory Footprint & Parameter Count**:
  - Uncompressed Projection Parameters: $4 D^2$
  - Factorized Parameters: $4 \times 2 r D = 8 r D$
  - Compression Ratio: $\frac{4 D^2}{8 r D} = \frac{D}{2r}$. For $D=768, r=32$, parameter reduction is $\frac{768}{64} = 12.0\times$.
  - Activation Memory: Intermediate low-rank activations require storing $[B, N, r]$ tensors per layer.

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Uniform Rank Bottleneck**:
  Assigning identical static rank $r$ across Query, Key, Value, and Output projections ignores non-uniform singular value spectra across heads. Query and Key projections often possess higher effective rank than Value projections, leading to key-query spatial collapse if $r$ is set too small.
- **Attention Map Complexity Persistence**:
  While projection parameter footprint shrinks by $12\times$, self-attention matrix calculation scales quadratically as $\mathcal{O}(N^2 D)$. At long context lengths ($N \gg D$), the attention score tensor $S \in \mathbb{R}^{B \times H \times N \times N}$ dominates memory bandwidth.
- **Unbalanced Factor Conditioning**:
  Initializing $U = \bar{U}_{:,:r} \Sigma_{:r}$ and $V = \bar{V}_{:r,:}^\top$ leaves $V$ orthogonal ($V V^\top = I_r$) while $U$ absorbs all singular magnitudes. This asymmetry creates high condition number disparities during gradient updates.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Low-Rank Down/Up Matmul Kernel**:
  Construct a custom Triton kernel performing fused GEMM: $Y = (X V^\top) U^\top + b$. Tiling sequence length by $B_N = 64$ and hidden dimension by $B_D = 64$ allows keeping intermediate low-rank tiles $B_N \times r$ ($64 \times 32$) in GPU L1 cache/SRAM, avoiding HBM writes.
- **Symmetric Factor Initialization**:
  Re-balance singular values symmetrically during weight assignment: $U \leftarrow \bar{U}_{:,:r} \Sigma_{:r}^{1/2}$ and $V \leftarrow \Sigma_{:r}^{1/2} \bar{V}_{:r,:}^\top$. This ensures $\kappa(U) = \kappa(V) = \sqrt{\kappa(W)}$, stabilizing FP16/BF16 tensor core GEMMs.
- **FlashAttention Integration**:
  Pass output of factorized $Q, K, V$ directly into fused FlashAttention-2 kernels to execute softmax attention and value reduction without writing intermediate $N \times N$ matrices to global HBM.