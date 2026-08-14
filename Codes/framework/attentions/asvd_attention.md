# ASVD Attention: Activation-Aware Singular Value Factorized Multi-Head Attention

## 1. Abstract & Theoretical Motivation
ASVD Attention (`ASVDAttention`, registered as `asvd`) implements an Activation-Aware Singular Value Decomposition framework designed for deep neural network parameter compression and zero-shot multi-head self-attention layer replacement. In transformer self-attention, parameter complexity is dominated by dense linear transformations ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$). ASVD Attention factorizes each projection matrix into three decoupled, trainable factor tensors: an orthogonal left-singular matrix $U \in \mathbb{R}^{d_{\text{out}} \times r}$, a singular value vector $S \in \mathbb{R}^r$, and an orthogonal right-singular matrix $V \in \mathbb{R}^{r \times d_{\text{in}}}$. Initialized via truncated SVD from pretrained checkpoints under the Eckart–Young–Mirsky theorem, ASVD Attention preserves activation energy across sequence dimensions while enabling parameter fine-tuning. By maintaining standard multi-head softmax attention over sequence length $N$, ASVD Attention achieves up to $12\times$ parameter savings (for $r=32, D=768$).

## 2. Mathematical Derivations & Formulations
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote an uncompressed projection weight matrix with bias $b \in \mathbb{R}^{d_{\text{out}}}$. Its full-rank Singular Value Decomposition is:
$$W = \bar{U} \Sigma \bar{V}^\top = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i \bar{u}_i \bar{v}_i^\top$$
where $\bar{U} \in \mathbb{R}^{d_{\text{out}} \times k}$, $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_k)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$, and $\bar{V} \in \mathbb{R}^{d_{\text{in}} \times k}$.

Under a target rank constraint $r \le k$, ASVD Attention initializes factor parameters $(U, S, V)$ as:
$$U_{:, :r} = \bar{U}_{:, :r}, \qquad S_{1:r} = (\sigma_1, \dots, \sigma_r)^\top, \qquad V_{:r, :} = \bar{V}_{:r, :}^\top$$

### Activation-Aware Feature Rescaling (Conceptual Framework)
Let $D_{\text{act}} = \text{diag}(\operatorname{std}(X_{:, :, 1}), \dots, \operatorname{std}(X_{:, :, d_{\text{in}}}))$ denote a diagonal matrix holding activation standard deviations over calibration sequences $X$. Activation-aware SVD solves the weighted Frobenius norm minimization problem:
$$\min_{\operatorname{rank}(\tilde{W}) \le r} \| (W - \tilde{W}) D_{\text{act}} \|_F^2$$
which is solved by computing SVD on the rescaled matrix $\hat{W} = W D_{\text{act}}$ and post-multiplying by $D_{\text{act}}^{-1}$.

The forward pass of `ASVDLinear` evaluates the factorized transformation:
$$y = \text{ASVDLinear}(X) = ((X V^\top) \odot S) U^\top + b = X V^\top \text{diag}(S) U^\top + b$$

Applying this to Query, Key, Value, and Output projections gives:
$$Q = \text{ASVDLinear}_Q(X) \in \mathbb{R}^{B \times N \times D}, \quad K = \text{ASVDLinear}_K(X) \in \mathbb{R}^{B \times N \times D}, \quad V = \text{ASVDLinear}_V(X) \in \mathbb{R}^{B \times N \times D}$$
$$S_{\text{scores}} = \frac{Q K^\top}{\sqrt{d_h}} + M_{\text{attn}}, \qquad P = \text{Dropout}(\text{Softmax}(S_{\text{scores}}, \text{dim}=-1))$$
$$C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$O = \text{ASVDLinear}_O(\text{Reshape}(C_{\text{head}})) \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step tensor operations across ASVD Attention proceed as follows:

1. **Input Tensor**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Factorized Linear Projections ($Q, K, V$)**:
   - Right-singular reduction: $Q_V = X V_Q^\top \in \mathbb{R}^{B \times N \times r}$
   - Singular value scaling: $Q_S = Q_V \odot S_Q \in \mathbb{R}^{B \times N \times r}$
   - Left-singular up-projection: $Q_{\text{full}} = Q_S U_Q^\top + b_Q \in \mathbb{R}^{B \times N \times D}$
   - Key projections: $K_{\text{full}} = ((X V_K^\top) \odot S_K) U_K^\top + b_K \in \mathbb{R}^{B \times N \times D}$
   - Value projections: $V_{\text{full}} = ((X V_V^\top) \odot S_V) U_V^\top + b_V \in \mathbb{R}^{B \times N \times D}$
3. **Multi-Head Layout Conversion**:
   - $Q, K, V \to [B, N, H, d_h] \xrightarrow{\text{transpose}(1, 2)} [B, H, N, d_h]$
4. **Attention Score & Masking**:
   - $S_{\text{scores}} = Q K^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
   - Mask processing via `apply_attention_mask`
5. **Softmax & Context Aggregation**:
   - $P = \text{Dropout}(\text{Softmax}(S_{\text{scores}}, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$
   - $C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$
6. **Head Concatenation & Output ASVD Projection**:
   - $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
   - $O = ((\tilde{C} V_O^\top) \odot S_O) U_O^\top + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Right-projection ($X V^\top$): $4 \times 2 B N D r$ FLOPs
  - Singular scaling ($\odot S$): $4 \times B N r$ FLOPs
  - Left-projection ($Q_S U^\top$): $4 \times 2 B N r D$ FLOPs
  - Projection Subtotal: $16 B N D r + 4 B N r$ FLOPs (vs uncompressed $8 B N D^2$)
  - Multi-Head Score ($Q K^\top$): $2 B N^2 D$ FLOPs
  - Value Aggregation ($P V$): $2 B N^2 D$ FLOPs
  - Total Module FLOPs: $16 B N D r + 4 B N r + 4 B N^2 D$
- **MAC Operations**:
  - ASVD Projections: $8 B N D r$ MACs
  - Attention Matrices: $2 B N^2 D$ MACs
- **Memory Footprint & Parameter Reduction**:
  - Dense Projection Parameters: $4 D^2$
  - ASVD Parameters: $4 \times (D r + r + r D) = 4 r (2 D + 1)$
  - For $D = 768, r = 32$: Uncompressed $= 2,359,296$, ASVD $= 196,736$ (Compression Ratio $= 12.0\times$).

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Orthonormality Loss During Fine-Tuning**:
  While $U$ and $V$ are initialized as orthogonal matrices ($U^\top U = I_r, V V^\top = I_r$), standard AdamW gradient updates break orthogonality during training, causing rank degeneracy.
- **Unconstrained Singular Value Drift**:
  Allowing singular values $S \in \mathbb{R}^r$ to float freely can cause numerical underflow/overflow or sign flips ($S_i < 0$), destroying the geometric interpretation of singular spaces.
- **Uniform Rank Bottleneck**:
  Enforcing uniform rank $r$ across all layers ignores the fact that early layers require higher effective rank than deeper layers.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused ASVD GEMM Triton Kernel**:
  Implement a unified Triton kernel performing fused sequence reduction: $y = \left((X V^\top) \odot S\right) U^\top + b$. By keeping intermediate tile $B_N \times r$ ($64 \times 32$) in SRAM, global DRAM roundtrips are completely eliminated.
- **Orthogonal Projection Stiefel Manifold Optimizer**:
  Use Stiefel manifold optimization or orthogonality penalty loss $\mathcal{L}_{\text{ortho}} = \| U^\top U - I_r \|_F^2 + \| V V^\top - I_r \|_F^2$ during backpropagation to preserve orthogonality of basis factors.
- **FlashAttention Integration**:
  Stream ASVD-projected $Q, K, V$ directly into FlashAttention-2 / FlashAttention-3 kernels to execute fused online softmax attention without allocating $N \times N$ score matrices in GPU DRAM.