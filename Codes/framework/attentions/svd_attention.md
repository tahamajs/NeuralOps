# SVD Sequence Compression Attention: Data-Adaptive Subspace Sequence Projection

## 1. Abstract & Theoretical Motivation
SVD Sequence Compression Attention (`SVDLinearAttention` registered as `svd`) presents a data-adaptive, sequence-length compressing multi-head attention mechanism. Traditional Transformer attention requires $\mathcal{O}(N^2)$ time and space complexity due to full pairwise key-query matrix multiplication. SVD Sequence Compression Attention reduces sequence length complexity from $\mathcal{O}(N^2)$ to linear scaling $\mathcal{O}(N \cdot r)$ by dynamically constructing an orthonormal subspace basis $U_r \in \mathbb{R}^{B \times N \times r}$ via randomized QR / truncated Singular Value Decomposition (SVD) of the sequence representations. Keys and values are projected into this top-$r$ left-singular subspace, contracting key-value sequence length from $N$ to rank $r$. When sequence length $N \le r$, the architecture dynamically falls back to exact multi-head softmax attention. Randomized subspace projection (Halko et al., 2011) provides numerical stability under padding masks, enabling long-sequence processing with low memory overhead.

## 2. Mathematical Derivations & Formulations
Given sequence input hidden states $X \in \mathbb{R}^{B \times N \times D}$ and boolean valid token mask $M_{\text{valid}} \in \{0, 1\}^{B \times N}$, the zero-padded input state is formed:
$$X_{\text{masked}} = X \odot M_{\text{valid}} \in \mathbb{R}^{B \times N \times D}$$

For sequence length $N > r$, an orthonormal sequence subspace matrix $U_r \in \mathbb{R}^{B \times N \times r}$ is computed per batch sample $b \in \{1, \dots, B\}$ via randomized range finder with Gaussian random matrix $\Omega \in \mathbb{R}^{D \times r}$:
$$Y_b = X_{\text{masked}}[b] \cdot \Omega \in \mathbb{R}^{N \times r}$$
$$Q_{\text{mat}}, R_{\text{mat}} = \text{QR}(Y_b) \implies U_r[b] = Q_{\text{mat}}[:, :r] \in \mathbb{R}^{N \times r}$$
satisfying sequence-wise orthonormality: $U_r[b]^\top U_r[b] = I_r$.

Query, Key, and Value linear projections produce standard multi-head tensors:
$$Q, K, V = \text{MultiHeadSplit}(X W_Q), \text{MultiHeadSplit}(X W_K), \text{MultiHeadSplit}(X W_V) \in \mathbb{R}^{B \times H \times N \times d_h}$$

Keys and Values are compressed along sequence dimension $N \to r$ by left-multiplying with transpose $U_r^\top$:
$$K_{\text{proj}} = U_r^\top K \in \mathbb{R}^{B \times H \times r \times d_h}, \qquad V_{\text{proj}} = U_r^\top V \in \mathbb{R}^{B \times H \times r \times d_h}$$

Attention scoring and context aggregation proceed between uncompressed Query ($N$ tokens) and compressed Key-Value subspace ($r$ tokens):
$$S_{\text{compressed}} = \frac{Q K_{\text{proj}}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times N \times r}$$
$$P = \text{Dropout}(\text{Softmax}(S_{\text{compressed}}, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times r}$$
$$C_{\text{head}} = P V_{\text{proj}} \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$O = (\text{Reshape}(C_{\text{head}}) W_O + b_O) \odot M_{\text{valid}} \in \mathbb{R}^{B \times N \times D}$$

If $N \le r$, execution routes to standard softmax attention:
$$S_{\text{full}} = \frac{Q K^\top}{\sqrt{d_h}} + M_{\text{additive}}, \quad C_{\text{head}} = \text{Softmax}(S_{\text{full}}) V$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The precise dimensional shape transformations are traced below:

1. **Input Tensor**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Dense Linear Projections**:
   - $Q_{\text{flat}} = X W_Q + b_Q \in \mathbb{R}^{B \times N \times D}$
   - $K_{\text{flat}} = X W_K + b_K \in \mathbb{R}^{B \times N \times D}$
   - $V_{\text{flat}} = X W_V + b_V \in \mathbb{R}^{B \times N \times D}$
3. **Multi-Head Layout Conversion**:
   - $Q, K, V \to [B, H, N, d_h]$ where $d_h = D / H$
4. **Rank Evaluation & Branch Routing**:
   - Evaluate $r_{\text{eff}} = \min(r, N)$
   - If $N \le r$, execute standard $Q K^\top \to [B, H, N, N] \implies [B, N, D]$.
5. **Subspace Construction (for $N > r$)**:
   - Valid token mask $M_{\text{valid}} \in \{0, 1\}^{B \times N}$
   - Masked state: $X_{\text{masked}} = X \cdot M_{\text{valid}} \in \mathbb{R}^{B \times N \times D}$
   - Per-batch sample randomized QR projection: $Y = X_{\text{masked}} \Omega \in \mathbb{R}^{N \times r}$
   - Orthonormal decomposition: $U_r = \text{QR}(Y)[:, :r] \in \mathbb{R}^{B \times N \times r}$
6. **Key & Value Subspace Compression**:
   - Broadcast $U_r \to [B, 1, N, r]$
   - $K_{\text{proj}} = U_r^\top K \in \mathbb{R}^{B \times H \times r \times d_h}$
   - $V_{\text{proj}} = U_r^\top V \in \mathbb{R}^{B \times H \times r \times d_h}$
7. **Compressed Attention & Context Generation**:
   - $S = Q K_{\text{proj}}^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times r}$
   - $P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times r}$
   - $C_{\text{head}} = P V_{\text{proj}} \in \mathbb{R}^{B \times H \times N \times d_h}$
8. **Head Concatenation & Masked Output Projection**:
   - $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
   - $O = (\tilde{C} W_O^\top + b_O) \cdot M_{\text{valid}} \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Projections ($Q, K, V, O$): $4 \times 2 B N D^2 = 8 B N D^2$ FLOPs
  - Randomized Subspace Projection ($X \Omega$): $2 B N D r$ FLOPs
  - Subspace QR Factorization: $\mathcal{O}(B N r^2)$ FLOPs
  - Key-Value Subspace Contraction ($U_r^\top K, U_r^\top V$): $2 \times 2 B H N r d_h = 4 B N r D$ FLOPs
  - Compressed Attention Score ($Q K_{\text{proj}}^\top$): $2 B H N r d_h = 2 B N r D$ FLOPs
  - Compressed Context Multiplication ($P V_{\text{proj}}$): $2 B H N r d_h = 2 B N r D$ FLOPs
  - Total FLOPs: $8 B N D^2 + 8 B N r D + 2 B N D r + \mathcal{O}(B N r^2)$ (Linear in $N$ for fixed $r$)
- **MAC Operations**:
  - Linear Projections: $4 B N D^2$ MACs
  - Attention Subspace Contractions & Matmuls: $4 B N r D$ MACs
- **Memory Footprint & KV Cache**:
  - Attention Map Memory: $[B, H, N, r]$ vs $[B, H, N, N]$ (saving $\frac{N}{r}\times$ bandwidth)
  - KV Cache Storage: Storing compressed keys/values requires $2 B H r d_h = 2 B r D$ elements instead of $2 B N D$.

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Subspace Information Loss**:
  Projecting $N$ sequence positions onto $r$ basis vectors discards fine-grained local dependencies if sequence length $N \gg r$ and input tokens lack low-rank structure.
- **Randomized SVD Instability in Mixed Precision**:
  Computing QR decomposition (`torch.linalg.qr`) on FP16/BF16 tensors can produce illegal NaNs due to numerical underflow/overflow in triangular R matrices. Fallback to CPU float64 prevents crashes but introduces CPU-GPU synchronization latency.
- **Non-Differentiable Subspace Basis**:
  The subspace matrix $U_r$ is computed inside a `torch.no_grad()` block to stabilize QR autograd. Backpropagation gradients do not flow into the sequence subspace projection, limiting joint end-to-end adaptivity.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Fast Randomized Subspace Triton Kernel**:
  Implement a fused Triton kernel for $Y = X \Omega$ and Gram-Schmidt orthogonalization $Y \to U_r$ directly in GPU L1 cache. This avoids CPU fallback and keeps all tensor operations on CUDA devices.
- **Batched Fused Subspace Compression**:
  Combine Key/Value projections and subspace compression: compute $K_{\text{proj}} = U_r^\top (X W_K)$ in a single Triton block-tiled GEMM kernel without allocating full $[B, H, N, d_h]$ key tensors.
- **Orthogonality Loss Regularization**:
  Use `get_ortho_loss()` to penalize deviations of $U_r^\top U_r$ from identity $I_r$: $\mathcal{L}_{\text{ortho}} = \| U_r^\top U_r - I_r \|_F^2$, ensuring numerical stability during fine-tuning.