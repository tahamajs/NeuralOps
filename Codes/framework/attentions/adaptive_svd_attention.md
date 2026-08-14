# Adaptive SVD Attention: Frozen Truncated SVD Base with Trainable Low-Rank Residual Adapters

## 1. Abstract & Theoretical Motivation
Adaptive SVD Attention (`AdaptiveSVDAttention`, registered as `adaptive_svd_attn`) presents a parameter-efficient fine-tuning (PEFT) attention sub-layer that combines a frozen, truncated Singular Value Decomposition (SVD) base projection with lightweight, trainable low-rank residual adapters ($W \approx U V + A B$). Full-rank Multi-Head Attention linear transformations ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$) account for significant parameter memory overhead. Adaptive SVD Attention addresses this by factorizing each projection matrix into a high-energy frozen SVD base $W_{\text{svd}} = U V$ (where $U \in \mathbb{R}^{d_{\text{out}} \times r_{\text{svd}}}, V \in \mathbb{R}^{r_{\text{svd}} \times d_{\text{in}}}$) and a low-rank trainable residual adapter $W_{\text{adapter}} = A B$ (where $A \in \mathbb{R}^{d_{\text{out}} \times r_{\text{res}}}, B \in \mathbb{R}^{r_{\text{res}} \times d_{\text{in}}}$). By initializing the frozen base via truncated SVD from pretrained checkpoints under the Eckart–Young–Mirsky theorem, Adaptive SVD Attention provides zero-shot fidelity while restricting downstream parameter updates to $< 1\%$ of original projection weights.

## 2. Mathematical Derivations & Formulations
Let $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote an uncompressed pretrained linear weight matrix with optional bias $b \in \mathbb{R}^{d_{\text{out}}}$. Its full-rank Singular Value Decomposition is:
$$W_0 = \bar{U} \Sigma \bar{V}^\top = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i \bar{u}_i \bar{v}_i^\top$$
where $\bar{U} \in \mathbb{R}^{d_{\text{out}} \times k}$, $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_k)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$, and $\bar{V} \in \mathbb{R}^{d_{\text{in}} \times k}$.

Under SVD rank restriction $r_{\text{svd}} \le k$, the frozen base factors $U \in \mathbb{R}^{d_{\text{out}} \times r_{\text{svd}}}$ and $V \in \mathbb{R}^{r_{\text{svd}} \times d_{\text{in}}}$ are initialized as:
$$U = \bar{U}_{:, :r_{\text{svd}}} \text{diag}(\sigma_{1:r_{\text{svd}}}), \qquad V = \bar{V}_{:r_{\text{svd}}, :}^\top$$

To absorb truncation residual error and enable downstream task adaptation, a trainable low-rank adapter pair $A \in \mathbb{R}^{d_{\text{out}} \times r_{\text{res}}}$ and $B \in \mathbb{R}^{r_{\text{res}} \times d_{\text{in}}}$ ($r_{\text{res}} \ll r_{\text{svd}}$) is initialized via Kaiming uniform initialization:
$$A_{i, j} \sim \mathcal{U}\left(-\sqrt{\frac{5}{r_{\text{res}}}}, \sqrt{\frac{5}{r_{\text{res}}}}\right), \qquad B_{i, j} \sim \mathcal{U}\left(-\sqrt{\frac{5}{d_{\text{in}}}}, \sqrt{\frac{5}{d_{\text{in}}}}\right)$$

For input token activations $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$, the combined adaptive forward linear operator computes:
$$y = \text{AdaptiveSVDLinear}(X) = \underbrace{X V^\top U^\top}_{\text{Frozen SVD Base}} + \underbrace{X B^\top A^\top}_{\text{Trainable Adapter}} + b$$

Applying this factorized operator to $Q, K, V, O$ yields:
$$Q = \text{AdaptiveSVDLinear}_Q(X) \in \mathbb{R}^{B \times N \times D}, \quad K = \text{AdaptiveSVDLinear}_K(X) \in \mathbb{R}^{B \times N \times D}, \quad V = \text{AdaptiveSVDLinear}_V(X) \in \mathbb{R}^{B \times N \times D}$$
$$S = \frac{Q K^\top}{\sqrt{d_h}} + M_{\text{attn}}, \qquad P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1))$$
$$C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$O = \text{AdaptiveSVDLinear}_O(\text{Reshape}(C_{\text{head}})) \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step tensor operations across Adaptive SVD Attention layers proceed as follows:

1. **Input Tensor**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Parallel Base & Adapter Down-Projections ($Q, K, V$)**:
   - Query frozen SVD down-projection: $Q_{\text{svd, low}} = X V_Q^\top \in \mathbb{R}^{B \times N \times r_{\text{svd}}}$
   - Query adapter down-projection: $Q_{\text{adap, low}} = X B_Q^\top \in \mathbb{R}^{B \times N \times r_{\text{res}}}$
3. **Up-Projections & Combination**:
   - Query up-projection: $Q_{\text{full}} = Q_{\text{svd, low}} U_Q^\top + Q_{\text{adap, low}} A_Q^\top + b_Q \in \mathbb{R}^{B \times N \times D}$
   - Key projections: $K_{\text{full}} = (X V_K^\top) U_K^\top + (X B_K^\top) A_K^\top + b_K \in \mathbb{R}^{B \times N \times D}$
   - Value projections: $V_{\text{full}} = (X V_V^\top) U_V^\top + (X B_V^\top) A_V^\top + b_V \in \mathbb{R}^{B \times N \times D}$
4. **Multi-Head Layout Conversion**:
   - $Q, K, V \to [B, N, H, d_h] \xrightarrow{\text{transpose}(1, 2)} [B, H, N, d_h]$
5. **Attention Scoring & Masking**:
   - $S = Q K^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
   - Mask addition: $S \leftarrow \text{apply\_attention\_mask}(S, \text{mask})$
6. **Softmax & Context Weighted Aggregation**:
   - $P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$
   - $C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$
7. **Head Concatenation & Output Adaptive Projection**:
   - $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
   - $O = (\tilde{C} V_O^\top) U_O^\top + (\tilde{C} B_O^\top) A_O^\top + b_O \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Frozen SVD GEMMs ($X V^\top U^\top$): $4 \times 2 B N D r_{\text{svd}} = 8 B N D r_{\text{svd}}$ FLOPs
  - Adapter GEMMs ($X B^\top A^\top$): $4 \times 2 B N D r_{\text{res}} = 8 B N D r_{\text{res}}$ FLOPs
  - Subtotal Projection FLOPs: $8 B N D (r_{\text{svd}} + r_{\text{res}})$ FLOPs
  - Multi-Head Score ($Q K^\top$): $2 B N^2 D$ FLOPs
  - Context Aggregation ($P V$): $2 B N^2 D$ FLOPs
  - Total Layer FLOPs: $8 B N D (r_{\text{svd}} + r_{\text{res}}) + 4 B N^2 D$
- **MAC Operations**:
  - SVD + Adapter Projections: $4 B N D (r_{\text{svd}} + r_{\text{res}})$ MACs
  - Attention Matrices: $2 B N^2 D$ MACs
- **Memory Footprint & Trainable Parameter Reduction**:
  - Total Projection Parameters: $4 \times [(d_{\text{out}} + d_{\text{in}}) (r_{\text{svd}} + r_{\text{res}}) + d_{\text{out}}]$
  - Trainable Parameters (Adapters + Biases): $4 \times [ (d_{\text{out}} + d_{\text{in}}) r_{\text{res}} + d_{\text{out}} ]$
  - For $D = 768, r_{\text{svd}} = 32, r_{\text{res}} = 8$: Total parameters $= 494,592$ (Compression Ratio $= 4.77\times$ vs uncompressed $2,359,296$).
  - Trainable parameters $= 4 \times [1536 \times 8 + 768] = 52,224$ (Trainable Reduction Ratio $= 45.18\times$).

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Dual GEMM Execution Latency**:
  Evaluating two separate low-rank matrix pairs ($X V^\top U^\top$ and $X B^\top A^\top$) per linear projection increases kernel launch overhead and GPU memory instruction counts compared to fused LoRA execution.
- **Asymmetric Adapter Initialization**:
  Kaiming uniform initialization of both $A$ and $B$ matrices causes initial adapter output $A B X$ to be non-zero, perturbing zero-shot SVD output fidelity at step 0. Standard LoRA zero-initialization of $A$ ($A = 0$) should be enforced.
- **Fixed SVD Rank Allocation**:
  Fixing $r_{\text{svd}}$ uniformly across all attention projections ignores non-uniform singular value decay spectra, forcing high-entropy layers to suffer higher truncation errors.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused SVD + Adapter Matmul Kernel**:
  Construct a Triton kernel that computes $Y = X \begin{bmatrix} V \\ B \end{bmatrix}^\top \begin{bmatrix} U & A \end{bmatrix}^\top + b$ in a single pass. Concatenating factor matrices along rank dimension reduces 4 linear kernel calls to 2.
- **Zero-Initialized Adapter Scaling**:
  Initialize adapter matrix $A = 0$ and introduce scaling hyperparameter $\gamma = \frac{\alpha}{r_{\text{res}}}$: $W_{\text{eff}} = U V + \gamma A B$. This guarantees exact zero-shot equivalence to truncated SVD at step 0: $W_{\text{eff}}\big|_{t=0} = U V$.
- **FlashAttention Integration**:
  Stream intermediate adaptive projections $Q, K, V$ directly into FlashAttention-2 / FlashAttention-3 Triton kernels to compute online softmax attention without storing $N \times N$ score matrices in GPU memory.