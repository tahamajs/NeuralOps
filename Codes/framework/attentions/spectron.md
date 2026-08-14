# SpectronAttention: Importance-Aware Low-Rank + Sparse Residual Attention

## 1. Abstract & Theoretical Motivation
SpectronAttention (`SpectronAttention`, registered as `spectron`) introduces an importance-aware hybrid compression framework for Transformer Multi-Head Attention projections ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$). SpectronAttention combines non-uniform low-rank matrix factorizations via truncated Singular Value Decomposition (SVD) with magnitude-pruned trainable sparse residual matrices $S_{\text{sparse}}$. To optimize parameter efficiency across model layers, per-projection ranks $\{r_{i}\}$ are allocated via Lagrangian dual optimization on spectral energy loss cost functions under a global parameter budget $B_{\text{target}}$. To recover high-frequency singular modes discarded during rank truncation, an additive sparse residual matrix $S_i$ is initialized from SVD approximation errors and periodically re-pruned during downstream fine-tuning via a dynamic Trainer callback.

## 2. Mathematical Derivations & Formulations
Let $\mathcal{W} = \{W_1, W_2, \dots, W_M\}$ denote the set of $M = 4 \times L$ attention projection matrices across an $L$-layer Transformer. For matrix $W_i \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with singular values $\sigma_{i, 1} \ge \sigma_{i, 2} \ge \dots \ge \sigma_{i, k} \ge 0$, cumulative retained spectral energy $E_i(r)$ is:
$$E_i(r) = \frac{\sum_{j=1}^r \sigma_{i, j}^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_{i, j}^2}$$

The spectral information loss cost function $c_i(r)$ and parameter allocation function $P_i(r)$ are:
$$c_i(r) = 1 - E_i(r), \qquad P_i(r) = r \cdot (d_{\text{in}} + d_{\text{out}})$$

### Global Constrained Optimization & Lagrangian Dual
Given parameter budget constraint $B_{\text{target}} = \alpha \sum_{i=1}^M (d_{\text{in}} d_{\text{out}})$, the optimal rank allocation vector $\mathbf{r}^* = [r_1^*, \dots, r_M^*]$ solves:
$$\min_{\{r_i\}} \sum_{i=1}^M c_i(r_i) \quad \text{subject to} \quad \sum_{i=1}^M P_i(r_i) \le B_{\text{target}}, \quad 1 \le r_i \le r_{\text{max}}^{(i)}$$

Formulating the Lagrangian relaxation with dual multiplier $\lambda \ge 0$:
$$\mathcal{L}(\mathbf{r}, \lambda) = \sum_{i=1}^M c_i(r_i) + \lambda \left( \sum_{i=1}^M P_i(r_i) - B_{\text{target}} \right)$$

Discrete search per matrix evaluates:
$$r_i^*(\lambda) = \arg\min_{r \in [1, r_{\text{max}}^{(i)}]} \left\{ c_i(r) + \lambda \cdot r \cdot (d_{\text{in}} + d_{\text{out}}) \right\}$$

Binary search over $\lambda \in [\lambda_{\text{min}}, \lambda_{\text{max}}]$ determines the optimal multiplier $\lambda^*$ satisfying total budget constraint $\sum_{i=1}^M P_i(r_i^*(\lambda^*)) \le B_{\text{target}}$.

### Low-Rank Factorization with Magnitude-Pruned Sparse Residuals
For assigned rank $r_i^*$, truncated SVD factorizes base projections into $W_i^{\text{down}} \in \mathbb{R}^{r_i^* \times d_{\text{in}}}$ and $W_i^{\text{up}} \in \mathbb{R}^{d_{\text{out}} \times r_i^*}$:
$$W_i^{\text{down}} = V_{:r_i^*, :}^\top, \qquad W_i^{\text{up}} = U_{:, :r_i^*} \text{diag}(\sigma_{1:r_i^*})$$

An additive sparse residual matrix $S_i \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ captures truncation error $R_i = W_i - W_i^{\text{up}} W_i^{\text{down}}$ under target sparsity $\rho \in (0, 1)$:
$$\tau = \text{KthValue}(|R_i|, \lfloor \text{numel}(R_i) \cdot (1 - \rho) \rfloor)$$
$$S_i[j, k] = \begin{cases} R_i[j, k] & \text{if } |R_i[j, k]| \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

The combined forward projection applied to activation inputs $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ evaluates as:
$$\text{SpectronProj}(X) = X (W_i^{\text{down}})^\top (W_i^{\text{up}})^\top + X S_i^\top + b_i$$

Applying this to $Q, K, V, O$ yields:
$$Q, K, V = \text{Reshape}(\text{SpectronProj}_Q(X)), \text{Reshape}(\text{SpectronProj}_K(X)), \text{Reshape}(\text{SpectronProj}_V(X)) \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$A = \text{Dropout}\left(\text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_h}} + M_{\text{attn}}\right)\right) V \implies O = \text{SpectronProj}_O(\text{Reshape}(A)) \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step tensor operations across SpectronAttention proceed as follows:

1. **Input Tensor**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Layer Projection Execution ($Q, K, V$)**:
   - Low-rank down-projection: $Q_{\text{low}} = X (W_Q^{\text{down}})^\top \in \mathbb{R}^{B \times N \times r_Q}$
   - Low-rank up-projection: $Q_{\text{up}} = Q_{\text{low}} (W_Q^{\text{up}})^\top + b_Q \in \mathbb{R}^{B \times N \times D}$
   - Sparse residual GEMM (if active): $Q_{\text{sparse}} = F.linear(X, S_Q) \in \mathbb{R}^{B \times N \times D}$
   - Combined projection: $Q_{\text{full}} = Q_{\text{up}} + Q_{\text{sparse}} \in \mathbb{R}^{B \times N \times D}$
   - Key projections: $K_{\text{full}} = (X (W_K^{\text{down}})^\top) (W_K^{\text{up}})^\top + b_K (+ X S_K^\top) \in \mathbb{R}^{B \times N \times D}$
   - Value projections: $V_{\text{full}} = (X (W_V^{\text{down}})^\top) (W_V^{\text{up}})^\top + b_V (+ X S_V^\top) \in \mathbb{R}^{B \times N \times D}$
3. **Multi-Head Layout Conversion**:
   - $Q, K, V \to [B, N, H, d_h] \xrightarrow{\text{transpose}(1, 2)} [B, H, N, d_h]$
4. **Attention Scoring & Masking**:
   - $S = Q K^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
   - Mask addition: $S \leftarrow S + (1.0 - M) \cdot (-10^9)$
5. **Softmax & Context Weighted Aggregation**:
   - $P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$
   - $C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$
6. **Head Concatenation & Output Spectron Projection**:
   - $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
   - $O_{\text{low}} = \tilde{C} (W_O^{\text{down}})^\top \in \mathbb{R}^{B \times N \times r_O}$
   - $O = O_{\text{low}} (W_O^{\text{up}})^\top + b_O (+ \tilde{C} S_O^\top) \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Low-Rank Projections ($Q, K, V, O$): $2 B N D (r_Q + r_K + r_V + r_O)$ FLOPs
  - Sparse Residual GEMMs: $4 \times 2 B N D^2 (1 - \rho)$ FLOPs
  - Multi-Head Score ($Q K^\top$): $2 B N^2 D$ FLOPs
  - Value Aggregation ($P V$): $2 B N^2 D$ FLOPs
  - Total Layer FLOPs: $2 B N D \sum_m r_m + 8 B N D^2 (1 - \rho) + 4 B N^2 D$
- **MAC Operations**:
  - Low-Rank Projections: $B N D \sum_m r_m$ MACs
  - Sparse Residual Matmuls: $4 B N D^2 (1 - \rho)$ MACs
  - Multi-Head Attention: $2 B N^2 D$ MACs
- **Memory Footprint & Parameter Reduction**:
  - Original Model Parameters: $4 L D^2$
  - Spectron Parameters: $\sum_{i=1}^{M} \left[ r_i^* (d_{\text{in}} + d_{\text{out}}) + (1 - \rho) d_{\text{in}} d_{\text{out}} \right]$
  - Under $\alpha = 0.20$ budget ratio and $\rho = 0.70$ sparsity, total memory reduction is $3.85\times$.

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Dense Representation of Sparse Residual Parameters**:
  Storing sparse residual $S_i$ as a dense `nn.Parameter` of shape $[d_{\text{out}}, d_{\text{in}}]$ retains full $\mathcal{O}(D^2)$ memory storage without custom GPU sparse tensor formats, neutralizing parameter memory savings.
- **Pruning Callback Gradient Discontinuity**:
  Periodic magnitude re-pruning via `SparsePruningCallback` hard-zeroes weights during training steps ($w < \tau \implies w = 0$). Hard thresholding introduces gradient discontinuities that destabilize AdamW momentum states.
- **Spectral Rank Pre-Allocation Drift**:
  Ranks $\{r_i^*\}$ are pre-computed on zero-shot pretrained weights. Domain-specific fine-tuning alters singular spectra, making pre-allocated ranks suboptimal.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Fused Low-Rank GEMM + Block-Sparse Residual Triton Kernel**:
  Implement a Triton kernel that computes low-rank projection $X V^\top U^\top$ and sparse residual multiplication $X S^\top$ using 2:4 / block-sparse Tensor Cores. Staging intermediate tile $B_N \times r$ ($64 \times 32$) in SRAM avoids DRAM writes.
- **Soft Straight-Through Estimator (STE) Pruning**:
  Replace hard thresholding in `SparsePruningCallback` with a differentiable Straight-Through Estimator (STE) or soft sigmoidal mask $\hat{S} = S \odot \sigma(\beta (|S| - \tau))$, enabling continuous gradient flow into sparse residual weights.
- **FlashAttention Integration**:
  Stream intermediate factorized projections $Q, K, V$ directly into FlashAttention-2 / FlashAttention-3 Triton kernels to execute online softmax attention without allocating $N \times N$ score matrices in GPU memory.