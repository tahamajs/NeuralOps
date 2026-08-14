# SVD-Compressed Attention: Importance-Aware Lagrangian Spectral Rank Allocation

## 1. Abstract & Theoretical Motivation
SVD-Compressed Attention (`SVDCompressedAttention`) establishes a principled framework for compressing Transformer Multi-Head Attention projections ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$) using truncated Singular Value Decomposition (SVD) coupled with Lagrangian spectral rank budgeting and optional low-rank plus sparse residual decomposition. Uniform rank allocation across transformer layers is mathematically suboptimal, as individual projection matrices exhibit non-uniform singular value energy decay rates. SVD-Compressed Attention addresses this through a global, budget-constrained Lagrangian dual optimization problem over per-layer and per-projection ranks $\{r_{i, m}\}$. By minimizing cumulative discarded spectral energy under a target parameter budget $B_{\text{target}}$, the model dynamically concentrates rank capacity into high-entropy projections. Furthermore, an optional sparse residual matrix $S_{\text{sparse}}$ recovers high-frequency singular modes, mitigating rank collapse during fine-tuning.

## 2. Mathematical Derivations & Formulations
Let $\mathcal{W} = \{W_1, W_2, \dots, W_M\}$ denote the set of $M = 4 \times L$ projection weight matrices in an $L$-layer Transformer. For weight matrix $W_i \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with singular values $\sigma_{i, 1} \ge \sigma_{i, 2} \ge \dots \ge \sigma_{i, k} \ge 0$, the cumulative retained spectral energy at rank $r$ is:
$$E_i(r) = \frac{\sum_{j=1}^r \sigma_{i, j}^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_{i, j}^2}$$

The spectral information loss cost function $c_i(r)$ and parameter allocation function $P_i(r)$ are defined as:
$$c_i(r) = 1 - E_i(r), \qquad P_i(r) = r \cdot (d_{\text{in}} + d_{\text{out}})$$

### Global Constrained Optimization & Lagrangian Dual
Given parameter budget constraint $B_{\text{target}} = \alpha \sum_{i=1}^M (d_{\text{in}} d_{\text{out}})$, the rank allocation vector $\mathbf{r}^* = [r_1^*, \dots, r_M^*]$ is determined by:
$$\min_{\{r_i\}} \sum_{i=1}^M c_i(r_i) \quad \text{subject to} \quad \sum_{i=1}^M P_i(r_i) \le B_{\text{target}}, \quad 1 \le r_i \le r_{\text{max}}^{(i)}$$

Formulating the Lagrangian relaxation with dual multiplier $\lambda \ge 0$:
$$\mathcal{L}(\mathbf{r}, \lambda) = \sum_{i=1}^M c_i(r_i) + \lambda \left( \sum_{i=1}^M P_i(r_i) - B_{\text{target}} \right)$$

Because the objective decouples across projection matrices, optimal per-matrix rank for a fixed multiplier $\lambda$ is obtained independently via discrete search:
$$r_i^*(\lambda) = \arg\min_{r \in [1, r_{\text{max}}^{(i)}]} \left\{ c_i(r) + \lambda \cdot r \cdot (d_{\text{in}} + d_{\text{out}}) \right\}$$

Binary search over $\lambda \in [\lambda_{\text{min}}, \lambda_{\text{max}}]$ determines the optimal multiplier $\lambda^*$ satisfying total budget constraint $\sum_{i=1}^M P_i(r_i^*(\lambda^*)) \approx B_{\text{target}}$.

### Low-Rank Factorization with Pruned Sparse Residuals
For assigned rank $r_i^*$, truncated SVD yields factorized matrices $W_i^{\text{down}} \in \mathbb{R}^{r_i^* \times d_{\text{in}}}$ and $W_i^{\text{up}} \in \mathbb{R}^{d_{\text{out}} \times r_i^*}$:
$$W_i^{\text{down}} = V_{:r_i^*, :}^\top, \qquad W_i^{\text{up}} = U_{:, :r_i^*} \text{diag}(\sigma_{1:r_i^*})$$

An optional sparse residual matrix $R_{\text{sparse}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ captures truncation residual $E_{\text{res}} = W_i - W_i^{\text{up}} W_i^{\text{down}}$ using percentile-based magnitude pruning at target sparsity $\rho \in (0, 1)$:
$$\tau = \text{Percentile}(|E_{\text{res}}|, \rho \cdot 100)$$
$$R_{\text{sparse}}[j, k] = \begin{cases} E_{\text{res}}[j, k] & \text{if } |E_{\text{res}}[j, k]| \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

The factorized transformation applied to token activations $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ is:
$$\text{Proj}(X) = X (W_i^{\text{down}})^\top (W_i^{\text{up}})^\top + X R_{\text{sparse}}^\top + b_i$$

Subsequent multi-head softmax self-attention proceeds as:
$$Q, K, V = \text{Reshape}(\text{Proj}_Q(X)), \text{Reshape}(\text{Proj}_K(X)), \text{Reshape}(\text{Proj}_V(X)) \in \mathbb{R}^{B \times H \times N \times d_h}$$
$$A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_h}} + M_{\text{attn}}\right) V \implies O = \text{Proj}_O(\text{Reshape}(A)) \in \mathbb{R}^{B \times N \times D}$$

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step tensor operations across SVD-Compressed Attention layers proceed as follows:

1. **Input Representation**: $X \in \mathbb{R}^{B \times N \times D}$
2. **Layer-Specific Factorized Projections ($Q, K, V$)**:
   - Query down-projection: $Q_{\text{low}} = X (W_Q^{\text{down}})^\top \in \mathbb{R}^{B \times N \times r_Q}$
   - Query up-projection: $Q_{\text{up}} = Q_{\text{low}} (W_Q^{\text{up}})^\top + b_Q \in \mathbb{R}^{B \times N \times D}$
   - Optional sparse addition: $Q_{\text{full}} = Q_{\text{up}} + X R_Q^\top \in \mathbb{R}^{B \times N \times D}$
   - Key projections: $K_{\text{low}} = X (W_K^{\text{down}})^\top \in \mathbb{R}^{B \times N \times r_K} \implies K_{\text{full}} = K_{\text{low}} (W_K^{\text{up}})^\top + b_K (+ X R_K^\top)$
   - Value projections: $V_{\text{low}} = X (W_V^{\text{down}})^\top \in \mathbb{R}^{B \times N \times r_V} \implies V_{\text{full}} = V_{\text{low}} (W_V^{\text{up}})^\top + b_V (+ X R_V^\top)$
3. **Multi-Head Layout Conversion**:
   - $Q, K, V \to [B, N, H, d_h] \xrightarrow{\text{transpose}(1, 2)} [B, H, N, d_h]$
4. **Attention Score & Masking**:
   - $S = Q K^\top / \sqrt{d_h} \in \mathbb{R}^{B \times H \times N \times N}$
   - Add additive mask $M \in \mathbb{R}^{B \times 1 \times 1 \times N}$: $S = S + (1 - M) \cdot (-10^9)$
5. **Softmax & Value Weighted Aggregation**:
   - $P = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1)) \in \mathbb{R}^{B \times H \times N \times N}$
   - $C_{\text{head}} = P V \in \mathbb{R}^{B \times H \times N \times d_h}$
6. **Head Concatenation & Output Factorized Projection**:
   - $\tilde{C} = \text{Reshape}(C_{\text{head}}) \in \mathbb{R}^{B \times N \times D}$
   - Output down-projection: $O_{\text{low}} = \tilde{C} (W_O^{\text{down}})^\top \in \mathbb{R}^{B \times N \times r_O}$
   - Output up-projection: $O = O_{\text{low}} (W_O^{\text{up}})^\top + b_O (+ \tilde{C} R_O^\top) \in \mathbb{R}^{B \times N \times D}$

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Down/Up Linear Projections: $2 B N D (r_Q + r_K + r_V + r_O)$ FLOPs
  - Sparse Residual GEMM (if active): $4 \times 2 B N D^2 (1 - \rho)$ FLOPs
  - Score Matmul ($Q K^\top$): $2 B N^2 D$ FLOPs
  - Context Matmul ($P V$): $2 B N^2 D$ FLOPs
  - Total FLOPs: $2 B N D \sum_{m} r_m + 8 B N D^2 (1 - \rho) + 4 B N^2 D$
- **MAC Operations**:
  - Low-Rank Projections: $B N D \sum_m r_m$ MACs
  - Sparse Matmuls: $4 B N D^2 (1 - \rho)$ MACs
  - Multi-Head Attention: $2 B N^2 D$ MACs
- **Memory Footprint & Parameter Reduction**:
  - Original Parameters: $4 L D^2$
  - Lagrangian Compressed Parameters: $\sum_{i=1}^{M} r_i^* (d_{\text{in}} + d_{\text{out}})$
  - Under target budget $\alpha = 0.20$, parameter reduction is exactly $5.0\times$ across all attention sub-layers.

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Spectral Energy Mismatch in Downstream Tasks**:
  Lagrangian allocation uses zero-shot pre-trained SVD energy spectra. Fine-tuning on domain-specific GLUE tasks alters weight gradient directions, causing pre-allocated ranks $\{r_i^*\}$ to become suboptimal during late training epochs.
- **Sparse Residual Memory Latency**:
  Storing sparse residual $R_{\text{sparse}}$ as a dense zero-padded matrix incurs full $\mathcal{O}(D^2)$ memory storage per projection without custom sparse hardware kernels, neutralizing memory footprint gains.
- **Additive Mask Broadcasting Shift**:
  Directly adding 2D sequence masks without shape un-broadcasting leads to broadcasting errors when handling 3D/4D extended causal attention masks.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Lagrangian Dual Multiplier Binary Search Kernel**:
  Precompute singular values $\Sigma_i$ using CPU/GPU batched SVD in parallel. Execute binary search for $\lambda^*$ inside a PyTorch C++ extension to construct all $4 L$ rank allocations in $< 10$ milliseconds.
- **Fused Low-Rank GEMM + Sparse Addition**:
  Implement a Triton kernel that computes $Y = (X W^{\text{down}}) W^{\text{up}} + \text{SparseGEMM}(X, R_{\text{sparse}})$ in a single pass. Load tile $B_N \times r$ into SRAM to eliminate intermediate GPU DRAM allocation.
- **Dynamic Sparse Pruning Callback**:
  Incorporate periodic magnitude re-pruning during PyTorch Trainer backward steps to maintain target residual sparsity $\rho$ and prevent dense fill-in.