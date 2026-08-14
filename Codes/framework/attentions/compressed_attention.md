# Compressed Multi-Head Attention (SVD & Energy-Thresholded Low-Rank Approximations)

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Attention (MHA) modules in deep Transformer architectures rely on full-rank linear parameterizations for input ($W_{\text{qkv}} \in \mathbb{R}^{3D \times D}$) and output ($W_{\text{out}} \in \mathbb{R}^{D \times D}$) projections. Storing and evaluating these high-dimensional dense parameter matrices creates High-Bandwidth Memory (HBM) fetch latency and large activation storage during inference.

This module presents a Singular Value Decomposition (SVD) compression framework comprising two mechanisms: fixed-rank factorization (`CompressedAttentionLayer`) and cumulative spectral energy-preserving dynamic rank truncation (`CompressedAttentionLayer2`). By factorizing projection weights into orthogonal singular vector bases and splitting singular value square roots ($\sqrt{\Sigma}$) across left and right projections, the architecture enforces associative low-rank bottlenecks ($r_{\text{qkv}}, r_{\text{out}} \ll D$). This reduces parameter count and FLOP complexity while maintaining representation energy above defined variance thresholds $\tau_{\text{energy}}$.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times N \times D}$ be the input activation sequence, where $B$ is batch size, $N$ is sequence length, and $D$ is model dimension. Let $H$ be the number of attention heads and $d_k = D / H$ the head dimension.

### 1. Truncated Singular Value Decomposition & Symmetric Square-Root Factorization:
For input projection matrix $W_{\text{qkv}} = [W_Q; W_K; W_V] \in \mathbb{R}^{3D \times D}$ and output projection matrix $W_{\text{out}} \in \mathbb{R}^{D \times D}$, full SVD yields:

\[
W_{\text{qkv}} = U_{\text{qkv}} \Sigma_{\text{qkv}} V_{\text{qkv}}^\top, \quad W_{\text{out}} = U_{\text{out}} \Sigma_{\text{out}} V_{\text{out}}^\top
\]

where $U_{\text{qkv}} \in \mathbb{R}^{3D \times 3D}$, $\Sigma_{\text{qkv}} \in \mathbb{R}^{3D \times D}$, and $V_{\text{qkv}}^\top \in \mathbb{R}^{D \times D}$.

Truncating to rank $r_{\text{qkv}}$ and distributing singular values symmetrically via $\Sigma^{1/2}$:

\[
W_{\text{qkv}} \approx \left( \widetilde{U}_{\text{qkv}} \widetilde{\Sigma}_{\text{qkv}}^{1/2} \right) \left( \widetilde{\Sigma}_{\text{qkv}}^{1/2} \widetilde{V}_{\text{qkv}}^\top \right)
\]

where $\widetilde{U}_{\text{qkv}} \in \mathbb{R}^{3D \times r_{\text{qkv}}}$, $\widetilde{\Sigma}_{\text{qkv}}^{1/2} \in \mathbb{R}^{r_{\text{qkv}} \times r_{\text{qkv}}}$, and $\widetilde{V}_{\text{qkv}}^\top \in \mathbb{R}^{r_{\text{qkv}} \times D}$.

Similarly for $W_{\text{out}}$ with target rank $r_{\text{out}}$:

\[
W_{\text{out}} \approx \left( \widetilde{U}_{\text{out}} \widetilde{\Sigma}_{\text{out}}^{1/2} \right) \left( \widetilde{\Sigma}_{\text{out}}^{1/2} \widetilde{V}_{\text{out}}^\top \right)
\]

### 2. Cumulative Spectral Energy Preservation Criteria:
In `CompressedAttentionLayer2`, target rank $r$ is dynamically determined by enforcing a cumulative singular value energy threshold $\tau_{\text{energy}} \in (0, 1]$ (e.g., $\tau_{\text{energy}} = 0.95$):

\[
r_{\text{qkv}} = \min \left\{ r \in \mathbb{N}^+ \; \Big| \; \frac{\sum_{i=1}^r \sigma_{\text{qkv}, i}}{\sum_{j=1}^R \sigma_{\text{qkv}, j}} \ge \tau_{\text{energy}} \text{ and } r \ge r_{\min} \right\}
\]
\[
r_{\text{out}} = \min \left\{ r \in \mathbb{N}^+ \; \Big| \; \frac{\sum_{i=1}^r \sigma_{\text{out}, i}}{\sum_{j=1}^R \sigma_{\text{out}, j}} \ge \tau_{\text{energy}} \text{ and } r \ge r_{\min} \right\}
\]

where $R$ is full rank and $r_{\min}$ is a minimum threshold constraint (e.g., $r_{\min} = 16$).

### 3. Forward Associative Factorization Execution Pass:
1. **Low-Rank Intermediate Projection**:
   \[
   Z_{\text{qkv}} = \left( X \widetilde{V}_{\text{qkv}} \right) \odot \widetilde{\Sigma}_{\text{qkv}}^{1/2} \in \mathbb{R}^{B \times N \times r_{\text{qkv}}}
   \]
2. **Up-Projection to QKV Subspace**:
   \[
   QKV_{\text{comp}} = Z_{\text{qkv}} \widetilde{U}_{\text{qkv}}^\top + b_{\text{qkv}} \in \mathbb{R}^{B \times N \times 3D}
   \]
   Partitioning into Query, Key, and Value components:
   \[
   Q = QKV_{\text{comp}}[:, :, 0:D], \quad K = QKV_{\text{comp}}[:, :, D:2D], \quad V = QKV_{\text{comp}}[:, :, 2D:3D]
   \]
3. **Scaled Dot-Product Attention**:
   \[
   S = \frac{1}{\sqrt{d_k}} Q K^\top + M_{\text{attn}} + M_{\text{pad}} \in \mathbb{R}^{B \times H \times N \times N}
   \]
   \[
   A = \text{Softmax}(S, \text{dim}=-1), \quad \widehat{A} = \text{Dropout}(A, p)
   \]
   \[
   C = \widehat{A} V \in \mathbb{R}^{B \times N \times D}
   \]
4. **Low-Rank Output Projection**:
   \[
   Z_{\text{out}} = \left( C \widetilde{V}_{\text{out}} \right) \odot \widetilde{\Sigma}_{\text{out}}^{1/2} \in \mathbb{R}^{B \times N \times r_{\text{out}}}
   \]
   \[
   Y = Z_{\text{out}} \widetilde{U}_{\text{out}}^\top + b_{\text{out}} \in \mathbb{R}^{B \times N \times D}
   \]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: query (x) [B, N, D]
  │
  ├──> MatMul(Vh_qkv.T) [r_qkv, D] ──> [B, N, r_qkv]
  ├──> Multiply S_qkv_sqrt [r_qkv] ──> projected_x: [B, N, r_qkv]
  ├──> MatMul(U_qkv.T) [3D, r_qkv] ──> [B, N, 3D]
  ├──> Add bias_qkv [3D] ──> qkv_compressed: [B, N, 3D]
  │
  ├──> Split [0:D], [D:2D], [2D:3D] ──> Q, K, V: [B, N, D]
  │
  ├──> Scaled Dot-Product Attention:
  │      scores = Q @ K.T / sqrt(d_k)
  │      Add attn_mask & key_padding_mask (-1e9)
  │      Softmax(dim=-1) & Dropout ──> attn_weights: [B, N, N]
  │      context = attn_weights @ V ──> context: [B, N, D]
  │
  ├──> MatMul(Vh_out.T) [r_out, D] ──> [B, N, r_out]
  ├──> Multiply S_out_sqrt [r_out] ──> context_fused: [B, N, r_out]
  ├──> MatMul(U_out.T) [D, r_out] ──> [B, N, D]
  └──> Add bias_out [D] ──> Output: [B, N, D]
```

---

## 4. Analytical Complexity Profiling

### Computational Workload & Memory Footprint

Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ heads, $d_k = D / H$, $r_{\text{qkv}}$ QKV rank, and $r_{\text{out}}$ output rank.

1. **Parameter Savings**:
   - Original Uncompressed Parameters: $3 D^2 (W_{\text{in}}) + D^2 (W_{\text{out}}) = 4 D^2$.
   - SVD Parameters (`U_qkv`, `S_qkv_sqrt`, `Vh_qkv`, `U_out`, `S_out_sqrt`, `Vh_out`):
     - $W_{\text{qkv}}$ factors: $3 D r_{\text{qkv}} + r_{\text{qkv}} + r_{\text{qkv}} D = r_{\text{qkv}} (4D + 1)$.
     - $W_{\text{out}}$ factors: $D r_{\text{out}} + r_{\text{out}} + r_{\text{out}} D = r_{\text{out}} (2D + 1)$.
   - Total SVD Parameters: $r_{\text{qkv}} (4D + 1) + r_{\text{out}} (2D + 1)$.
   - Compression Ratio (e.g., $D=768, r_{\text{qkv}}=64, r_{\text{out}}=64$):
     - Original: $2,359,296$ parameters ($9.44 \text{ MB}$).
     - SVD Compressed: $64(3073) + 64(1537) = 196,672 + 98,368 = 295,040$ parameters ($1.18 \text{ MB}$).
     - **Parameter Footprint Savings**: $\approx 8.0\times$ compression ($87.5\%$ savings).

2. **Computational FLOPs & MACs**:
   - QKV Low-Rank Down-Projection: $2 B N D r_{\text{qkv}}$ FLOPs ($B N D r_{\text{qkv}}$ MACs).
   - QKV Low-Rank Up-Projection: $2 B N r_{\text{qkv}} (3D) = 6 B N D r_{\text{qkv}}$ FLOPs ($3 B N D r_{\text{qkv}}$ MACs).
   - Core Attention Scores & Context ($Q K^\top$ & $A V$): $4 B N^2 D$ FLOPs ($2 B N^2 D$ MACs).
   - Output Low-Rank Down-Projection: $2 B N D r_{\text{out}}$ FLOPs ($B N D r_{\text{out}}$ MACs).
   - Output Low-Rank Up-Projection: $2 B N r_{\text{out}} D$ FLOPs ($B N D r_{\text{out}}$ MACs).

- **Total MACs**: $B N \left[ 4 D r_{\text{qkv}} + 2 D r_{\text{out}} + 2 N D \right]$
- **Total FLOPs**: $B N \left[ 8 D r_{\text{qkv}} + 4 D r_{\text{out}} + 4 N D \right]$

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Spectral Energy Tail Loss & Accuracy Drop
When energy threshold $\tau_{\text{energy}}$ is set aggressively low ($\tau_{\text{energy}} < 0.85$), significant singular values in the tail of $\Sigma_{\text{qkv}}$ are discarded. This causes degradation in fine-grained syntactic relationships and cross-head contextual routing.

```
Singular Energy Cumulative Spectrum:
Sum(σ_1..r) / Sum(σ_1..R)  ──>  [██████████████░░░░░]  (Threshold τ = 0.95)
                                  │ Preserved Rank r │ Discarded Tail │
```

### CPU-GPU NumPy SVD Initialization Bottleneck
In `CompressedAttentionLayer`, SVD factorization is performed on CPU using NumPy (`np.linalg.svd`). For large Transformer models ($D > 4096$), computing CPU SVD during module instantiation causes long startup latency and CPU-to-GPU memory transfer overheads.

### Intermediate Activation Allocation Inefficiency
The PyTorch forward pass calculates low-rank projections sequentially (`projected_x = x @ Vh.T`, `qkv = projected_x @ U.T`), instantiating intermediate activation tensors $[B, N, r_{\text{qkv}}]$ in GPU DRAM. This introduces memory bandwidth pressure compared to fused kernels.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Double-GEMM SVD Triton Kernel Architecture
To saturate GPU Tensor Cores and eliminate intermediate DRAM activation writes, Triton fuses the low-rank down-projection, singular scaling, and up-projection into a single memory-efficient kernel.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ Input x [B, N, D]                                      │
      │ Vh_qkv [r_qkv, D], S_qkv_sqrt [r_qkv], U_qkv [3D, r_qkv]│
      └───────────────────────────┬────────────────────────────┘
                                  │ Coalesced Tile Load
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM Scratchpad                         │
 │ 1. Load Input Tile: x_tile [BLOCK_M, D]                          │
 │ 2. Compute Intermediate Down-GEMM: Z = x_tile @ Vh_qkv.T         │
 │ 3. Scale Singular Values: Z_scaled = Z * S_qkv_sqrt              │
 │ 4. Compute Up-GEMM: QKV_tile = Z_scaled @ U_qkv.T               │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ Direct Stream to Attention
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Q, K, V Tensors [B, N, 3D]                             │
      └────────────────────────────────────────────────────────┘
```

### Key Triton Implementation Guidelines:
1. **Pre-Absorbed Weight Buffers**:
   Absorb singular square roots $\sqrt{\Sigma}$ directly into left singular matrices $\widetilde{U}$ at initialization:
   \[
   \widehat{U}_{\text{qkv}} = \widetilde{U}_{\text{qkv}} \cdot \text{diag}\left(\sqrt{\Sigma_{\text{qkv}}}\right) \in \mathbb{R}^{3D \times r_{\text{qkv}}}
   \]
   This reduces the kernel workload to two sequential GEMM calls ($x \widetilde{V}^\top$ and $Z \widehat{U}^\top$), removing elementwise multiplication overhead.
2. **SRAM Scratchpad Reuse**:
   Retain low-rank activation vectors $Z \in \mathbb{R}^{\text{BLOCK\_M} \times r_{\text{qkv}}}$ in Streaming Multiprocessor (SM) L1 registers, skipping intermediate DRAM writes entirely.
3. **Dynamic GPU-Native Batched SVD**:
   Replace CPU NumPy initialization calls with GPU-accelerated PyTorch batched SVD primitives (`torch.linalg.svd`) to accelerate model compression during online training pipelines.
