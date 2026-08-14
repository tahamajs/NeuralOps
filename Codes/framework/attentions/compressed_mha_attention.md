# Compressed Multi-Head Attention: Joint Truncated SVD Subspace Decomposition

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Attention (MHA) layers in Transformer models rely on four dense linear projections ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$), requiring $4D^2$ parameters per layer. When scaling Transformer architectures or deploying models in memory-constrained environments, storing and fetching these dense weight matrices creates substantial memory footprint and memory bandwidth overhead.

Compressed Multi-Head Attention (`CompressedDistilBertAttention`) addresses this by factorizing the joint input space and output space through low-rank Truncated Singular Value Decomposition (SVD). By stacking Query, Key, and Value projection matrices into a unified vertical block matrix $W_{\text{qkv}} = [W_Q; W_K; W_V] \in \mathbb{R}^{3D \times D}$ and decomposing both $W_{\text{qkv}}$ and the output projection $W_{\text{out}} \in \mathbb{R}^{D \times D}$ into orthogonal singular vector bases, the module projects high-dimensional activations through low-rank bottlenecks ($r_{\text{qkv}}, r_{\text{out}} \ll D$). This provides significant parameter compression and computational speedup without structural fine-tuning.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times L \times D}$ be the input hidden state tensor, where $B$ is batch size, $L$ is sequence length, and $D$ is model hidden dimension. Let $H$ be the number of heads and $d_k = D / H$ the head dimension.

### Stacked QKV Matrix Singular Value Decomposition:
Pretrained projection weights $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ are vertically concatenated to form $W_{\text{qkv}} \in \mathbb{R}^{3D \times D}$:

\[
W_{\text{qkv}} = \begin{bmatrix} W_Q \\ W_K \\ W_V \end{bmatrix} \in \mathbb{R}^{3D \times D}
\]

Applying full SVD to $W_{\text{qkv}}$ yields:

\[
W_{\text{qkv}} = U_{\text{qkv}} \Sigma_{\text{qkv}} V_{\text{qkv}}^\top
\]

where $U_{\text{qkv}} \in \mathbb{R}^{3D \times 3D}$ (left singular vectors), $\Sigma_{\text{qkv}} \in \mathbb{R}^{3D \times D}$ (singular values), and $V_{\text{qkv}}^\top \in \mathbb{R}^{D \times D}$ (right singular vectors). Truncating to rank $r_{\text{qkv}} \le \min(3D, D)$:

\[
W_{\text{qkv}} \approx \widetilde{U}_{\text{qkv}} \widetilde{\Sigma}_{\text{qkv}} \widetilde{V}_{\text{qkv}}^\top
\]

where $\widetilde{U}_{\text{qkv}} \in \mathbb{R}^{3D \times r_{\text{qkv}}}$, $\widetilde{\Sigma}_{\text{qkv}} \in \mathbb{R}^{r_{\text{qkv}} \times r_{\text{qkv}}}$, and $\widetilde{V}_{\text{qkv}}^\top \in \mathbb{R}^{r_{\text{qkv}} \times D}$.

### Joint QKV Projection Forward Pass:
Given input tensor $X \in \mathbb{R}^{B \times L \times D}$:
1. **Low-Rank Down-Projection**:
   \[
   Z_{\text{qkv}} = X \widetilde{V}_{\text{qkv}} \in \mathbb{R}^{B \times L \times r_{\text{qkv}}}
   \]
2. **Singular Value Scaling**:
   \[
   \widetilde{Z}_{\text{qkv}} = Z_{\text{qkv}} \odot \text{diag}(\widetilde{\Sigma}_{\text{qkv}}) \in \mathbb{R}^{B \times L \times r_{\text{qkv}}}
   \]
3. **Up-Projection & Unstacking**:
   \[
   QKV_{\text{comp}} = \widetilde{Z}_{\text{qkv}} \widetilde{U}_{\text{qkv}}^\top \in \mathbb{R}^{B \times L \times 3D}
   \]
   Partitioning along the last dimension yields queries, keys, and values:
   \[
   Q = QKV_{\text{comp}}[:, :, 0:D] + b_q \in \mathbb{R}^{B \times L \times D}
   \]
   \[
   K = QKV_{\text{comp}}[:, :, D:2D] + b_k \in \mathbb{R}^{B \times L \times D}
   \]
   \[
   V = QKV_{\text{comp}}[:, :, 2D:3D] + b_v \in \mathbb{R}^{B \times L \times D}
   \]

### Scaled Dot-Product Attention:
Reshaping $Q, K, V$ into $[B, H, L, d_k]$:

\[
S = \frac{Q K^\top}{\sqrt{d_k}} + M \in \mathbb{R}^{B \times H \times L \times L}
\]
\[
A = \text{Softmax}(S, \text{dim}=-1), \quad \widehat{A} = \text{Dropout}(A, p)
\]
\[
\text{Context} = \widehat{A} V \in \mathbb{R}^{B \times H \times L \times d_k} \implies \text{Reshaped Context} \in \mathbb{R}^{B \times L \times D}
\]

### Truncated SVD Output Projection:
Pretrained output projection matrix $W_{\text{out}} \in \mathbb{R}^{D \times D}$ is factorized into rank $r_{\text{out}}$ matrices:

\[
W_{\text{out}} \approx \widetilde{U}_{\text{out}} \widetilde{\Sigma}_{\text{out}} \widetilde{V}_{\text{out}}^\top
\]

where $\widetilde{U}_{\text{out}} \in \mathbb{R}^{D \times r_{\text{out}}}$, $\widetilde{\Sigma}_{\text{out}} \in \mathbb{R}^{r_{\text{out}} \times r_{\text{out}}}$, and $\widetilde{V}_{\text{out}}^\top \in \mathbb{R}^{r_{\text{out}} \times D}$.

The output transformation is computed as:

\[
Z_{\text{out}} = \text{Context} \cdot \widetilde{V}_{\text{out}} \in \mathbb{R}^{B \times L \times r_{\text{out}}}
\]
\[
\widetilde{Z}_{\text{out}} = Z_{\text{out}} \odot \text{diag}(\widetilde{\Sigma}_{\text{out}}) \in \mathbb{R}^{B \times L \times r_{\text{out}}}
\]
\[
Y = \widetilde{Z}_{\text{out}} \widetilde{U}_{\text{out}}^\top + b_{\text{out}} \in \mathbb{R}^{B \times L \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: hidden_states [B, L, D]
  │
  ├──> MatMul(Vh_qkv.T) ──> [B, L, r_qkv]
  ├──> Multiply S_qkv ──> proj: [B, L, r_qkv]
  ├──> MatMul(U_qkv.T) ──> qkv_comp: [B, L, 3D]
  │
  ├──> Split into Q, K, V [B, L, D] & Add Biases (b_q, b_k, b_v)
  ├──> Reshape & Transpose ──> Q, K, V: [B, H, L, d_k]
  │
  ├──> MatMul: Q @ K.transpose(-2, -1) / sqrt(d_k) ──> scores: [B, H, L, L]
  ├──> Add attention_mask ──> [B, H, L, L]
  ├──> Softmax(dim=-1) & Dropout ──> attn: [B, H, L, L]
  ├──> MatMul: attn @ V ──> context: [B, H, L, d_k]
  │
  ├──> Transpose & Contiguous View ──> context: [B, L, D]
  ├──> MatMul(Vh_out.T) ──> [B, L, r_out]
  ├──> Multiply S_out ──> fused: [B, L, r_out]
  ├──> MatMul(U_out.T) ──> out: [B, L, D]
  └──> Add bias_out ──> Output: [B, L, D]
```

---

## 4. Analytical Complexity Profiling

### Parameter & Computational Breakdown

Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H$ heads, $d_k = D / H$, $r_{\text{qkv}}$ joint QKV SVD rank, and $r_{\text{out}}$ output SVD rank.

1. **Parameter Counts**:
   - Original Uncompressed Attention Parameters: $3 D^2 (W_Q, W_K, W_V) + D^2 (W_O) = 4 D^2$.
   - SVD Compressed Parameters:
     - $V_{\text{qkv}}^\top$: $r_{\text{qkv}} D$
     - $\Sigma_{\text{qkv}}$: $r_{\text{qkv}}$
     - $U_{\text{qkv}}^\top$: $3 D r_{\text{qkv}}$
     - $V_{\text{out}}^\top$: $r_{\text{out}} D$
     - $\Sigma_{\text{out}}$: $r_{\text{out}}$
     - $U_{\text{out}}^\top$: $D r_{\text{out}}$
   - Total Compressed Parameters: $4 D r_{\text{qkv}} + 2 D r_{\text{out}} + r_{\text{qkv}} + r_{\text{out}}$.

2. **Parameter Reduction Ratio (for $D=768, r_{\text{qkv}}=32, r_{\text{out}}=32$)**:
   - Original: $4 \times 768^2 = 2,359,296$ parameters ($9.44 \text{ MB}$ in FP32).
   - Compressed: $4(768)(32) + 2(768)(32) + 32 + 32 = 147,520$ parameters ($0.59 \text{ MB}$ in FP32).
   - **Compression Factor**: $\approx 16.0\times$ parameter footprint reduction ($93.75\%$ savings).

3. **FLOPs & MACs**:
   - Joint QKV Projections:
     - Down-projection: $2 B L D r_{\text{qkv}}$ FLOPs ($B L D r_{\text{qkv}}$ MACs).
     - Up-projection: $2 B L r_{\text{qkv}} (3D) = 6 B L D r_{\text{qkv}}$ FLOPs ($3 B L D r_{\text{qkv}}$ MACs).
   - Core Attention ($Q K^\top$ & $A V$): $4 B L^2 D$ FLOPs ($2 B L^2 D$ MACs).
   - Output Projection:
     - Down-projection: $2 B L D r_{\text{out}}$ FLOPs ($B L D r_{\text{out}}$ MACs).
     - Up-projection: $2 B L r_{\text{out}} D$ FLOPs ($B L D r_{\text{out}}$ MACs).

- **Total FLOPs**: $8 B L D r_{\text{qkv}} + 4 B L D r_{\text{out}} + 4 B L^2 D$
- **Total MACs**: $4 B L D r_{\text{qkv}} + 2 B L D r_{\text{out}} + 2 B L^2 D$

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Singular Value Energy Truncation Loss
Truncating SVD matrices to low ranks ($r_{\text{qkv}} \ll D$) drops lower-spectrum singular values. If the singular value spectrum of $W_{\text{qkv}}$ decays slowly (high spectral entropy), truncating at $r = 32$ discards significant variance, causing loss of performance in zero-shot tasks without post-SVD fine-tuning.

```
Singular Value Index (i)  ───>  1 ....... r_qkv ....... D
Singular Value Magnitude  ───> [██████████░░░░░░░░░░░░░░░]
                                  │ Retained │ Discarded  │
```

### Buffer Device Synchronization Latency
In `CompressedDistilBertAttention`, `Vh_qkv`, `S_qkv`, `U_qkv`, etc., are registered as buffers. If `hidden_states` is moved to GPU dynamically while buffers remain on CPU, runtime device checks trigger `self.U_qkv.to(device)` during the forward pass, inducing host-to-device PCI-e transfers and kernel launch stalls.

### Head Coupling via Joint SVD
By stacking $[W_Q; W_K; W_V]$ into a single matrix before SVD, the singular bases $V_{\text{qkv}}$ are shared across queries, keys, and values. This coupling restricts independent subspace alignment between query and key vectors across distinct heads.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Low-Rank Linear GEMM Triton Architecture
Instead of executing two separate sequential PyTorch `torch.matmul` kernel calls for down-projection and up-projection, Triton fuses the low-rank factorized linear transformations into a single CUDA kernel.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ X [B, L, D]                                            │
      │ Vh_qkv [r_qkv, D]                                      │
      │ S_qkv [r_qkv]                                          │
      │ U_qkv [3D, r_qkv]                                      │
      └───────────────────────────┬────────────────────────────┘
                                  │ Fused Tile Read
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM Scratchpad                         │
 │ 1. Load X Tile [BLOCK_M, D] & Vh_qkv Tile [r_qkv, D]             │
 │ 2. Compute Intermediate GEMM: Z_tile = X_tile @ Vh_qkv.T         │
 │ 3. Scale by S_qkv: Z_scaled = Z_tile * S_qkv                      │
 │ 4. Compute Up-Projection GEMM: QKV_tile = Z_scaled @ U_qkv.T     │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ Write Direct to Q, K, V Registers
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Q, K, V Tensors [B, L, D]                              │
      └────────────────────────────────────────────────────────┘
```

### Triton Optimization Guidelines:
1. **Pre-Multiplied Weight Buffers**:
   During initialization, absorb singular values $S_{\text{qkv}}$ directly into $U_{\text{qkv}}$:
   \[
   \widehat{U}_{\text{qkv}} = U_{\text{qkv}} \cdot \text{diag}(S_{\text{qkv}}) \in \mathbb{R}^{3D \times r_{\text{qkv}}}
   \]
   This reduces the forward pass computation to two pure GEMM calls ($X V_{\text{qkv}}^\top$ and $Z \widehat{U}_{\text{qkv}}^\top$), eliminating per-element scaling kernel overhead.
2. **Fused Double-GEMM Kernel**:
   Write a specialized 2-stage GEMM Triton kernel that retains intermediate low-rank representations $Z \in \mathbb{R}^{\text{BLOCK\_M} \times r_{\text{qkv}}}$ in GPU L1 cache registers, writing output directly into the FlashAttention input layout.
3. **Weight Fusion for Fine-Tuning**:
   If fine-tuning is completed, re-materialize dense weights $W_{\text{qkv}}^{\text{reconstructed}} = \widehat{U}_{\text{qkv}} V_{\text{qkv}}^\top \in \mathbb{R}^{3D \times D}$ back into standard single matrix projections, removing all low-rank overhead during inference.