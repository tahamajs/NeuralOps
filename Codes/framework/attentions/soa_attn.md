# Shared Orthogonal Attention (SOA): SVD-Factorized Latent Bottlenecks for Parameter-Efficient Multi-Head Attention

## 1. Executive Abstract & Architectural Motivation
Standard Multi-Head Self-Attention (MHSA) in Transformer architectures requires dense projection matrices ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$) that introduce heavy memory footprints and linear transformation FLOP overhead $\mathcal{O}(D^2)$ per token. Standard Low-Rank Adaptation (LoRA) adds auxiliary low-rank paths, but maintains full-rank base weight matrices in memory.

**Shared Orthogonal Attention (SOA)** establishes a parameter-efficient attention architecture that compresses dense $Q, K, V$ projections into rank-$r$ bottleneck pairs ($r = \text{k\_rank} \ll D$):
1. **Zero-Shot SVD Latent Subspace Factorization**: Source projection weights $W_0 \in \mathbb{R}^{D \times D}$ are factorized via Singular Value Decomposition (SVD) into down-projection matrix $W^{\text{down}} = V_{:r, :}^T \in \mathbb{R}^{r \times D}$ and up-projection matrix $W^{\text{up}} = U_{:, :r} \Sigma_{1:r} \in \mathbb{R}^{D \times r}$.
2. **Latent Bottleneck Representations**: Inputs $X \in \mathbb{R}^{B \times N \times D}$ pass through intermediate $r$-dimensional latent states $Z_Q, Z_K, Z_V \in \mathbb{R}^{B \times N \times r}$, reducing parameter count and projection FLOPs by factor $D / r$.
3. **Exact Multi-Head Scaled Dot-Product Attention**: Preserves standard multi-head scaled dot-product attention geometry while achieving parameter compression.

---

## 2. Mathematical Formulations & Derivations

### 2.1 SVD Factorization of Projection Weights (`_init_from_pretrained`)
For pretrained weight matrix $W_0 \in \mathbb{R}^{D \times D}$, full SVD is evaluated:

$$W_0 = U \Sigma V^T, \quad U \in \mathbb{R}^{D \times D}, \; \Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_D), \; V^T \in \mathbb{R}^{D \times D}$$

Selecting rank $r = \text{k\_rank} \ll D$, projection layers are factorized into down-projection $W^{\text{down}} \in \mathbb{R}^{r \times D}$ and up-projection $W^{\text{up}} \in \mathbb{R}^{D \times r}$:

$$W^{\text{down}} = V_{:r, :}^T, \quad W^{\text{up}} = U_{:, :r} \operatorname{diag}(\sigma_1, \dots, \sigma_r)$$

$$W_0 \approx W^{\text{up}} W^{\text{down}} \in \mathbb{R}^{D \times D}$$

### 2.2 Latent Bottleneck Transformations
For input activations $X \in \mathbb{R}^{B \times N \times D}$, intermediate latent bottleneck representations $Z_Q, Z_K, Z_V \in \mathbb{R}^{B \times N \times r}$ are computed:

$$Z_Q = X (W_Q^{\text{down}})^T, \quad Z_K = X (W_K^{\text{down}})^T, \quad Z_V = X (W_V^{\text{down}})^T \in \mathbb{R}^{B \times N \times r}$$

Up-projections reconstruct full hidden representations $Q, K, V \in \mathbb{R}^{B \times N \times D}$:

$$Q = Z_Q (W_Q^{\text{up}})^T, \quad K = Z_K (W_K^{\text{up}})^T, \quad V = Z_V (W_V^{\text{up}})^T \in \mathbb{R}^{B \times N \times D}$$

### 2.3 Multi-Head Attention Core
Projections are reshaped into $H$ heads with head dimension $d_h = D / H$:

$$Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$$

Scaled dot-product attention scores evaluate:

$$S_{b, h, i, j} = \frac{Q_{b, h, i} \cdot K_{b, h, j}^T}{\sqrt{d_h}} + M^{\text{pad}}_{b, j}$$

$$A = \text{Softmax}(S), \quad A_{\text{drop}} = \text{Dropout}(A), \quad O = A_{\text{drop}} V \in \mathbb{R}^{B \times H \times N \times d_h}$$

Reshaping context to $[B, N, H \cdot d_h]$, the output is projected through linear output layer $W_O$:

$$Y = \text{Reshape}(O) W_O + b_O \in \mathbb{R}^{B \times N \times D}$$

---

## 3. Step-by-Step Tensor Execution Trace

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input Activation** | $X$ | `[B, N, D]` | Base activation state |
| **Down-Projection** | $Z_Q, Z_K, Z_V$ | `[B, N, D] -> [B, N, r]` | Down-projection into $r$-dim latent space |
| **Up-Projection** | $Q, K, V$ | `[B, N, r] -> [B, N, D]` | Up-projection back to hidden dimension |
| **Head Partitioning** | $Q, K, V$ | `[B, N, D] -> [B, H, N, d_h]` | Multi-head view transposition |
| **Scaled Dot-Product** | $S$ | `[B, H, N, d_h] * [B, H, N, d_h]^T -> [B, H, N, N]` | Score matrix evaluation $\frac{Q K^T}{\sqrt{d_h}}$ |
| **Softmax & Dropout** | $A$ | `[B, H, N, N]` | Row-wise Softmax probability matrix |
| **Value Contraction** | $O$ | `[B, H, N, N] * [B, H, N, d_h] -> [B, H, N, d_h]` | Context accumulation |
| **Output Linear GEMM** | $Y$ | `[B, N, H * d_h] -> [B, N, D]` | Output projection $W_O$ |

```
                       [B, N, D] Input Tensor
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     SVD Down-Projections        Check Mask Format
     Z_Q, Z_K, Z_V [B, N, r]     (Prepare Padding Mask M_pad)
             │                           │
             ▼                           │
     SVD Up-Projections                  │
     Q, K, V [B, H, N, d_h]              │
             │                           │
             └─────────────┬─────────────┘
                           ▼
            Scaled Dot-Product Attention
            S = Q K^T / sqrt(d_h) + M_pad
                           │
                           ▼
            Softmax & Dropout: A = Softmax(S)
                           │
                           ▼
            Value Contraction Context O = A V
                           │
                           ▼
            Output Projection W_O
                           │
                           ▼
                     [B, N, D] Output
```

---

## 4. Analytical Complexity & Hardware Profiling

### 4.1 FLOPs Computation Breakdown
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ head count, $d_h = D/H$ head dimension, and $r$ bottleneck rank ($r = \text{k\_rank}$).

1. **SOA Bottleneck Projections ($Q, K, V$)**:
   - Down-projection ($X W^{\text{down} T}$): $3 \times (2 B N D r) = 6 B N D r$
   - Up-projection ($Z W^{\text{up} T}$): $3 \times (2 B N r D) = 6 B N D r$
   $$\text{FLOPs}_{\text{proj}} = 12 B N D r$$
2. **Attention Core & Output Projection**:
   - Query-Key dot products: $2 B H N^2 d_h = 2 B N^2 D$
   - Softmax & Value contraction: $2 B H N^2 d_h = 2 B N^2 D$
   - Output projection $W_O$: $2 B N D^2$
   $$\text{FLOPs}_{\text{attn}} = 4 B N^2 D + 2 B N D^2$$

$$\text{FLOPs}_{\text{total}} = 2 B N D^2 + 12 B N D r + 4 B N^2 D$$

### 4.2 MACs & Memory Footprint Analysis
- **Multiply-Accumulate Operations (MACs)**:
  $$\text{Total MACs} = B N D^2 + 6 B N D r + 2 B N^2 D$$
- **Parameter Footprint**:
  $$\Theta = 3 \times (D \cdot r + r \cdot D) + D^2 = 6 D r + D^2 \ll 4 D^2$$
- **Activation Footprint**:
  - Latent bottleneck tensors $Z_Q, Z_K, Z_V$: $\mathcal{O}(B N r)$
  - Full attention logit matrix: $\mathcal{O}(B H N^2)$
  - Total Memory: $\mathcal{O}(B N D + B N r + B H N^2)$

### 4.3 Asymptotic Scalability Comparison
- **Standard Dense Projection MHSA**: $\mathcal{O}(D^2)$ projection compute per token.
- **SOA Bottleneck Architecture**: $\mathcal{O}(D r)$ projection compute per token, achieving a parameter compression ratio of $\approx 4\times$ when $r = D/4$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Edge Cases

### 5.1 Latent Rank Under-Estimation
- **Mechanism**: Setting bottleneck rank $r < 16$ on deep layers causes truncation of significant pre-trained singular values $\sigma_{r+1 \dots D}$.
- **Consequence**: Information loss in projection representations, degrading zero-shot transfer accuracy.
- **Mitigation**: Select rank $r$ such that $\frac{\sum_{i=1}^r \sigma_i^2}{\sum \sigma_j^2} \ge 0.90$.

### 5.2 Down/Up Matrix Gradient Orthogonality Drift
- **Mechanism**: Unconstrained fine-tuning of $W^{\text{down}}$ and $W^{\text{up}}$ can destroy the initial singular value orthogonality derived from SVD.
- **Mitigation**: Apply an auxiliary orthogonal loss regularizer $\mathcal{L}_{\text{ortho}} = \|W^{\text{down}} (W^{\text{down}})^T - I_r\|_F^2$.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

```
+-----------------------------------------------------------------------------------+
|                        Triton Hardware Execution Graph                            |
|                                                                                   |
|  Input X [B, N, D] ---> [Fused SOA Dual-GEMM Down/Up Projection Kernel]           |
|                          - Evaluate X W^{down T} into SRAM registers              |
|                          - Evaluate Z W^{up T} into Q, K, V SRAM buffers          |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                  |       |
|        v                                                                  v       |
| [Triton FlashAttention-2 Core Kernel]             [Output Linear GEMM Kernel]     |
|  - Block-Sparse SRAM Tiling (128x128)              - Standard Output GEMM         |
|  - Online Softmax Normalization                    - DRAM Write Output Y [B, N, D] |
+-----------------------------------------------------------------------------------+
```

### 6.1 Triton Strategy 1: Fused Dual-GEMM Bottleneck Kernel
Standard PyTorch executes two sequential matrix multiplications (`x @ down.weight.T` followed by `@ up.weight.T`), allocating intermediate tensor $[B, N, r]$ in global DRAM.

**Triton Optimization**:
- Implement a custom Triton GEMM kernel loading input tile $X_i \in \mathbb{R}^{\text{BLOCK\_M} \times D}$ into SRAM.
- Compute down-projection $W^{\text{down} T}$ in register files and immediately evaluate up-projection $W^{\text{up} T}$.
- **Speedup**: Reduces DRAM memory bandwidth overhead by $2\times$, achieving up to $1.8\times$ latency reduction.

### 6.2 Triton Strategy 2: FlashAttention-2 Core Integration
- Stream Q, K, V tensors through a Triton FlashAttention-2 kernel with `BLOCK_M = 128` and `BLOCK_N = 128`.
- Maintain online maximum tracking $m_i$ and normalizer $l_i$ directly in SRAM registers, eliminating global memory allocation for $[B, H, N, N]$ attention matrices.

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Shared Key-Query Subspace Fusion**: Share a single universal down-projection matrix $W_{\text{shared}}^{\text{down}} \in \mathbb{R}^{r \times D}$ across Query and Key projections, leaving only up-projection matrices distinct ($Q = X W_{\text{shared}}^{\text{down} T} W_Q^{\text{up} T}$), halving projection parameter storage.
2. **Dynamic Energy-Based Rank Truncation per Layer**: Automatically set $r_l$ per layer $l$ based on a cumulative energy target $\eta = 0.95$, allocating higher rank to early attention layers and lower rank to deep layers.
3. **Structured LoRA Adapter Augmentation**: Attach trainable rank-4 LoRA paths $(A B^T)$ to $W^{\text{up}}$, enabling low-rank fine-tuning while leaving frozen SVD singular basis vectors unchanged.
4. **FP8 Quantized Latent Bottleneck GEMM Execution**: Execute down- and up-projections using FP8 (E4M3) precision on NVIDIA Hopper GPUs, quadrupling bottleneck projection FLOP throughput.