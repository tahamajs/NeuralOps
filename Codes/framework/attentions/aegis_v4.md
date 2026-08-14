# AEGIS-Attention V4: Adaptive Enhanced Global-Local Integrated Sparse Attention with Energy-Preserving Factorization

## 1. Abstract & Theoretical Motivation

Standard multi-head self-attention (MHSA) in Transformer architectures exhibits quadratic computational scaling $\mathcal{O}(N^2)$ and memory allocation growth, severely hindering long-context language modeling and sequence processing. While sparse localized sliding windows restrict attention computation to $\mathcal{O}(N \cdot W_{\text{local}})$, they restrict global information propagation. Conversely, global kernelized linear attention achieves linear complexity $\mathcal{O}(N \cdot d_h^2)$ but suffers from spatial over-smoothing and loss of fine-grained local syntactic structures.

**AEGIS-Attention V4** resolves this fundamental trade-off through a decoupled multi-branch architecture incorporating four core principles:
1. **Energy-Preserving Dynamic SVD Factorization**: Linear projection matrices ($W_Q, W_K, W_V, W_O$) are decomposed via Singular Value Decomposition (SVD). The truncation rank $r$ is dynamically selected per weight tensor to preserve a predefined cumulative singular value energy threshold $\tau_{\text{energy}} = 0.95$.
2. **Receptive Field Split-Head Partitioning**: Multi-head attention is explicitly partitioned into $H_l$ local heads focused on local spatial receptive fields and $H_g$ global heads handling sequence-wide context aggregation ($H = H_l + H_g$).
3. **Softpick Rectified Local Windowing**: Local heads employ a tail-thresholded non-linear probability mapping (`softpick`), which active-prunes background noise by zeroing low-probability attention tails.
4. **Random Fourier Feature (RFF) Global Linear Kernel**: Global heads project L2-normalized, zero-mean centered query and key representations into an $F$-dimensional positive RFF space, executing global memory accumulation in linear $\mathcal{O}(N \cdot F \cdot d_h)$ time.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Energy-Preserving Dynamic SVD Linear Projection (`FactorisedLinear`)
For any projection layer weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the full SVD is evaluated:

$$W = U \Sigma V^T, \quad \Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min(d_{\text{out}}, d_{\text{in}})})$$

The optimal low-rank projection rank $r$ is determined by the minimum index satisfying cumulative energy ratio $\tau_{\text{energy}} \in (0, 1]$:

$$\frac{\sum_{k=1}^r \sigma_k^2}{\sum_{k=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_k^2} \ge \tau_{\text{energy}}, \quad \text{subject to } r \in [r_{\min}, r_{\max}]$$

The weight matrix is factorized into $W_{\text{down}} \in \mathbb{R}^{r \times d_{\text{in}}}$ and $W_{\text{up}} \in \mathbb{R}^{d_{\text{out}} \times r}$:

$$W_{\text{down}} = V_{:, :r}^T, \quad W_{\text{up}} = U_{:, :r} \Sigma_{:r, :r}$$

For activation tensor $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$, forward transformation proceeds via dual linear evaluations:

$$Y = (X W_{\text{down}}^T) W_{\text{up}}^T + b \in \mathbb{R}^{B \times N \times d_{\text{out}}}$$

### 2.2 Head Partitioning into Local and Global Branches
The projected representations $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ are split along the head dimension into local heads ($H_l$) and global heads ($H_g$):

$$Q_{\text{local}}, K_{\text{local}}, V_{\text{local}} \in \mathbb{R}^{B \times H_l \times N \times d_h}, \quad Q_{\text{global}}, K_{\text{global}}, V_{\text{global}} \in \mathbb{R}^{B \times H_g \times N \times d_h}$$

### 2.3 Local Branch with Softpick Rectified Activation
Local heads compute scaled dot-product attention logits within symmetric local sliding window radius $W_{\text{local}}$:

$$S_{b, h, i, j} = \frac{Q_{\text{local}, b, h, i} \cdot K_{\text{local}, b, h, j}^T}{\sqrt{d_h}} + M^{\text{band}}_{i, j} + M^{\text{pad}}_{b, j}$$

where the spatial window mask $M^{\text{band}}$ enforces $|i - j| \le \lfloor W_{\text{local}} / 2 \rfloor$. The **Softpick** probability distribution is calculated with temperature scaling $T$:

$$m_i = \max_j S_{i, j}, \quad e_{i, j} = \exp\left( \frac{S_{i, j} - m_i}{T} \right)$$

$$p_{i, j} = \frac{\max\left(0, e_{i, j} - \exp(-m_i / T)\right)}{\sum_{k} \max\left(0, e_{i, k} - \exp(-m_i / T)\right) + \epsilon}$$

$$O^{\text{local}}_{b, h, i} = \sum_{j=1}^N p_{b, h, i, j} V_{\text{local}, b, h, j}$$

### 2.4 Global Branch with Zero-Sum RFF Linear Kernel
Global heads compute zero-mean centered keys $\tilde{K} = K_{\text{global}} - \bar{K}_{\text{global}}$ and apply L2-normalization:

$$\hat{Q} = \frac{Q_{\text{global}}}{\|Q_{\text{global}}\|_2}, \quad \hat{K} = \frac{\tilde{K}}{\|\tilde{K}\|_2}$$

Using positive feature mapping with random projection parameters $W_{\text{rff}} \in \mathbb{R}^{H_g \times d_h \times F}$:

$$\phi_Q(Q) = \frac{\exp\left( \hat{Q} W_{\text{rff}} \right)}{\sqrt{F}}, \quad \phi_K(K) = \frac{\exp\left( \hat{K} W_{\text{rff}} \right)}{\sqrt{F}}$$

The recurrent global memory accumulation matrix $KV \in \mathbb{R}^{B \times H_g \times F \times d_h}$ and normalizer $K_{\text{sum}} \in \mathbb{R}^{B \times H_g \times F}$ aggregate token representations:

$$KV_{b, h} = \sum_{j=1}^N \phi_K(K)_{b, h, j}^T \otimes V_{\text{global}, b, h, j}, \quad K_{\text{sum}, b, h} = \sum_{j=1}^N \phi_K(K)_{b, h, j}^T$$

The global context output is retrieved via linear query contraction:

$$O^{\text{global}}_{b, h, i} = \frac{\phi_Q(Q)_{b, h, i} KV_{b, h}}{\phi_Q(Q)_{b, h, i} K_{\text{sum}, b, h}^T + \epsilon}$$

### 2.5 Multi-Head Gating & Adaptive Fusion
For sequence lengths $N \ge W_{\text{local}}$, local and global branch outputs are combined via head-wise sigmoidal gating:

$$g_h = \sigma\left( \text{logits}_h + \text{bias}_{\text{gate}} \right)$$

$$O^{\text{fused}} = g \odot O^{\text{local\_padded}} + (1 - g) \odot O^{\text{global\_broadcast}} \in \mathbb{R}^{B \times H \times N \times d_h}$$

The fused multi-head representation is reshaped and passed through the SVD factorized output layer:

$$Y = \text{FactorisedLinear}_{\text{out}}\left( \text{Reshape}(O^{\text{fused}}) \right) \in \mathbb{R}^{B \times N \times D}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input Hidden State** | $X$ | $[B, N, D]$ | Base activation state |
| **SVD Projection Q/K/V** | $X W_{\text{down}}^T W_{\text{up}}^T$ | $[B, N, D] \to [B, N, r] \to [B, N, H \cdot d_h]$ | SVD dual-linear matrix transformation |
| **Head Decomposition** | $Q, K, V$ | $[B, N, H \cdot d_h] \to [B, H, N, d_h]$ | Multi-head view transposition |
| **Receptive Field Split** | $Q_l, Q_g$ | $[B, H_l, N, d_h], [B, H_g, N, d_h]$ | Head partition into local ($H_l$) & global ($H_g$) |
| **Local Softpick Attention**| $O^{\text{local}}$ | $[B, H_l, N, d_h] \times [B, H_l, N, d_h] \to [B, H_l, N, d_h]$ | Band-masked Softpick score contraction |
| **RFF Kernel Mapping** | $\phi_Q, \phi_K$ | $[B, H_g, N, d_h] \times [H_g, d_h, F] \to [B, H_g, N, F]$ | Positive exponential feature mapping |
| **Global Memory Matrix** | $KV$ | $[B, H_g, N, F]^T \times [B, H_g, N, d_h] \to [B, H_g, F, d_h]$| Sequence-wide key-value outer product |
| **Global Querying** | $O^{\text{global}}$ | $[B, H_g, N, F] \times [B, H_g, F, d_h] \to [B, H_g, N, d_h]$ | Linear query context retrieval |
| **Adaptive Head Gating** | $g$ | $[H] \to [1, H, 1, 1]$ | Head-wise sigmoidal blending gate |
| **Fused Head Concatenation**| $O^{\text{fused}}$ | $[B, H, N, d_h]$ | Convex head combination |
| **Output Low-Rank GEMM** | $Y$ | $[B, N, H \cdot d_h] \to [B, N, r] \to [B, N, D]$ | SVD factorized output linear projection |

---

## 4. Analytical Complexity Profiling

### 4.1 Floating Point Operations (FLOPs)
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ total heads, $d_h = D/H$ head dimension, $r$ average SVD rank, $W$ local window radius, and $F$ RFF feature dimension ($F = \text{rff\_features}$).

1. **SVD Linear Projections ($Q, K, V, O$)**:
   $$\text{FLOPs}_{\text{proj}} = 4 \times \left( 2 B N D r + 2 B N r D \right) = 16 B N D r$$
2. **Local Softpick Attention ($H_l$ heads)**:
   - Band score matmul: $2 B H_l N W d_h$
   - Softpick rectification & Value contraction: $2 B H_l N W d_h + \mathcal{O}(B H_l N W)$
   $$\text{FLOPs}_{\text{local}} = 4 B H_l N W d_h$$
3. **Global RFF Linear Attention ($H_g$ heads)**:
   - RFF Einstein sum projections ($\hat{Q} W_{\text{rff}}, \hat{K} W_{\text{rff}}$): $4 B H_g N d_h F$
   - Memory State accumulation ($KV$): $2 B H_g N F d_h$
   - Global Context retrieval: $2 B H_g N F d_h$
   $$\text{FLOPs}_{\text{global}} = 8 B H_g N d_h F$$

**Total FLOP Count**:

$$\text{FLOPs}_{\text{total}} = 16 B N D r + 4 B H_l N W d_h + 8 B H_g N d_h F$$

### 4.2 Memory Footprint & Bandwidth
- **Parameter Footprint**:
  $$\Theta = 4 \times (D \cdot r + r \cdot D) + (H_g \cdot d_h \cdot F) + H = 8 D r + H_g d_h F + H$$
- **Activation Footprint**:
  - Local Softpick logits: $\mathcal{O}(B H_l N W)$
  - RFF feature activations: $\mathcal{O}(B H_g N F)$
  - Global memory matrix state: $\mathcal{O}(B H_g F d_h)$
  - Total Memory: $\mathcal{O}(B N D + B H_l N W + B H_g N F + B H_g F d_h)$

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Energy Truncation Loss in Pretrained Linear Projections
- **Mechanism**: The energy threshold parameter $\tau_{\text{energy}} = 0.95$ dictates singular rank truncation. For pre-trained models exhibiting slow singular value decay (high spectral entropy), capping rank to $r_{\max} = 64$ discards more than $5\%$ of total Frobenius energy.
- **Consequence**: Severe drop in zero-shot task performance prior to fine-tuning.

### 5.2 RFF Feature Map Numerical Instability
- **Mechanism**: RFF kernel evaluation computes $\exp(\hat{Q} W_{\text{rff}})$. If input normalization fails or $W_{\text{rff}}$ scale std dev exceeds $0.05$, exponentiation under FP16 precision causes numerical overflow (`inf`), triggering NaN loss cascades.
- **Mitigation**: Enforce strict L2 normalization $\hat{Q} = Q / \|Q\|_2$ and initialize $W_{\text{rff}}$ with standard deviation $\sigma = 0.02$.

### 5.3 Softpick Temperature Gradient Lock
- **Mechanism**: Setting Softpick temperature $T < 0.2$ results in aggressive tail subtraction, zeroing all off-diagonal probabilities. The resulting gradient $\frac{\partial p_{i, j}}{\partial S_{i, j}}$ collapses to zero everywhere except at the maximum index, locking local heads in a static identity routing pattern.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

```
+-----------------------------------------------------------------------------------+
|                        Triton Hardware Optimization Blueprint                     |
|                                                                                   |
|  Input X [B, N, D] ---> [Fused Energy-SVD GEMM Kernel] ---> Q, K, V               |
|                                                                                   |
|      +-------------------------------------+-----------------------------------+  |
|      |                                     |                                   |  |
|      v (H_l Local Heads)                   v (H_g Global Heads)                |  |
| [Triton Fused Softpick Kernel]      [Triton Streaming RFF Kernel]              |  |
|  - Block-Sparse Tiling (64x64)       - Fused L2-Norm + Exp RFF Projection       |  |
|  - SRAM Maximum & Thresholding       - Shared Memory Atomic Reduction (F, d_h)   |  |
|  - Zero DRAM Write for Logits        - Linear Context Query Contraction          |  |
|                                                                                   |
|      +-------------------------------------+-----------------------------------+  |
|      |                                                                            |
|      +---> Fused Gated Blend & Output SVD GEMM Kernel ---> Output Y [B, N, D]     |
+-----------------------------------------------------------------------------------+
```

### 6.1 Triton Strategy 1: Fused Energy-SVD GEMM Execution
Instead of executing standard two-stage PyTorch matrix multiplications (`x @ W_down.T` followed by `@ W_up.T`), implement a custom fused Triton kernel:
- Load tile $X_i \in \mathbb{R}^{\text{BLOCK\_M} \times D}$ directly into SRAM.
- Compute down-projection in SRAM registers, perform singular scaling, and multiply by $W_{\text{up}}^T$ before writing back to global memory.
- Reduces memory traffic by avoiding intermediate rank-$r$ tensor HBM allocation.

### 6.2 Triton Strategy 2: Block-Sparse Softpick Window Kernel
- Process sliding window attention using SRAM blocks of size `64 x 64`.
- Execute online maximum tracking $m_i$, exponential subtraction, thresholding, and value reduction within GPU register files.
- Eliminates global memory allocation for the $[B, H_l, N, W]$ Softpick probability tensor.

### 6.3 Triton Strategy 3: Fused RFF Memory Accumulation Grid
- Stream sequence tiles of size `BLOCK_N = 128` across global heads $H_g$.
- Compute L2 normalization, RFF projection, and exponential activation in shared memory, accumulating outer products into $KV \in \mathbb{R}^{F \times d_h}$.
- Avoids storing full sequence $[B, H_g, N, F]$ feature maps in DRAM.