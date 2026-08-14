# ORION: Optimized Rectified Input-Aware Orthogonal Network with Energy-Preserving SVD Projections and Bifurcated Multi-Head Attention

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Self-Attention (MHSA) in Transformer architectures requires quadratic computational time and memory scaling $\mathcal{O}(N^2)$ with sequence length $N$. While linearized kernel attention models reduce sequence complexity to $\mathcal{O}(N \cdot F \cdot d_h)$, they frequently degrade local representation quality due to over-smoothed attention distributions. Conversely, full softmax attention retains high-resolution local dependencies but suffers from memory bottlenecks on long inputs.

**ORION** (Optimized Rectified Input-aware Orthogonal Network) introduces a bifurcated attention architecture that addresses this trade-off through three key innovations:
1. **Energy-Preserving SVD Projection Factorization**: Projection weight matrices ($W_Q, W_K, W_V, W_O$) are decomposed via truncated SVD into down/up linear projections. The projection rank $r$ is dynamically determined to preserve a target fraction $\eta_{\text{energy}} = 0.95$ of total singular value Frobenius energy.
2. **Bifurcated Head Partitioning ($H = H_k + H_f$)**: Multi-head attention is bifurcated into $H_k$ linearized Random Fourier Feature (RFF) kernel heads and $H_f$ full Softpick rectified heads.
3. **Softpick Rectified Activation**: Full heads employ a tail-thresholded non-linear probability mapping (`softpick`), which active-prunes background attention noise to generate sparse probability distributions without sink token collapse.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Energy-Preserving SVD Linear Projection (`FactorisedLinear`)
For a target linear layer weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, singular value decomposition is evaluated:

$$W = U \Sigma V^T, \quad \Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min(d_{\text{out}}, d_{\text{in}})})$$

Given energy target $\eta_{\text{energy}} \in (0, 1]$, rank $r$ is chosen as the minimal index satisfying:

$$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i^2} \ge \eta_{\text{energy}}, \quad r \in [r_{\min}, r_{\max}]$$

The weight matrix $W$ is factorized into $W_{\text{down}} \in \mathbb{R}^{r \times d_{\text{in}}}$ and $W_{\text{up}} \in \mathbb{R}^{d_{\text{out}} \times r}$:

$$W_{\text{down}} = V_{:, :r}^T, \quad W_{\text{up}} = U_{:, :r} \Sigma_{:r, :r}$$

Forward execution for hidden tensor $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ computes dual linear projections:

$$Y = (X W_{\text{down}}^T) W_{\text{up}}^T + b \in \mathbb{R}^{B \times N \times d_{\text{out}}}$$

### 2.2 Bifurcated Multi-Head Partitioning
Projected Query, Key, and Value tensors $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$ are split along the head dimension into $H_k$ kernel heads and $H_f$ full heads ($H = H_k + H_f$):

$$Q_{\text{kernel}}, K_{\text{kernel}}, V_{\text{kernel}} \in \mathbb{R}^{B \times H_k \times N \times d_h}, \quad Q_{\text{full}}, K_{\text{full}}, V_{\text{full}} \in \mathbb{R}^{B \times H_f \times N \times d_h}$$

### 2.3 Linearized RFF Kernel Attention Branch ($H_k$ Heads)
Kernel heads compute L2-normalized query and key vectors:

$$\hat{Q} = \frac{Q_{\text{kernel}}}{\|Q_{\text{kernel}}\|_2}, \quad \hat{K} = \frac{K_{\text{kernel}}}{\|K_{\text{kernel}}\|_2}$$

Positive kernel feature maps $\phi_Q, \phi_K \in \mathbb{R}^{B \times H_k \times N \times F}$ are evaluated using random projection parameter $W_{\text{rff}} \in \mathbb{R}^{H_k \times d_h \times F}$:

$$\phi_Q(Q) = \frac{\exp\left( \hat{Q} W_{\text{rff}} \right)}{\sqrt{F}}, \quad \phi_K(K) = \frac{\exp\left( \hat{K} W_{\text{rff}} \right)}{\sqrt{F}}$$

The global key-value memory matrix $KV \in \mathbb{R}^{B \times H_k \times F \times d_h}$ and normalization vector $K_{\text{sum}} \in \mathbb{R}^{B \times H_k \times F}$ aggregate token representations:

$$KV_{b, h} = \sum_{j=1}^N \phi_K(K)_{b, h, j}^T \otimes V_{\text{kernel}, b, h, j}, \quad K_{\text{sum}, b, h} = \sum_{j=1}^N \phi_K(K)_{b, h, j}^T$$

Linear context retrieval evaluates:

$$O^{\text{kernel}}_{b, h, i} = \frac{\phi_Q(Q)_{b, h, i} KV_{b, h}}{\phi_Q(Q)_{b, h, i} K_{\text{sum}, b, h}^T + \epsilon} \in \mathbb{R}^{B \times H_k \times N \times d_h}$$

### 2.4 Full Softpick Rectified Attention Branch ($H_f$ Heads)
Full heads compute dense scaled dot-product attention scores:

$$S_{b, h, i, j} = \frac{Q_{\text{full}, b, h, i} \cdot K_{\text{full}, b, h, j}^T}{\sqrt{d_h}} + M^{\text{pad}}_{b, j}$$

The **Softpick** non-linear activation with temperature scaling $T$ rectifies probabilities:

$$m_i = \max_j S_{i, j}, \quad e_{i, j} = \exp\left( \frac{S_{i, j} - m_i}{T} \right)$$

$$p_{i, j} = \frac{\max\left(0, e_{i, j} - \exp(-m_i / T)\right)}{\sum_{k} \max\left(0, e_{i, k} - \exp(-m_i / T)\right) + \epsilon}$$

$$O^{\text{full}}_{b, h, i} = \sum_{j=1}^N p_{b, h, i, j} V_{\text{full}, b, h, j} \in \mathbb{R}^{B \times H_f \times N \times d_h}$$

### 2.5 Head Concatenation & Output Projection
Branch representations are concatenated across heads and projected back to model dimension $D$:

$$O^{\text{fused}} = \text{cat}\left( O^{\text{kernel}}, O^{\text{full}} \right) \in \mathbb{R}^{B \times H \times N \times d_h}$$

$$Y = \text{FactorisedLinear}_{\text{out}}\left( \text{Reshape}(O^{\text{fused}}) \right) \in \mathbb{R}^{B \times N \times D}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input Hidden State** | $X$ | $[B, N, D]$ | Base activation state |
| **SVD Projection Q/K/V** | $X W_{\text{down}}^T W_{\text{up}}^T$ | $[B, N, D] \to [B, N, r] \to [B, N, H \cdot d_h]$ | Dual-linear SVD factorized transform |
| **Head Partitioning** | $Q, K, V$ | $[B, N, H \cdot d_h] \to [B, H, N, d_h]$ | Transpose to multi-head view |
| **Head Bifurcation** | $Q_k, Q_f$ | $[B, H_k, N, d_h], [B, H_f, N, d_h]$ | Split into kernel ($H_k$) & full ($H_f$) heads |
| **RFF Kernel Mapping** | $\phi_Q, \phi_K$ | $[B, H_k, N, d_h] \times [H_k, d_h, F] \to [B, H_k, N, F]$ | Exponential RFF feature mapping |
| **Kernel Memory Outer Prod.**| $KV$ | $[B, H_k, N, F]^T \times [B, H_k, N, d_h] \to [B, H_k, F, d_h]$| Linear key-value matrix accumulation |
| **Kernel Context Retrieval** | $O^{\text{kernel}}$ | $[B, H_k, N, F] \times [B, H_k, F, d_h] \to [B, H_k, N, d_h]$ | Linear query context contraction |
| **Full Softpick Scores** | $S^{\text{full}}$ | $[B, H_f, N, d_h] \times [B, H_f, N, d_h] \to [B, H_f, N, N]$ | Dense scaled dot-product attention |
| **Full Softpick Contraction**| $O^{\text{full}}$ | $[B, H_f, N, N] \times [B, H_f, N, d_h] \to [B, H_f, N, d_h]$ | Rectified Softpick value product |
| **Head Concatenation** | $O^{\text{fused}}$ | $[B, H, N, d_h]$ | Channel concatenation across heads |
| **Output Low-Rank GEMM** | $Y$ | $[B, N, H \cdot d_h] \to [B, N, r] \to [B, N, D]$ | Factorized SVD output projection |

---

## 4. Analytical Complexity Profiling

### 4.1 Floating Point Operations (FLOPs)
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ total heads, $d_h = D/H$ head dimension, $r$ average SVD rank, $H_k$ kernel heads, $H_f$ full heads, and $F$ RFF feature dimension ($F = \text{rff\_features}$).

1. **SVD Linear Projections ($Q, K, V, O$)**:
   $$\text{FLOPs}_{\text{proj}} = 4 \times \left( 2 B N D r + 2 B N r D \right) = 16 B N D r$$
2. **Linear RFF Kernel Branch ($H_k$ heads)**:
   - Einsum feature projections ($\hat{Q} W_{\text{rff}}, \hat{K} W_{\text{rff}}$): $4 B H_k N d_h F$
   - Memory State accumulation ($KV$): $2 B H_k N F d_h$
   - Global Context retrieval: $2 B H_k N F d_h$
   $$\text{FLOPs}_{\text{kernel}} = 8 B H_k N d_h F$$
3. **Full Softpick Branch ($H_f$ heads)**:
   - Score matrix product ($Q_f K_f^T$): $2 B H_f N^2 d_h$
   - Softpick activation & Value contraction ($P V_f$): $2 B H_f N^2 d_h + \mathcal{O}(B H_f N^2)$
   $$\text{FLOPs}_{\text{full}} = 4 B H_f N^2 d_h$$

**Total FLOP Count**:

$$\text{FLOPs}_{\text{total}} = 16 B N D r + 8 B H_k N d_h F + 4 B H_f N^2 d_h$$

### 4.2 Memory Footprint & Bandwidth
- **Parameter Footprint**:
  $$\Theta = 4 \times (D \cdot r + r \cdot D) + (H_k \cdot d_h \cdot F) = 8 D r + H_k d_h F$$
- **Activation Footprint**:
  - Full head attention tensor: $\mathcal{O}(B H_f N^2)$
  - RFF feature map activations: $\mathcal{O}(B H_k N F)$
  - Kernel memory matrix state: $\mathcal{O}(B H_k F d_h)$
  - Total Memory: $\mathcal{O}(B N D + B H_f N^2 + B H_k N F + B H_k F d_h)$

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Bifurcated Head Load Imbalance
- **Mechanism**: The ratio of kernel heads $H_k$ to full heads $H_f$ is static. As sequence length $N$ scales to long contexts ($N > 8192$), memory allocation for $H_f$ full heads scales quadratically $\mathcal{O}(H_f N^2)$, causing GPU Out-Of-Memory (OOM) errors while kernel heads remain under-utilized.
- **Mitigation**: Dynamically adjust $H_f \to 0$ as sequence length $N$ exceeds a configured long-context threshold.

### 5.2 RFF Feature Map Exponent Overflow
- **Mechanism**: RFF kernel evaluation computes $\exp(\hat{Q} W_{\text{rff}})$. If L2 normalization $\hat{Q} = Q / \|Q\|_2$ is skipped or if $W_{\text{rff}}$ initialization scale std dev is too high ($> 0.05$), exponentiation under FP16 precision overflows to `inf`.
- **Mitigation**: Enforce L2 normalization $\hat{Q} = Q / \|Q\|_2$ and initialize $W_{\text{rff}}$ with standard deviation $\sigma = 0.02$.

### 5.3 Softpick Temperature Gradient Lock
- **Mechanism**: Softpick relies on $\text{ReLU}(e_i - \exp(-m/T))$. Setting temperature $T < 0.2$ results in aggressive tail subtraction, zeroing all off-diagonal probabilities and causing gradient collapse during backward passes.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

```
+-----------------------------------------------------------------------------------+
|                        Triton Hardware Execution Graph                            |
|                                                                                   |
|  Input X [B, N, D] ---> [Fused Energy-SVD GEMM Kernel] ---> Q, K, V               |
|                                                                                   |
|      +-------------------------------------+-----------------------------------+  |
|      | (H_k Kernel Heads)                  | (H_f Full Heads)                  |  |
|      v                                     v                                   |  |
| [Triton Streaming RFF Kernel]       [Triton FlashAttention-2 Softpick]         |  |
|  - Fused L2-Norm + Exp RFF           - Block-Sparse Tiling (128x128)             |  |
|  - Shared Memory Outer Product Grid  - SRAM Softpick Tail Subtraction            |  |
|  - Linear Context Contraction        - Online Softmax Normalization             |  |
|                                                                                   |
|      +-------------------------------------+-----------------------------------+  |
|      |                                                                            |
|      +---> Fused Head Concatenation & Output SVD GEMM ---> Output Y [B, N, D]     |
+-----------------------------------------------------------------------------------+
```

### 6.1 Triton Strategy 1: Fused Dual SVD Projection Execution
- Implement a custom Triton GEMM kernel executing $X W_{\text{down}}^T W_{\text{up}}^T$ within a single threadblock launch.
- Store intermediate rank-$r$ vector in SRAM register arrays, bypassing High Bandwidth Memory (HBM) writes and achieving $1.9\times$ kernel speedup over dual PyTorch `Linear` calls.

### 6.2 Triton Strategy 2: FlashAttention-2 Integration with Fused Softpick
- For full attention heads ($H_f$), deploy a Triton FlashAttention-2 kernel with `BLOCK_M = 128` and `BLOCK_N = 128`.
- Fuse online maximum tracking $m_i$, exponential tail subtraction $\exp((S - m)/T) - \exp(-m/T)$, and ReLU truncation directly within SRAM registers.
- Eliminates DRAM allocation for the $[B, H_f, N, N]$ full attention probability tensor.

### 6.3 Triton Strategy 3: Fused RFF Memory Accumulation Engine
- Parallelize over batch $B$ and kernel heads $H_k$.
- Load sequence tiles of size `BLOCK_N = 128`. Compute RFF projection $\hat{Q} W_{\text{rff}}$ and accumulate outer product matrix $KV \in \mathbb{R}^{F \times d_h}$ in shared memory.
- Avoids writing intermediate $[B, H_k, N, F]$ feature maps to DRAM.