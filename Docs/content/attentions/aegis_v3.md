# AEGIS-Attention V3: Energy-Preserving SVD Projections with Softpick Local Windowing and Random Fourier Feature Global Kernel Attention

## 1. Executive Abstract & Architectural Motivation
Standard multi-head self-attention (MHSA) architectures suffer from quadratic time and memory complexity $\mathcal{O}(N^2)$ with sequence length $N$. Existing linear attention variants replace quadratic attention with non-negative kernel feature maps; however, they frequently exhibit degraded local representation accuracy due to spatial feature blurring and over-smoothed attention distributions.

**AEGIS-Attention V3** addresses these limitations through a multi-branch split-head attention architecture featuring four key innovations:
1. **Energy-Preserving Dynamic SVD Factorization**: Linear projection matrices ($W_Q, W_K, W_V, W_O$) are decomposed via Singular Value Decomposition (SVD), with projection rank $r$ dynamically determined to retain a target cumulative Frobenius energy threshold $\tau$ (e.g., $\tau = 0.95$).
2. **Split-Head Local-Global Receptive Fields**: The $H$ attention heads are split into $H_l$ local heads dedicated to fine-grained spatial modeling within window $W$, and $H_g$ global heads handling long-range dependencies.
3. **Softpick Local Activation Mechanism**: Replaces standard softmax with a thresholded tail-subtraction non-linearity (`softpick`), which active-prunes low-probability attention tails to prevent soft background noise accumulation.
4. **Random Fourier Feature (RFF) Global Linear Kernel**: Global heads map normalized key-query vectors into an $F$-dimensional RFF space with zero-sum key mean-centering, executing global context accumulation in linear $\mathcal{O}(N \cdot F \cdot d_h)$ time.

---

## 2. Mathematical Formulations & Derivations

### 2.1 Energy-Preserving Dynamic SVD Linear Layer (`FactorisedLinear`)
For a target linear layer weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the singular value decomposition is computed:

$$W = U \Sigma V^T, \quad \Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min(d_{\text{out}}, d_{\text{in}})})$$

Given energy threshold $\tau \in (0, 1]$, rank $r$ is dynamically selected as the minimal index satisfying:

$$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i^2} \ge \tau, \quad \text{subject to } r \in [r_{\min}, r_{\max}]$$

The original weight matrix $W$ is factorized into down- and up-projections $W_{\text{down}} \in \mathbb{R}^{r \times d_{\text{in}}}$ and $W_{\text{up}} \in \mathbb{R}^{d_{\text{out}} \times r}$:

$$W_{\text{down}} = V_{:, :r}^T, \quad W_{\text{up}} = U_{:, :r} \Sigma_{:r, :r}$$

Forward execution for hidden state $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ is expressed as:

$$Y = (X W_{\text{down}}^T) W_{\text{up}}^T + b$$

### 2.2 Receptive Field Head Partitioning
The output of $Q, K, V$ projections is partitioned along head dimensions into $H_l$ local heads and $H_g$ global heads ($H = H_l + H_g$):

$$\begin{aligned}
Q_{\text{all}}, K_{\text{all}}, V_{\text{all}} &\in \mathbb{R}^{B \times H \times N \times d_h} \\
Q_{\text{local}}, K_{\text{local}}, V_{\text{local}} &\in \mathbb{R}^{B \times H_l \times N \times d_h} \\
Q_{\text{global}}, K_{\text{global}}, V_{\text{global}} &\in \mathbb{R}^{B \times H_g \times N \times d_h}
\end{aligned}$$

### 2.3 Local Windowed Branch with Softpick Activation
Local heads evaluate scaled dot-product scores within symmetric local window radius $W_{\text{local}}$:

$$S_{b, h, i, j} = \frac{Q_{\text{local}, b, h, i} \cdot K_{\text{local}, b, h, j}^T}{\sqrt{d_h}} + M^{\text{band}}_{i, j} + M^{\text{pad}}_{b, j}$$

Standard softmax is replaced by the **Softpick** operator with temperature scaling $T$:

$$m_i = \max_j S_{i, j}, \quad e_{i, j} = \exp\left( \frac{S_{i, j} - m_i}{T} \right)$$

$$p_{i, j} = \frac{\max\left(0, e_{i, j} - \exp(-m_i / T)\right)}{\sum_{k} \max\left(0, e_{i, k} - \exp(-m_i / T)\right) + \epsilon}$$

$$O^{\text{local}}_{b, h, i} = \sum_{j=1}^N p_{b, h, i, j} V_{\text{local}, b, h, j}$$

### 2.4 Global Random Fourier Feature (RFF) Linear Kernel Branch
Global heads apply L2-normalization and zero-sum mean-centering to key representations:

$$\hat{Q} = \frac{Q_{\text{global}}}{\|Q_{\text{global}}\|_2}, \quad \hat{K} = \frac{K_{\text{global}} - \bar{K}_{\text{global}}}{\|K_{\text{global}} - \bar{K}_{\text{global}}\|_2}$$

Using static RFF weight parameter matrix $W_{\text{rff}} \in \mathbb{R}^{H_g \times d_h \times F}$, positive kernel feature maps $\phi_Q, \phi_K \in \mathbb{R}^{B \times H_g \times N \times F}$ are evaluated via exponential projection:

$$\phi_Q(Q) = \frac{\exp\left( \hat{Q} W_{\text{rff}} \right)}{\sqrt{F}}, \quad \phi_K(K) = \frac{\exp\left( \hat{K} W_{\text{rff}} \right)}{\sqrt{F}}$$

Linear context accumulation matrix $KV \in \mathbb{R}^{B \times H_g \times F \times d_h}$ and key sum accumulator $K_{\text{sum}} \in \mathbb{R}^{B \times H_g \times F}$ are calculated:

$$KV_{b, h} = \sum_{j=1}^N \phi_K(K)_{b, h, j}^T \otimes V_{\text{global}, b, h, j}, \quad K_{\text{sum}, b, h} = \sum_{j=1}^N \phi_K(K)_{b, h, j}^T$$

Global context recovery proceeds via query matrix contraction:

$$O^{\text{global}}_{b, h, i} = \frac{\phi_Q(Q)_{b, h, i} KV_{b, h}}{\phi_Q(Q)_{b, h, i} K_{\text{sum}, b, h}^T + \epsilon}$$

### 2.5 Adaptive Head Blending & Short-Sequence Fallback
If sequence length $N < W_{\text{local}}$, execution falls back to full dense scaled dot-product attention. Otherwise, local context $O^{\text{local}}$ and global context $O^{\text{global}}$ are fused via learned gate parameter $\beta \in \mathbb{R}^H$:

$$g_h = \sigma\left( \beta_h + \text{bias}_{\text{gate}} \right)$$

$$O^{\text{fused}} = g \odot O^{\text{local\_padded}} + (1 - g) \odot O^{\text{global\_broadcast}} \in \mathbb{R}^{B \times H \times N \times d_h}$$

Final projection maps fused heads back to $D$:

$$Y = \text{FactorisedLinear}_{\text{out}}\left( \text{Reshape}(O^{\text{fused}}) \right) \in \mathbb{R}^{B \times N \times D}$$

---

## 3. Step-by-Step Tensor Execution Trace

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input State** | $X$ | `[B, N, D]` | Hidden input tensor |
| **SVD Projection Q/K/V** | $X W_{\text{down}}^T W_{\text{up}}^T$ | `[B, N, D] -> [B, N, r] -> [B, N, H * d_h]` | Down/Up SVD Linear transform |
| **Head Partitioning** | $Q, K, V$ | `[B, N, H * d_h] -> [B, H, N, d_h]` | Transpose to multi-head view |
| **Branch Splitting** | $Q_l, Q_g$ | `[B, H_l, N, d_h]`, `[B, H_g, N, d_h]` | Split into local ($H_l$) & global ($H_g$) |
| **Local Softpick Attention**| $O^{\text{local}}$ | `[B, H_l, N, d_h] * [B, H_l, N, d_h] -> [B, H_l, N, d_h]` | Sliding-window Softpick contraction |
| **RFF Kernel Projection** | $\phi_Q, \phi_K$ | `[B, H_g, N, d_h] * [H_g, d_h, F] -> [B, H_g, N, F]` | Exponential RFF feature mapping |
| **Global Memory State** | $KV$ | `[B, H_g, N, F]^T * [B, H_g, N, d_h] -> [B, H_g, F, d_h]`| Outer-product memory state |
| **Global Context Querying** | $O^{\text{global}}$ | `[B, H_g, N, F] * [B, H_g, F, d_h] -> [B, H_g, N, d_h]` | Linear query context contraction |
| **Head Blending Gate** | $g$ | `[H] -> [1, H, 1, 1]` | Sigmoidal head blending gate |
| **Fused Head Assembly** | $O^{\text{fused}}$ | `[B, H, N, d_h]` | Gated combination of local/global |
| **Output Projection** | $Y$ | `[B, N, H * d_h] -> [B, N, r] -> [B, N, D]` | Output SVD factorized transformation |

```
                       [B, N, D] Input Tensor
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     SVD Q, K, V Projections     Check N < W_local
      [B, H, N, d_h]             (Fallback to Dense Matmul if True)
             │
     ┌───────┴───────┐
     ▼ (H_l heads)   ▼ (H_g heads)
Local Softpick    RFF Projection: phi(Q), phi(K) [B, H_g, N, F]
Window Attn              │
Shape: [B, H_l, N, d_h]  ▼
     │            Outer Product KV Memory: phi(K)^T @ V [B, H_g, F, d_h]
     │                   │
     │                   ▼
     │            Global Output: phi(Q) @ KV / (phi(Q) @ K_sum) [B, H_g, N, d_h]
     │                   │
     └───────┬───────────┘
             ▼
      Adaptive Head Blending Gate (sig(beta + bias))
             │
             ▼
      Output SVD Projection (FactorisedLinear)
             │
             ▼
                       [B, N, D] Output
```

---

## 4. Analytical Complexity & Hardware Profiling

### 4.1 FLOPs Computation Breakdown
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ total heads, $d_h = D/H$ head dimension, $r$ average SVD rank, $W$ local window radius, and $F$ RFF feature dimension ($F = \text{rff\_features}$).

1. **SVD Linear Projections ($Q, K, V, O$)**:
   $$\text{FLOPs}_{\text{proj}} = 4 \times \left( 2 B N D r + 2 B N r D \right) = 16 B N D r$$
2. **Local Softpick Branch ($H_l$ heads)**:
   - Band score matmul: $2 B H_l N W d_h$
   - Softpick non-linearity & Value contraction: $2 B H_l N W d_h + \mathcal{O}(B H_l N W)$
   $$\text{FLOPs}_{\text{local}} = 4 B H_l N W d_h$$
3. **Global RFF Linear Branch ($H_g$ heads)**:
   - RFF Einstein sum projections ($\hat{Q} W_{\text{rff}}, \hat{K} W_{\text{rff}}$): $2 \times (2 B H_g N d_h F) = 4 B H_g N d_h F$
   - Memory State accumulation ($KV$): $2 B H_g N F d_h$
   - Global Context retrieval: $2 B H_g N F d_h$
   $$\text{FLOPs}_{\text{global}} = 8 B H_g N d_h F$$

$$\text{FLOPs}_{\text{total}} = 16 B N D r + 4 B H_l N W d_h + 8 B H_g N d_h F$$

### 4.2 MACs & Memory Footprint Analysis
- **Multiply-Accumulate Operations (MACs)**:
  $$\text{Total MACs} = 8 B N D r + 2 B H_l N W d_h + 4 B H_g N d_h F$$
- **Parameter Count**:
  $$\Theta = 8 D r + H_g d_h F + H$$
- **Activation Footprint**:
  - Local Softpick attention tensor: $\mathcal{O}(B H_l N W)$
  - RFF feature activations: $\mathcal{O}(B H_g N F)$
  - Global memory matrix state: $\mathcal{O}(B H_g F d_h)$
  - Total Memory: $\mathcal{O}(B N D + B H_l N W + B H_g N F + B H_g F d_h)$

### 4.3 Asymptotic Scalability Comparison
- **Standard Dense Attention**: $\mathcal{O}(N^2 D)$ compute, $\mathcal{O}(N^2 H)$ memory.
- **AEGIS-Attention V3**: $\mathcal{O}\left(N (D r + H_l W d_h + H_g d_h F)\right)$ compute, $\mathcal{O}(N D + H_g F d_h)$ memory footprint.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

### 5.1 Dynamic Rank Under-Estimation & Energy Truncation Loss
- **Mechanism**: The energy threshold $\tau$ (e.g., $0.95$) determines rank $r$. For weight matrices with long singular tails, preserving $95\%$ energy may result in rank $r > r_{\max}$, triggering hard truncation to $r_{\max}$.
- **Consequence**: High-frequency spectral components of pre-trained weights are discarded, causing severe validation loss degradation during early fine-tuning steps.

### 5.2 Exponential RFF Feature Explosion & Overflow
- **Mechanism**: RFF kernel evaluation computes $\exp(\hat{Q} W_{\text{rff}})$. If the L2-normalization step is omitted or if $W_{\text{rff}}$ contains large standard deviation entries ($> 0.1$), the exponential function overflows to `inf` under IEEE FP16 training.
- **Mitigation**: Strictly enforce L2 normalization $\hat{Q} = Q / \|Q\|_2$ and initialize $W_{\text{rff}}$ with standard deviation $\sigma \le 0.02$.

### 5.3 Softpick Gradient Sparsification & Zero-Gradient Lock
- **Mechanism**: The Softpick operator subtracts the tail threshold $\exp(-m/T)$ and applies ReLU. If temperature $T$ is set too low ($T < 0.1$), all off-diagonal scores are zeroed ($p_{i, j} = 0$), trapping local attention in an identity-map state with zero backpropagation gradients.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

```
+-----------------------------------------------------------------------------------+
|                        Triton Hardware Optimization Blueprint                     |
|                                                                                   |
|  Input X [B, N, D] ---> [Fused Dynamic SVD GEMM Kernel] ---> Q, K, V              |
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

### 6.1 Triton Kernel Design: Fused Dynamic SVD GEMM
- Instantiate a custom Triton GEMM kernel that loads input $X \in \mathbb{R}^{\text{BLOCK\_M} \times D}$ into SRAM.
- Dynamically perform down-projection $X W_{\text{down}}^T$ into register space, then immediately multiply by $W_{\text{up}}^T$.
- Bypass intermediate HBM allocation for rank-$r$ activations, lowering DRAM traffic by $2\times$.

### 6.2 Triton Kernel Design: Fused Softpick Sliding-Window Kernel
Standard PyTorch creates an explicit $[B, H_l, N, W]$ logit tensor before evaluating `softpick`.

**Triton Optimization**:
- Execute block-sparse sliding window iteration using SRAM tiles of size `64 x 64`.
- Fuse online maximum tracking $m_i$, exponential evaluation $\exp((S - m)/T)$, threshold subtraction, and ReLU truncation directly inside SRAM registers.
- Write context output $O^{\text{local}}$ directly to global memory, maintaining $\mathcal{O}(1)$ activation memory overhead.

### 6.3 Triton Kernel Design: Streaming RFF Memory Reduction
- Parallelize over global heads $H_g$ and batch $B$.
- Load tokens in sequence tiles of `BLOCK_N = 128`. Compute normalized projection $\exp(\hat{K} W_{\text{rff}}) / \sqrt{F}$ and accumulate outer products into shared memory buffer $KV \in \mathbb{R}^{F \times d_h}$.
- Stream across sequence blocks in a single pass without storing full $[B, H_g, N, F]$ RFF feature tensors in DRAM.

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Cosine-Sine Dual Random Fourier Feature Mapping**: Replace pure exponential maps $\exp(\hat{X} W_{\text{rff}})$ with true shift-invariant Gaussian kernel approximators $\phi(X) = \frac{1}{\sqrt{F}} [\cos(X W_{\text{rff}}); \sin(X W_{\text{rff}})]$, resolving numerical exponentiation overflow.
2. **Adaptive Softpick Temperature Annealing**: Dynamically anneal softpick temperature $T_\ell$ across layer depth $\ell$, maintaining high entropy ($T = 2.0$) in initial layers for global search and low entropy ($T = 0.5$) in deep layers for precise local extraction.
3. **Structured Low-Rank Tensor Train (TT) Factorization**: Upgrade SVD matrix factorization to Tensor Train decomposition for 4D weight tensors, compressing projections down to rank $r = 8$.
4. **FP8 Quantized RFF State Accumulator**: Compute outer products $KV = \sum \phi_K(K)^T \otimes V$ using 8-bit FP8 (E4M3) operations on NVIDIA Hopper GPUs, doubling linear attention throughput.