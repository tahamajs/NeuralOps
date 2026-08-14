# LEKAv2: Linear-Time Efficient Kernel Attention with Landmark Distillation, Dual-Scale RFF, and Gated Local Sliding Window

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Self-Attention (MHSA) incurs quadratic time and memory complexity $\mathcal{O}(N^2)$ with sequence length $N$. While linearized kernel attention models reduce sequence complexity to $\mathcal{O}(N \cdot F \cdot d_h)$, they suffer from spatial over-smoothing and loss of non-local associative recall. Conversely, sparse landmark attention methods select key-value subsets but risk semantic representation collapse if landmarks are poorly sampled.

**LEKAv2** (Linear-time Efficient Kernel Attention v2) introduces a multi-scale hybrid attention mechanism combining four architectural components:
1. **Dual-Scale Random Fourier Feature (RFF) Mapping**: Maps query and key vectors using dual-frequency random projections ($\phi_1$ with feature dimension $F_1$, $\phi_2$ with $F_2 = F_1 / 2$), capturing both high-frequency local details and low-frequency global trends.
2. **MLP Landmark Distillation**: Dynamically scores key feature vectors using a light MLP network, extracting the top-$L_{\max}$ landmark tokens to construct a compact key-value summary.
3. **Value Correction Transformation**: Applies a learnable per-head matrix $W_{\text{corr}} \in \mathbb{R}^{H \times d_h \times d_h}$ to adjust landmark value representations.
4. **Gated Local Sliding Window Integration**: Interpolates between rectified Softpick landmark attention context ($C_{\text{leka}}$) and local sliding-window softmax attention context ($C_{\text{local}}$) via a learnable per-head sigmoidal gate $g_{\text{local}}$.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Dual-Scale Random Fourier Feature Mapping
For projected queries and keys $Q, K \in \mathbb{R}^{B \times H \times N \times d_h}$, dual random projection parameters $W_1 \in \mathbb{R}^{d_h \times F_1}$ and $W_2 \in \mathbb{R}^{d_h \times F_2}$ (where $F_2 = F_1 / 2$) evaluate multi-scale trigonometric features:

$$P_1 = x W_1 \implies \phi_1(x) = \frac{1}{\sqrt{F_1}} \left[ \cos(P_1), \; \sin(P_1) \right] \in \mathbb{R}^{B \times H \times N \times 2F_1}$$

$$P_2 = x W_2 \implies \phi_2(x) = \frac{1}{\sqrt{F_2}} \left[ \cos(P_2), \; \sin(P_2) \right] \in \mathbb{R}^{B \times H \times N \times 2F_2}$$

$$\phi(x) = \left[ \phi_1(x), \; \phi_2(x) \right] \in \mathbb{R}^{B \times H \times N \times d_{\text{feat}}}, \quad d_{\text{feat}} = 2F_1 + 2F_2$$

### 2.2 MLP-Based Landmark Selection & Value Correction
Key feature vectors $\phi(K) \in \mathbb{R}^{B \times H \times N \times d_{\text{feat}}}$ are scored by a two-layer MLP:

$$\text{Scores}_{\text{land}} = \text{Linear}_2\left( \text{GELU}\left( \text{Linear}_1( \phi(K) ) \right) \right) \in \mathbb{R}^{B \times H \times N}$$

Top-$L_{\max}$ landmark indices $\mathcal{I}_{\text{land}} \in \{1, \dots, N\}^{L_{\max}}$ are extracted:

$$\mathcal{I}_{\text{land}} = \text{TopK}\left( \text{Scores}_{\text{land}}, L_{\max}, \text{dim}=-1 \right)$$

Key features and value vectors are gathered at landmark indices:

$$K_{\text{land}} = \text{gather}\left( \phi(K), \mathcal{I}_{\text{land}} \right) \in \mathbb{R}^{B \times H \times L_{\max} \times d_{\text{feat}}}$$

$$V_{\text{land}} = \text{gather}\left( V, \mathcal{I}_{\text{land}} \right) \in \mathbb{R}^{B \times H \times L_{\max} \times d_h}$$

Landmark values are transformed by learnable correction matrix $W_{\text{corr}} \in \mathbb{R}^{H \times d_h \times d_h}$:

$$V_{\text{adj}} = V_{\text{land}} W_{\text{corr}} \in \mathbb{R}^{B \times H \times L_{\max} \times d_h}$$

### 2.3 Rectified Softpick Landmark Attention
Landmark score logits are computed by multiplying query features with landmark key features:

$$S_{\text{land}} = \phi(Q) K_{\text{land}}^T \in \mathbb{R}^{B \times H \times N \times L_{\max}}$$

**Softpick** rectified activation with temperature scaling $T$ yields sparse landmark attention weights:

$$m_i = \max_l S_{\text{land}, i, l}, \quad e_{i, l} = \exp\left( \frac{S_{\text{land}, i, l} - m_i}{T} \right)$$

$$p_{i, l} = \frac{\max\left(0, e_{i, l} - \exp(-m_i / T)\right)}{\sum_k \max\left(0, e_{i, k} - \exp(-m_i / T)\right) + \epsilon}$$

$$C_{\text{leka}} = p \cdot V_{\text{adj}} \in \mathbb{R}^{B \times H \times N \times d_h}$$

### 2.4 Gated Local Sliding Window Integration
Local sliding window attention computes scaled dot-product attention within local radius $W_{\text{local}}$:

$$S_{\text{local}, i, j} = \frac{Q_i K_j^T}{\sqrt{d_h}} + M^{\text{window}}_{i, j}, \quad M^{\text{window}}_{i, j} = \begin{cases} 0 & |i - j| \le \lfloor W_{\text{local}} / 2 \rfloor \\ -\infty & \text{otherwise} \end{cases}$$

$$C_{\text{local}} = \text{Softmax}\left( S_{\text{local}} \right) V \in \mathbb{R}^{B \times H \times N \times d_h}$$

Landmark and local contexts are interpolated via a learnable per-head gating parameter $g_{\text{local}} \in \mathbb{R}^H$:

$$\alpha = \sigma(g_{\text{local}}) \in [0, 1]^H$$

$$C = \alpha \odot C_{\text{leka}} + (1 - \alpha) \odot C_{\text{local}} \in \mathbb{R}^{B \times H \times N \times d_h}$$

The fused tensor is reshaped and projected through output linear layer $W_O$:

$$Y = \text{Reshape}(C) W_O + b_O \in \mathbb{R}^{B \times N \times D}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input State** | $X$ | $[B, N, D]$ | Base activation state |
| **Linear Projections Q/K/V** | $Q, K, V$ | $[B, N, D] \to [B, H, N, d_h]$ | Multi-head view split |
| **Dual-Scale RFF Mapping** | $\phi(Q), \phi(K)$ | $[B, H, N, d_h] \to [B, H, N, d_{\text{feat}}]$ | Dual projection $\cos/\sin$ concatenation |
| **Landmark MLP Scoring** | $\text{Scores}_{\text{land}}$ | $[B, H, N, d_{\text{feat}}] \to [B, H, N]$ | Two-layer MLP scoring network |
| **Top-L Landmark Gather** | $K_{\text{land}}, V_{\text{land}}$| $[B, H, N, \dots] \to [B, H, L_{\max}, \dots]$ | Top-$L_{\max}$ token index gather |
| **Value Correction Matrix** | $V_{\text{adj}}$ | $[B, H, L_{\max}, d_h] \times [H, d_h, d_h] \to [B, H, L_{\max}, d_h]$ | Tensor contraction $V_{\text{land}} W_{\text{corr}}$ |
| **Landmark Softpick Attn** | $C_{\text{leka}}$ | $[B, H, N, d_{\text{feat}}] \times [B, H, L_{\max}, d_{\text{feat}}]^T \to [B, H, N, d_h]$ | Rectified Softpick value contraction |
| **Local Sliding Window** | $C_{\text{local}}$ | $[B, H, N, d_h] \times [B, H, N, d_h]^T \to [B, H, N, d_h]$ | Band-masked softmax attention |
| **Sigmoidal Gate Fusion** | $C$ | $[B, H, N, d_h]$ | $\sigma(g) C_{\text{leka}} + (1 - \sigma(g)) C_{\text{local}}$ |
| **Output GEMM** | $Y$ | $[B, N, H \cdot d_h] \to [B, N, D]$ | Layer output projection |

---

## 4. Analytical Complexity Profiling

### 4.1 Floating Point Operations (FLOPs)
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ head count, $d_h = D/H$ head dimension, $F_1, F_2$ RFF feature dimensions ($d_{\text{feat}} = 2F_1 + 2F_2$), $L_{\max}$ landmark count, and $W_{\text{local}}$ local window width.

1. **Linear Q, K, V Projections**:
   $$\text{FLOPs}_{\text{proj}} = 3 \times 2 B N D^2 = 6 B N D^2$$
2. **Dual-Scale RFF Mapping**:
   - Dual projections ($x W_1, x W_2$): $2 B H N d_h (F_1 + F_2) = 2 B N D (F_1 + F_2)$
   $$\text{FLOPs}_{\text{rff}} = 2 B N D (F_1 + F_2)$$
3. **Landmark MLP Scoring & Gathering**:
   - MLP scorer: $2 B H N d_{\text{feat}} (F_1) + 2 B H N (F_1) (1) = 2 B H N d_{\text{feat}} F_1$
   $$\text{FLOPs}_{\text{score}} = 2 B H N d_{\text{feat}} F_1$$
4. **Value Correction & Landmark Softpick Attention**:
   - Value correction matrix product $V_{\text{land}} W_{\text{corr}}$: $2 B H L_{\max} d_h^2 = 2 B L_{\max} D d_h$
   - Landmark score product $\phi(Q) K_{\text{land}}^T$: $2 B H N L_{\max} d_{\text{feat}}$
   - Softpick & Value contraction $p V_{\text{adj}}$: $2 B H N L_{\max} d_h = 2 B N L_{\max} D$
   $$\text{FLOPs}_{\text{land}} = 2 B L_{\max} D d_h + 2 B H N L_{\max} d_{\text{feat}} + 2 B N L_{\max} D$$
5. **Local Sliding Window Attention**:
   $$\text{FLOPs}_{\text{local}} = 4 B N D W_{\text{local}}$$

**Total FLOP Count**:

$$\text{FLOPs}_{\text{total}} \approx 6 B N D^2 + 2 B H N L_{\max} d_{\text{feat}} + 2 B N L_{\max} D + 4 B N D W_{\text{local}}$$

### 4.2 Memory Footprint & Bandwidth
- **Parameter Footprint**:
  $$\Theta = 4 D^2 + (d_h F_1 + d_h F_2) + \Theta_{\text{scorer}} + H d_h^2 = 4 D^2 + d_h (F_1 + F_2) + d_{\text{feat}} F_1 + H d_h^2$$
- **Activation Footprint**:
  - Landmark feature buffers: $\mathcal{O}(B H L_{\max} d_{\text{feat}})$
  - Landmark attention matrix: $\mathcal{O}(B H N L_{\max})$
  - Local window logits: $\mathcal{O}(B H N W_{\text{local}})$
  - Total Memory: $\mathcal{O}\left(B N D + B H N L_{\max} + B H N W_{\text{local}}\right)$

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Landmark Collapse in Uniform Sequences
- **Mechanism**: The landmark MLP scorer evaluates $\text{MLP}(\phi(K))$. If key features exhibit low variance across the sequence (e.g., repeating tokens or uniform padding), the Top-k operator selects redundant landmark indices.
- **Consequence**: The landmark summary loses diversity, degrading global attention to a localized repeating context.

### 5.2 Value Correction Matrix Ill-Conditioning
- **Mechanism**: The value correction tensor $W_{\text{corr}} \in \mathbb{R}^{H \times d_h \times d_h}$ is initialized to the identity matrix. Unconstrained optimization can cause singular values of $W_{\text{corr}}$ to collapse to zero, corrupting value representations.
- **Mitigation**: Add an orthogonal regularization penalty $\mathcal{L}_{\text{ortho}} = \|W_{\text{corr}}^T W_{\text{corr}} - I\|_F^2$.

### 5.3 Softpick Temperature Gradient Lock
- **Mechanism**: Setting Softpick temperature $T < 0.2$ causes aggressive tail subtraction, zeroing off-diagonal landmark probabilities and locking backpropagation gradients.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

```
+-----------------------------------------------------------------------------------+
|                        Triton Hardware Execution Graph                            |
|                                                                                   |
|  Input X [B, N, D] ---> [Fused QKV Projection GEMM Kernel] ---> Q, K, V           |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                  |       |
|        v                                                                  v       |
| [Triton Dual-Scale RFF Kernel]                    [Triton Landmark Gather Engine] |
|  - SRAM Parallel Dual Cos/Sin Projections          - SRAM Top-k Bitonic Sort     |
|  - Write Concatenated \phi(Q), \phi(K) Features    - Gather K_land, V_land Buffers |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                          |
|        +---> [Triton Fused Landmark Softpick & Local Window Kernel]              |
|               - Load Landmark K_land, V_land into SRAM                            |
|               - Compute Softpick Attn + Tile Local Window SDPA                    |
|               - Elementwise Sigmoidal Gate Fusion                                 |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                          |
|        +---> [Fused Output GEMM Kernel] ---> Output Y [B, N, D]                   |
+-----------------------------------------------------------------------------------+
```

### 6.1 Triton Strategy 1: Dual-Scale RFF Mapping & Feature Concatenation
- Construct a single vectorized Triton kernel that loads input $X_i \in \mathbb{R}^{\text{BLOCK\_M} \times d_h}$ into SRAM.
- Compute dual projections $X_i W_1$ and $X_i W_2$ in SRAM registers, evaluate $\cos/\sin$ trigonometric functions, and write concatenated feature map $\phi(X_i) \in \mathbb{R}^{\text{BLOCK\_M} \times d_{\text{feat}}}$ directly to main DRAM.

### 6.2 Triton Strategy 2: Shared-Memory Landmark Top-k Gather Kernel
- Compute landmark MLP scores in shared memory.
- Perform a parallel bitonic top-$k$ selection across sequence blocks to extract landmark indices $\mathcal{I}_{\text{land}}$.
- Gather $K_{\text{land}}$ and $V_{\text{land}}$ vectors into continuous SRAM buffers, avoiding fragmented global memory reads.

### 6.3 Triton Strategy 3: Fused Landmark Softpick & Local Window Kernel
- Process sequence blocks of size `BLOCK_M = 64`.
- Load continuous landmark key-value buffers $K_{\text{land}}, V_{\text{land}}$ into shared memory. Compute landmark Softpick attention context $C_{\text{leka}}$ in register files.
- Simultaneously compute local sliding-window attention $C_{\text{local}}$ and perform element-wise sigmoidal gate interpolation in SRAM before writing final context output $C$ to global DRAM.