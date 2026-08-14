# LEKAv3: Linear-Time Efficient Kernel Attention with Value-Scored Semantic Landmarks and Multi-Scale RFF

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Self-Attention (MHSA) scales quadratically $\mathcal{O}(N^2)$ with sequence length $N$, posing prohibitive computational and memory bottlenecks. While random feature kernel attention linearizes attention to $\mathcal{O}(N \cdot F \cdot d_h)$, it struggles with spatial over-smoothing and loss of associative memory. Furthermore, key-based landmark selection heuristics frequently select redundant tokens that carry minimal semantic value.

**LEKAv3** (Linear-time Efficient Kernel Attention v3) presents an advanced hybrid attention mechanism incorporating four core innovations:
1. **Value-State Semantic Landmark Selection**: Unlike traditional landmark models that evaluate keys or queries, LEKAv3 dynamically scores tokens via an MLP operating directly on Value representations $V \in \mathbb{R}^{B \times H \times N \times d_h}$, ensuring landmarks are selected based on semantic output information content.
2. **Multi-Scale Random Fourier Feature (RFF) Mapping**: Query and key vectors are projected into a multi-scale trigonometric space using dual random projection matrices ($F_1$ and $F_2 = F_1 / 2$), capturing both broad sequence context and fine-grained feature correlations.
3. **Value Correction Tensor Transformation**: Employs a learnable per-head transformation tensor $W_{\text{corr}} \in \mathbb{R}^{H \times d_h \times d_h}$ to adapt gathered landmark values prior to context contraction.
4. **Gated Softpick Landmark & Local Window Fusion**: Blends Softpick rectified landmark attention ($C_{\text{land}}$) with local sliding-window softmax attention ($C_{\text{local}}$) via a learnable per-head sigmoidal gating parameter $g$.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Multi-Scale Random Fourier Feature Mapping
For projected queries and keys $Q, K \in \mathbb{R}^{B \times H \times N \times d_h}$, multi-scale random projections $W_1 \in \mathbb{R}^{d_h \times F_1}$ and $W_2 \in \mathbb{R}^{d_h \times F_2}$ (where $F_2 = F_1 / 2$) generate trigonometric feature maps:

$$P_1 = x W_1 \implies \phi_1(x) = \frac{1}{\sqrt{F_1}} \left[ \cos(P_1), \; \sin(P_1) \right] \in \mathbb{R}^{B \times H \times N \times 2F_1}$$

$$P_2 = x W_2 \implies \phi_2(x) = \frac{1}{\sqrt{F_2}} \left[ \cos(P_2), \; \sin(P_2) \right] \in \mathbb{R}^{B \times H \times N \times 2F_2}$$

$$\phi(x) = \left[ \phi_1(x), \; \phi_2(x) \right] \in \mathbb{R}^{B \times H \times N \times d_{\text{feat}}}, \quad d_{\text{feat}} = 2F_1 + 2F_2$$

### 2.2 Value-State Semantic Landmark Selection (`select_landmarks`)
Token importance scores are predicted directly from Value states $V \in \mathbb{R}^{B \times H \times N \times d_h}$ using a two-layer bottleneck network:

$$\text{Scores}_{\text{land}} = \text{Linear}_2\left( \text{GELU}\left( \text{Linear}_1( V ) \right) \right) \in \mathbb{R}^{B \times H \times N}$$

The top-$L_{\max}$ landmark token indices $\mathcal{I}_{\text{land}} \in \{1, \dots, N\}^{L_{\max}}$ are extracted:

$$\mathcal{I}_{\text{land}} = \text{TopK}\left( \text{Scores}_{\text{land}}, L_{\max}, \text{dim}=-1 \right)$$

Key features $\phi(K)$ and Value representations $V$ are gathered at landmark indices:

$$K_{\text{land}} = \text{gather}\left( \phi(K), \mathcal{I}_{\text{land}} \right) \in \mathbb{R}^{B \times H \times L_{\max} \times d_{\text{feat}}}$$

$$V_{\text{land}} = \text{gather}\left( V, \mathcal{I}_{\text{land}} \right) \in \mathbb{R}^{B \times H \times L_{\max} \times d_h}$$

Gathered landmark values are adjusted by learnable correction tensor $W_{\text{corr}} \in \mathbb{R}^{H \times d_h \times d_h}$:

$$V_{\text{adj}} = V_{\text{land}} W_{\text{corr}} \in \mathbb{R}^{B \times H \times L_{\max} \times d_h}$$

### 2.3 Rectified Softpick Landmark Attention
Landmark score logits evaluate inner products between query features $\phi(Q)$ and landmark key features $K_{\text{land}}$:

$$S_{\text{land}} = \phi(Q) K_{\text{land}}^T \in \mathbb{R}^{B \times H \times N \times L_{\max}}$$

**Softpick** rectified activation with temperature scaling $T$ normalizes probabilities:

$$m_i = \max_l S_{\text{land}, i, l}, \quad e_{i, l} = \exp\left( \frac{S_{\text{land}, i, l} - m_i}{T} \right)$$

$$p_{i, l} = \frac{\max\left(0, e_{i, l} - \exp(-m_i / T)\right)}{\sum_k \max\left(0, e_{i, k} - \exp(-m_i / T)\right) + \epsilon}$$

$$C_{\text{land}} = p \cdot V_{\text{adj}} \in \mathbb{R}^{B \times H \times N \times d_h}$$

### 2.4 Local Window Attention & Sigmoidal Gate Fusion
Local sliding window attention evaluates scaled dot-product attention within local radius $W_{\text{local}}$:

$$S_{\text{local}, i, j} = \frac{Q_i K_j^T}{\sqrt{d_h}} + M^{\text{window}}_{i, j}, \quad M^{\text{window}}_{i, j} = \begin{cases} 0 & |i - j| \le \lfloor W_{\text{local}} / 2 \rfloor \\ -\infty & \text{otherwise} \end{cases}$$

$$C_{\text{local}} = \text{Softmax}\left( S_{\text{local}} \right) V \in \mathbb{R}^{B \times H \times N \times d_h}$$

Landmark context $C_{\text{land}}$ and local context $C_{\text{local}}$ are interpolated via learnable per-head gate parameter $g \in \mathbb{R}^H$:

$$\alpha = \sigma(g) \in [0, 1]^H$$

$$C = \alpha \odot C_{\text{land}} + (1 - \alpha) \odot C_{\text{local}} \in \mathbb{R}^{B \times H \times N \times d_h}$$

Reshaping into $[B, N, H \cdot d_h]$, the output is projected back to model dimension $D$:

$$Y = \text{Reshape}(C) W_O + b_O \in \mathbb{R}^{B \times N \times D}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input State** | $X$ | $[B, N, D]$ | Base activation state |
| **Linear Projections Q/K/V** | $Q, K, V$ | $[B, N, D] \to [B, H, N, d_h]$ | Multi-head view split |
| **Multi-Scale RFF Mapping** | $\phi(Q), \phi(K)$ | $[B, H, N, d_h] \to [B, H, N, d_{\text{feat}}]$ | Dual projection $\cos/\sin$ concatenation |
| **Value Semantic Scoring** | $\text{Scores}_{\text{land}}$ | $[B, H, N, d_h] \to [B, H, N]$ | Two-layer MLP operating on Value $V$ |
| **Top-L Landmark Gather** | $K_{\text{land}}, V_{\text{land}}$| $[B, H, N, \dots] \to [B, H, L_{\max}, \dots]$ | Value-driven top-$L_{\max}$ token gather |
| **Value Correction Tensor** | $V_{\text{adj}}$ | $[B, H, L_{\max}, d_h] \times [H, d_h, d_h] \to [B, H, L_{\max}, d_h]$ | Tensor product $V_{\text{land}} W_{\text{corr}}$ |
| **Landmark Softpick Attn** | $C_{\text{land}}$ | $[B, H, N, d_{\text{feat}}] \times [B, H, L_{\max}, d_{\text{feat}}]^T \to [B, H, N, d_h]$ | Rectified Softpick value contraction |
| **Local Sliding Window** | $C_{\text{local}}$ | $[B, H, N, d_h] \times [B, H, N, d_h]^T \to [B, H, N, d_h]$ | Band-masked softmax attention |
| **Sigmoidal Gate Fusion** | $C$ | $[B, H, N, d_h]$ | $\sigma(g) C_{\text{land}} + (1 - \sigma(g)) C_{\text{local}}$ |
| **Output GEMM** | $Y$ | $[B, N, H \cdot d_h] \to [B, N, D]$ | Layer output projection |

---

## 4. Analytical Complexity Profiling

### 4.1 Floating Point Operations (FLOPs)
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ head count, $d_h = D/H$ head dimension, $F_1, F_2$ RFF feature dimensions ($d_{\text{feat}} = 2F_1 + 2F_2$), $L_{\max}$ landmark count, and $W_{\text{local}}$ local window width.

1. **Linear Q, K, V Projections**:
   $$\text{FLOPs}_{\text{proj}} = 3 \times 2 B N D^2 = 6 B N D^2$$
2. **Multi-Scale RFF Feature Mapping**:
   $$\text{FLOPs}_{\text{rff}} = 2 B N D (F_1 + F_2)$$
3. **Value Semantic Landmark Scoring**:
   - Bottleneck MLP operating on $V$: $2 B H N d_h (32) + 2 B H N (32) (1) = 64 B N D + 64 B H N$
   $$\text{FLOPs}_{\text{score}} \approx 64 B N D$$
4. **Value Correction & Landmark Softpick Attention**:
   - Value correction tensor product $V_{\text{land}} W_{\text{corr}}$: $2 B H L_{\max} d_h^2 = 2 B L_{\max} D d_h$
   - Landmark score product $\phi(Q) K_{\text{land}}^T$: $2 B H N L_{\max} d_{\text{feat}}$
   - Softpick & Value contraction $p V_{\text{adj}}$: $2 B H N L_{\max} d_h = 2 B N L_{\max} D$
   $$\text{FLOPs}_{\text{land}} = 2 B L_{\max} D d_h + 2 B H N L_{\max} d_{\text{feat}} + 2 B N L_{\max} D$$
5. **Local Sliding Window Attention**:
   $$\text{FLOPs}_{\text{local}} = 4 B N D W_{\text{local}}$$

**Total FLOP Count**:

$$\text{FLOPs}_{\text{total}} \approx 6 B N D^2 + 2 B H N L_{\max} d_{\text{feat}} + 2 B N L_{\max} D + 4 B N D W_{\text{local}}$$

### 4.2 Memory Footprint & Bandwidth
- **Parameter Footprint**:
  $$\Theta = 4 D^2 + (d_h F_1 + d_h F_2) + \Theta_{\text{v\_scorer}} + H d_h^2 = 4 D^2 + d_h (F_1 + F_2) + 33 d_h + H d_h^2$$
- **Activation Footprint**:
  - Landmark feature buffers: $\mathcal{O}(B H L_{\max} d_{\text{feat}})$
  - Landmark attention matrix: $\mathcal{O}(B H N L_{\max})$
  - Local window logits: $\mathcal{O}(B H N W_{\text{local}})$
  - Total Memory: $\mathcal{O}\left(B N D + B H N L_{\max} + B H N W_{\text{local}}\right)$

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Value-State Landmark Score Saturation
- **Mechanism**: The landmark MLP scorer evaluates $\text{MLP}(V)$. If value states $V$ saturate due to large norm values in deep network layers, the MLP output logits flatten, leading to random Top-k landmark selection.
- **Mitigation**: Apply RMSNorm to Value states $V$ before feeding into `landmark_scorer`.

### 5.2 Value Correction Matrix Identity Collapse
- **Mechanism**: $W_{\text{corr}}$ is initialized to identity matrices $I_{d_h}$. If learning rates are too high, $W_{\text{corr}}$ weights can drift towards zero, truncating landmark value output.

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
| [Triton Multi-Scale RFF Kernel]                   [Triton Value Scorer Engine]    |
|  - SRAM Parallel Dual Cos/Sin Projections          - SRAM Bottleneck MLP on V     |
|  - Write Concatenated \phi(Q), \phi(K) Features    - Top-k Bitonic Landmark Gather|
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

### 6.1 Triton Strategy 1: Multi-Scale RFF Mapping & Feature Concatenation
- Construct a single vectorized Triton kernel loading input $X_i \in \mathbb{R}^{\text{BLOCK\_M} \times d_h}$ into SRAM.
- Evaluate dual projections $X_i W_1$ and $X_i W_2$ in SRAM registers, compute trigonometric features, and write concatenated feature map $\phi(X_i) \in \mathbb{R}^{\text{BLOCK\_M} \times d_{\text{feat}}}$ directly to main DRAM.

### 6.2 Triton Strategy 2: Shared-Memory Value Landmark Scorer Kernel
- Compute Value-based MLP scores $\text{MLP}(V)$ in shared memory.
- Perform a parallel bitonic top-$k$ selection across sequence blocks to extract landmark indices $\mathcal{I}_{\text{land}}$.
- Gather $K_{\text{land}}$ and $V_{\text{land}}$ vectors into continuous SRAM buffers, avoiding fragmented global memory reads.

### 6.3 Triton Strategy 3: Fused Landmark Softpick & Local Window Kernel
- Process sequence blocks of size `BLOCK_M = 64`.
- Load continuous landmark key-value buffers $K_{\text{land}}, V_{\text{land}}$ into shared memory. Compute landmark Softpick attention context $C_{\text{land}}$ in register files.
- Simultaneously compute local sliding-window attention $C_{\text{local}}$ and perform element-wise sigmoidal gate interpolation in SRAM before writing final context output $C$ to global DRAM.