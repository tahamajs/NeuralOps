# NEXUS: Neural Efficient X-scoring Unified Sparse Attention with Multi-Scale Block Gating and Random Fourier Feature Rectification

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Self-Attention (MHSA) in Transformer models requires quadratic computational time and memory scaling $\mathcal{O}(N^2)$ with sequence length $N$. Existing sparse attention techniques enforce rigid local windows (losing global context), heuristic token dropping (degrading accuracy), or linearized kernels (suffering from spatial over-smoothing).

**NEXUS** (Neural Efficient X-scoring Unified Sparse Attention) introduces a unified multi-scale sparse attention paradigm incorporating six synergistic mechanisms:
1. **XAttention Antidiagonal Block Scoring**: Subsamples query-key tokens at stride $\sigma$ to compute coarse inter-block correlation maps $S_{bi, bj}$ in training-free sub-quadratic time.
2. **SeerAttention Learnable Block Gating**: Employs a low-rank query-key pooling network to predict head-wise dynamic sparsity budgets and refine block retention masks.
3. **DuSA Dual-Stage Sparse Engine**: Combines full intra-block token pairs ($bi = bj$) with top-$k$ inter-block token pairs.
4. **AnchorAttention Fine-Grained Stripe Masking**: Applies intra-block diagonal band-stripe masking of width $S_w$ to achieve sub-block token-level sparsity.
5. **Spectraformer RFF Global Rectification**: Interleaves global Random Fourier Feature (RFF) linear attention layers every $L_{\text{rectify}}$ transformer blocks, guaranteeing sequence-wide associative context recovery in $\mathcal{O}(N \cdot F \cdot d_h)$ time.
6. **Softpick Rectified Activation**: Replaces standard softmax with a thresholded tail-subtraction non-linearity (`softpick`), producing non-negative sparse probabilities without artificial sink token collapse.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Sequence Partitioning & XAttention Antidiagonal Block Scoring
Input sequence $X \in \mathbb{R}^{B \times N \times D}$ is partitioned into $M = \lceil N / B_s \rceil$ blocks of block size $B_s$. Query and Key tensors $Q, K \in \mathbb{R}^{B \times H \times N \times d_h}$ are subsampled at stride $\sigma$:

$$Q_{\text{sample}} = Q_{:, :, ::\sigma, :}, \quad K_{\text{sample}} = K_{:, :, ::\sigma, :}$$

Coarse block interaction scores $S_{bi, bj} \in \mathbb{R}^{B \times H \times M \times M}$ evaluate block-averaged cross-correlations:

$$S_{bi, bj} = \frac{1}{|B_s / \sigma|^2} \sum_{i \in \text{Block}_{bi}} \sum_{j \in \text{Block}_{bj}} \frac{Q_{\text{sample}, i} \cdot K_{\text{sample}, j}^T}{\sqrt{d_h}}$$

For each query block $bi$, the top-$k$ key blocks $\mathcal{K}_{bi}$ yielding the highest inter-block interaction scores are retained:

$$\mathcal{K}_{bi} = \text{TopK}\left( S_{bi, :}, k, \text{dim}=-1 \right)$$

### 2.2 SeerAttention Learnable Block Gating (`SeerGating`)
In parallel, average-pooled Query and Key representations are projected through bottleneck MLPs:

$$q_{\text{gate}} = \text{Linear}_2\left( \text{GELU}\left( \text{Linear}_1( \bar{Q} ) \right) \right) \in \mathbb{R}^{B \times H \times 1 \times r_{\text{gate}}}$$

$$k_{\text{gate}} = \text{Linear}_2\left( \text{GELU}\left( \text{Linear}_1( \bar{K} ) \right) \right) \in \mathbb{R}^{B \times H \times r_{\text{gate}} \times 1}$$

Dynamic block capacity $k_{\text{seer}} \in \{1, \dots, M\}$ is calculated per head:

$$g_{\text{seer}} = \sigma\left( \frac{q_{\text{gate}} k_{\text{gate}}}{\tau_{\text{seer}}} \right) \in [0, 1]^{B \times H}, \quad k_{\text{seer}} = \text{clamp}\left( \lfloor g_{\text{seer}} \cdot M \rfloor, 1, M \right)$$

### 2.3 Dual-Stage & AnchorAttention Fine-Grained Stripe Masking
For each query block $bi$, the active key token set consists of full intra-block keys ($bj = bi$) plus top-$k$ inter-block keys ($bj \in \mathcal{K}_{bi}$):

$$K_{\text{active}}^{(bi)} = \text{cat}\left( K_{\text{intra}}^{(bi)}, K_{\text{inter}}^{(bi)} \right) \in \mathbb{R}^{B \times H \times N_{\text{active}} \times d_h}, \quad N_{\text{active}} = B_s + k \cdot B_s$$

Within selected blocks, AnchorAttention stripe diagonal band masking of width $S_w$ restricts token interactions to adjacent stripe groups:

$$M^{\text{stripe}}_{i, j} = \begin{cases} 0 & \text{if } |\lfloor i / S_w \rfloor - \lfloor j / S_w \rfloor| \le 1 \\ -\infty & \text{otherwise} \end{cases}$$

Raw dot-product scores are masked:

$$\text{Scores}_{bi} = \frac{Q_{bi} \left( K_{\text{active}}^{(bi)} \right)^T}{\sqrt{d_h}} + M^{\text{stripe}} + M^{\text{pad}}$$

Intra-block probabilities are normalized via **Softpick** with temperature scaling $\tau$:

$$m_i = \max_j \text{Scores}_{i, j}, \quad e_{i, j} = \exp\left( \frac{\text{Scores}_{i, j} - m_i}{\tau} \right)$$

$$p_{i, j} = \frac{\max\left(0, e_{i, j} - \exp(-m_i / \tau)\right)}{\sum_k \max\left(0, e_{i, k} - \exp(-m_i / \tau)\right) + \epsilon}$$

$$O_{bi}^{\text{dss}} = \sum_j p_{i, j} V_{\text{active}, j}^{(bi)} \in \mathbb{R}^{B \times H \times B_s \times d_h}$$

### 2.4 Spectraformer Random Fourier Feature Global Rectification (`SpectraformerRectify`)
At periodic layer intervals ($l \pmod{L_{\text{rectify}}} == 0$), execution switches from dual-stage sparse attention to global Random Fourier Feature (RFF) linear attention. Queries and keys are mapped via static parameter $W_{\text{rff}} \in \mathbb{R}^{d_h \times F}$:

$$\phi(Q) = \frac{\exp\left( \text{normalize}(Q) W_{\text{rff}} \right)}{\sqrt{F}}, \quad \phi(K) = \frac{\exp\left( \text{normalize}(K) W_{\text{rff}} \right)}{\sqrt{F}}$$

Linear context accumulation matrix $KV \in \mathbb{R}^{B \times H \times F \times d_h}$ and normalization vector $Z \in \mathbb{R}^{B \times H \times F}$ compute global attention in $\mathcal{O}(N \cdot F \cdot d_h)$ time:

$$KV = \sum_{j=1}^N \phi(K)_j^T \otimes V_j, \quad Z = \sum_{j=1}^N \phi(K)_j^T, \quad O^{\text{rectify}}_i = \frac{\phi(Q)_i KV}{\phi(Q)_i Z^T + \epsilon}$$

### 2.5 Layer Output Projection
Context representations from dual-stage sparse or global RFF layers are reshaped and projected through output linear layer $W_O$:

$$Y = \text{Reshape}\left( O^{\text{dss}} \text{ or } O^{\text{rectify}} \right) W_O + b_O \in \mathbb{R}^{B \times N \times D}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input Hidden State** | $X$ | $[B, N, D]$ | Base activation state |
| **Linear Projection Q/K/V** | $Q, K, V$ | $[B, N, D] \to [B, H, N, d_h]$ | Multi-head view transposition |
| **Subsampled Token Scoring**| $Q_{\text{sample}}, K_{\text{sample}}$ | $[B, H, N, d_h] \to [B, H, N/\sigma, d_h]$ | Strided sequence subsampling |
| **XAttention Block Scoring**| $S_{bi, bj}$ | $[B, H, N/\sigma, d_h] \to [B, H, M, M]$ | Coarse block interaction matmul |
| **Top-k Inter-Block Selection**| $\mathcal{K}_{bi}$ | $[B, H, M, M] \to [B, H, M, k]$ | Top-$k$ block index extraction |
| **Seer Gating Prediction** | $g_{\text{seer}}$ | $[B, H, D] \to [B, H, 1, 1]$ | Bottleneck MLP capacity prediction |
| **Active Key-Value Concatenation**| $K_{\text{active}}, V_{\text{active}}$ | $[B, H, N_{\text{active}}, d_h]$ | Intra + inter-block tensor cat |
| **Stripe Granularity Masking**| $M^{\text{stripe}}$ | $[B_s, N_{\text{active}}]$ | Fine-grained band-stripe mask |
| **Softpick Score Contraction**| $O^{\text{dss}}$ | $[B, H, B_s, N_{\text{active}}] \times [B, H, N_{\text{active}}, d_h] \to [B, H, N, d_h]$ | Rectified Softpick value product |
| **Global RFF Rectification** | $O^{\text{rectify}}$ | $[B, H, N, d_h] \times [d_h, F] \to [B, H, N, d_h]$ | Periodic RFF linear attention |
| **Output Linear GEMM** | $Y$ | $[B, N, H \cdot d_h] \to [B, N, D]$ | Layer output projection |

---

## 4. Analytical Complexity Profiling

### 4.1 Floating Point Operations (FLOPs)
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ head count, $d_h = D/H$ head dimension, $B_s$ block size, $k$ top-k inter-blocks, $\sigma$ XAttention stride, $S_w$ stripe width, and $F$ RFF feature dimension.

1. **Linear Projections ($Q, K, V, O$)**:
   $$\text{FLOPs}_{\text{proj}} = 4 \times 2 B N D^2 = 8 B N D^2$$
2. **XAttention Subsampled Block Scoring**:
   - Subsampled score matmul ($N/\sigma$ tokens): $2 B H (N/\sigma)^2 d_h = \frac{2 B N^2 D}{\sigma^2}$
   $$\text{FLOPs}_{\text{xattn}} = \frac{2 B N^2 D}{\sigma^2}$$
3. **Dual-Stage Stripe Attention ($M = N / B_s$ query blocks)**:
   - Active keys per query block $N_{\text{active}} = B_s + k B_s = B_s (1 + k)$
   - Intra + Inter dot product: $M \times \left( 2 B H B_s N_{\text{active}} d_h \right) = 2 B N D B_s (1 + k)$
   - Softpick & Value contraction: $2 B N D B_s (1 + k)$
   $$\text{FLOPs}_{\text{dss}} = 4 B N D B_s (1 + k)$$
4. **Global RFF Rectification Layer (Every $L_{\text{rectify}}$ layers)**:
   $$\text{FLOPs}_{\text{rff}} = 8 B H N d_h F = 8 B N D F$$

**Total FLOP Count (Sparse Layer)**:

$$\text{FLOPs}_{\text{sparse}} = 8 B N D^2 + \frac{2 B N^2 D}{\sigma^2} + 4 B N D B_s (1 + k)$$

### 4.2 Memory Footprint & Bandwidth
- **Parameter Footprint**:
  $$\Theta = 4 D^2 + \Theta_{\text{seer}} + \Theta_{\text{rff}} = 4 D^2 + 4 d_h r_{\text{gate}} + d_h F$$
- **Activation Footprint**:
  - Block score matrix: $\mathcal{O}(B H M^2)$
  - Active key/value buffers: $\mathcal{O}(B H N (1 + k) B_s)$
  - Total Memory: $\mathcal{O}\left(B N D + B H \left( \frac{N}{B_s} \right)^2 + B H N k B_s\right) \ll \mathcal{O}(B H N^2)$

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 XAttention Subsampling Misalignment & Block Omission
- **Mechanism**: XAttention evaluates coarse block interactions using strided subsampling ($\sigma = 2$). If critical query-key interactions occur in a localized spike between sampled token indices, the block score $S_{bi, bj}$ drops below the top-$k$ threshold.
- **Consequence**: The inter-block pair is omitted from active computation, causing lost context and retrieval failures.
- **Mitigation**: Ensure $B_s / \sigma \ge 8$ so that sampled tokens represent block statistics accurately.

### 5.2 Seer Gating Dynamic Budget Collapse
- **Mechanism**: The Seer gating network predicts block retention capacity $k_{\text{seer}} = \lfloor \sigma(\cdot) \cdot M \rfloor$. If initialization biases drive $\sigma(\cdot) \to 0$, $k_{\text{seer}}$ collapses to $1$, reducing sparse attention to isolated intra-block diagonal interactions.
- **Mitigation**: Initialize Seer gating bias to $+2.0$, forcing full block retention early in training.

### 5.3 Softpick Tail Subtraction Discontinuity
- **Mechanism**: Softpick relies on $\text{ReLU}(e_{i, j} - \exp(-m/\tau))$. Under extremely low temperatures ($\tau < 0.2$), all off-diagonal scores are zeroed, causing zero-gradient propagation across active inter-block keys.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

```
+-----------------------------------------------------------------------------------+
|                        Triton Hardware Execution Graph                            |
|                                                                                   |
|  Input X [B, N, D] ---> [Fused QKV Projection Kernel] ---> Q, K, V               |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                  |       |
|        v                                                                  v       |
| [Triton Strided XAttention Block Scorer]           [Triton Top-k Block Indexer]   |
|  - Subsampled Tile Matmul (32x32)                  - SRAM Parallel Top-k Bitonic  |
|  - Coarse Block Score Grid Output                  - Active Key Pointer Generation|
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                          |
|        +---> [Triton Block-Sparse Dual-Stage Stripe Engine]                       |
|               - Load Intra + Top-k Inter-Block Keys into SRAM                     |
|               - Apply Stripe Band Mask & Online Softpick Normalization            |
|               - Write Context Output to Global Memory                             |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                          |
|        +---> [Fused Output GEMM Kernel] ---> Output Y [B, N, D]                   |
+-----------------------------------------------------------------------------------+
```

### 6.1 Triton Strategy 1: Subsampled Strided Block Scorer Kernel
- Implement a lightweight Triton kernel executing strided query-key inner products over subsampled sequence tiles of size `32 x 32`.
- Perform spatial block reduction directly in GPU shared memory, writing coarse block interaction matrix $S_{bi, bj}$ to global DRAM in a single pass.

### 6.2 Triton Strategy 2: Block-Sparse Dual-Stage Stripe Engine
Standard PyTorch implementations collect inter-block key tensors via `torch.cat`, creating high HBM allocation overhead.

**Triton Optimization**:
- Construct a block-sparse Triton kernel where each threadblock handles a query block $bi$.
- Use indirection pointers $\mathcal{K}_{bi}$ to load active inter-block keys directly from main key memory into shared memory.
- Fuse stripe diagonal mask checks and online Softpick tail subtraction into register iteration loops.

### 6.3 Triton Strategy 3: Periodic RFF Streaming Linear Engine
- For rectification layers ($l \pmod{L_{\text{rectify}}} == 0$), execute a streaming RFF linear attention kernel.
- Stream sequence blocks of size `BLOCK_N = 128` to accumulate outer products into shared memory buffer $KV \in \mathbb{R}^{F \times d_h}$, bypassing global memory allocations for feature activations $\phi(K)$.