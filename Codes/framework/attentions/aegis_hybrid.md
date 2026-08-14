# Aegis-Hybrid Attention: Low-Rank Factorized Multi-Scale Hybrid Attention with Adaptive Gated Pruning

## 1. Abstract & Theoretical Motivation

Standard multi-head self-attention (MHSA) in Transformer architectures presents quadratic computational complexity $\mathcal{O}(N^2)$ with respect to sequence length $N$, restricting long-context deployment. While localized sliding-window attention restricts receptive fields to achieve linear complexity $\mathcal{O}(N \cdot W)$, it loses global contextual dependencies. Conversely, linearized kernel attention achieves global scope in $\mathcal{O}(N \cdot d_h^2)$ complexity but degrades local representation fidelity due to unconstrained feature projection smoothing.

**Aegis-Hybrid Attention** presents a unified multi-scale hybrid paradigm designed to bridge this efficiency-expressivity divide:
1. **Low-Rank Singular Value Decomposition (SVD) Projections**: Linear projection operators $W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$ are factorized into rank-$r$ bottleneck matrices ($W_1 \in \mathbb{R}^{D \times r}, W_2 \in \mathbb{R}^{r \times D}$) initialized via truncated SVD of pre-trained parameters, drastically reducing parameter count while retaining spectral energy.
2. **Multi-Scale Hybrid Execution**: Sequences undergo parallel evaluation through a local sliding-window dense attention branch (preserving high-frequency syntactic interactions) and a global kernelized linear attention branch utilizing non-negative feature activations $\phi(x) = \text{ELU}(x) + 1$.
3. **Adaptive Gated Soft-Pruning**: Prior to global kernel aggregation, sequence tokens pass through a token-importance sigmoid gate $g \in [0, 1]^{B \times N \times H}$, softly masking non-informative tokens to mitigate noise accumulation in the linear memory accumulator.
4. **Dynamic Dynamic Fusion**: A learned per-token, per-head gating parameter $\alpha \in [0, 1]$ dynamically balances local high-resolution details against global summary representations.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Low-Rank Factorized Projections
Given an input activation tensor $X \in \mathbb{R}^{B \times N \times D}$, the low-rank projection for query, key, value, and output representations is defined as:

$$\begin{aligned}
Q &= (X W_{Q1}) W_{Q2} + b_{Q2}, \quad W_{Q1} \in \mathbb{R}^{D \times r}, \, W_{Q2} \in \mathbb{R}^{r \times D} \\
K &= (X W_{K1}) W_{K2} + b_{K2}, \quad W_{K1} \in \mathbb{R}^{D \times r}, \, W_{K2} \in \mathbb{R}^{r \times D} \\
V &= (X W_{V1}) W_{V2} + b_{V2}, \quad W_{V1} \in \mathbb{R}^{D \times r}, \, W_{V2} \in \mathbb{R}^{r \times D}
\end{aligned}$$

The weight matrices are initialized via truncated SVD of pre-trained weight $W_0 \in \mathbb{R}^{D \times D}$:

$$W_0 = U \Sigma V^T \implies W_{1} = V_{:, :r}^T, \quad W_{2} = U_{:, :r} \Sigma_{:r, :r}$$

Reshaping into $H$ heads with head dimension $d_h = D / H$:

$$Q, K, V \in \mathbb{R}^{B \times N \times H \times d_h}$$

### 2.2 Local Sliding Window Attention Branch
For context window radius $W_{\text{local}}$, the band-masked score matrix $S^{\text{local}} \in \mathbb{R}^{B \times H \times N \times N}$ is formulated as:

$$S^{\text{local}}_{b, h, i, j} = \frac{Q_{b, i, h} \cdot K_{b, j, h}^T}{\sqrt{d_h}} + M^{\text{band}}_{i, j} + M^{\text{pad}}_{b, j}$$

where the spatial sliding-window mask $M^{\text{band}}$ and padding mask $M^{\text{pad}}$ are defined by:

$$M^{\text{band}}_{i, j} = \begin{cases} 0 & \text{if } |i - j| \le W_{\text{local}} \\ -\infty & \text{otherwise} \end{cases}, \quad M^{\text{pad}}_{b, j} = \begin{cases} 0 & \text{if } \text{valid}(b, j) \\ -\infty & \text{otherwise} \end{cases}$$

The local output is computed via row-wise softmax normalization:

$$A^{\text{local}} = \text{Softmax}\left( S^{\text{local}} \right), \quad O^{\text{local}}_{b, i, h} = \sum_{j=1}^N A^{\text{local}}_{b, h, i, j} V_{b, j, h}$$

### 2.3 Global Kernel Linear Attention with Adaptive Pruning
The global branch projects tokens into a non-negative feature space using $\phi(x) = \text{ELU}(x) + 1.0$. The token pruning mask $g \in \mathbb{R}^{B \times N \times H}$ computes token importance via a multi-layer perceptron:

$$g = \sigma\left( \text{ReLU}\left( X W_{g1} + b_{g1} \right) W_{g2} + b_{g2} \right)$$

Key and Value representations are modulated by $g$:

$$\tilde{K} = g \odot K, \quad \tilde{V} = g \odot V$$

The global memory accumulation matrix $M \in \mathbb{R}^{B \times H \times d_h \times d_h}$ and normalization vector $Z \in \mathbb{R}^{B \times H \times d_h}$ aggregate token information across the sequence in linear time:

$$M_{b, h} = \sum_{j=1}^N \phi\left(\tilde{K}_{b, j, h}\right)^T \otimes \tilde{V}_{b, j, h}, \quad Z_{b, h} = \sum_{j=1}^N \phi\left(\tilde{K}_{b, j, h}\right)^T$$

The unnormalized and normalized global outputs are:

$$O^{\text{global}}_{b, i, h} = \frac{\phi\left(Q_{b, i, h}\right) M_{b, h}}{\phi\left(Q_{b, i, h}\right) Z_{b, h} + \epsilon}$$

### 2.4 Gated Fusion & Final Projection
Per-token and per-head blending coefficient $\alpha \in \mathbb{R}^{B \times N \times H \times 1}$ is computed via linear transformation:

$$\alpha = \sigma\left( X W_{\text{fusion}} + b_{\text{fusion}} \right)$$

The local and global representations are fused and projected back to the hidden dimension $D$:

$$O^{\text{fused}} = \alpha \odot O^{\text{local}} + (1 - \alpha) \odot O^{\text{global}} \in \mathbb{R}^{B \times N \times H \times d_h}$$

$$Y = \left( \text{Reshape}(O^{\text{fused}}) W_{O1} \right) W_{O2} + b_{O2}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Stage | Input Tensor Shape | Output Tensor Shape | Mathematical Operation |
| :--- | :--- | :--- | :--- |
| **Input Activation** | $[B, N, D]$ | $[B, N, D]$ | Base representation $X$ |
| **Low-Rank Q Projection** | $[B, N, D]$ | $[B, N, r] \to [B, N, D]$ | $X W_{Q1} W_{Q2}$ |
| **Head Reshaping** | $[B, N, D]$ | $[B, N, H, d_h]$ | Split into $H$ heads |
| **Local Score Matmul** | $[B, N, H, d_h]$ | $[B, H, N, N]$ | $\frac{Q K^T}{\sqrt{d_h}} + M^{\text{band}}$ |
| **Local Softmax & Value Agg.** | $[B, H, N, N], [B, N, H, d_h]$ | $[B, N, H, d_h]$ | $\text{Softmax}(S^{\text{local}}) V$ |
| **Pruning Gate Evaluation** | $[B, N, D]$ | $[B, N, H]$ | $\sigma(\text{MLP}(X))$ |
| **Kernel Feature Mapping** | $[B, N, H, d_h]$ | $[B, N, H, d_h]$ | $\phi(x) = \text{ELU}(x) + 1.0$ |
| **Global Memory Outer Prod.**| $[B, H, d_h, N], [B, H, N, d_h]$ | $[B, H, d_h, d_h]$ | $\phi(\tilde{K})^T \tilde{V}$ |
| **Global Context Querying** | $[B, H, N, d_h], [B, H, d_h, d_h]$| $[B, N, H, d_h]$ | $\frac{\phi(Q) M}{\phi(Q) Z + \epsilon}$ |
| **Fusion Gating** | $[B, N, D]$ | $[B, N, H, 1]$ | $\alpha = \sigma(X W_{\text{fusion}})$ |
| **Branch Combination** | $[B, N, H, d_h]$ | $[B, N, H, d_h]$ | $\alpha O^{\text{local}} + (1 - \alpha) O^{\text{global}}$ |
| **Low-Rank Out Projection** | $[B, N, D]$ | $[B, N, D]$ | $(O^{\text{fused}} W_{O1}) W_{O2}$ |

---

## 4. Analytical Complexity Profiling

### 4.1 Floating Point Operations (FLOPs)
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ head count, $d_h = D/H$ head dimension, $r$ bottleneck rank, and $W$ local window radius ($W = W_{\text{local}}$).

1. **Low-Rank Projections ($Q, K, V, O$)**:
   $$\text{FLOPs}_{\text{proj}} = 4 \times 2 \cdot B \cdot N \cdot D \cdot r = 8 B N D r$$
2. **Local Sliding-Window Branch**:
   - Query-Key product within window ($W$ elements per token): $2 \cdot B \cdot H \cdot N \cdot W \cdot d_h = 2 B N D W$
   - Softmax & Value contraction: $2 \cdot B \cdot H \cdot N \cdot W \cdot d_h = 2 B N D W$
   $$\text{FLOPs}_{\text{local}} = 4 B N D W$$
3. **Global Linear Branch**:
   - Pruning Gate MLP: $2 B N D (D/4) + 2 B N (D/4) H \approx \frac{1}{2} B N D^2$
   - Feature map $\phi(\cdot)$: $\mathcal{O}(B N D)$ transcendental operations.
   - Memory State outer product $\phi(\tilde{K})^T \tilde{V}$: $2 B H N d_h^2 = 2 B N D d_h$
   - Global Context Matmul $\phi(Q) M$: $2 B H N d_h^2 = 2 B N D d_h$
   $$\text{FLOPs}_{\text{global}} = \frac{1}{2} B N D^2 + 4 B N D d_h$$
4. **Fusion Gate & Combination**:
   $$\text{FLOPs}_{\text{fusion}} = 2 B N D H + 3 B N D$$

**Total FLOP Count**:

$$\text{FLOPs}_{\text{total}} \approx 8 B N D r + 4 B N D W + \frac{1}{2} B N D^2 + 4 B N D d_h$$

### 4.2 Memory Footprint & Bandwidth
- **Parameters**: 
  $$\Theta = 4 \times (D \cdot r + r \cdot D) + \left(D \cdot \frac{D}{4} + \frac{D}{4} \cdot H\right) + D \cdot H = 8 D r + \frac{D^2}{4} + \frac{D H}{4} + D H$$
- **Activation Memory**:
  - Local logits matrix (if un-fused PyTorch implementation): $\mathcal{O}(B \cdot H \cdot N^2)$ or $\mathcal{O}(B \cdot H \cdot N \cdot W)$ if sparse-banded.
  - Global memory states: $\mathcal{O}(B \cdot H \cdot d_h^2)$.
  - Total Memory: $\mathcal{O}(B N D + B H N W + B H d_h^2)$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Low-Rank Projection Collapse
- **Mechanism**: Restricting intermediate projections to rank $r \ll D$ forces $Q, K, V$ into an $r$-dimensional subspace. If the singular value spectrum of the pre-trained weight matrix exhibits high entropy (slow decay), truncation to rank $r$ discards significant information, leading to catastrophic representation capacity loss.
- **Detection**: Calculate the nuclear norm ratio $\frac{\|W_{:r}\|_*}{\|W\|_*}$. A ratio $< 0.85$ indicates severe energy truncation.

### 5.2 Global Accumulator Saturation & Vanishing Gradients
- **Mechanism**: In the global branch, denominator $Z = \sum_{j=1}^N \phi(\tilde{K}_j)^T$ grows linearly with sequence length $N$. As $N \to \infty$, the magnitude of $Z$ scales up to $\mathcal{O}(N)$, causing $\frac{\phi(Q) M}{\phi(Q) Z + \epsilon}$ gradients with respect to early key representations to decay as $\mathcal{O}(1/N^2)$.
- **Mitigation**: Introduce zero-sum key centering or length-normalized scaling factors $\frac{1}{\sqrt{N}}$ within the kernel feature map.

### 5.3 Local-Global Gating Dominance Degeneracy
- **Mechanism**: The fusion parameter $\alpha = \sigma(X W_{\text{fusion}})$ can experience saturated logit distributions ($\alpha \to 1.0$ or $\alpha \to 0.0$). If $\alpha \to 1.0$, the global kernel branch receives zero backpropagation gradient, rendering the long-range sub-network un-trainable.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

```
+-----------------------------------------------------------------------------------+
|                            Triton Kernel Pipeline                                 |
|                                                                                   |
|  Input X [B, N, D] ---> [Low-Rank GEMM Fused Kernel] ---> Q, K, V [B, N, H, d_h] |
|                                                                                   |
|        +-----------------------------------+-----------------------------------+  |
|        |                                   |                                   |  |
|        v                                   v                                   v  |
| [Block-Sparse Local Attn]       [Gated Kernel Global Attn]     [Fusion Gate Kernel]|
|  Tile-based SRAM Loading         Fused ELU+1 & Outer Prod       Fused Sigmoid     |
|  Receptive Band Masking          SRAM Grid Reduction (d_h, d_h) Interpolation      |
|                                                                                   |
|        +-----------------------------------+-----------------------------------+  |
|        |                                                                          |
|        +---> Fused Combination & Low-Rank Out GEMM ---> Output Y [B, N, D]       |
+-----------------------------------------------------------------------------------+
```

### 6.1 Triton Kernel Design: Fused Low-Rank GEMM
Standard PyTorch low-rank execution invokes two distinct GPU kernel launches per projection (`x @ W1` followed by `@ W2`), incurring DRAM round-trips for the intermediate rank-$r$ tensor. 

**Triton Strategy**: Fuse `W1` and `W2` into a single custom C++/Triton kernel using persistent SRAM registers for intermediate rank outputs:
- Block Size Configuration: `BLOCK_M = 64`, `BLOCK_N = 64`, `BLOCK_K = 32`.
- Load input block $X_{i} \in \mathbb{R}^{64 \times D}$ into SRAM. Compute tile $X_{i} W_1 \in \mathbb{R}^{64 \times r}$ in registers, then immediately multiply by $W_2$ before writing back to global memory.

### 6.2 Triton Kernel Design: Fused Local Sliding-Window Attention
Instead of instantiating the dense $[B, H, N, N]$ tensor and applying masked fill:
- Use a block-sparse Triton kernel with `BLOCK_SIZE_M = 64` and `BLOCK_SIZE_N = 64`.
- Skip block pairs $(i, j)$ where $|i \cdot 64 - j \cdot 64| > W_{\text{local}} + 64$.
- Keep online Softmax accumulators (FlashAttention-style $m_i, l_i$) directly in SRAM registers to execute local windowed attention in $\mathcal{O}(N \cdot W)$ IO complexity.

### 6.3 Triton Kernel Design: Global Accumulator Grid Reduction
For the global branch:
- Parallelize over batch $B$ and heads $H$.
- Fuse the $\text{ELU}(x) + 1$ computation, sigmoid token gate, and outer product matrix contraction into a single streaming loop across sequence blocks of size `BLOCK_N = 128`.
- Accumulate intermediate $KV \in \mathbb{R}^{d_h \times d_h}$ in shared memory (SRAM), avoiding intermediate global memory writes for feature representations $\phi(K)$.