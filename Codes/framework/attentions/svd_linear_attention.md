# SVD Linear Attention: Semi-Orthogonal Low-Rank Projection Factorization

## 1. Executive Abstract & Architectural Motivation

SVD Linear Attention (`SVDLinearAttention`) introduces a zero-shot model compression framework for Multi-Head Attention by replacing standard dense linear projection layers ($W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$) with low-rank factorized modules (`SVDLinear`). Standard dense projections consume $4 D^2$ parameter storage per layer, creating significant memory bandwidth constraints during inference.

`SVDLinearAttention` factorizes each projection matrix into two lower-rank matrices $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and $V \in \mathbb{R}^{r \times d_{\text{in}}}$, where $r \ll D$ is a configurable bottleneck rank. The parameters are initialized zero-shot from pre-trained model checkpoints via truncated Singular Value Decomposition ($W \approx U \Sigma V^\top$). Semi-orthogonal initialization (`reset_parameters`) maintains gradient stability and representation capacity during fine-tuning while reducing projection parameter counts and FLOPs.

---

## 2. Mathematical Formulations & Derivations

Let $X \in \mathbb{R}^{B \times N \times D}$ denote input activations. Let $H$ be the number of query heads, $d_k = D / H$ head dimension, and $r$ the factor rank per projection layer.

### Low-Rank Matrix Factorization & Semi-Orthogonal Initialization (`SVDLinear`):
A dense parameter matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with bias $b \in \mathbb{R}^{d_{\text{out}}}$ is factorized via truncated SVD:

\[
W = U_{\text{full}} \Sigma_{\text{full}} V_{\text{full}}^\top \implies U = U_{\text{full}}[:, :r] \cdot \text{diag}(\Sigma_{\text{full}}[:r]) \in \mathbb{R}^{d_{\text{out}} \times r}, \quad V = V_{\text{full}}[:r, :]^\top \in \mathbb{R}^{r \times d_{\text{in}}}
\]

The forward pass computes down-projection followed by up-projection:

\[
Z = X V^\top \in \mathbb{R}^{B \times N \times r}, \quad Y = Z U^\top + b = X (U V)^\top + b \in \mathbb{R}^{B \times N \times d_{\text{out}}}
\]

### SVD-Factorized Multi-Head Projections:
Query, Key, Value, and Output linear transformations execute via factorized modules:

\[
Q = \text{SVDLinear}_Q(X) \in \mathbb{R}^{B \times N \times (H \cdot d_k)} \implies Q \in \mathbb{R}^{B \times H \times N \times d_k}
\]
\[
K = \text{SVDLinear}_K(X) \in \mathbb{R}^{B \times N \times (H \cdot d_k)} \implies K \in \mathbb{R}^{B \times H \times N \times d_k}
\]
\[
V = \text{SVDLinear}_V(X) \in \mathbb{R}^{B \times N \times (H \cdot d_k)} \implies V \in \mathbb{R}^{B \times H \times N \times d_k}
\]

### Scaled Dot-Product Attention & Output Projection:
Affinity score calculation and value aggregation proceed as:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} \cdot K_{b, h, j, m}
\]
\[
A_{b, h, i, j} = \text{Softmax}_j (\text{apply\_attention\_mask}(S, M)_{b, h, i, j})
\]
\[
C_{b, h, i, m} = \sum_{j=1}^N \text{Dropout}(A)_{b, h, i, j} \cdot V_{b, h, j, m} \implies C \in \mathbb{R}^{B \times N \times D}
\]
\[
Y = \text{SVDLinear}_O(C) \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Step-by-Step Tensor Execution Trace

### Tensor Shape Transformation Table

| Execution Stage | Input Shape | Transformation / Operation | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input** | `[B, N, D]` | Input activation batch | `[B, N, D]` |
| **Q Down Projection** | `[B, N, D]` | `F.linear(X, query.V)` | `[B, N, r]` |
| **Q Up Projection** | `[B, N, r]` | `F.linear(Vx, query.U) + bias` | `[B, N, H*d_k]` |
| **Multi-Head Reshape** | `[B, N, H*d_k]` | `.reshape(B, N, H, d_k).transpose(1, 2)` | `[B, H, N, d_k]` each |
| **Score Matrix MatMul** | `Q`, `K` | `torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)` | `[B, H, N, N]` |
| **Mask & Softmax** | `scores` | `apply_attention_mask` + `F.softmax` | `[B, H, N, N]` |
| **Value Aggregation** | `attn`, `V` | `torch.matmul(attn, V)` | `[B, H, N, d_k]` |
| **Out Down Projection**| `context: [B, N, D]`| `F.linear(context, out_proj.V)` | `[B, N, r]` |
| **Out Up Projection** | `[B, N, r]` | `F.linear(Vx, out_proj.U) + bias` | `Output: [B, N, D]` |

### Architectural Flowchart Diagram

```
                        Input: hidden_states [B, N, D]
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
     SVDLinear_Q                   SVDLinear_K                   SVDLinear_V
  X -> V_q(r) -> U_q(D)         X -> V_k(r) -> U_k(D)         X -> V_v(r) -> U_v(D)
          │                             │                             │
    Q: [B, H, N, d_k]             K: [B, H, N, d_k]             V: [B, H, N, d_k]
          │                             │                             │
          └──────────────┬──────────────┘                             │
                         ▼                                            │
           MatMul: Q @ K.T / sqrt(d_k)                                │
                         │                                            │
                 scores: [B, H, N, N]                                 │
                         │                                            │
             Apply Attention Mask & Softmax                           │
                         │                                            │
                attn: [B, H, N, N]                                    │
                         │                                            │
                         └──────────────────────┬─────────────────────┘
                                                ▼
                                      MatMul: attn @ V
                                                │
                                     context: [B, N, D]
                                                │
                                            SVDLinear_O
                                       C -> V_o(r) -> U_o(D)
                                                │
                                        Output: [B, N, D]
```

---

## 4. Analytical Complexity & Hardware Profiling

### Computational Workload (FLOPs & MACs)
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ heads, $d_k = D/H$, and $r$ factor rank.

1. **SVD Linear Projections**:
   - Down-projections ($X V^\top$): $4 \times (2 B N D r) = 8 B N D r$ FLOPs.
   - Up-projections ($Z U^\top$): $4 \times (2 B N r D) = 8 B N D r$ FLOPs.
   - *Total Projection Workload*: $16 B N D r$ FLOPs ($8 B N D r$ MACs).

2. **Core Scaled Attention**:
   - $Q K^\top$ & $A V$: $4 B N^2 D$ FLOPs.
   - Softmax & Masking: $7 B H N^2$ FLOPs.

3. **Total Workload Summary**:
   - **Total MACs**: $8 B N D r + 2 B N^2 D$
   - **Total FLOPs**: $16 B N D r + 4 B N^2 D + 7 B H N^2$

### Parameter Footprint Comparison
- **Dense MHA Projection Parameters**: $4 D^2 + 4 D$ floats.
- **SVD Linear MHA Parameter Count**:
  \[
  \text{Params}_{\text{SVDLinear}} = 4 \cdot r(D + D) + 4 D = 8 D r + 4 D \text{ floats}
  \]
  For $D = 4096, r = 64$:
  - Dense MHA: $4 \times 4096^2 \approx 67.1 \text{ million parameters}$.
  - SVD Linear MHA: $8 \times 4096 \times 64 \approx 2.09 \text{ million parameters}$ (**$96.8\%$ parameter compression**).

---

## 5. Architectural Bottlenecks, Rank Collapse & Edge Cases

### Rank Deficiency in Deep Transformer Layers
In deep layers where singular value spectra decay slowly, low rank $r \le 16$ can truncate critical feature directions, leading to perplexity degradation unless fine-tuned with small learning rates.

### Semi-Orthogonal Gradient Drift
During long fine-tuning runs, unconstrained backpropagation can cause matrices $U$ and $V$ to lose their initial semi-orthogonality. Periodic orthogonal re-projection or Gram-Schmidt regularization preserves gradient flow.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton kernel for SVD Linear Attention computes the low-rank product $V^\top U^\top$ directly inside register tiles, bypassing intermediate memory writes.

```
                    Global Memory (HBM)
   ┌─────────────────────────────────────────────────────────────┐
   │ Input: X [B, N, D]                                          │
   │ Low-Rank Buffers: V [r, D], U [D, r], bias [D]               │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Vectorized Memory Streaming
                                  ▼
 ┌────────────────────────────────────────────────────────────────┐
 │                     Triton SRAM Scratchpad                     │
 │ 1. Load Activation Tile X_i [BLOCK_M, D]                       │
 │ 2. Compute Intermediate Tile: Vx_i = X_i @ V.T [BLOCK_M, r]   │
 │ 3. Reconstruct Head Projection: Out_i = Vx_i @ U.T + bias      │
 │ 4. Execute Online Scaled Attention                             │
 └────────────────────────────────┬───────────────────────────────┘
                                  │ Write Output
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Output Tensor: Y [B, N, D]                                  │
   └─────────────────────────────────────────────────────────────┘
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Orthogonal Singular Regularization**: Add an explicit orthogonality penalty $\mathcal{L}_{\text{ortho}} = \|U^\top U - I\|_F^2 + \|V V^\top - I\|_F^2$ during fine-tuning to preserve subspace stability.
2. **Dynamic Rank Adapters (AdaSVDLinear)**: Dynamically scale rank $r$ across transformer layers based on gradient magnitude or feature entropy.
3. **Quantized SVD Linear (Q-SVDLinear)**: Store $V$ and $U$ in FP8 format while maintaining accumulator registers in FP32 to optimize memory throughput on NVIDIA H100 hardware.