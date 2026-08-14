# Full Attention (Multi-Head Scaled Dot-Product Attention): Standard Pairwise Quadratic Contextualization

## 1. Abstract & Theoretical Motivation

Multi-Head Scaled Dot-Product Attention (MHA), introduced by Vaswani et al. (2017), serves as the foundational architectural block of modern Transformer language models. MHA computes dense, content-dependent, pairwise token interactions by mapping a sequence of input representations into query, key, and value subspaces distributed across $H$ independent representation heads. The core theoretical premise of MHA is to enable the model to jointly attend to information from different representation subspaces at different positions. 

Mathematically, standard MHA relies on dynamic rank-one interactions formed by the inner products of query and key vectors. Scaled dot-product scoring measures the affinity between pairs of positions, producing a full sequence-length similarity matrix $A \in \mathbb{R}^{L \times L}$ per head. By normalizing this matrix row-wise with the Softmax operator, MHA models discrete context-dependent routing distributions over the key-value sequence. While highly expressive and capable of modeling arbitrary dependencies without spatial inductive bias, the explicit materialization of the $L \times L$ attention matrix incurs quadratic time and memory complexity $\mathcal{O}(L^2)$ with respect to sequence length $L$. This establishes the primary computational bottleneck in high-throughput and long-context Transformer execution.

---

## 2. Mathematical Derivations & Formulations

Given an input hidden state sequence tensor $X \in \mathbb{R}^{B \times L \times D}$, where $B$ denotes batch size, $L$ is sequence length, and $D$ is the model hidden dimension, the Multi-Head Scaled Dot-Product Attention module projects $X$ using linear transformations:

\[
Q = X W_Q \in \mathbb{R}^{B \times L \times D}, \quad K = X W_K \in \mathbb{R}^{B \times L \times D}, \quad V = X W_V \in \mathbb{R}^{B \times L \times D}
\]

where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$. The projected representations are partitioned into $H$ parallel heads, each operating on a sub-space dimension $d_k = D / H$:

\[
Q_h = Q[:, :, h \cdot d_k : (h+1) \cdot d_k] \in \mathbb{R}^{B \times L \times d_k}
\]
\[
K_h = K[:, :, h \cdot d_k : (h+1) \cdot d_k] \in \mathbb{R}^{B \times L \times d_k}
\]
\[
V_h = V[:, :, h \cdot d_k : (h+1) \cdot d_k] \in \mathbb{R}^{B \times L \times d_k}
\]

Reshaping and transposing into multi-head layouts yields tensors $Q_h, K_h, V_h \in \mathbb{R}^{B \times H \times L \times d_k}$.

The unnormalized dot-product score tensor $S \in \mathbb{R}^{B \times H \times L \times L}$ is calculated as:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} \cdot K_{b, h, j, m} = \left( \frac{Q_h K_h^\top}{\sqrt{d_k}} \right)_{b, h, i, j}
\]

The scaling factor $\frac{1}{\sqrt{d_k}}$ counteracts the growing magnitude of inner products in higher-dimensional vector spaces ($d_k \gg 1$), preventing Softmax saturation and vanishing gradients during backward propagation.

When an additive attention mask $M \in \mathbb{R}^{B \times 1 \times L \times L}$ (or broadcastable variant) is present, it modifies score computation:

\[
\widetilde{S}_{b, h, i, j} = S_{b, h, i, j} + M_{b, 1, i, j}
\]

where $M_{b, 1, i, j} = -\infty$ for disallowed sequence positions (e.g., causal padding or future context) and $0$ otherwise.

The normalized attention map $A \in \mathbb{R}^{B \times H \times L \times L}$ is derived via row-wise Softmax:

\[
A_{b, h, i, j} = \text{Softmax}_j (\widetilde{S}_{b, h, i, j}) = \frac{\exp\left(\widetilde{S}_{b, h, i, j} - \max_k \widetilde{S}_{b, h, i, k}\right)}{\sum_{l=1}^L \exp\left(\widetilde{S}_{b, h, i, l} - \max_k \widetilde{S}_{b, h, i, k}\right)}
\]

To prevent overfitting, attention dropout with probability $p_\text{drop}$ is applied during training:

\[
\widehat{A} = \text{Dropout}(A, p_\text{drop}) = \frac{1}{1 - p_\text{drop}} A \odot R
\]

where $R \sim \text{Bernoulli}(1 - p_\text{drop})$. If a head mask $H_\text{mask} \in \mathbb{R}^{B \times H \times 1 \times 1}$ is supplied, head-level pruning is enforced:

\[
\overline{A}_{b, h, i, j} = \widehat{A}_{b, h, i, j} \cdot (H_\text{mask})_{b, h, 1, 1}
\]

The head-specific contextual output $O_h \in \mathbb{R}^{B \times H \times L \times d_k}$ is produced via matrix multiplication with value states:

\[
O_{b, h, i, m} = \sum_{j=1}^L \overline{A}_{b, h, i, j} \cdot V_{b, h, j, m} \implies O_h = \overline{A}_h V_h
\]

Finally, head outputs are concatenated and projected back to the hidden space via $W_O \in \mathbb{R}^{D \times D}$:

\[
\text{Context} = \text{Concat}(O_1, O_2, \dots, O_H) \in \mathbb{R}^{B \times L \times D}
\]
\[
Y = \text{Context} \cdot W_O \in \mathbb{R}^{B \times L \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

The complete execution path within `FullAttention` processes tensors as follows:

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, D) [W_q] ──> Q: [B, L, D] ──> view [B, L, H, d_k] ──> transpose(1, 2) ──> Q: [B, H, L, d_k]
  ├──> Linear(D, D) [W_k] ──> K: [B, L, D] ──> view [B, L, H, d_k] ──> transpose(1, 2) ──> K: [B, H, L, d_k]
  └──> Linear(D, D) [W_v] ──> V: [B, L, D] ──> view [B, L, H, d_k] ──> transpose(1, 2) ──> V: [B, H, L, d_k]
                                │
                                ├──> MatMul: Q @ K.transpose(-2, -1) ──> [B, H, L, L]
                                ├──> Scale: * (d_k ** -0.5) ──> [B, H, L, L]
                                ├──> apply_attention_mask(mask) ──> [B, H, L, L]
                                ├──> Softmax(dim=-1) ──> attn: [B, H, L, L]
                                ├──> Dropout(p) ──> [B, H, L, L]
                                ├──> Head Masking (* head_mask) ──> [B, H, L, L]
                                │
                                └──> MatMul: attn @ V ──> context: [B, H, L, d_k]
                                        │
                                        └──> transpose(1, 2) ──> [B, L, H, d_k]
                                        └──> contiguous().view([B, L, D])
                                        └──> Linear(D, D) [W_o] ──> Output: [B, L, D]
```

### Detailed Tensor Transformations:
1. `hidden_states`: $[B, L, D]$
2. `self.query(hidden_states)`: $[B, L, D]$
3. `q.view(B, L, H, d_k).transpose(1, 2)`: $[B, H, L, d_k]$
4. `k.view(B, L, H, d_k).transpose(1, 2)`: $[B, H, L, d_k]$
5. `v.view(B, L, H, d_k).transpose(1, 2)`: $[B, H, L, d_k]$
6. `k.transpose(-2, -1)`: $[B, H, d_k, L]$
7. `scores = torch.matmul(q, k.transpose(-2, -1)) * scale`: $[B, H, L, L]$
8. `apply_attention_mask(scores, mask)`: $[B, H, L, L]$
9. `attn = F.softmax(scores, dim=-1)`: $[B, H, L, L]$
10. `context = torch.matmul(attn, v)`: $[B, H, L, d_k]$
11. `context.transpose(1, 2).contiguous().view(B, L, D)`: $[B, L, D]$
12. `self.out_proj(context)`: $[B, L, D]$

---

## 4. Analytical Complexity Profiling

### FLOP and MAC Breakdown

Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H$ number of heads, and $d_k = D / H$.

1. **Input Projections ($W_Q, W_K, W_V$)**:
   - Operations: 3 linear projections of shape $[B, L, D] \times [D, D]$.
   - MACs: $3 \times B \cdot L \cdot D^2 = 3 B L D^2$
   - FLOPs: $6 B L D^2$

2. **Query-Key Score Computation ($Q K^\top$)**:
   - Matrix product $[B, H, L, d_k] \times [B, H, d_k, L]$.
   - MACs: $B \cdot H \cdot L \cdot L \cdot d_k = B L^2 D$
   - FLOPs: $2 B L^2 D$

3. **Score Scaling & Masking**:
   - FLOPs: $B H L^2$ scaling $+ B H L^2$ masking $\approx 2 B H L^2$

4. **Softmax Normalization**:
   - Max reduction, subtraction, exponentiation, sum reduction, division:
   - FLOPs: $\approx 5 B H L^2$

5. **Attention Context Aggregation ($A V$)**:
   - Matrix product $[B, H, L, L] \times [B, H, L, d_k]$.
   - MACs: $B \cdot H \cdot L \cdot L \cdot d_k = B L^2 D$
   - FLOPs: $2 B L^2 D$

6. **Output Projection ($W_O$)**:
   - Operation: $[B, L, D] \times [D, D]$.
   - MACs: $B L D^2$
   - FLOPs: $2 B L D^2$

### Total Complexity:
- **Total MACs**: $4 B L D^2 + 2 B L^2 D$
- **Total FLOPs**: $8 B L D^2 + 4 B L^2 D$

### Memory Footprint & Peak Allocations:
- **Static Weights**: $4 \times D^2 = 4 D^2$ parameters (FP32: $16 D^2$ bytes, FP16/BF16: $8 D^2$ bytes).
- **Transient Activation Memory**:
  - $Q, K, V$ states: $3 \times B \cdot L \cdot D \times \text{bytes\_per\_element}$
  - Raw Score matrix $S$: $B \cdot H \cdot L^2 \times \text{bytes\_per\_element}$
  - Softmax Attention matrix $A$: $B \cdot H \cdot L^2 \times \text{bytes\_per\_element}$
  - Context matrix: $B \cdot L \cdot D \times \text{bytes\_per\_element}$
- **Peak Activation Memory**: $\mathcal{O}(B \cdot L \cdot D + B \cdot H \cdot L^2)$ during forward pass for backpropagation gradient retention.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Quadratic Memory & Compute Scaling
The explicit generation of $A \in \mathbb{R}^{B \times H \times L \times L}$ creates an $\mathcal{O}(L^2)$ memory wall. For sequence lengths exceeding $L > 8192$, activation storage for attention maps dominates GPU DRAM allocations, inducing High Memory Bandwidth (HBM) exhaustion and out-of-memory (OOM) faults.

### Low-Rank Bottleneck (Sequence Length vs Head Dimension)
When sequence length $L$ exceeds the head dimension $d_k$ ($L > d_k$), the unnormalized score matrix $S_h = \frac{Q_h K_h^\top}{\sqrt{d_k}}$ is formed by the multiplication of two rank-$d_k$ matrices. Consequently:

\[
\text{rank}(S_h) \le \min(L, d_k) = d_k
\]

Even though $S_h \in \mathbb{R}^{L \times L}$, its maximum rank is bounded by $d_k$. Softmax application generates a full-rank matrix numerically due to the non-linear exponential operator, but the underlying manifold remains strictly constrained to a low-dimensional subspace.

### Softmax Over-Concentration & Entropy Collapse
Deep Transformer architectures often suffer from attention entropy collapse in late layers, where row distributions $A_{i, :}$ degenerate into near one-hot delta distributions. This diminishes gradient flow across sequence steps and leads to representation degeneration where distinct head channels collapse into redundant subspace projections.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Standard PyTorch Execution Inefficiencies
Standard PyTorch implementations write intermediate tensors ($Q, K, V$, raw scores $S$, masked scores $\widetilde{S}$, Softmax probabilities $A$, and output context $O$) back to high-bandwidth GPU memory (HBM) between successive kernel calls. The arithmetic intensity of standard softmax and masking operations is low ($\ll 1 \text{ FLOP/byte}$), making execution heavily **memory-bandwidth bound**.

### Triton Kernel Architecture (FlashAttention Paradigm)
To achieve peak performance, standard MHA must be optimized using fused block-tiling algorithms (Dao et al., 2022 / 2023) implemented in Triton:

```
      High-Bandwidth Memory (HBM)
     ┌──────────────────────────┐
     │   Q, K, V Tensors (HBM)  │
     └────────────┬─────────────┘
                  │ Tile Load [BLOCK_M, d_k], [BLOCK_N, d_k]
                  ▼
┌───────────────────────────────────────────────┐
│        SRAM (On-Chip L1 Scratchpad)           │
│  1. Compute Tile: S_ij = Q_i @ K_j.T / sqrt(d)│
│  2. Online Softmax Update:                     │
│     m_new = max(m_prev, max(S_ij))            │
│     l_new = l_prev * exp(m_prev - m_new) +    │
│             sum(exp(S_ij - m_new))            │
│  3. Accumulate Output Tile:                   │
│     O_i = O_i * exp(m_prev - m_new) +         │
│           P_ij @ V_j                          │
└─────────────────┬─────────────────────────────┘
                  │ Stream Write Final Tile O_i
                  ▼
     ┌──────────────────────────┐
     │   Output Tensor (HBM)    │
     └──────────────────────────┘
```

### Key Triton Implementation Strategies:
1. **Online Softmax Tiling**:
   Avoid materializing the full $L \times L$ score matrix in HBM by computing attention block-by-block using safe online Softmax updates. Track running max $m_i$ and running sum $l_i$ across blocks along sequence length $L$.
2. **SRAM Block Sizing**:
   Choose block dimensions `BLOCK_M` (e.g., 64 or 128) and `BLOCK_N` (e.g., 64 or 128) tuned for GPU SRAM capacities (e.g., 100 KB to 228 KB per Streaming Multiprocessor).
3. **Memory Access Coalescing & Vectorization**:
   Ensure contiguous memory access patterns for outer loop loads over sequence length blocks. Utilize 128-bit vector loads (`float4` / `bfloat16x8`) to saturate GPU memory bus lanes.
4. **Warp-level Synchronization**:
   Perform row-max and row-sum reductions across 32 threads using warp-shuffle primitives (`__shfl_xor_sync`), eliminating inter-thread shared memory barrier latency.
5. **Recomputation in Backward Pass**:
   Do not store the $L \times L$ attention matrix $A$ during the forward pass. Instead, recompute $S_{ij}$ and $P_{ij}$ on-the-fly in SRAM during the backward pass using stored $Q, K, V$ vectors and saved Softmax log-sum-exp vectors $L_i = m_i + \log(l_i)$, reducing activation memory from $\mathcal{O}(L^2)$ to $\mathcal{O}(L)$.