# ASVD Attention: Activation-Aware Singular Value Decomposition Architecture

## 1. Executive Abstract & Architectural Motivation

Activation-aware Singular Value Decomposition Attention (`ASVDAttention`, Yuan et al., 2023) provides an activation-guided low-rank model compression framework for Multi-Head Attention. Standard SVD parameter reduction factorizes dense projection matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ solely based on static weight magnitude matrices, ignoring the non-uniform variance and magnitude distributions of dynamic activation vectors ($X$).

ASVD reformulates factorized linear layers by incorporating explicit singular value scaling vectors $S \in \mathbb{R}^r$ alongside orthogonal left and right singular bases ($U \in \mathbb{R}^{d_{\text{out}} \times r}, V \in \mathbb{R}^{r \times d_{\text{in}}}$). By computing linear transformations as $y = ((x V^\top) \odot S) U^\top + b$ (`ASVDLinear`), ASVD allows individual tuning of singular value scales during post-compression fine-tuning, preserving sensitivity to critical activation channels while reducing parameter footprints by over 80%.

---

## 2. Mathematical Formulations & Derivations

Let $X \in \mathbb{R}^{B \times N \times D}$ denote input activations. Let $H$ be query heads, $d_k = D / H$ head dimension, and $r$ the rank per ASVD factorized layer.

### Factorised ASVD Linear Transformation (`ASVDLinear`):
For dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with bias $b \in \mathbb{R}^{d_{\text{out}}}$, truncated SVD factorizes $W$:

\[
W \approx U \text{diag}(S) V \implies U \in \mathbb{R}^{d_{\text{out}} \times r}, \, S \in \mathbb{R}^r, \, V \in \mathbb{R}^{r \times d_{\text{in}}}
\]

Where $U$ and $V$ are semi-orthogonal matrices ($U^\top U = I_r, V V^\top = I_r$).

For input activation tensor $X \in \mathbb{R}^{B \times N \times d_{\text{in}}}$:

1. **Down-Projection to Rank-r Bottleneck**:
   \[
   Z_{\text{down}} = X V^\top \in \mathbb{R}^{B \times N \times r}
   \]
2. **Explicit Singular Value Scaling**:
   \[
   Z_{\text{scaled}} = Z_{\text{down}} \odot S \in \mathbb{R}^{B \times N \times r}
   \]
3. **Up-Projection & Bias Aggregation**:
   \[
   Y = Z_{\text{scaled}} U^\top + b \in \mathbb{R}^{B \times N \times d_{\text{out}}}
   \]

### Multi-Head ASVD Scaled Attention:
ASVD factorized layers map hidden states to Query, Key, Value, and Output multi-head tensors:

\[
Q = \text{ASVDLinear}_Q(X) \in \mathbb{R}^{B \times H \times N \times d_k}
\]
\[
K = \text{ASVDLinear}_K(X) \in \mathbb{R}^{B \times H \times N \times d_k}
\]
\[
V = \text{ASVDLinear}_V(X) \in \mathbb{R}^{B \times H \times N \times d_k}
\]

Affinity matrix evaluation and context output projection proceed as:

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
Y = \text{ASVDLinear}_O(C) \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Step-by-Step Tensor Execution Trace

### Tensor Shape Transformation Table

| Execution Stage | Input Shape | Transformation / Operation | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input** | `[B, N, D]` | Input activation batch | `[B, N, D]` |
| **Q Down Projection** | `[B, N, D]` | `F.linear(X, query.V)` | `[B, N, r]` |
| **Q Singular Scaling** | `[B, N, r]` | `Vx * query.S` | `[B, N, r]` |
| **Q Up Projection** | `[B, N, r]` | `F.linear(SVx, query.U) + bias` | `[B, N, H*d_k]` |
| **Multi-Head Transpose** | `[B, N, H*d_k]` | `.view(B, N, H, d_k).transpose(1, 2)` | `[B, H, N, d_k]` each |
| **Score Matrix MatMul** | `Q`, `K` | `torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)` | `[B, H, N, N]` |
| **Mask & Softmax** | `scores` | `apply_attention_mask` + `F.softmax` | `[B, H, N, N]` |
| **Value Aggregation** | `attn`, `V` | `torch.matmul(attn, V)` | `[B, H, N, d_k]` |
| **Out Down & Scale** | `context: [B, N, D]`| `F.linear(context, out.V) * out.S` | `[B, N, r]` |
| **Out Up Projection** | `[B, N, r]` | `F.linear(SVx, out.U) + bias` | `Output: [B, N, D]` |

### Architectural Flowchart Diagram

```
                        Input: hidden_states [B, N, D]
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
      ASVDLinear_Q                 ASVDLinear_K                  ASVDLinear_V
X->V_q->scale(S_q)->U_q       X->V_k->scale(S_k)->U_k       X->V_v->scale(S_v)->U_v
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
                                           ASVDLinear_O
                                     C->V_o->scale(S_o)->U_o
                                                │
                                        Output: [B, N, D]
```

---

## 4. Analytical Complexity & Hardware Profiling

### Computational Workload (FLOPs & MACs)
Let $B$ be batch size, $N$ sequence length, $D$ model dimension, $H$ heads, $d_k = D/H$, and $r$ factor rank.

1. **ASVD Projection Layers**:
   - Down-projections ($X V^\top$): $4 \times (2 B N D r) = 8 B N D r$ FLOPs.
   - Singular scaling ($\odot S$): $4 B N r$ FLOPs.
   - Up-projections ($Z U^\top$): $4 \times (2 B N r D) = 8 B N D r$ FLOPs.
   - *Total Projection Workload*: $16 B N D r + 4 B N r$ FLOPs ($8 B N D r$ MACs).

2. **Core Scaled Dot-Product Attention**:
   - $Q K^\top$ & $A V$: $4 B N^2 D$ FLOPs.
   - Softmax & Masking: $7 B H N^2$ FLOPs.

3. **Total Workload Summary**:
   - **Total MACs**: $8 B N D r + 2 B N^2 D$
   - **Total FLOPs**: $16 B N D r + 4 B N r + 4 B N^2 D + 7 B H N^2$

### Parameter Footprint Comparison
- **Dense Uncompressed MHA**: $4 D^2 + 4 D$ floats.
- **ASVD MHA Parameter Count**:
  \[
  \text{Params}_{\text{ASVD}} = 4 \cdot [D \cdot r + r + r \cdot D] + 4 D = 8 D r + 4 r + 4 D \text{ floats}
  \]
  For $D = 4096, r = 32$:
  - Dense MHA: $4 \times 4096^2 \approx 67.1 \text{ million parameters}$.
  - ASVD MHA: $8 \times 4096 \times 32 + 128 + 16384 \approx 1.06 \text{ million parameters}$ (**$98.4\%$ parameter compression**).

---

## 5. Architectural Bottlenecks, Rank Collapse & Edge Cases

### Activation Outlier Shift Sensitivity
If activation scales shift significantly during task fine-tuning, fixed singular values $S$ derived from initial calibration can cause feature clipping. Allowing parameter vector $S$ to be trainable during fine-tuning resolves activation mismatch.

### Subnormal Singular Value Scaling
If $S_k \to 0$ for channel $k$, gradient flow into corresponding rows of $V$ and columns of $U$ drops to zero. Adding lower bound regularizers ($S \ge 1e-4$) prevents singular value collapse.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton kernel for ASVD Attention fuses down-projection ($V$), singular value Hadamard scaling ($S$), and up-projection ($U$) into a single SIMD thread loop.

```
                    Global Memory (HBM)
   ┌─────────────────────────────────────────────────────────────┐
   │ Input: X [B, N, D]                                          │
   │ ASVD Factors: V [r, D], S [r], U [D, r], bias [D]            │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Vectorized Block Load
                                  ▼
 ┌────────────────────────────────────────────────────────────────┐
 │                     Triton SRAM Scratchpad                     │
 │ 1. Load Activation Tile X_i [BLOCK_M, D]                       │
 │ 2. Compute Down-Projection: Vx_i = X_i @ V.T [BLOCK_M, r]      │
 │ 3. Apply In-Register Scaling: SVx_i = Vx_i * S                 │
 │ 4. Reconstruct Output Tile: Out_i = SVx_i @ U.T + bias         │
 │ 5. Execute Online Scaled Attention                             │
 └────────────────────────────────┬───────────────────────────────┘
                                  │ Write Context Output
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Context Output: Y [B, N, D]                                 │
   └─────────────────────────────────────────────────────────────┘
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Activation-Aware Fisher Information Weighting**: Weight initial singular value extraction $S_i$ using diagonal Fisher Information metrics calculated over calibration datasets.
2. **ASVD + Grouped-Query Attention (ASVD-GQA)**: Combine ASVD factorized projections with Grouped-Query KV head sharing for maximum decoding throughput.
3. **Quantized ASVD (Q-ASVD)**: Quantize orthogonal bases $U, V$ to INT4 precision while keeping scalar vector $S$ in FP16, achieving up to $32\times$ memory bandwidth compression.