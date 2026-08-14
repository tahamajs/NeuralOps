# Weighted Grouped-Query Attention (WGQA): Learnable Per-Head Importance-Scaled Shared KV Subspaces

## 1. Executive Abstract & Architectural Motivation

Weighted Grouped-Query Attention (WGQA) extends standard Grouped-Query Attention (GQA) by introducing learnable scalar importance parameters per Key and Value head prior to query-group expansion. Standard GQA enforces equal weight across all shared Key-Value heads, forcing query heads within a group to consume identical unweighted representations. This strict equality can restrict head specialization across multi-query groups.

WGQA overcomes this bottleneck by attaching learnable per-channel importance vectors $w_K, w_V \in \mathbb{R}^{H_{KV}}$ to the compressed Key and Value projections. By dynamically scaling the magnitude of each KV head before broadcast expansion (`repeat_interleave`), WGQA allows the model to learn fine-grained head prioritization with zero overhead to the autoregressive KV cache memory footprint.

---

## 2. Mathematical Formulations & Derivations

Let $X \in \mathbb{R}^{B \times N \times D}$ denote input activations. Let $H_Q$ be query heads, $H_{KV}$ key-value heads ($H_{KV} \le H_Q$), and $d_k = D / H_Q$. The expansion ratio is $R = H_Q / H_{KV}$.

### Linear Subspace Projections:
Input states are projected into query space and compressed key-value space:

\[
Q = X W_Q \in \mathbb{R}^{B \times N \times (H_Q \cdot d_k)}
\]
\[
K = X W_K \in \mathbb{R}^{B \times N \times (H_{KV} \cdot d_k)}, \quad V = X W_V \in \mathbb{R}^{B \times N \times (H_{KV} \cdot d_k)}
\]

Reshaping into multi-head layouts yields $Q \in \mathbb{R}^{B \times N \times H_Q \times d_k}$ and $K, V \in \mathbb{R}^{B \times N \times H_{KV} \times d_k}$.

### Learnable Per-KV-Head Importance Scaling:
WGQA introduces learnable parameter vectors $w_K \in \mathbb{R}^{H_{KV} \times 1}$ and $w_V \in \mathbb{R}^{H_{KV} \times 1}$ initialized to ones:

\[
\widetilde{K}_{b, n, h, m} = w_{K, h} \cdot K_{b, n, h, m}, \quad \forall h \in \{1, \dots, H_{KV}\}
\]
\[
\widetilde{V}_{b, n, h, m} = w_{V, h} \cdot V_{b, n, h, m}, \quad \forall h \in \{1, \dots, H_{KV}\}
\]

### Group Expansion & Transposition:
The scaled KV heads are expanded via `repeat_interleave` across factor $R$:

\[
K_{\text{exp}} = \text{repeat\_interleave}(\widetilde{K}, R, \text{dim}=\text{head}) \in \mathbb{R}^{B \times N \times H_Q \times d_k}
\]
\[
V_{\text{exp}} = \text{repeat\_interleave}(\widetilde{V}, R, \text{dim}=\text{head}) \in \mathbb{R}^{B \times N \times H_Q \times d_k}
\]

Transposing layout to standard multi-head indexing:

\[
Q \in \mathbb{R}^{B \times H_Q \times N \times d_k}, \quad K_{\text{exp}} \in \mathbb{R}^{B \times H_Q \times N \times d_k}, \quad V_{\text{exp}} \in \mathbb{R}^{B \times H_Q \times N \times d_k}
\]

### Scaled Dot-Product Attention & Output Synthesis:
Attention affinity matrices and value aggregation are computed as:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} \cdot (K_{\text{exp}})_{b, h, j, m}
\]
\[
A_{b, h, i, j} = \text{Softmax}_j (S_{b, h, i, j} + M_{b, 1, i, j})
\]
\[
O_{b, h, i, m} = \sum_{j=1}^N \text{Dropout}(A)_{b, h, i, j} \cdot (V_{\text{exp}})_{b, h, j, m}
\]
\[
Y = \text{Concat}(O_1, \dots, O_{H_Q}) W_O \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Step-by-Step Tensor Execution Trace

### Tensor Shape Transformation Table

| Execution Stage | Input Shape | Transformation / Operation | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input** | `[B, N, D]` | Input activation batch | `[B, N, D]` |
| **Linear Projections** | `[B, N, D]` | `self.q_proj`, `self.k_proj`, `self.v_proj` | `Q: [B, N, H_Q, d_k]`, `K, V: [B, N, H_KV, d_k]` |
| **Importance Scaling** | `K, V: [B, N, H_KV, d_k]` | `K * w_K.view(1, 1, H_KV, 1)`, `V * w_V.view(1, 1, H_KV, 1)` | `[B, N, H_KV, d_k]` each |
| **Group Interleaving** | `[B, N, H_KV, d_k]` | `repeat_interleave(R, dim=2)` | `[B, N, H_Q, d_k]` each |
| **Transpose Multi-Head** | `[B, N, H_Q, d_k]` | `.transpose(1, 2)` | `[B, H_Q, N, d_k]` each |
| **Score Matrix MatMul** | `Q`, `K_exp` | `torch.matmul(q, k_exp.transpose(-2, -1)) / sqrt(d_k)` | `[B, H_Q, N, N]` |
| **Mask & Softmax** | `scores` | `scores + attention_mask.unsqueeze(1)` + `softmax` | `[B, H_Q, N, N]` |
| **Value Aggregation** | `attn`, `V_exp` | `torch.matmul(attn, v_exp)` | `[B, H_Q, N, d_k]` |
| **Concat & Out Projection** | `context` | `.transpose(1, 2).view(B, N, D)` + `self.out_proj` | `[B, N, D]` |

### Architectural Flowchart Diagram

```
                         Input: hidden_states [B, N, D]
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   Linear(D, H_Q*d_k) [Q]       Linear(D, H_KV*d_k) [K]       Linear(D, H_KV*d_k) [V]
         │                             │                             │
   Q: [B, N, H_Q, d_k]           K: [B, N, H_KV, d_k]          V: [B, N, H_KV, d_k]
         │                             │                             │
         │                   Multiply by w_K [H_KV, 1]     Multiply by w_V [H_KV, 1]
         │                             │                             │
         │                    repeat_interleave(R)          repeat_interleave(R)
         │                             │                             │
   transpose(1, 2)              transpose(1, 2)               transpose(1, 2)
         │                             │                             │
   Q: [B, H_Q, N, d_k]         K_exp: [B, H_Q, N, d_k]       V_exp: [B, H_Q, N, d_k]
         │                             │                             │
         └──────────────┬──────────────┘                             │
                        ▼                                            │
           MatMul: Q @ K_exp.transpose(-2, -1)                       │
                        │                                            │
               Scale by 1/sqrt(d_k)                                  │
                        │                                            │
            Apply Attention Mask & Softmax                           │
                        │                                            │
               attn: [B, H_Q, N, N]                                  │
                        │                                            │
                        └─────────────────────┬──────────────────────┘
                                              ▼
                                   MatMul: attn @ V_exp
                                              │
                                   context: [B, H_Q, N, d_k]
                                              │
                                   view(B, N, D) & out_proj
                                              │
                                      Output: [B, N, D]
```

---

## 4. Analytical Complexity & Hardware Profiling

### Computational Workload (FLOPs & MACs)
For batch size $B$, sequence length $N$, hidden dimension $D$, query heads $H_Q$, KV heads $H_{KV}$, and factor $R = H_Q / H_{KV}$:

1. **Linear Projections**:
   - $W_Q$: $2 B N D^2$ FLOPs.
   - $W_K, W_V$: $\frac{4}{R} B N D^2$ FLOPs.
   - $W_O$: $2 B N D^2$ FLOPs.
2. **Importance Scaling**:
   - $w_K, w_V$ Hadamard scaling: $2 B N H_{KV} d_k = \frac{2}{R} B N D$ FLOPs (negligible).
3. **Core Attention Computation**:
   - $Q K_{\text{exp}}^\top$ & $A V_{\text{exp}}$: $4 B N^2 D$ FLOPs.
   - Softmax & Masking: $7 B H_Q N^2$ FLOPs.

4. **Total Workload Summary**:
   - **Total MACs**: $2 \left(1 + \frac{1}{R}\right) B N D^2 + 2 B N^2 D$
   - **Total FLOPs**: $4 \left(1 + \frac{1}{R}\right) B N D^2 + 4 B N^2 D + 7 B H_Q N^2 + \frac{2}{R} B N D$

### Memory Footprint & KV Cache Analysis
- **Parameter Count**: $D^2 \left(2 + \frac{2}{R}\right) + 2 H_{KV}$ (only $2 H_{KV}$ extra parameters over standard GQA).
- **Autoregressive KV Cache Size**: Unchanged from standard GQA:
  \[
  \text{KV Cache Size}_{\text{WGQA}} = 2 \cdot B \cdot H_{KV} \cdot N \cdot d_k \cdot b = \frac{2 B N D \cdot b}{R} \text{ bytes}
  \]

---

## 5. Architectural Bottlenecks, Rank Collapse & Edge Cases

### Weight Collapse & Degenerate Head Pruning
If learnable weight $w_{K, h}$ decays to zero ($\lim w_{K, h} \to 0$), Key head $h$ is entirely muted, causing score collapse across all query heads assigned to group $h$. Regularization or lower bound constraints ($w_K \ge \epsilon$) prevent premature head pruning.

### Gradient Instability in High Precision Ratios
Because $w_K, w_V$ directly scale key and value norms prior to Softmax exponentiation, large gradients flowing into $w_K$ during initial training phases can cause rapid shifts in Softmax attention entropy.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

The Triton kernel fuses scalar importance scaling ($w_K, w_V$) directly into SRAM tile loading, eliminating additional memory writes.

```
                    Global Memory (HBM)
   ┌─────────────────────────────────────────────────────────────┐
   │ Q: [B, H_Q, N, d_k]                                         │
   │ K: [B, H_KV, N, d_k], V: [B, H_KV, N, d_k]                   │
   │ w_K: [H_KV], w_V: [H_KV]                                    │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Direct In-SRAM Scaled Tile Load
                                  ▼
 ┌────────────────────────────────────────────────────────────────┐
 │                     Triton SRAM Scratchpad                     │
 │ 1. Load Query Tile Q_i [BLOCK_M, d_k]                          │
 │ 2. Load Key Tile K_j [BLOCK_N, d_k]                            │
 │ 3. Fuse Weight: K_j_scaled = K_j * w_K[kv_head]               │
 │ 4. Compute Raw Score: S_ij = Q_i @ K_j_scaled.T / sqrt(d_k)    │
 │ 5. Online Softmax & Accumulate V_j * w_V[kv_head]             │
 └────────────────────────────────┬───────────────────────────────┘
                                  │ Write Context Output
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Context Output: O [B, H_Q, N, d_k]                          │
   └─────────────────────────────────────────────────────────────┘
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Per-Channel Temperature Gating**: Expand $w_K$ from scalar vector $[H_{KV}, 1]$ to full matrix $[H_{KV}, d_k]$ for fine-grained per-frequency channel weighting.
2. **Dynamic GQA Weighting**: Condition $w_K, w_V$ on sequence representations via lightweight MLP routing for input-dependent capacity allocation.
3. **Sparse WGQA Pruning**: Zero out heads where $|w_{K, h}| < \theta$ during inference to dynamically drop uninformative KV heads and further compress KV cache.