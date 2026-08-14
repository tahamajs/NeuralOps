# Block Transformer HIBA: Hierarchical Intra-Block and Inter-Block Attention with Periodic Dense Rectification

## 1. Abstract & Theoretical Motivation
Dense Multi-Head Self-Attention (MHSA) computes pairwise interaction matrices of size $L \times L$, scaling quadratically $\mathcal{O}(L^2 d)$ in time and memory. Purely localized block-sparse attention models isolate token attention within non-overlapping contiguous windows of size $B_{\text{size}}$, reducing compute to $\mathcal{O}(L \cdot B_{\text{size}} \cdot d)$. However, local block isolation prevents information propagation across distant sequence segments, causing **attention fragmentation** and loss of global context.

**Block Transformer HIBA (Hierarchical Intra-Block and Inter-Block Attention)** addresses this trade-off through a tri-part architectural design:
1. **Intra-Block Dense Attention**: Evaluates exact causal self-attention within local sequence blocks of size $B_{\text{size}}$, capturing high-frequency local syntax.
2. **Inter-Block Top-$k$ Sparse Attention**: Dynamically routes queries in block $b$ to non-local keys $K_{\neg b}$ across all remaining sequence blocks, retaining only the top-$k_{\text{inter}}$ cross-block connections.
3. **Periodic Global Dense Rectification**: Every $T_{\text{rect}}$ layers, the model executes a full, un-constrained global dense attention pass over all $L \times L$ token pairs, repairing accumulated representation fragmentation.

This hierarchical approach achieves $\mathcal{O}\left(L \cdot (B_{\text{size}} + k_{\text{inter}}) \cdot d\right)$ sub-quadratic complexity on sparse layers, maintaining strict global coherence via periodic dense rectification.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Multi-Head Projection and Spatial Partitioning
Given input sequence activations $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$, Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) projections are defined as:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_k, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_v \quad \in \mathbb{R}^{B \times H \times L \times d_h}$$
where $H$ is the number of heads and $d_h = D / H$.

The sequence is partitioned into $M = \lceil L / B_{\text{size}} \rceil$ contiguous blocks. For block $b \in \{0, \dots, M-1\}$, token index range $\mathcal{I}_b = [b \cdot B_{\text{size}}, \min((b+1) \cdot B_{\text{size}}, L))$ defines local block slices $\mathbf{Q}_b, \mathbf{K}_b, \mathbf{V}_b \in \mathbb{R}^{B \times H \times L_b \times d_h}$, where $L_b = |\mathcal{I}_b| \le B_{\text{size}}$.

### 2.2 Stage 1: Local Intra-Block Causal Attention
Within block $b$, exact causal dot-product attention computes local context:
$$\mathbf{S}_{b, \text{intra}} = \frac{\mathbf{Q}_b \mathbf{K}_b^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times L_b \times L_b}$$

A lower-triangular causal mask $\mathbf{M}_{\text{causal}} \in \{0, -\infty\}^{L_b \times L_b}$ is applied:
$$\mathbf{M}_{\text{causal}, i, j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
$$\mathbf{A}_{b, \text{intra}} = \operatorname{Softmax}\left( \mathbf{S}_{b, \text{intra}} + \mathbf{M}_{\text{causal}} \right) \in \mathbb{R}^{B \times H \times L_b \times L_b}$$
$$\mathbf{O}_{b, \text{intra}} = \mathbf{A}_{b, \text{intra}} \mathbf{V}_b \quad \in \mathbb{R}^{B \times H \times L_b \times d_h}$$

### 2.3 Stage 2: Inter-Block Top-$k$ Sparse Attention
For query block $b$, non-local key and value tokens belonging to all remaining blocks $j \neq b$ are concatenated into global neighbor matrices:
$$\mathbf{K}_{\neg b} = \operatorname{Concat}_{j \neq b} \left( \mathbf{K}_j \right) \quad \in \mathbb{R}^{B \times H \times (L - L_b) \times d_h}$$
$$\mathbf{V}_{\neg b} = \operatorname{Concat}_{j \neq b} \left( \mathbf{V}_j \right) \quad \in \mathbb{R}^{B \times H \times (L - L_b) \times d_h}$$

The inter-block affinity score matrix $\mathbf{S}_{b, \text{inter}} \in \mathbb{R}^{B \times H \times L_b \times (L - L_b)}$ is evaluated as:
$$\mathbf{S}_{b, \text{inter}} = \frac{\mathbf{Q}_b \mathbf{K}_{\neg b}^\top}{\sqrt{d_h}}$$

For each query token $i \in \{1, \dots, L_b\}$, a discrete top-$k$ routing step extracts the indices of the $k_{\text{inter}} = \min(k_{\text{inter}}, L - L_b)$ most salient non-local keys:
$$\mathcal{T}_{b, i} = \operatorname{TopK}\left( \mathbf{S}_{b, \text{inter}, :, :, i, :}, k_{\text{inter}} \right)$$

Non-top-$k$ scores are masked out with $-\infty$:
$$\tilde{\mathbf{S}}_{b, \text{inter}, :, :, i, j} = \begin{cases} \mathbf{S}_{b, \text{inter}, :, :, i, j} & \text{if } j \in \mathcal{T}_{b, i} \\ -\infty & \text{otherwise} \end{cases}$$
$$\mathbf{A}_{b, \text{inter}} = \operatorname{Softmax}\left( \tilde{\mathbf{S}}_{b, \text{inter}} \right) \in \mathbb{R}^{B \times H \times L_b \times (L - L_b)}$$
$$\mathbf{O}_{b, \text{inter}} = \mathbf{A}_{b, \text{inter}} \mathbf{V}_{\neg b} \quad \in \mathbb{R}^{B \times H \times L_b \times d_h}$$

The composite sparse output for block $b$ sums local and non-local contexts:
$$\mathbf{O}_b = \mathbf{O}_{b, \text{intra}} + \mathbf{O}_{b, \text{inter}} \quad \in \mathbb{R}^{B \times H \times L_b \times d_h}$$

The individual block outputs $\mathbf{O}_b$ are concatenated along the sequence axis:
$$\mathbf{O}_{\text{sparse}} = \operatorname{Concat}_{b=0}^{M-1} \left( \mathbf{O}_b \right) \quad \in \mathbb{R}^{B \times H \times L \times d_h}$$

### 2.4 Stage 3: Periodic Global Dense Rectification
Let $\ell = \text{layer\_idx}$ denote the layer depth counter. If $\ell \equiv 0 \pmod{T_{\text{rect}}}$ (where $T_{\text{rect}} = \text{rectification\_interval}$), the block-sparse steps are bypassed in favor of a full dense attention pass:
$$\mathbf{S}_{\text{dense}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}} + \mathbf{M}_{\text{attn}} \quad \in \mathbb{R}^{B \times H \times L \times L}$$
$$\mathbf{O}_{\text{rectified}} = \operatorname{Softmax}\left( \mathbf{S}_{\text{dense}} \right) \mathbf{V} \quad \in \mathbb{R}^{B \times H \times L \times d_h}$$

### 2.5 Final Output Projection
The selected context $\mathbf{O} \in \{\mathbf{O}_{\text{sparse}}, \mathbf{O}_{\text{rectified}}\}$ is reshaped to $\mathbb{R}^{B \times L \times D}$ and projected:
$$\mathbf{Y} = \operatorname{Reshape}(\mathbf{O}) \mathbf{W}_O \quad \in \mathbb{R}^{B \times L \times D}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Step | Operation | Input Shape | Output Shape | Tensor Transformation Description |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Projections | `[B, L, D]` | `[B, H, L, d_h]` | Linear maps via $W_q, W_k, W_v$ & split heads |
| 2 | Layer Mode Check | `layer_idx` | `bool` | Check if `layer_idx % rectification_interval == 0` |
| 3a | Dense Rectify Pass | `Q`, `K`, `V` | `[B, H, L, d_h]` | Full dense matmul $\text{Softmax}(Q K^T / \sqrt{d_h}) V$ |
| 3b | Block Partition | `Q`, `K`, `V` | $M \times$ `[B, H, L_b, d_h]`| Slice sequence into $M$ blocks of size $B_{\text{size}}$ |
| 4 | Intra-Block Attn | `Q_b`, `K_b`, `V_b` | `[B, H, L_b, d_h]` | Causal dense attention within block slice |
| 5 | Inter-Block Extract | `K`, `V`, `b` | `[B, H, L-L_b, d_h]`| Concatenate out-of-block keys & values $K_{\neg b}, V_{\neg b}$ |
| 6 | Inter-Block Top-$k$ | `Q_b`, `K_other` | `[B, H, L_b, top_k]`| Extract top-$k_{\text{inter}}$ scores & mask non-top-$k$ to $-\infty$ |
| 7 | Inter-Block Context | `Scores`, `V_other` | `[B, H, L_b, d_h]` | Softmax attention weight matmul with $V_{\neg b}$ |
| 8 | Add Block Contexts | `Intra`, `Inter` | `[B, H, L_b, d_h]` | `O_b = intra_out + inter_out` per block |
| 9 | Reassemble Sequence | $M \times$ `O_b` | `[B, H, L, d_h]` | Concatenate block outputs along sequence axis |
| 10| Output Projection | `[B, L, D]` | `[B, L, D]` | Reshape multi-head context & map via $W_O$ |

```
                       [B, L, D] Input Tensor
                           │
                           ▼
                  Q, K, V Projections [B, H, L, d_h]
                           │
        ┌──────────────────┴──────────────────┐
        │ layer_idx % interval == 0           │ layer_idx % interval != 0
        ▼                                     ▼
Full Dense Global Attention          HIBA Block-Sparse Partition
Softmax(Q K^T / sqrt(d_h)) V         Sequence split into M Blocks
  Shape: [B, H, L, d_h]                       │
        │                                     ▼
        │                    Iterate over Blocks b (0..M-1)
        │                    ┌────────────────┼────────────────┐
        │                    ▼                                 ▼
        │           Intra-Block Causal Attn          Inter-Block Top-k Attn
        │           Softmax(Q_b K_b^T) V_b           Q_b @ K_other^T -> Top-k
        │            Shape: [B, H, L_b, d_h]          Shape: [B, H, L_b, d_h]
        │                    │                                 │
        │                    └────────────────┬────────────────┘
        │                                     ▼
        │                           Sum: O_b = Intra + Inter
        │                                     │
        │                                     ▼
        │                           Concat Blocks [B, H, L, d_h]
        └──────────────────┬──────────────────┘
                           ▼
             Reshape & Linear Projection W_O
                           │
                           ▼
                      [B, L, D] Output
```

---

## 4. Analytical Complexity Profiling

### 4.1 FLOPs Computation Breakdown
Let $B$ be batch size, $L$ sequence length, $D$ hidden dimension, $H$ heads, $d_h = D/H$, $B_{\text{size}}$ block size, $M = L / B_{\text{size}}$, $k_{\text{inter}}$ top-$k$ inter-block tokens, and $T_{\text{rect}}$ rectification interval.

1. **Linear Projections ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$)**:
   $$\operatorname{FLOPs}_{\text{proj}} = 3 \times 2 B L D^2 = 6 B L D^2$$
2. **Sparse HIBA Layers (Fraction $\frac{T_{\text{rect}} - 1}{T_{\text{rect}}}$ of layers)**:
   - **Intra-Block Causal Attention**:
     Across $M$ blocks: $M \times (4 B H B_{\text{size}}^2 d_h) = 4 B H L B_{\text{size}} d_h = 4 B D L B_{\text{size}}$ FLOPs.
   - **Inter-Block Top-$k$ Attention**:
     Across $M$ blocks: $M \times (2 B H B_{\text{size}} (L - B_{\text{size}}) d_h) = 2 B D L (L - B_{\text{size}})$ FLOPs for raw scoring, plus $2 B D L k_{\text{inter}}$ FLOPs for top-$k$ value multiplication.
     $$\operatorname{FLOPs}_{\text{sparse\_layer}} = 4 B D L B_{\text{size}} + 2 B D L^2 + 2 B D L k_{\text{inter}}$$
3. **Dense Rectification Layers (Fraction $\frac{1}{T_{\text{rect}}}$ of layers)**:
   $$\operatorname{FLOPs}_{\text{dense\_layer}} = 4 B H L^2 d_h = 4 B D L^2$$
4. **Output Projection ($\mathbf{W}_O$)**:
   $$\operatorname{FLOPs}_{\text{out}} = 2 B L D^2$$

$$\text{Average Per-Layer FLOPs} = 8 B L D^2 + 4 B D L B_{\text{size}} + 2 B D L k_{\text{inter}} + \left(2 + \frac{2}{T_{\text{rect}}}\right) B D L^2$$

### 4.2 MACs & Memory Footprint Analysis
- **Multiply-Accumulate Operations (MACs)**:
  $$\text{Average MACs} \approx 4 B L D^2 + 2 B D L B_{\text{size}} + B D L k_{\text{inter}} + \left(1 + \frac{1}{T_{\text{rect}}}\right) B D L^2$$
- **Peak Activation Memory**:
  - Sparse Layer Attention Score Tensors: $M \times B \cdot H \cdot B_{\text{size}} \cdot (L - B_{\text{size}}) \times 2 \text{ bytes}$
  - Dense Rectification Layer Score Tensor: $B \cdot H \cdot L \cdot L \times 2 \text{ bytes}$
  - Peak Memory: Bounded by global dense rectification layer $\mathcal{O}(B H L^2)$.

### 4.3 Asymptotic Scalability Comparison
- **Standard Dense Attention**: $\mathcal{O}(L^2 D)$ compute, $\mathcal{O}(L^2 H)$ memory.
- **HIBA Sparse Layers**: $\mathcal{O}\left(L D (D + B_{\text{size}} + k_{\text{inter}} + L)\right)$ compute.
- *Note*: Because `_inter_block_sparse` computes raw scores $\mathbf{Q}_b \mathbf{K}_{\neg b}^\top$ over all $(L - B_{\text{size}})$ non-local keys before extracting top-$k$, the scoring step still scales quadratically with $L$, but with a reduced top-$k$ value matmul stage.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 The Inter-Block Score Matrix Memory Leak
In `_inter_block_sparse`:
```python
scores = torch.matmul(q_b, k_neighbors.transpose(-2, -1)) * scale # [B, H, L_b, L - L_b]
top_k = min(self.top_k_inter, k_neighbors.size(2))
_, topk_idx = torch.topk(scores, top_k, dim=-1)
mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
scores = scores.masked_fill(mask == 0, float("-inf"))
```
- **Quadratic Scoring Memory**: The kernel computes an explicit score tensor of size `[B, H, L_b, L - L_b]` and instantiates a dense zero mask `mask = torch.zeros_like(scores)`.
- **VRAM Spikes**: Creating a dense floating-point mask of shape `[B, H, L_b, L - L_b]` per block allocates as much VRAM as full attention ($\mathcal{O}(L^2)$), negating memory savings during sparse layers.

### 5.2 Layer Counter Instance Desynchronization
In `BlockTransformerHIBA.forward`:
```python
use_rectify = self.layer_idx % self.rectification_interval == 0
self.layer_idx += 1
```
- **Independent State Trap**: `self.layer_idx` is stored as an instance attribute within each individual layer module (`self.layer_idx = 0`).
- **Schedule Failure**: Every layer module maintains its own isolated counter (`0, 1, 2...`). Consequently, **every layer in the model executes dense rectification on its very first forward call** because `0 % T_rect == 0`. Subsequent calls increment each layer independently, breaking synchronized global rectification schedules across Transformer depth.

### 5.3 Non-Causal Inter-Block Leakage in Autoregressive Tasks
In `_inter_block_sparse`:
```python
if (block_idx + 1) * self.block_size < total_len:
    k_select.append(k_all[:, :, end_b:, :]) # FEEDS FUTURE KEYS!
```
- **Future Token Leakage**: The inter-block attention routine appends keys from future blocks (`end_b:`).
- **Causality Violation**: If deployed in autoregressive decoder models (e.g., GPT), query block $b$ attends to future keys $k > b$, leaking future token information and invalidating generative inference.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

To make HIBA truly sub-quadratic and eliminate quadratic VRAM allocations, inter-block top-$k$ scoring must use coarse block centroids instead of per-token dense comparisons.

### 6.1 Triton Hardware Acceleration Pipeline Layout

```
+-----------------------------------------------------------------------------------+
|                        TRITON FUSED HIBA PIPELINE                                 |
+-----------------------------------------------------------------------------------+
|  1. COARSE INTER-BLOCK CENTROID ROUTING KERNEL                                    |
|     - Computes block centroids Q_mean, K_mean [M, d_h].                          |
|     - Evaluates coarse inter-block score matrix S_block [M, M].                   |
|     - Extracts top-k block indices TopK_blocks [M, k_blk].                        |
|                                                                                   |
|  2. FUSED INTRA-BLOCK FLASHATTENTION KERNEL                                       |
|     - Computes exact causal FlashAttention strictly within [B_size, B_size] tile. |
|     - Writes local context O_intra [B, L, D] without global matrix allocation.    |
|                                                                                   |
|  3. FUSED INTER-BLOCK ROUTED FLASHATTENTION KERNEL                                |
|     - Loads Q_b [B_size, d_h].                                                    |
|     - Indirect loads ONLY the selected k_blk key/value blocks into SRAM.          |
|     - Evaluates online softmax context O_inter over selected candidate blocks.    |
|     - Combines O_final = O_intra + O_inter in SRAM registers.                    |
+-----------------------------------------------------------------------------------+
```

### 6.2 Implementation Code Blueprint (Triton Fused Inter-Block Kernel)

```python
import triton
import triton.language as tl

@triton.jit
def _hiba_fused_sparse_inter_kernel(
    Q_ptr, K_ptr, V_ptr, TopK_Block_Idx_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_idx_b, stride_idx_h, stride_idx_m, stride_idx_k,
    scale,
    B_SIZE: tl.constexpr,
    D_K: tl.constexpr,
    K_BLOCKS: tl.constexpr
):
    block_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=2)

    # Calculate token index range for current block
    start_q = block_id * B_SIZE
    offs_m = start_q + tl.arange(0, B_SIZE)
    offs_d = tl.arange(0, D_K)

    # Load Query block tile [B_SIZE, D_K] into SRAM
    q_ptrs = Q_ptr + batch_id * stride_qb + head_id * stride_qh + (offs_m[:, None] * stride_qn + offs_d[None, :])
    q = tl.load(q_ptrs)

    # Online Softmax Accumulators
    m_i = tl.full([B_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([B_SIZE], dtype=tl.float32)
    acc = tl.zeros([B_SIZE, D_K], dtype=tl.float32)

    idx_base = TopK_Block_Idx_ptr + batch_id * stride_idx_b + head_id * stride_idx_h + block_id * stride_idx_m

    # Loop ONLY over routed top-k inter-block indices
    for k_idx in range(K_BLOCKS):
        target_block = tl.load(idx_base + k_idx * stride_idx_k)
        start_k = target_block * B_SIZE
        offs_n = start_k + tl.arange(0, B_SIZE)

        # Indirect load selected Key block tile [B_SIZE, D_K]
        k_ptrs = K_ptr + batch_id * stride_kb + head_id * stride_kh + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs)

        # Matmul S = Q @ K.T -> [B_SIZE, B_SIZE]
        s = tl.dot(q, k) * scale

        # Online Softmax update
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)

        acc_scale = tl.exp(m_i - m_new)
        acc = acc * acc_scale[:, None]

        # Indirect load selected Value block tile [B_SIZE, D_K]
        v_ptrs = V_ptr + batch_id * stride_vb + head_id * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs)
        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_new
        m_i = m_new

    # Store normalized inter-block context
    acc = acc / l_i[:, None]
    out_ptrs = Out_ptr + batch_id * stride_ob + head_id * stride_oh + (offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty))
```

### 6.3 Hardware Optimization Impact
1. **Elimination of Dense Inter-Block Masks**: Centroid block routing eliminates the allocation of $[L_b, L - L_b]$ dense score matrices and masks, dropping memory usage to $\mathcal{O}(L \cdot B_{\text{size}})$.
2. **Synchronized Global Rectification**: Explicit layer depth passing (`layer_idx`) ensures dense rectification passes occur concurrently across all layers at exact intervals $T_{\text{rect}}$.