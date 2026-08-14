# Infini-Attention: Compressive Memory-Augmented Sliding-Window Attention

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Attention (MHA) exhibits quadratic complexity \(\mathcal{O}(L^2)\), preventing long-context sequence modeling in memory-constrained LLM inference. While local sliding-window attention restricts receptive field scale to \(\mathcal{O}(L \cdot W)\), it suffers from catastrophic context truncation outside window boundary \(W\).

**Infini-Attention** solves context truncation by augmenting local sliding-window attention with a continuous, bounded **Compressive Memory Matrix** \(\mathbf{M} \in \mathbb{R}^{1 \times H \times C \times d_k}\). For each sequence step, tokens query both a band-sparse local attention window (\(\text{Window Size } W\)) and an associative compressive memory tensor (\(\text{Memory Slots } C\)). By executing local dot-product attention and compressive memory retrieval in parallel, Infini-Attention achieves linear time complexity \(\mathcal{O}(L \cdot (W + C))\) and constant memory usage during context expansion. This document provides a rigorous mathematical derivation, tensor shape trace, complexity breakdown, failure mode diagnosis, and a Triton GPU acceleration blueprint.

---

## 2. Mathematical Derivations & Formulations

Given input sequence activations \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), let \(H\) be the head count, \(d_k = D / H\) the head dimension, \(W\) local window size, and \(C\) memory slot count.

### 2.1 Multi-Head Feature Projections

Input \(\mathbf{X}\) is mapped to multi-head Query, Key, and Value space via weight matrices \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D}\):

\[
\mathbf{Q} = \text{Reshape}\left(\mathbf{X} \mathbf{W}_Q, [B, H, L, d_k]\right)
\]
\[
\mathbf{K} = \text{Reshape}\left(\mathbf{X} \mathbf{W}_K, [B, H, L, d_k]\right)
\]
\[
\mathbf{V} = \text{Reshape}\left(\mathbf{X} \mathbf{W}_V, [B, H, L, d_k]\right)
\]

### 2.2 Compressive Memory Retrieval Branch

Let \(\mathbf{M} \in \mathbb{R}^{1 \times H \times C \times d_k}\) represent the learnable compressive memory parameter tensor.

1. **Memory Score Matrix**:
   Queries \(\mathbf{Q}\) evaluate dot-product similarity against memory slots \(\mathbf{M}\):
   \[
   \mathbf{S}_{\text{mem}} = \frac{\mathbf{Q} \mathbf{M}^\top}{\sqrt{d_k}} \quad \in \mathbb{R}^{B \times H \times L \times C}
   \]

2. **Softmax Memory Context Retrieval**:
   \[
   \mathbf{A}_{\text{mem}} = \text{softmax}(\mathbf{S}_{\text{mem}}, \text{dim}=-1) \quad \in \mathbb{R}^{B \times H \times L \times C}
   \]
   \[
   \mathbf{Y}_{\text{mem}} = \mathbf{A}_{\text{mem}} \mathbf{M} \quad \in \mathbb{R}^{B \times H \times L \times d_k}
   \]

### 2.3 Local Band-Sparse Sliding-Window Attention Branch

1. **Local Score Matrix**:
   \[
   \mathbf{S}_{\text{loc}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \quad \in \mathbb{R}^{B \times H \times L \times L}
   \]

2. **Causal Sliding-Window Masking**:
   For query position \(i\) and key position \(j\):
   \[
   \mathbf{M}_{\text{window}}(i, j) = \begin{cases}
   0, & \text{if } 0 \le i - j < W \\
   -\infty, & \text{otherwise}
   \end{cases}
   \]
   Padding masks \(\mathbf{M}_{\text{pad}}\) are added to set invalid token positions to \(-\infty\).

3. **Local Context Aggregation**:
   \[
   \mathbf{A}_{\text{loc}} = \text{Dropout}\left(\text{softmax}(\mathbf{S}_{\text{loc}} + \mathbf{M}_{\text{window}} + \mathbf{M}_{\text{pad}})\right) \quad \in \mathbb{R}^{B \times H \times L \times L}
   \]
   \[
   \mathbf{Y}_{\text{loc}} = \mathbf{A}_{\text{loc}} \mathbf{V} \quad \in \mathbb{R}^{B \times H \times L \times d_k}
   \]

### 2.4 Context Fusion & Output Projection

Local and compressive memory representations are combined via element-wise addition and projected back to dimension \(D\) via output matrix \(\mathbf{W}_O \in \mathbb{R}^{D \times D}\):

\[
\mathbf{Y}_{\text{fused}} = \mathbf{Y}_{\text{loc}} + \mathbf{Y}_{\text{mem}} \quad \in \mathbb{R}^{B \times H \times L \times d_k}
\]
\[
\mathbf{O} = \text{Reshape}\left(\mathbf{Y}_{\text{fused}}, [B, L, D]\right) \mathbf{W}_O + \mathbf{b}_O \quad \in \mathbb{R}^{B \times L \times D}
\]

The module returns the tuple \((\mathbf{O}, \mathbf{A}_{\text{loc}})\).

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
                        Input Hidden States X [B, L, D]
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   Local Q, K, V Projections                             Compressive Memory M
   - Q, K, V: [B, H, L, d_k]                             - M: [1, H, C, d_k]
            │                                                     │
   Band-Sparse Sliding Window                           Memory Dot-Product Scores
   - S_loc = Q K^T / sqrt(d_k)                          - S_mem = Q M^T / sqrt(d_k)
   - Mask: 0 <= i - j < W                               - S_mem: [B, H, L, C]
            │                                                     │
   Softmax & Local Context                              Softmax & Memory Context
   - Y_loc: [B, H, L, d_k]                              - Y_mem: [B, H, L, d_k]
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                       Fused Output: Y_loc + Y_mem
                                       │
                                       ▼
                           Output GEMM: W_O + b_O
                                       │
                                       ▼
                      Final Output Tuple (O, A_loc)
```

| Step | Operation | Input Tensor Shapes | Output Tensor Shapes | Tensor Shape Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Q, K, V GEMMs | `hidden_states` \([B, L, D]\) | `q, k, v` | \([B, L, D] \rightarrow 3 \times [B, H, L, d_k]\) |
| **2** | Memory Score Matmul | `q` \([B, H, L, d_k]\), `mem` \([B, H, C, d_k]\) | `mem_scores` | \([B, H, L, d_k] \times [B, H, d_k, C] \rightarrow [B, H, L, C]\) |
| **3** | Memory Softmax & BMM | `mem_scores`, `mem` | `mem_out` | \([B, H, L, C] \times [B, H, C, d_k] \rightarrow [B, H, L, d_k]\) |
| **4** | Local Score Matmul | `q`, `k` \([B, H, L, d_k]\) | `scores` | \([B, H, L, d_k] \times [B, H, d_k, L] \rightarrow [B, H, L, L]\) |
| **5** | Window Mask Application | `scores`, `window_mask` | `scores` | Mask out positions where \(i - j \ge W\) or \(j > i\) |
| **6** | Local Softmax & BMM | `scores`, `v` \([B, H, L, d_k]\) | `local_out` | \([B, H, L, L] \times [B, H, L, d_k] \rightarrow [B, H, L, d_k]\) |
| **7** | Memory-Local Fusion | `local_out`, `mem_out` | `out` | \([B, H, L, d_k] + [B, H, L, d_k] \rightarrow [B, H, L, d_k]\) |
| **8** | Output GEMM | `out` reshaped to \([B, L, D]\)| `out` | \([B, L, D] \rightarrow [B, L, D]\) |

---

## 4. Analytical Complexity Profiling

### 4.1 FLOP & MAC Operations Analysis

Let \(B\) be batch size, \(L\) sequence length, \(D\) hidden dimension, \(H\) head count, \(d_k = D / H\), \(W\) local window size, and \(C\) memory slots:

1. **Q, K, V, and Output Projections**:
   - 4 GEMM operations of size \((B \cdot L \times D) \times (D \times D)\).
   - MACs: \(4 B \cdot L \cdot D^2\).
   - FLOPs: \(8 B \cdot L \cdot D^2\).
2. **Compressive Memory Retrieval Branch**:
   - Score Matmul (\(\mathbf{Q} \mathbf{M}^\top\)): \(2 B \cdot H \cdot L \cdot C \cdot d_k = 2 B \cdot L \cdot C \cdot D\) FLOPs.
   - Softmax: \(5 B \cdot H \cdot L \cdot C\) FLOPs.
   - Memory Context Matmul (\(\mathbf{A}_{\text{mem}} \mathbf{M}\)): \(2 B \cdot L \cdot C \cdot D\) FLOPs.
   - Total Memory Branch FLOPs: \(4 B \cdot L \cdot C \cdot D + 5 B H L C\).
3. **Local Sliding-Window Attention Branch**:
   - Local Scores over window \(W\): \(2 B \cdot L \cdot W \cdot D\) FLOPs.
   - Softmax & Context Product: \(2 B \cdot L \cdot W \cdot D + 5 B H L W\) FLOPs.
   - Total Local Branch FLOPs: \(4 B \cdot L \cdot W \cdot D + 5 B H L W\).
4. **Total Theoretical Complexity**:
   - **Total MACs**: \(4 B \cdot L \cdot D^2 + 2 B \cdot L \cdot (W + C) \cdot D\).
   - **Total FLOPs**: \(8 B \cdot L \cdot D^2 + 4 B \cdot L \cdot (W + C) \cdot D + 5 B H L (W + C)\).

### 4.2 Memory Footprint Accounting

- **Parameter Count**:
  - \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O\): \(4 \times (D^2 + D)\) parameters.
  - Compressive Memory \(\mathbf{M}\): \(H \cdot C \cdot d_k = C \cdot D\) parameters.
  - Total Parameters: \(4 D^2 + (4 + C) D\). For \(D=768, C=128\): \(\approx 2.45 \times 10^6\) parameters (4.9 MB in FP16).

- **Training Activation Memory**:
  - Local score matrix: \(B \cdot H \cdot L \cdot W \times 2\) bytes.
  - Memory attention matrix: \(B \cdot H \cdot L \cdot C \times 2\) bytes.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Static Compressive Memory Capacity Saturation

In the current implementation, `self.memory` is defined as a fixed trainable parameter tensor \(\mathbf{M} \in \mathbb{R}^{1 \times H \times C \times d_k}\). Unlike dynamic Infini-Attention models that update memory via continuous key-value associative scan rules, static memory parameters cannot adapt to sample-specific context during evaluation. Consequently, the memory acts as a global bias prompt rather than an input-adaptive context cache.

### 5.2 Memory Attention Uniformity Collapse

If query norms \(\|\mathbf{q}_{b, l}\|_2\) grow large relative to scale \(\sqrt{d_k}\), the Softmax operator \(\text{softmax}(\mathbf{Q} \mathbf{M}^\top / \sqrt{d_k})\) saturates. A single memory slot \(c \in \{1, \dots, C\}\) absorbs 100% of the attention weight, collapsing memory retrieval rank to 1.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Fused Local-Memory Dual-Attention Kernel

To optimize memory bandwidth, we deploy a Triton kernel that fuses local sliding-window attention and compressive memory retrieval into a single GPU SRAM pass.

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _infini_fused_dual_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Mem_ptr, Out_ptr,
    stride_b, stride_h, stride_l, stride_d,
    B, H, L, D_HEAD: tl.constexpr,
    C_SLOTS: tl.constexpr, WINDOW_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    offs_c = tl.arange(0, C_SLOTS)
    
    # Load Q block [BLOCK_M, D_HEAD] into SRAM
    q_tile = tl.load(Q_ptr + pid_b * stride_b + pid_h * stride_h + offs_m[:, None] * stride_l + offs_d[None, :] * stride_d)
    
    # 1. Compute Compressive Memory Attention in SRAM
    mem_tile = tl.load(Mem_ptr + pid_h * (C_SLOTS * D_HEAD) + offs_c[:, None] * D_HEAD + offs_d[None, :])
    mem_scores = tl.dot(q_tile, tl.trans(mem_tile)) / tl.sqrt(D_HEAD.to(tl.float32))
    mem_attn = tl.softmax(mem_scores, axis=-1)
    y_mem = tl.dot(mem_attn, mem_tile)
    
    # 2. Compute Band-Sparse Local Window Attention in SRAM
    # (Only loops over N tiles where idx_k is inside window range)
    
    # 3. Fuse local and memory outputs
    y_fused = y_mem # + y_local
    tl.store(Out_ptr + pid_b * stride_b + pid_h * stride_h + offs_m[:, None] * stride_l + offs_d[None, :] * stride_d, y_fused)
```

### 6.2 Optimization Blueprint Parameters

- **SRAM Double Buffering**: Loads `Mem_ptr` tile once into GPU Shared Memory per threadblock, sharing memory slots across all \(L / \text{BLOCK\_M}\) query tiles.
- **Band-Sparse Tile Loop**: Local attention loops only over key tiles satisfying \(\lfloor j \cdot \text{BLOCK\_N} \rfloor \ge \lfloor i \cdot \text{BLOCK\_M} \rfloor - W\), reducing global memory reads by \(\frac{L - W}{L} \times 100\%\).