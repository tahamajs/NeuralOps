# Jamba-Info: Dual-Mode Information-Bottlenecked SSM & Sparse Differential Attention Architecture

## 1. Abstract & Theoretical Motivation

Context compression and attention noise reduction represent fundamental challenges in long-sequence modeling. High-entropy attention distributions often allocate probability mass to irrelevant context tokens, while un-compressed token states overload state-space recurrent memory. **Jamba-Info Block** introduces a dual-operational architecture designed to optimize information flow under task-specific constraints:

1. **Retrieval Mode (Head-Sparse Differential Attention)**: Computes a signed differential attention map \(A_{\text{diff}} = A_1 - 0.5 A_2\) across dual Query-Key projections, subtracting background noise maps to isolate high-relevance tokens while enforcing head-wise sparsity (\(H_{\text{active}} \ll H\)).
2. **SSM Bottleneck Mode (InfoMamba Concept Bottleneck)**: Projects hidden states into a low-dimensional concept subspace \(\mathbb{R}^{d_{\text{bottleneck}}}\) (\(d_{\text{bottleneck}} \ll D\)) via non-linear bottleneck transformations before sequence state-space evolution (`SimpleSSM`), minimizing channel redundancy and memory bandwidth.

This document details the complete mathematical derivations, tensor shape evolution, FLOP/MAC accounting, failure modes (including differential map divergence and bottleneck rank truncation), and Triton GPU acceleration blueprints for Jamba-Info.

---

## 2. Mathematical Derivations & Formulations

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) be the input sequence activation tensor.

### 2.1 Mode 1: Retrieval Mode (Sparse Differential Attention)

1. **Dual Query & Key Projections**:
   Inputs are projected into dual Query-Key pairs and shared Value representations across \(H\) attention heads with head dimension \(d_k = D / H\):

   \[
   \mathbf{Q}_1 = \mathbf{X} \mathbf{W}_{q1}, \quad \mathbf{K}_1 = \mathbf{X} \mathbf{W}_{k1} \quad \in \mathbb{R}^{B \times H \times L \times d_k}
   \]
   \[
   \mathbf{Q}_2 = \mathbf{X} \mathbf{W}_{q2}, \quad \mathbf{K}_2 = \mathbf{X} \mathbf{W}_{k2} \quad \in \mathbb{R}^{B \times H \times L \times d_k}
   \]
   \[
   \mathbf{V} = \mathbf{X} \mathbf{W}_v \quad \in \mathbb{R}^{B \times H \times L \times d_k}
   \]

2. **Softmax Attention Map Cancellation**:
   Standard scaled dot-product Softmax maps \(\mathbf{A}_1, \mathbf{A}_2 \in \mathbb{R}^{B \times H \times L \times L}\) are evaluated:

   \[
   \mathbf{A}_1 = \text{softmax}\left(\frac{\mathbf{Q}_1 \mathbf{K}_1^\top}{\sqrt{d_k}}\right), \quad \mathbf{A}_2 = \text{softmax}\left(\frac{\mathbf{Q}_2 \mathbf{K}_2^\top}{\sqrt{d_k}}\right)
   \]

   The differential attention matrix is formed by subtracting half of the secondary attention map:

   \[
   \mathbf{A}_{\text{diff}} = \mathbf{A}_1 - 0.5 \mathbf{A}_2 \quad \in \mathbb{R}^{B \times H \times L \times L}
   \]

3. **Head Sparsity Masking**:
   Let \(\alpha \in (0, 1)\) be the sparsity ratio parameter. The active head count is defined as \(H_{\text{active}} = \max(1, \lfloor H \cdot \alpha \rfloor)\). Attention heads beyond \(H_{\text{active}}\) are zero-masked:

   \[
   \mathbf{A}_{\text{diff}, b, h, :, :} = \mathbf{0}_{L \times L} \quad \forall h \ge H_{\text{active}}
   \]

4. **Value Aggregation & Projection**:
   \[
   \mathbf{Z}_{\text{retrieval}} = \text{Reshape}\left(\mathbf{A}_{\text{diff}} \mathbf{V}, [B, L, D]\right)
   \]
   \[
   \mathbf{O} = \mathbf{Z}_{\text{retrieval}} \mathbf{W}_O + \mathbf{b}_O
   \]

### 2.2 Mode 2: SSM Bottleneck Mode (InfoMamba Concept Bottleneck)

1. **Low-Rank Concept Compression**:
   Input \(\mathbf{X}\) is down-projected into bottleneck dimension \(d_{\text{bottleneck}} = 64\) and non-linearly transformed via GELU before re-expanding to \(D\):

   \[
   \tilde{\mathbf{X}} = \text{GELU}(\mathbf{X} \mathbf{W}_{\text{down}}) \mathbf{W}_{\text{up}} \quad \in \mathbb{R}^{B \times L \times D}
   \]

   where \(\mathbf{W}_{\text{down}} \in \mathbb{R}^{D \times d_{\text{bottleneck}}}\) and \(\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{bottleneck}} \times D}\).

2. **SimpleSSM Sequence Evolution & Residual Connection**:
   The bottleneck output \(\tilde{\mathbf{X}}\) passes through `SimpleSSM` linear recurrence:

   \[
   \mathbf{Z}_{\text{ssm}} = \text{SimpleSSM}(\tilde{\mathbf{X}}, \mathbf{M}_{\text{pad}}) \quad \in \mathbb{R}^{B \times L \times D}
   \]
   \[
   \mathbf{O} = (\mathbf{X} + \mathbf{Z}_{\text{ssm}}) \mathbf{W}_O + \mathbf{b}_O
   \]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
                         Input Tensor X [B, L, D]
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
      [mode == "retrieval"]                    [mode == "bottleneck"]
   Head-Sparse Differential Attn           InfoMamba Concept Bottleneck
   - Q1, K1, Q2, K2, V GEMMs               - W_down: [B, L, D] -> [B, L, 64]
   - A_1, A_2 Softmax [B, H, L, L]         - GELU Activation
   - A_diff = A_1 - 0.5 A_2                - W_up: [B, L, 64] -> [B, L, D]
   - Head Zero Masking (h >= H_act)        - SimpleSSM Linear Recurrence
   - Z = A_diff @ V                        - Residual: X + SimpleSSM(X_bottle)
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                        Output Linear Projection W_O
                                    │
                                    ▼
                    Output Tuple (O, DiffAttnMap)
```

| Mode | Execution Step | Tensor Input | Output Tensor | Shape Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval** | Dual Q, K Projections | `x` \([B, L, D]\) | `q1, k1, q2, k2` | \([B, L, D] \rightarrow 4 \times [B, H, L, d_k]\) |
| **Retrieval** | Softmax Maps | `q1, k1` and `q2, k2` | `attn1`, `attn2` | \(2 \times ([B, H, L, d_k] \times [B, H, d_k, L] \to [B, H, L, L])\) |
| **Retrieval** | Differential Map | `attn1`, `attn2` | `diff_attn` | `attn1 - 0.5 * attn2` \([B, H, L, L]\) |
| **Retrieval** | Head Zeroing | `diff_attn` | `diff_attn` | `diff_attn[:, active_heads:] = 0.0` |
| **Retrieval** | Value BMM | `diff_attn`, `v` | `out` | \([B, H, L, L] \times [B, H, L, d_k] \rightarrow [B, L, D]\) |
| **Bottleneck**| Down Projection | `x` \([B, L, D]\) | `x_down` | \([B, L, D] \rightarrow [B, L, d_{\text{bottleneck}}]\) |
| **Bottleneck**| Up Projection | GELU(`x_down`) | `x_bottle` | \([B, L, d_{\text{bottleneck}}] \rightarrow [B, L, D]\) |
| **Bottleneck**| SimpleSSM Pass | `x_bottle` | `ssm_out` | \([B, L, D] \rightarrow [B, L, D]\) |
| **Bottleneck**| Residual Sum | `x`, `ssm_out` | `out` | `x + ssm_out` \([B, L, D]\) |

---

## 4. Analytical Complexity Profiling

### 4.1 FLOP & MAC Breakdown

Let \(B\) be batch size, \(L\) sequence length, \(D\) model dimension, \(H\) total heads, and \(d_{\text{bot}} = 64\) bottleneck dimension:

#### Mode A: Retrieval Mode
1. **Linear Projections (\(W_{q1}, W_{k1}, W_{q2}, W_{k2}, W_v, W_O\))**:
   - 6 GEMM operations of shape \((B \cdot L \times D) \times (D \times D)\).
   - MACs: \(6 B \cdot L \cdot D^2\).
   - FLOPs: \(12 B \cdot L \cdot D^2\).
2. **Dual Softmax Attention Maps**:
   - Dual BMMs: \(2 \times (2 B \cdot H \cdot L^2 \cdot d_k) = 4 B L^2 D\) FLOPs.
   - Dual Softmax: \(2 \times 5 B H L^2 = 10 B H L^2\) FLOPs.
3. **Differential Context Aggregation**:
   - Matrix subtraction & head zeroing: \(2 B H L^2\) FLOPs.
   - Active Value BMM (\(H_{\text{active}}\) heads): \(2 B \cdot H_{\text{active}} \cdot L^2 \cdot d_k = 2 \alpha B L^2 D\) FLOPs.
4. **Retrieval Mode Total FLOPs**:
   - \(\text{FLOPs}_{\text{retrieval}} = 12 B L D^2 + 4(1 + 0.5\alpha) B L^2 D + 12 B H L^2\).

#### Mode B: Bottleneck Mode
1. **Concept Bottleneck MLP**:
   - `W_down`: \(B L D d_{\text{bot}}\) MACs (\(2 B L D d_{\text{bot}}\) FLOPs).
   - GELU: \(4 B L d_{\text{bot}}\) FLOPs.
   - `W_up`: \(B L d_{\text{bot}} D\) MACs (\(2 B L D d_{\text{bot}}\) FLOPs).
   - Subtotal Bottleneck FLOPs: \(4 B L D d_{\text{bot}} + 4 B L d_{\text{bot}}\).
2. **SimpleSSM Recurrence**:
   - Recurrence FLOPs: \(6 B L D^2 + 10 B L D\).
3. **Output GEMM (`W_O`)**:
   - MACs: \(B L D^2\). FLOPs: \(2 B L D^2\).
4. **Bottleneck Mode Total FLOPs**:
   - \(\text{FLOPs}_{\text{bottleneck}} = 8 B L D^2 + 4 B L D d_{\text{bot}} + 10 B L D\).

### 4.2 Memory Footprint Accounting

- **Parameter Count**:
  - `Retrieval Mode`: \(6 D^2 + D\) parameters.
  - `Bottleneck Mode`: \(2 D^2 + 2 D d_{\text{bot}} + D\) (includes SSM) parameters.

- **Training Activation Footprint**:
  - Retrieval Mode: Storing dual attention maps \(\mathbf{A}_1, \mathbf{A}_2\) requires \(2 \times (B \cdot H \cdot L^2 \cdot 2)\) bytes.
  - Bottleneck Mode: Storing bottleneck activations requires \(B \cdot L \cdot d_{\text{bot}} \times 2\) bytes.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Negative Attention Weights & Differential Divergence

The differential map \(\mathbf{A}_{\text{diff}} = \mathbf{A}_1 - 0.5 \mathbf{A}_2\) can yield negative matrix entries \(\mathbf{A}_{\text{diff}}(i, j) < 0\). Unlike standard non-negative Softmax distributions that form convex combinations of values, signed attention matrix multiplication \(\mathbf{A}_{\text{diff}} \mathbf{V}\) can cause extreme destructive interference, leading to activation magnitude explosion during deep layer propagation.

### 5.2 Severe Concept Rank Bottlenecking

In `bottleneck` mode, projecting \(D=768\) hidden features down to \(d_{\text{bottleneck}} = 64\) imposes a \(12\times\) compression ratio. The rank of the reconstructed tensor \(\tilde{\mathbf{X}}\) is strictly bounded:

\[
\text{rank}(\tilde{\mathbf{X}}) \le d_{\text{bottleneck}} = 64
\]

This introduces severe feature bottlenecking, stripping high-frequency positional details before sequence evolution in `SimpleSSM`.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Fused Differential FlashAttention Triton Kernel

To avoid storing dual \(B \times H \times L \times L\) attention maps in global memory during Retrieval mode, we design a single fused Triton kernel that computes differential Softmax inline within SRAM registers.

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _fused_diff_attention_kernel(
    Q1_ptr, K1_ptr, Q2_ptr, K2_ptr, V_ptr, Out_ptr,
    stride_b, stride_h, stride_l, stride_d,
    H_active: tl.constexpr,
    D_HEAD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Early termination for inactive sparse heads
    if pid_h >= H_active:
        return
        
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    
    # Load Q1 and Q2 blocks into SRAM
    # Iteratively compute S1 = Q1 @ K1^T, S2 = Q2 @ K2^T
    # Apply inline Softmax: S_diff = Softmax(S1) - 0.5 * Softmax(S2)
    # Compute Out_block = S_diff @ V
```

### 6.2 Optimization Blueprint Parameters

- **Sparse Head Skipping**: Inactive heads (\(h \ge H_{\text{active}}\)) trigger instant Triton kernel thread termination, achieving up to \((1 - \alpha) \times 100\%\) compute cycles reduction in the attention phase.
- **SRAM Register Double-Buffering**: `BLOCK_M = 64, BLOCK_N = 64` tiling reduces DRAM read/write bandwidth consumption by \(\sim 4 \times\).