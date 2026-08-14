# Mamba: Gated Selective State-Space Gated Projection Layer

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Self-Attention (MHA) mechanism exhibits quadratic time and memory complexity \(\mathcal{O}(L^2)\) with respect to sequence length \(L\), rendering long-context scaling computationally prohibitive for edge and real-time LLM inference. State-Space Models (SSMs), derived from continuous-time linear dynamical systems \(\dot{h}(t) = \mathbf{A}h(t) + \mathbf{B}x(t)\), offer linear time complexity \(\mathcal{O}(L)\) and constant memory usage during inference. The Mamba architecture advances state-space modelling by introducing selective parameters (\(\mathbf{B}\), \(\mathbf{C}\), and \(\mathbf{\Delta}\)) that vary dynamically based on the input sequence, overcoming the contextual capacity bottleneck of linear time-invariant (LTI) systems.

This module implements the **Mamba Gated Selective State-Space Projection** component within the modular attention framework. Designed as a direct drop-in replacement for standard self-attention modules, it couples an expanded dual-path projection mechanism with non-linear Swish/SiLU multiplication gating. The primary theoretical motivation is to achieve high-capacity contextual gating with linear sequence-length scaling, enabling efficient token transformation while maintaining interface compatibility with standard transformer stacks.

---

## 2. Mathematical Derivations & Formulations

Given an input hidden sequence tensor \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), where \(B\) is batch size, \(L\) is sequence length, and \(D\) is the model hidden dimension:

### 2.1 Dual-Branch Linear Projection

The input hidden state \(\mathbf{X}\) is projected into a expanded space of dimension \(2D\) via a learned parameter weight matrix \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 2D}\) and bias vector \(\mathbf{b}_{\text{in}} \in \mathbb{R}^{2D}\):

\[
\mathbf{P} = \mathbf{X} \mathbf{W}_{\text{in}} + \mathbf{b}_{\text{in}} \quad \in \mathbb{R}^{B \times L \times 2D}
\]

The projected tensor \(\mathbf{P}\) is split along the channel dimension into two symmetric activation sub-branches \(\mathbf{P}_1 \in \mathbb{R}^{B \times L \times D}\) and \(\mathbf{P}_2 \in \mathbb{R}^{B \times L \times D}\):

\[
[\mathbf{P}_1, \mathbf{P}_2] = \text{Chunk}(\mathbf{P}, \text{chunks}=2, \text{dim}=-1)
\]

### 2.2 Non-Linear Gated Fusion

The first branch \(\mathbf{P}_1\) undergoes element-wise non-linear transformation via the Sigmoid Linear Unit (SiLU / Swish) activation function:

\[
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
\]

The non-linearly transformed branch is modulated by Hadamard (element-wise) multiplication with the un-activated residual branch \(\mathbf{P}_2\):

\[
\mathbf{Y} = \text{SiLU}(\mathbf{P}_1) \odot \mathbf{P}_2 \quad \in \mathbb{R}^{B \times L \times D}
\]

This Hadamard product dynamically gates feature activation flow: \(\text{SiLU}(\mathbf{P}_1)\) acts as a continuous feature filter selecting non-linear spatial representations, while \(\mathbf{P}_2\) supplies linear context.

### 2.3 Output Projection & Padding Masking

The gated tensor \(\mathbf{Y}\) is mapped back to the original model dimension \(D\) via an output projection matrix \(\mathbf{W}_{\text{out}} \in \mathbb{R}^{D \times D}\) and bias \(\mathbf{b}_{\text{out}} \in \mathbb{R}^D\):

\[
\mathbf{O} = \mathbf{Y} \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \quad \in \mathbb{R}^{B \times L \times D}
\]

If a sequence padding mask \(\mathbf{M} \in \{0, 1\}^{B \times L}\) or boolean attention mask is supplied:

\[
\hat{\mathbf{O}}_{b, l, d} = \mathbf{O}_{b, l, d} \cdot \mathbf{M}_{b, l}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Stage | Input Tensor | Operation / Kernel | Output Tensor | Shape Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **0. Layout Check** | `hidden_states` | `.is_contiguous()` check / `.contiguous()` | `hidden_states` | \([B, L, D]\) |
| **1. Expanded Linear Projection** | `hidden_states` \([B, L, D]\) | `self.in_proj` \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 2D}\) | `proj` | \([B, L, D] \rightarrow [B, L, 2D]\) |
| **2. Channel Splitting** | `proj` \([B, L, 2D]\) | `.chunk(2, dim=-1)` | `p1`, `p2` | \([B, L, 2D] \rightarrow 2 \times [B, L, D]\) |
| **3. Non-Linear Activation** | `p1` \([B, L, D]\) | `F.silu(p1)` | `p1_act` | \([B, L, D] \rightarrow [B, L, D]\) |
| **4. Element-wise Gating** | `p1_act`, `p2` | `p1_act * p2` (Hadamard product) | `gated` | \([B, L, D] \times [B, L, D] \rightarrow [B, L, D]\) |
| **5. Output Linear Projection** | `gated` \([B, L, D]\) | `self.out_proj` \(\mathbf{W}_{\text{out}} \in \mathbb{R}^{D \times D}\) | `out` | \([B, L, D] \rightarrow [B, L, D]\) |
| **6. Attention Masking** | `out`, `attention_mask` | Broadcasting element-wise product | `out` | \([B, L, D] \times [B, L, 1] \rightarrow [B, L, D]\) |

---

## 4. Analytical Complexity Profiling

### 4.1 FLOP & MAC Count Breakdown

Let \(B\) denote batch size, \(L\) sequence length, and \(D\) hidden dimension.

1. **Input Linear Projection (`in_proj`)**:
   - Matrix Multiplication: \(\mathbf{X} \in \mathbb{R}^{(BL) \times D}\) by \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 2D}\).
   - MACs: \(B \cdot L \cdot D \cdot 2D = 2 B L D^2\).
   - FLOPs: \(2 \cdot (2 B L D^2) = 4 B L D^2\).
2. **SiLU Activation & Gating (`F.silu(p1) * p2`)**:
   - SiLU FLOPs: \(\sim 4\) FLOPs per element (exponential, addition, division, multiplication) \(\rightarrow 4 B L D\).
   - Hadamard Multiplication: \(1 B L D\) FLOPs.
   - Total Gating FLOPs: \(5 B L D\).
3. **Output Linear Projection (`out_proj`)**:
   - Matrix Multiplication: \(\mathbf{Y} \in \mathbb{R}^{(BL) \times D}\) by \(\mathbf{W}_{\text{out}} \in \mathbb{R}^{D \times D}\).
   - MACs: \(B \cdot L \cdot D \cdot D = B L D^2\).
   - FLOPs: \(2 B L D^2\).
4. **Total Theoretical Compute**:
   - **Total MACs**: \(3 B L D^2\).
   - **Total FLOPs**: \(6 B L D^2 + 5 B L D \approx 6 B L D^2\).

### 4.2 Memory Footprint & Activation Accounting

- **Parameter Storage**:
  - \(\mathbf{W}_{\text{in}}\): \(D \times 2D = 2D^2\) parameters.
  - \(\mathbf{b}_{\text{in}}\): \(2D\) parameters.
  - \(\mathbf{W}_{\text{out}}\): \(D \times D = D^2\) parameters.
  - \(\mathbf{b}_{\text{out}}\): \(D\) parameters.
  - Total Parameters: \(3 D^2 + 3 D\). In 16-bit precision (FP16/BF16), memory footprint is \(6 D^2 + 6 D\) bytes.

- **Training Activation Memory (Backward Pass)**:
  - Input \(\mathbf{X}\): \(B \cdot L \cdot D \times 2\) bytes.
  - Projection \(\mathbf{P}\): \(B \cdot L \cdot 2D \times 2\) bytes.
  - Chunked tensors \(\mathbf{P}_1, \mathbf{P}_2\): \(2 \times (B \cdot L \cdot D \times 2)\) bytes.
  - Activated \(\text{SiLU}(\mathbf{P}_1)\): \(B \cdot L \cdot D \times 2\) bytes.
  - Gated result \(\mathbf{Y}\): \(B \cdot L \cdot D \times 2\) bytes.
  - Total Saved Activations: \(\approx 14 \cdot B \cdot L \cdot D\) bytes.

- **Inference State Size**:
  - Standard MHA KV Cache per layer: \(\mathcal{O}(2 \cdot B \cdot L \cdot D)\).
  - Mamba Gated Layer Cache: \(\mathcal{O}(1)\) sequence memory (no historic token KV-cache required for position-wise operations).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Absence of Cross-Token Sequence Mixing

In this direct projection variant, sequence mixing relies on upstream spatial/temporal operations or surrounding convolutional/SSM recurrence blocks. Because operations across dimension \(L\) are strictly position-wise:

\[
\frac{\partial \mathbf{O}_{b, l_1, d}}{\partial \mathbf{X}_{b, l_2, d'}} = 0 \quad \forall l_1 \neq l_2
\]

If deployed without explicit sequence-mixing kernels (such as 1D causal convolutions or selective discretization recurrence \(\mathbf{A}, \mathbf{B}, \mathbf{C}\)), the layer operates as a position-wise feed-forward net (FFN) and cannot propagate information across temporal timesteps.

### 5.2 Low-Rank Feature Collapse in Gating

The Hadamard product \(\mathbf{Y} = \text{SiLU}(\mathbf{P}_1) \odot \mathbf{P}_2\) can induce rank collapse in the channel representation. The rank of the product of two matrices is bounded by:

\[
\text{rank}(\mathbf{Y}) \le \min(\text{rank}(\text{SiLU}(\mathbf{P}_1)), \text{rank}(\mathbf{P}_2))
\]

If \(\mathbf{P}_1\) produces saturated negative values where \(\text{SiLU}(p_1) \rightarrow 0\), entire feature channels undergo zero-gating, reducing the effective numerical rank of \(\mathbf{Y}\) and inducing gradient dead-zones during backpropagation.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Memory Bandwidth & Kernel Fusion Bottleneck

In standard PyTorch, executing `mamba.py` involves launching multiple separate GPU kernels:
1. `GEMM 1`: `in_proj` (\(D \to 2D\)) writing \([B, L, 2D]\) to HBM.
2. `Slice & Chunk`: Reading \([B, L, 2D]\) from HBM, writing `p1`, `p2` to HBM.
3. `SiLU & Mul`: Reading `p1`, `p2` from HBM, computing \(\text{SiLU}(p1) \odot p2\), writing \(\mathbf{Y}\) to HBM.
4. `GEMM 2`: `out_proj` (\(D \to D\)) reading \(\mathbf{Y}\) from HBM.

This incurs high DRAM read/write overhead. A fused Triton kernel consolidates Chunk, SiLU, and Hadamard Gating into a single pass directly inside SRAM before writing out to global memory or passing directly into GEMM 2.

### 6.2 Fused Triton Gating Kernel Design

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _fused_silu_gating_kernel(
    proj_ptr,       # Pointer to in_proj output [B, L, 2D]
    out_gated_ptr,  # Pointer to gated output [B, L, D]
    stride_proj_bl, stride_proj_d,
    stride_out_bl,  stride_out_d,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Program identifier across (B * L) dimension
    pid = tl.program_id(axis=0)
    
    # Offsets inside the D dimension
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # Compute pointers for branch 1 and branch 2
    # p1 is located at offset [0..D-1], p2 is at offset [D..2D-1]
    p1_ptr = proj_ptr + pid * stride_proj_bl + cols * stride_proj_d
    p2_ptr = proj_ptr + pid * stride_proj_bl + (cols + D) * stride_proj_d
    
    # Load from DRAM to SRAM
    p1 = tl.load(p1_ptr, mask=mask, other=0.0)
    p2 = tl.load(p2_ptr, mask=mask, other=0.0)
    
    # Compute SiLU in SRAM: silu(x) = x * sigmoid(x)
    sigmoid_p1 = tl.sigmoid(p1.to(tl.float32)).to(p1.dtype)
    silu_p1 = p1 * sigmoid_p1
    
    # Compute Hadamard Gated output
    gated = silu_p1 * p2
    
    # Write back to DRAM
    out_ptr = out_gated_ptr + pid * stride_out_bl + cols * stride_out_d
    tl.store(out_ptr, gated, mask=mask)
```

### 6.3 Memory Access & SRAM Tiling Strategy

- **Tile Sizing**: `BLOCK_SIZE` is tuned to the nearest power of 2 greater than or equal to \(D\) (e.g., 64, 128, 256).
- **VMM & Tensor Core Alignment**: Matrices should be aligned to 128-byte boundaries to maximize L2 cache line efficiency and enable 16-bit vectorized load instruction (`ld.global.v4`).
- **Pipeline Fusion**: By chaining `in_proj` output straight to SRAM, intermediate DRAM memory traffic is reduced by \(\sim 3 \times (B \cdot L \cdot D \times \text{bytes})\), transforming an memory-bound activation pipeline into a compute-bound GEMM pipeline.