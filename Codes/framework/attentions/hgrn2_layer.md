# HGRN2 Layer: Hierarchically Gated Recurrent Network v2

## 1. Abstract & Theoretical Motivation

The **Hierarchically Gated Recurrent Network v2 (HGRN2 Layer)** provides a linear-time \(\mathcal{O}(L)\) sequence mixing layer designed as a direct substitute for quadratic Multi-Head Self-Attention (MHA). While classical RNNs suffer from gradient instability and state explosion over long context horizons, and Transformers incur quadratic memory cost \(\mathcal{O}(L^2)\), HGRN2 combines input-dependent dual-gating mechanisms (forget and input gates) with hard residual state boundary clipping (\(\tau_{\text{clip}} = 1.0\)).

By eliminating multi-head key-query matrix inner products in favor of state-space element-wise recurrence, the HGRN2 Layer compresses variable-length token contexts into a compact state vector \(\mathbf{h}_t \in \mathbb{R}^{B \times D}\). This document details the exact mathematical formulation, step-by-step tensor execution tracing, computational FLOP/MAC profiling, architectural failure modes (including state saturation and Python loop overhead), and a hardware-optimized GPU Triton kernel blueprint.

---

## 2. Mathematical Derivations & Formulations

Given an input sequence tensor \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), where \(B\) is batch size, \(L\) is sequence length, and \(D\) is the model feature dimension, the recurrent state sequence \(\mathbf{h}_t \in \mathbb{R}^{B \times D}\) is initialized as \(\mathbf{h}_0 = \mathbf{0}\).

### 2.1 Dual Input-Dependent Gate Formulations

At each sequential token step \(t \in \{1, \dots, L\}\) with input token feature \(\mathbf{x}_t = \mathbf{X}_{:, t, :} \in \mathbb{R}^{B \times D}\):

1. **Forget Gate**: Controls the retention scale of historical memory \(\mathbf{h}_{t-1}\):
   \[
   \mathbf{f}_t = \sigma(\mathbf{x}_t \mathbf{W}_f + \mathbf{b}_f) \quad \in (0, 1)^{B \times D}
   \]

2. **Input Gate**: Controls the injection scale of the current token signal \(\mathbf{x}_t\):
   \[
   \mathbf{i}_t = \sigma(\mathbf{x}_t \mathbf{W}_i + \mathbf{b}_i) \quad \in (0, 1)^{B \times D}
   \]

   where \(\mathbf{W}_f, \mathbf{W}_i \in \mathbb{R}^{D \times D}\) are parameter projection matrices and \(\mathbf{b}_f, \mathbf{b}_i \in \mathbb{R}^D\) are bias vectors. \(\sigma(z) = \frac{1}{1 + e^{-z}}\) denotes element-wise Sigmoidal activation.

### 2.2 Gated Recurrence & Bounded Clamping

The unclipped candidate state \(\hat{\mathbf{h}}_t\) is formed by Hadamard element-wise products:

\[
\hat{\mathbf{h}}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t \quad \in \mathbb{R}^{B \times D}
\]

To prevent state magnitude explosion during long sequence iteration, a hard residual state clipping threshold \(C = \text{res\_clip} = 1.0\) is enforced:

\[
\mathbf{h}_t = \text{clamp}(\hat{\mathbf{h}}_t, -C, C) = \min\left(\max(\hat{\mathbf{h}}_t, -C), C\right)
\]

### 2.3 Readout Projection & Output Assembly

The clamped hidden state \(\mathbf{h}_t\) is transformed into the output space by linear mapping \(\mathbf{W}_o \in \mathbb{R}^{D \times D}\) and bias \(\mathbf{b}_o \in \mathbb{R}^D\):

\[
\mathbf{y}_t = \mathbf{h}_t \mathbf{W}_o + \mathbf{b}_o \quad \in \mathbb{R}^{B \times D}
\]

The sequence of output vectors \(\mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_L\) is stacked along dimension 1:

\[
\mathbf{Y} = \text{Stack}([\mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_L], \text{dim}=1) \quad \in \mathbb{R}^{B \times L \times D}
\]

The layer returns the tuple \((\mathbf{Y}, \text{None})\) to maintain strict interface compliance with standard attention layers.

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Stage | Operation / Function | Input Tensors & Shapes | Output Tensors & Shapes | Transformation Details |
| :--- | :--- | :--- | :--- | :--- |
| **0. State Init** | `torch.zeros` | Shape parameters \(B, D\) | `h` \([B, D]\) | Zero-initialized state vector |
| **1. Token Extraction** | Tensor Slicing | `hidden_states` \([B, L, D]\) | `xt` \([B, D]\) | Extract timestep \(t\): `hidden_states[:, t]` |
| **2. Forget Gate** | `self.forget_gate` + `sigmoid` | `xt` \([B, D]\), \(\mathbf{W}_f \in \mathbb{R}^{D \times D}\) | `f` \([B, D]\) | \(\mathbf{f}_t = \sigma(\text{Linear}_f(\mathbf{x}_t))\) |
| **3. Input Gate** | `self.input_gate` + `sigmoid` | `xt` \([B, D]\), \(\mathbf{W}_i \in \mathbb{R}^{D \times D}\) | `i` \([B, D]\) | \(\mathbf{i}_t = \sigma(\text{Linear}_i(\mathbf{x}_t))\) |
| **4. Recurrent Accumulation** | Hadamard Multiplication & Sum | `f`, `h`, `i`, `xt` \([B, D]\) | `h_accum` \([B, D]\) | \(\hat{\mathbf{h}}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t\) |
| **5. Boundary Clamp** | `torch.clamp` | `h_accum` \([B, D]\) | `h` \([B, D]\) | Hard clip bounds \([-1.0, 1.0]\) |
| **6. Output Projection** | `self.output` | `h` \([B, D]\), \(\mathbf{W}_o \in \mathbb{R}^{D \times D}\) | `yt` \([B, D]\) | \(\mathbf{y}_t = \mathbf{h}_t \mathbf{W}_o + \mathbf{b}_o\) |
| **7. Tensor Stacking** | `torch.stack` | List of \(L \times [B, D]\) | `out` \([B, L, D]\) | Combine temporal outputs along `dim=1` |

---

## 4. Analytical Complexity Profiling

### 4.1 FLOP & MAC Operations Analysis

For sequence length \(L\), hidden size \(D\), and batch size \(B\):

1. **Gate Projections (`forget_gate`, `input_gate`)**:
   - Two GEMMs of size \((B \times D) \times (D \times D)\) per timestep \(t\).
   - MACs per step: \(2 \cdot B \cdot D^2\).
   - FLOPs per step: \(4 \cdot B \cdot D^2\).
2. **Gated Update & Sigmoid Activations**:
   - Two Sigmoids: \(2 \times 4 B D = 8 B D\) FLOPs.
   - Two Hadamard Products + One Addition: \(3 B D\) FLOPs.
   - Clamping (`torch.clamp`): \(B D\) FLOPs.
   - Total Gating FLOPs per step: \(12 B D\).
3. **Output Readout Projection (`output`)**:
   - GEMM of size \((B \times D) \times (D \times D)\) per step.
   - MACs per step: \(B \cdot D^2\).
   - FLOPs per step: \(2 \cdot B \cdot D^2\).
4. **Total Sequence Complexity**:
   - **Total MACs**: \(3 B \cdot L \cdot D^2\).
   - **Total FLOPs**: \(6 B \cdot L \cdot D^2 + 12 B \cdot L \cdot D\).

### 4.2 Memory Footprint & Activation Accounting

- **Parameter Memory**:
  - \(\mathbf{W}_f, \mathbf{W}_i, \mathbf{W}_o\): \(3 \times (D \cdot D + D) = 3 D^2 + 3 D\) parameters.
  - At FP16/BF16 (2 bytes per parameter): \(6 D^2 + 6 D\) bytes.

- **Training Activation Footprint (Backward Pass)**:
  - Preserving intermediate states \(\mathbf{h}_t\), gates \(\mathbf{f}_t, \mathbf{i}_t\), and token inputs \(\mathbf{x}_t\) across all \(L\) sequence steps requires:
    \[
    \text{Activation Memory} = 4 \times (B \cdot L \cdot D \times 2 \text{ bytes}) = 8 B L D \text{ bytes}
    \]

- **Inference State Size**:
  - The recurrent state footprint is strictly \(\mathcal{O}(B \cdot D)\) elements (\(2 B D\) bytes in FP16), enabling constant-memory inference regardless of whether \(L=10^3\) or \(L=10^6\).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Python Recurrence Loop & CUDA Overhead

Executing the time-step loop `for t in range(L):` in Python generates substantial interpreter overhead. Each iteration dispatches individual PyTorch C++ CUDA kernel launches for `Linear`, `Sigmoid`, element-wise product, `Clamp`, and `Output Linear`. This causes GPU kernel launch latency to dominate overall runtime, resulting in severe GPU compute starvation.

### 5.2 Dynamic State Saturation & Vanishing Gradient Dead-Zones

Because the hidden state is clamped to \([-1.0, 1.0]\), if accumulated inputs consistently drive \(\hat{\mathbf{h}}_t > 1.0\) or \(\hat{\mathbf{h}}_t < -1.0\), the gradient of the clamping function evaluates to zero:

\[
\frac{\partial \text{clamp}(\hat{h}, -1, 1)}{\partial \hat{h}} = \begin{cases} 1 & \text{if } -1 < \hat{h} < 1 \\ 0 & \text{otherwise} \end{cases}
\]

Saturated states completely sever gradient propagation back to earlier timesteps \(\tau < t\), causing vanishing gradient dead-zones in long-context training.

### 5.3 Absence of Head-Wise Subspace Specialization

Unlike Multi-Head Attention, which projects inputs into \(H\) distinct attention heads of dimension \(d_k = D / H\), HGRN2 maintains a single non-partitioned vector state \(\mathbf{h}_t \in \mathbb{R}^D\). This limits the model's capacity to simultaneously track diverse linguistic or positional relationships in parallel feature subspaces.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Kernel Fusion Strategy & SRAM State Caching

To solve the Python loop bottleneck, the linear projections \(\mathbf{X} \mathbf{W}_f\) and \(\mathbf{X} \mathbf{W}_i\) are pre-computed in a single batched GEMM pass \([B, L, D] \to [B, L, 2D]\). The sequential recurrence loop is then executed within a single fused Triton kernel where state updates reside entirely inside GPU SRAM (Shared Memory / Register Files).

### 6.2 Fused Triton Recurrence Kernel Code Blueprint

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _hgrn2_fused_recurrent_kernel(
    Proj_ptr,      # Pointer to pre-computed gates [B, L, 2D]
    X_ptr,         # Pointer to input tensor [B, L, D]
    Out_ptr,       # Pointer to state output [B, L, D]
    stride_proj_b, stride_proj_l, stride_proj_d,
    stride_x_b,    stride_x_l,    stride_x_d,
    B, L, D: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0) # Batch block
    pid_d = tl.program_id(1) # Feature dimension block
    
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # Fast SRAM register state storage
    h = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for t in range(L):
        # Memory offsets for timestep t
        proj_f_offset = pid_b * stride_proj_b + t * stride_proj_l + offs_d * stride_proj_d
        proj_i_offset = pid_b * stride_proj_b + t * stride_proj_l + (offs_d + D) * stride_proj_d
        x_offset      = pid_b * stride_x_b    + t * stride_x_l    + offs_d * stride_x_d
        
        # Load values into registers
        f_raw = tl.load(Proj_ptr + proj_f_offset, mask=mask_d, other=0.0)
        i_raw = tl.load(Proj_ptr + proj_i_offset, mask=mask_d, other=0.0)
        x_val = tl.load(X_ptr + x_offset, mask=mask_d, other=0.0)
        
        # Compute activations inline in register file
        f_gate = tl.sigmoid(f_raw.to(tl.float32))
        i_gate = tl.sigmoid(i_raw.to(tl.float32))
        
        # Recurrent state accumulation
        h_next = f_gate * h + i_gate * x_val
        
        # Hard clamping [-1.0, 1.0]
        h = tl.minimum(tl.maximum(h_next, -1.0), 1.0)
        
        # Store state back to global DRAM for output projection GEMM
        tl.store(Out_ptr + x_offset, h.to(X_ptr.dtype.element_ty), mask=mask_d)
```

### 6.3 Memory Access Optimization & Block Tiling Recommendations

- **Grid Allocation**: Launched with grid shape `(B, triton.cdiv(D, BLOCK_D))`.
- **SRAM Register Persistence**: The recurrent vector `h` stays mapped directly to SM register hardware across the loop \(t = 0 \dots L-1\), eliminating DRAM state read/writes.
- **Memory Coalescing**: `BLOCK_D` set to 128 or 256 to ensure 128-byte aligned memory transactions (`ld.global.nc` / `st.global.wb`), saturating GPU memory bandwidth.