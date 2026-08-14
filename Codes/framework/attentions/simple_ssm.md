# SimpleSSM: Selective State-Space Recurrence with Multiplicative Gating

## 1. Abstract & Theoretical Motivation

Standard Transformer self-attention modules incur quadratic time and memory cost \(\mathcal{O}(L^2)\) with respect to sequence length \(L\). State-Space Models (SSMs) provide an efficient linear-time alternative \(\mathcal{O}(L)\) by parameterizing continuous linear dynamical systems:

\[
\dot{\mathbf{h}}(t) = \mathbf{A} \mathbf{h}(t) + \mathbf{B} \mathbf{x}(t), \quad \mathbf{y}(t) = \mathbf{C} \mathbf{h}(t) + \mathbf{D} \mathbf{x}(t)
\]

**SimpleSSM** is a selective state-space model layer designed to act as a lightweight, drop-in replacement for self-attention layers. It incorporates:
1. Expanded dual-branch linear projections with SiLU multiplicative gating,
2. Causal depthwise 1D convolutions for local temporal smoothing,
3. Input-dependent discretization timescales \(\Delta t = \text{softplus}(\text{Linear}(\mathbf{x}))\), and
4. Diagonal state transition matrices \(\mathbf{A} = -\exp(\mathbf{A}_{\text{log}})\).

This document presents a comprehensive theoretical derivation, dimensional shape tracing, analytical FLOP/MAC profiling, architectural failure mode diagnosis, and a Triton hardware acceleration blueprint for SimpleSSM.

---

## 2. Mathematical Derivations & Formulations

Given input sequence activations \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\):

### 2.1 Dual-Branch Linear Projection & Causal 1D Convolution

Input \(\mathbf{X}\) is projected into expanded dimension \(2 D_{\text{inner}}\) (where \(D_{\text{inner}} = E \cdot D\) with expansion multiplier \(E=2\)):

\[
\mathbf{P} = \mathbf{X} \mathbf{W}_{\text{in}} \quad \in \mathbb{R}^{B \times L \times 2 D_{\text{inner}}} \quad (\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 2 D_{\text{inner}}})
\]
\[
[\mathbf{X}_{\text{ssm}}, \mathbf{Z}] = \text{Chunk}(\mathbf{P}, \text{chunks}=2, \text{dim}=-1) \quad \in \mathbb{R}^{B \times L \times D_{\text{inner}}}
\]

If mask \(\mathbf{M} \in \{0, 1\}^{B \times L}\) is present: \(\mathbf{X}_{\text{ssm}} \leftarrow \mathbf{X}_{\text{ssm}} \odot \mathbf{M}, \quad \mathbf{Z} \leftarrow \mathbf{Z} \odot \mathbf{M}\).

The SSM path \(\mathbf{X}_{\text{ssm}}\) passes through a causal depthwise 1D convolution over sequence dimension \(L\) with kernel size \(d_{\text{conv}} = 4\):

\[
\tilde{\mathbf{X}}_{\text{ssm}} = \text{Truncate}_{:L}\left(\text{Conv1D}_{\text{depthwise}}(\mathbf{X}_{\text{ssm}}^\top)\right)^\top \quad \in \mathbb{R}^{B \times L \times D_{\text{inner}}}
\]

### 2.2 Input-Dependent Discretization & Diagonal Recurrence

1. **Timescale Discretization Projection \(\Delta t\)**:
   \[
   \mathbf{\Delta t} = \text{softplus}(\tilde{\mathbf{X}}_{\text{ssm}} \mathbf{W}_{\Delta t} + \mathbf{b}_{\Delta t}) \quad \in \mathbb{R}^{B \times L \times D} \quad (\mathbf{W}_{\Delta t} \in \mathbb{R}^{D_{\text{inner}} \times D})
   \]

2. **Diagonal Transition Matrix Parameterization**:
   The continuous transition parameter \(\mathbf{A}_{\text{log}} \in \mathbb{R}^{D \times d_{\text{state}}}\) is exponentiated and negated to ensure strictly negative real eigenvalues:
   \[
   \mathbf{A} = -\exp(\mathbf{A}_{\text{log}}) \quad \in \mathbb{R}^{D \times d_{\text{state}}}
   \]

3. **Discretized Zero-Order Hold (ZOH) State Recurrence**:
   For timestep \(t \in \{1, \dots, L\}\), state tensor \(\mathbf{h}_t \in \mathbb{R}^{B \times D \times d_{\text{state}}}\) (initialized to \(\mathbf{h}_0 = \mathbf{0}\)) updates via:
   \[
   \bar{\mathbf{A}}_t = \exp(\mathbf{\Delta t}_{:, t, :, \text{none}} \odot \mathbf{A}) \quad \in \mathbb{R}^{B \times D \times d_{\text{state}}}
   \]
   \[
   \bar{\mathbf{B}}_t = \tilde{\mathbf{X}}_{\text{ssm}, :, t, :D, \text{none}} \odot \mathbf{\Delta t}_{:, t, :, \text{none}} \quad \in \mathbb{R}^{B \times D \times 1}
   \]
   \[
   \mathbf{h}_t = \bar{\mathbf{A}}_t \odot \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t \otimes \mathbf{1}_{d_{\text{state}}}^\top \quad \in \mathbb{R}^{B \times D \times d_{\text{state}}}
   \]

4. **State Readout & Direct Skip Summation**:
   Let \(\mathbf{D}_{\text{skip}} \in \mathbb{R}^D\) be a learnable direct feedthrough parameter vector:
   \[
   \mathbf{y}_t = \sum_{s=1}^{d_{\text{state}}} \mathbf{h}_{t, :, :, s} \odot \mathbf{D}_{\text{skip}} \quad \in \mathbb{R}^{B \times D}
   \]
   Stacking yields sequence tensor \(\mathbf{Y}_{\text{ssm}} \in \mathbb{R}^{B \times L \times D}\).

### 2.3 Multiplicative SiLU Gating & Output Projection

The SSM readout \(\mathbf{Y}_{\text{ssm}}\) is modulated by Hadamard multiplication with the SiLU-activated residual branch \(\mathbf{Z}\):

\[
\mathbf{G} = \text{SiLU}(\mathbf{Z}_{:, :, :D}) = \mathbf{Z}_{:, :, :D} \odot \sigma(\mathbf{Z}_{:, :, :D}) \quad \in \mathbb{R}^{B \times L \times D}
\]
\[
\mathbf{O}_{\text{gated}} = \mathbf{Y}_{\text{ssm}} \odot \mathbf{G} \quad \in \mathbb{R}^{B \times L \times D}
\]
\[
\mathbf{O} = \mathbf{O}_{\text{gated}} \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \quad \in \mathbb{R}^{B \times L \times D} \quad (\mathbf{W}_{\text{out}} \in \mathbb{R}^{D \times D})
\]

The module outputs the tuple \((\mathbf{O}, \text{None})\).

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
                     Input Hidden States X [B, L, D]
                                    │
                                    ▼
                    Dual Projection W_in: [B, L, 4D]
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   SSM Path X_ssm [B, L, 2D]                       Gate Path Z [B, L, 2D]
            │                                               │
   Causal 1D Conv (k=4)                            Slice & SiLU Activation
   - X_ssm: [B, 2D, L] -> [B, 2D, L]                - G = SiLU(Z[:, :, :D])
            │                                               │
   Discretization dt_proj                                  │
   - dt = softplus(Linear(X_ssm)): [B, L, D]                │
            │                                               │
   Selective State Recurrence Loop                         │
   - h_t = exp(dt * A) * h_{t-1} + dt * x_t                │
   - out_ssm = sum(h_t * D_skip, dim=-1): [B, L, D]        │
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                   Multiplicative Gating: out_ssm * G
                                    │
                                    ▼
                      Output GEMM: W_out + b_out
                                    │
                                    ▼
                    Final Output Tuple (out, None)
```

| Step | Operation / Layer | Input Tensor Shapes | Output Tensor Shapes | Tensor Dimensional Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Input Projection | `hidden_states` \([B, L, D]\) | `x_proj` | \([B, L, D] \rightarrow [B, L, 4D]\) |
| **2** | Branch Splitting | `x_proj` \([B, L, 4D]\) | `x_ssm`, `z` | \([B, L, 4D] \rightarrow 2 \times [B, L, 2D]\) |
| **3** | Mask Multiplication | `x_ssm`, `attention_mask` | `x_ssm` | \([B, L, 2D] \odot [B, L, 1] \rightarrow [B, L, 2D]\) |
| **4** | Causal Conv1D | `x_ssm` \([B, 2D, L]\) | `x_ssm` | \([B, 2D, L] \rightarrow [B, 2D, L]\) |
| **5** | Timescale GEMM | `x_ssm` \([B, L, 2D]\) | `dt` | \([B, L, 2D] \rightarrow [B, L, D]\) |
| **6** | Recurrence Step \(t\)| `x_ssm[:, t, :D]`, `dt[:, t]`| `h` | \([B, D] \times [D, d_{\text{state}}] \rightarrow [B, D, d_{\text{state}}]\) |
| **7** | State Readout | `h` \([B, D, d_{\text{state}}]\), `D` | `out_t` | \([B, D, d_{\text{state}}] \rightarrow [B, D]\) |
| **8** | Sequence Stack | List of \(L \times [B, D]\) | `out_ssm` | \(L \times [B, D] \rightarrow [B, L, D]\) |
| **9** | SiLU Gating | `out_ssm`, `z` | `out_gated` | \([B, L, D] \odot \text{SiLU}([B, L, D]) \rightarrow [B, L, D]\) |
| **10**| Output GEMM | `out_gated` \([B, L, D]\) | `out` | \([B, L, D] \rightarrow [B, L, D]\) |

---

## 4. Analytical Complexity Profiling

### 4.1 Computational Complexity (FLOPs & MACs)

Let \(B\) be batch size, \(L\) sequence length, \(D\) hidden dimension, \(E=2\) expansion multiplier, \(D_{\text{inner}} = 2D\), \(d_{\text{conv}} = 4\), and \(d_{\text{state}} = 16\):

1. **Input Dual Linear Projection (`in_proj`)**:
   - Matrix Multiplication \(\mathbf{X} \in \mathbb{R}^{(BL) \times D}\) by \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 4D}\).
   - MACs: \(4 B L D^2\).
   - FLOPs: \(8 B L D^2\).
2. **Causal Depthwise 1D Convolution (`conv1d`)**:
   - Depthwise Convolution over \(2D\) channels with kernel \(d_{\text{conv}} = 4\):
   - FLOPs: \(2 \cdot B \cdot L \cdot (2D) \cdot d_{\text{conv}} = 16 B L D\).
3. **Discretization Timescale Projection (`dt_proj`)**:
   - Matrix Multiplication \((B L \times 2D) \times (2D \times D)\) + Softplus:
   - MACs: \(2 B L D^2\).
   - FLOPs: \(4 B L D^2 + 4 B L D\).
4. **State Space Recurrence Loop**:
   - Exponential discretization \(\exp(\Delta t \cdot \mathbf{A})\): \(4 B L D d_{\text{state}}\) FLOPs.
   - Recurrent update \(\mathbf{h}_t = \mathbf{A} \mathbf{h}_{t-1} + \mathbf{B} \mathbf{x}_t\): \(3 B L D d_{\text{state}}\) FLOPs.
   - Readout sum: \(B L D d_{\text{state}}\) FLOPs.
   - Subtotal Recurrence FLOPs: \(8 B L D d_{\text{state}} = 128 B L D\).
5. **SiLU Gating & Output Projection (`out_proj`)**:
   - SiLU + Hadamard Gating: \(5 B L D\) FLOPs.
   - Output GEMM \((BL \times D) \times (D \times D)\): \(B L D^2\) MACs / \(2 B L D^2\) FLOPs.
6. **Total Combined Complexity**:
   - **Total MACs**: \(7 B \cdot L \cdot D^2\).
   - **Total FLOPs**: \(14 B \cdot L \cdot D^2 + (16 + 4 + 128 + 5) B L D = 14 B \cdot L \cdot D^2 + 153 B L D\).

### 4.2 Memory Footprint Accounting

- **Parameter Count**:
  - `in_proj`: \(D \times 4D = 4D^2\).
  - `conv1d`: \(2D \times 1 \times 4 = 8D\).
  - `dt_proj`: \(2D \times D + D = 2D^2 + D\).
  - `A_log`: \(D \times d_{\text{state}} = 16D\).
  - `D` (skip): \(D\).
  - `out_proj`: \(D \times D + D = D^2 + D\).
  - Total Parameters: \(7 D^2 + 26 D\). For \(D=768\): \(\approx 4.15 \times 10^6\) parameters (8.3 MB in FP16).

- **Training Activation Footprint**:
  - Preserving convolved states \(\tilde{\mathbf{X}}_{\text{ssm}}\) and recurrence states \(\mathbf{h}_1 \dots \mathbf{h}_L\) requires \(\approx 4 B L D + 2 B L D d_{\text{state}} = 36 B L D\) bytes.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Sequential Iterative Recurrence GPU Starvation

Executing the sequential loop `for t in range(seq_len):` in PyTorch dispatches \(L\) individual sets of CUDA operations (`torch.exp`, element-wise product, `sum`). At sequence length \(L = 8192\), Python interpreter overhead and GPU kernel launch latency dominate overall execution time, severely throttling GPU Streaming Multiprocessor (SM) utilization.

### 5.2 Discretization Timescale Saturation & Zero-Gradient Plateau

The discretization parameter \(\Delta t = \text{softplus}(\mathbf{z}_{\Delta t})\) uses the Softplus activation \(\ln(1 + e^z)\). If \(\mathbf{z}_{\Delta t} \ll -10\), \(\Delta t \to 0\). The discretized transition matrix \(\bar{\mathbf{A}}_t = \exp(\Delta t \cdot \mathbf{A}) \to \mathbf{I}\) (identity matrix), while the input multiplier \(\bar{\mathbf{B}}_t = \Delta t \cdot \mathbf{x}_t \to \mathbf{0}\). Under this condition, new input tokens are zeroed out, freezing state evolution and causing zero-gradient plateaus.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Parallel Prefix Associative Scan Triton Kernel

To bypass the sequential Python loop, the linear recurrence equation \(\mathbf{h}_t = \bar{\mathbf{A}}_t \odot \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t\) is reformulated as an associative binary scan operator \(\bullet\):

\[
(a_i, b_i) \bullet (a_j, b_j) = (a_j \odot a_i, \; a_j \odot b_i + b_j)
\]

This allows parallel prefix scan execution (Blelloch algorithm) in GPU Shared Memory (SRAM), reducing recurrence time complexity from \(\mathcal{O}(L)\) sequential steps to \(\mathcal{O}(\log_2 L)\) parallel steps.

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _simple_ssm_parallel_scan_kernel(
    X_ssm_ptr, DT_ptr, A_ptr, Out_ptr,
    stride_b, stride_l, stride_d,
    B, L, D: tl.constexpr, N_STATE: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # State register buffer in fast GPU SRAM
    h_state = tl.zeros([BLOCK_D, N_STATE], dtype=tl.float32)
    
    for t in range(L):
        off_dt = pid_b * stride_b + t * stride_l + offs_d * stride_d
        off_x  = pid_b * stride_b + t * stride_l + offs_d * stride_d
        
        dt_val = tl.load(DT_ptr + off_dt, mask=mask_d, other=0.0) # [BLOCK_D]
        x_val  = tl.load(X_ssm_ptr + off_x, mask=mask_d, other=0.0) # [BLOCK_D]
        
        # Continuous to discrete ZOH decay factors
        # Compute A_bar = exp(dt * A) and B_bar = dt * x
        # Update state h_state in registers
        # Sum out_t = sum(h_state * D_skip, axis=-1)
        # Store out_t to Out_ptr
```

### 6.2 Optimization Blueprint Parameters

- **SRAM Register Persistence**: The recurrent state matrix `h_state` stays residing inside SM Register Files across loop timesteps \(t\), eliminating intermediate DRAM reads/writes.
- **Asynchronous Global-to-Shared Copy (`cp.async`)**: Prefetches convolved activations `x_ssm` for step \(t+1\) while simultaneously computing discretization parameters \(\Delta t\) for step \(t\).