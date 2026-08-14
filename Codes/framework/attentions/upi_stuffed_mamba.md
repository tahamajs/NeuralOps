# UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Selective State-Space Model

## 1. Abstract & Theoretical Motivation

Selective State-Space Models (such as standard Mamba) leverage continuous linear dynamic systems to process sequence information in linear time \(\mathcal{O}(L)\). However, standard SSMs assume deterministic historical state accumulation, leaving them vulnerable to error propagation, state saturation, and noisy token hallucination during long-context autoregressive processing.

**UPI-Stuffed Mamba (Uncertainty-Penalized Interpolated Mamba)** extends the selective state-space architecture by introducing a learnable exponential decay penalty \(\gamma \in \mathbb{R}\). This parameter interpolates dynamically between full associative state retention (\(\gamma = 0\)) and exponential memory fading (\(\gamma > 0\)), enforcing uncertainty-calibrated context forgetting. Furthermore, the architecture decouples spatial gating capacity (\(D_{\text{inner}} = 2D\)) from state-space recurrence footprint (\(D \times d_{\text{state}}\)) using zero-padded feature stuffed gating. This document provides formal derivations, shape traces, FLOP/MAC profiling, architectural failure mode analyses, and Triton acceleration blueprints.

---

## 2. Mathematical Derivations & Formulations

Given input hidden sequence tensor \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\):

### 2.1 Dual-Branch Expansion & Depthwise Convolution

Input \(\mathbf{X}\) is projected into an expanded feature space of dimension \(2 D_{\text{inner}}\) (where \(D_{\text{inner}} = E \cdot D\) with expansion multiplier \(E=2\)):

\[
\mathbf{P} = \mathbf{X} \mathbf{W}_{\text{in}} + \mathbf{b}_{\text{in}} \quad \in \mathbb{R}^{B \times L \times 2 D_{\text{inner}}} \quad (\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 2 D_{\text{inner}}})
\]
\[
[\mathbf{X}_{\text{ssm}}, \mathbf{Z}] = \text{Chunk}(\mathbf{P}, \text{chunks}=2, \text{dim}=-1) \quad \in \mathbb{R}^{B \times L \times D_{\text{inner}}}
\]

The SSM branch \(\mathbf{X}_{\text{ssm}}\) passes through a causal depthwise 1D convolution over sequence dimension \(L\) with kernel size \(d_{\text{conv}} = 4\):

\[
\tilde{\mathbf{X}}_{\text{ssm}} = \text{Truncate}_{:L}\left(\text{Conv1D}_{\text{depthwise}}(\mathbf{X}_{\text{ssm}}^\top)\right)^\top \quad \in \mathbb{R}^{B \times L \times D_{\text{inner}}}
\]

### 2.2 Uncertainty-Penalized Discretized Recurrence

1. **Discretization Timescale Projection \(\Delta t\)**:
   \[
   \mathbf{\Delta t} = \text{softplus}(\tilde{\mathbf{X}}_{\text{ssm}} \mathbf{W}_{\Delta t} + \mathbf{b}_{\Delta t}) \quad \in \mathbb{R}^{B \times L \times D} \quad (\mathbf{W}_{\Delta t} \in \mathbb{R}^{D_{\text{inner}} \times D})
   \]

2. **Diagonal Transition Parameterization**:
   \[
   \mathbf{A} = -\exp(\mathbf{A}_{\text{log}}) \quad \in \mathbb{R}^{D \times d_{\text{state}}} \quad (\mathbf{A}_{\text{log}} \in \mathbb{R}^{D \times d_{\text{state}}})
   \]

3. **Truncated Channel Extraction**:
   To reduce state memory overhead, only the first \(D\) feature channels of \(\tilde{\mathbf{X}}_{\text{ssm}}\) are fed to the recurrence:
   \[
   \mathbf{x}_t^{\text{trunc}} = \tilde{\mathbf{X}}_{\text{ssm}, :, t, :D} \quad \in \mathbb{R}^{B \times D}
   \]

4. **State Evolution with Uncertainty Penalty \(\gamma\)**:
   For timestep \(t \in \{1, \dots, L\}\), state matrix \(\mathbf{h}_t \in \mathbb{R}^{B \times D \times d_{\text{state}}}\) (initialized to \(\mathbf{h}_0 = \mathbf{0}\)) updates via:
   \[
   \hat{\mathbf{h}}_t = \mathbf{h}_{t-1} \odot \exp(\mathbf{\Delta t}_{:, t, :, \text{none}} \odot \mathbf{A}_{\text{none}, :, :}) + \mathbf{x}_t^{\text{trunc}} \otimes \mathbf{\Delta t}_{:, t, :, \text{none}}
   \]
   \[
   \mathbf{h}_t = \hat{\mathbf{h}}_t - \gamma \cdot \hat{\mathbf{h}}_t = (1 - \gamma) \cdot \hat{\mathbf{h}}_t \quad \in \mathbb{R}^{B \times D \times d_{\text{state}}}
   \]
   where \(\gamma \in \mathbb{R}\) is a learnable scalar initialized to 0.

5. **State Readout & Skip Connection**:
   \[
   \mathbf{y}_t = \sum_{s=1}^{d_{\text{state}}} \left(\mathbf{h}_{t, :, :, s} \odot \mathbf{D}\right) \quad \in \mathbb{R}^{B \times D} \quad (\mathbf{D} \in \mathbb{R}^D)
   \]
   Stacking yields \(\mathbf{Y}_{\text{ssm}} \in \mathbb{R}^{B \times L \times D}\).

### 2.3 Stuffed Zero-Padding & SiLU Gated Output

1. **Zero-Padded Feature Stuffing**:
   The \(D\)-dimensional readout \(\mathbf{Y}_{\text{ssm}}\) is padded with zeros along the channel dimension to match expanded inner size \(D_{\text{inner}}\):
   \[
   \tilde{\mathbf{Y}}_{\text{ssm}} = \text{Pad}_{\text{channel}}\left(\mathbf{Y}_{\text{ssm}}, [0, D_{\text{inner}} - D]\right) \quad \in \mathbb{R}^{B \times L \times D_{\text{inner}}}
   \]

2. **Multiplicative Gating & Projection**:
   \[
   \mathbf{O}_{\text{gated}} = \tilde{\mathbf{Y}}_{\text{ssm}} \odot \text{SiLU}(\mathbf{Z}) \quad \in \mathbb{R}^{B \times L \times D_{\text{inner}}}
   \]
   \[
   \mathbf{O} = \mathbf{O}_{\text{gated}} \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \quad \in \mathbb{R}^{B \times L \times D} \quad (\mathbf{W}_{\text{out}} \in \mathbb{R}^{D_{\text{inner}} \times D})
   \]

The block returns the tuple \((\mathbf{O}, \text{None})\).

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
   Causal 1D Conv (k=4)                            SiLU Activation
   - X_ssm: [B, 2D, L]                             - SiLU(Z): [B, L, 2D]
            │                                               │
   Discretization & Truncation                              │
   - dt = softplus(Linear(X_ssm)): [B, L, D]                │
   - x_trunc = X_ssm[:, :, :D]: [B, L, D]                   │
            │                                               │
   Uncertainty-Penalized Recurrence Loop                    │
   - h_t = (1 - gamma) * (h_{t-1} * exp(dt*A) + dt*x_trunc) │
   - out_ssm = sum(h_t * D, dim=-1): [B, L, D]              │
            │                                               │
   Zero-Padded Stuffing                                     │
   - out_ssm_full = Pad(out_ssm, [0, D]): [B, L, 2D]        │
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                   Multiplicative Gating: out_ssm_full * SiLU(Z)
                                    │
                                    ▼
                      Output GEMM: W_out + b_out
                                    │
                                    ▼
                    Final Output Tuple (out, None)
```

| Step | Layer / Operation | Input Tensor Shapes | Output Tensor Shapes | Tensor Dimensional Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Input Projection | `x` \([B, L, D]\) | `x_proj` | \([B, L, D] \rightarrow [B, L, 4D]\) |
| **2** | Branch Chunking | `x_proj` \([B, L, 4D]\) | `x_ssm`, `z` | \([B, L, 4D] \rightarrow 2 \times [B, L, 2D]\) |
| **3** | Causal Conv1D | `x_ssm` \([B, 2D, L]\) | `x_ssm` | \([B, 2D, L] \rightarrow [B, 2D, L]\) |
| **4** | Timescale GEMM | `x_ssm` \([B, L, 2D]\) | `dt` | \([B, L, 2D] \rightarrow [B, L, D]\) |
| **5** | Feature Truncation | `x_ssm` \([B, L, 2D]\) | `x_ssm_trunc` | \([B, L, 2D] \rightarrow [B, L, D]\) |
| **6** | Recurrence Step \(t\)| `x_ssm_trunc[:, t]`, `dt[:, t]`| `h` | \([B, D] \times [D, d_{\text{state}}] \rightarrow [B, D, d_{\text{state}}]\) |
| **7** | Uncertainty Decay | `h` \([B, D, d_{\text{state}}]\), `gamma` | `h` | `h = (1.0 - gamma) * h` |
| **8** | Readout Stack | List of \(L \times [B, D]\) | `out_ssm` | \(L \times [B, D] \rightarrow [B, L, D]\) |
| **9** | Zero-Padded Stuffing | `out_ssm` \([B, L, D]\) | `out_ssm_full` | \([B, L, D] \xrightarrow{\text{pad}} [B, L, 2D]\) |
| **10**| SiLU Gating | `out_ssm_full`, `z` | `out_gated` | \([B, L, 2D] \odot \text{SiLU}([B, L, 2D]) \rightarrow [B, L, 2D]\) |
| **11**| Output GEMM | `out_gated` \([B, L, 2D]\) | `out` | \([B, L, 2D] \rightarrow [B, L, D]\) |

---

## 4. Analytical Complexity Profiling

### 4.1 Computational Complexity (FLOPs & MACs)

Let \(B\) be batch size, \(L\) sequence length, \(D\) model hidden dimension, \(E=2\) expansion multiplier, \(D_{\text{inner}} = 2D\), \(d_{\text{conv}} = 4\), and \(d_{\text{state}} = 16\):

1. **Input Linear Projection (`in_proj`)**:
   - \(\mathbf{X} \in \mathbb{R}^{(BL) \times D}\) by \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 4D}\).
   - MACs: \(4 B L D^2\). FLOPs: \(8 B L D^2\).
2. **Causal Depthwise 1D Convolution (`conv1d`)**:
   - Depthwise Conv over \(2D\) channels: \(2 \cdot B \cdot L \cdot (2D) \cdot d_{\text{conv}} = 16 B L D\) FLOPs.
3. **Discretization Timescale Projection (`dt_proj`)**:
   - \((BL \times 2D) \times (2D \times D)\) GEMM + Softplus:
   - MACs: \(2 B L D^2\). FLOPs: \(4 B L D^2 + 4 B L D\).
4. **State Space Recurrence Loop with Uncertainty Decay**:
   - Exp discretization: \(4 B L D d_{\text{state}}\) FLOPs.
   - Recurrent update: \(3 B L D d_{\text{state}}\) FLOPs.
   - Uncertainty Decay subtraction (\(\gamma \cdot \mathbf{h}_t\)): \(2 B L D d_{\text{state}}\) FLOPs.
   - Readout sum: \(B L D d_{\text{state}}\) FLOPs.
   - Subtotal Recurrence FLOPs: \(10 B L D d_{\text{state}} = 160 B L D\).
5. **Zero-Stuffing, SiLU Gating & Output Projection (`out_proj`)**:
   - SiLU + Gating over \(2D\) channels: \(5 B L (2D) = 10 B L D\) FLOPs.
   - Output GEMM \((BL \times 2D) \times (2D \times D)\): \(2 B L D^2\) MACs / \(4 B L D^2\) FLOPs.
6. **Total Combined Complexity**:
   - **Total MACs**: \(8 B \cdot L \cdot D^2\).
   - **Total FLOPs**: \(16 B \cdot L \cdot D^2 + (16 + 4 + 160 + 10) B L D = 16 B \cdot L \cdot D^2 + 190 B L D\).

### 4.2 Memory Footprint Accounting

- **Parameter Count**:
  - `in_proj`: \(D \times 4D = 4D^2\).
  - `conv1d`: \(2D \times 1 \times 4 = 8D\).
  - `A_log`: \(D \times d_{\text{state}} = 16D\).
  - `D` (skip): \(D\).
  - `dt_proj`: \(2D \times D + D = 2D^2 + D\).
  - `gamma`: 1 scalar parameter.
  - `out_proj`: \(2D \times D + D = 2D^2 + D\).
  - Total Parameters: \(8 D^2 + 27 D + 1\). For \(D=768\): \(\approx 4.74 \times 10^6\) parameters (9.5 MB in FP16).

- **Training Activation Footprint**:
  - Storing convolved features and recurrent states requires \(\approx 4 B L (2D) + 2 B L D d_{\text{state}} = 40 B L D\) bytes.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Uncertainty Decay Over-Forgetting (\(\gamma \to 1\))

The uncertainty penalty term updates states via \(\mathbf{h}_t = (1 - \gamma) \cdot \hat{\mathbf{h}}_t\). If gradient updates push \(\gamma \ge 1.0\), the multiplier \((1 - \gamma) \le 0\). This forces state values to 0 or flips their signs, wiping out long-term historical memory and causing catastrophic forgetting across long context windows.

### 5.2 Zero-Padded Feature Information Wastage

In the current implementation, `out_ssm` of shape \([B, L, D]\) is padded with zeros to match expanded gating dimension \(D_{\text{inner}} = 2D\):

\[
\tilde{\mathbf{Y}}_{\text{ssm}} = [\mathbf{Y}_{\text{ssm}} \;\|\; \mathbf{0}_{B \times L \times D}]
\]

Consequently, the upper \(D\) channels of the gate tensor \(\mathbf{Z}\) (\(\text{SiLU}(\mathbf{Z}_{:, :, D:2D})\)) are multiplied by exact zeros, completely wasting 50% of the parameter capacity and FLOPs of `in_proj` and `conv1d`.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Fused Uncertainty Recurrence Triton Kernel

To eliminate Python sequential loop overhead and integrate uncertainty decay directly inside GPU Shared Memory (SRAM), we deploy a custom fused Triton kernel.

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _upi_mamba_fused_recurrent_kernel(
    X_ssm_ptr, DT_ptr, A_ptr, Gamma_ptr, Out_ptr,
    stride_b, stride_l, stride_d,
    B, L, D: tl.constexpr, N_STATE: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # Load scalar gamma parameter into register
    gamma_val = tl.load(Gamma_ptr)
    decay_multiplier = 1.0 - gamma_val
    
    h_state = tl.zeros([BLOCK_D, N_STATE], dtype=tl.float32)
    
    for t in range(L):
        off_dt = pid_b * stride_b + t * stride_l + offs_d * stride_d
        off_x  = pid_b * stride_b + t * stride_l + offs_d * stride_d
        
        dt_val = tl.load(DT_ptr + off_dt, mask=mask_d, other=0.0)
        x_val  = tl.load(X_ssm_ptr + off_x, mask=mask_d, other=0.0)
        
        # Discretization update
        # h_candidate = h_state * exp(dt * A) + dt * x
        # Uncertainty Penalty application inside SRAM register:
        # h_state = decay_multiplier * h_candidate
        
        # Readout sum: out_t = sum(h_state * D_skip, axis=-1)
        # Write out_t back to DRAM
```

### 6.2 Optimization Blueprint Parameters

- **Register-Level Decay Application**: The scalar multiplier `(1.0 - gamma)` is computed once and held in GPU Register Files, scaling state updates inline with zero additional DRAM memory transactions.
- **Asynchronous Global Memory Prefetching**: Prefetches convolved activations for step \(t+1\) while simultaneously computing discretization parameters for step \(t\).