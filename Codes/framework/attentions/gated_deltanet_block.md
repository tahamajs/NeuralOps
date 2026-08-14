# Gated DeltaNet Block: Hybrid Recurrent-Attention Transformer Architecture

## 1. Abstract & Theoretical Motivation

Standard Transformer blocks rely on Multi-Head Self-Attention (MHA) to capture global dependencies across token sequences. However, MHA enforces quadratic compute and memory scaling \(\mathcal{O}(L^2)\), posing severe hardware barriers for real-time inference and context-window expansion. **Gated DeltaNet Block** introduces a hybrid architectural paradigm that dynamically bridges non-linear state-space linear recurrence via Hierarchical Gated Recurrent Network v2 (HGRN-2) and classical quadratic self-attention.

When operating in HGRN-2 linear recurrence mode, the block reduces context processing complexity to strictly linear time \(\mathcal{O}(L)\) and constant memory \(\mathcal{O}(1)\) state caching during incremental decoding. The theoretical core of the block lies in its dual-gated multiplicative state update mechanism with hard state boundary clipping, coupled with a residual pre-normalization pipeline (\(\text{Pre-LayerNorm}\)). This document delivers an end-to-end mathematical analysis, shape tracing, complexity profiling, failure mode diagnosis, and a Triton hardware acceleration plan.

---

## 2. Mathematical Derivations & Formulations

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) denote the input hidden activation tensor, where \(B\) is batch size, \(L\) is sequence length, and \(D\) is the model feature dimension.

### 2.1 Dual-Branch Routing Logic

The module routes inputs based on the boolean flag `use_full_attention`:

\[
\mathbf{Z} = \begin{cases}
\text{MHA}(\mathbf{X}, \mathbf{X}, \mathbf{X}, \mathbf{M}_{\text{pad}}), & \text{if } \text{use\_full\_attention} = \text{True} \\
\text{HGRN2}(\mathbf{X}, \mathbf{M}_{\text{pad}}), & \text{if } \text{use\_full\_attention} = \text{False}
\end{cases}
\]

### 2.2 HGRN-2 Recurrent Sub-Layer Formulations

When `use_full_attention = False`, the block invokes `HGRN2Layer`. For each timestep \(t \in \{1, \dots, L\}\), token feature vector \(\mathbf{x}_t = \mathbf{X}_{:, t, :} \in \mathbb{R}^{B \times D}\):

1. **Gate Generation**:
   Inputs are projected via learned linear transformations \(\mathbf{W}_f, \mathbf{W}_i \in \mathbb{R}^{D \times D}\) and bias vectors \(\mathbf{b}_f, \mathbf{b}_i \in \mathbb{R}^D\):

   \[
   \mathbf{f}_t = \sigma(\mathbf{x}_t \mathbf{W}_f + \mathbf{b}_f) \in (0, 1)^{B \times D} \quad \text{(Forget Gate)}
   \]
   \[
   \mathbf{i}_t = \sigma(\mathbf{x}_t \mathbf{W}_i + \mathbf{b}_i) \in (0, 1)^{B \times D} \quad \text{(Input Gate)}
   \]

2. **Gated State Evolution & Hard Clipping**:
   The hidden state \(\mathbf{h}_t \in \mathbb{R}^{B \times D}\) is updated through element-wise Hadamard decay and input modulation, followed by hard clipping to ceiling \(C = \tau_{\text{clip}}\):

   \[
   \hat{\mathbf{h}}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t
   \]
   \[
   \mathbf{h}_t = \text{clamp}(\hat{\mathbf{h}}_t, -C, C)
   \]

3. **Output Linear Transformation**:
   The state \(\mathbf{h}_t\) is projected to output space using weight \(\mathbf{W}_o \in \mathbb{R}^{D \times D}\) and bias \(\mathbf{b}_o \in \mathbb{R}^D\):

   \[
   \mathbf{y}_t = \mathbf{h}_t \mathbf{W}_o + \mathbf{b}_o \in \mathbb{R}^{B \times D}
   \]

   Outputs are concatenated over sequence timesteps to form \(\mathbf{Y}_{\text{HGRN2}} \in \mathbb{R}^{B \times L \times D}\).

### 2.3 Layer Normalization & Residual Aggregation

The branch output \(\mathbf{Z}\) is combined with the original input \(\mathbf{X}\) via a skip connection and normalized using LayerNorm:

\[
\mathbf{O} = \text{LayerNorm}(\mathbf{X} + \mathbf{Z}) = \frac{(\mathbf{X} + \mathbf{Z}) - \mathbb{E}[\mathbf{X} + \mathbf{Z}]}{\sqrt{\text{Var}[\mathbf{X} + \mathbf{Z}] + \epsilon}} \odot \gamma + \beta
\]

where \(\gamma, \beta \in \mathbb{R}^D\) are learnable gain and bias parameters.

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
                Input Tensor [B, L, D]
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 [use_full_attention=True]     [use_full_attention=False]
          │                             │
  Multi-Head Attention          HGRN2 Recurrent Layer
  - Q, K, V Projections         - Gate Projections [B, L, D] -> [B, L, D]
  - Softmax(QK^T / sqrt(d_k))   - Sequential Recurrence Loop
  - Context Matrix [B, L, D]    - State Clamping [-C, C]
          │                             │
          └──────────────┬──────────────┘
                         ▼
             Residual Addition: X + Z [B, L, D]
                         │
                         ▼
             LayerNorm: [B, L, D] -> [B, L, D]
                         │
                         ▼
             Output Tuple: (O, AttnWeights)
```

| Execution Phase | Operation | Tensor Inputs | Tensor Outputs | Dimension Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **1. Branch Selection** | Mode Check | `hidden_states` | `x` | \([B, L, D]\) |
| **2a. Full Attention Path** | Multi-head Dot-Product | `x` \([B, L, D]\) | `attn_out` | \([B, L, D] \rightarrow [B, L, D]\) |
| **2b. HGRN2 Recurrence Path**| Linear Recurrence Pass | `x` \([B, L, D]\) | `hgrn_out` | \([B, L, D] \rightarrow [B, L, D]\) |
| **3. Residual Addition** | Element-wise Sum | `x`, `attn_out` / `hgrn_out` | `residual` | \([B, L, D] + [B, L, D] \rightarrow [B, L, D]\) |
| **4. Layer Normalization** | Feature Normalization | `residual` | `out` | \([B, L, D] \rightarrow [B, L, D]\) |
| **5. Output Formatting** | Pack Tuple | `out`, `attn_weights` | `(out, weights)`| Tuple of \([B, L, D]\) and `Optional[Tensor]` |

---

## 4. Analytical Complexity Profiling

### 4.1 Comparative FLOP & MAC Profiling

Let \(B\) be batch size, \(L\) sequence length, \(D\) hidden size, and \(H\) number of heads:

| Computation Stage | HGRN-2 Mode FLOPs | Full Attention Mode FLOPs | MACs (HGRN-2) | MACs (MHA) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Projections** | \(6 B L D^2\) (Gates + Out) | \(8 B L D^2\) (\(Q,K,V,O\)) | \(3 B L D^2\) | \(4 B L D^2\) |
| **Attention / Recurrence**| \(10 B L D\) (Gates + Clamp) | \(4 B L^2 D + 2 B L^2 H\) | \(2 B L D\) | \(2 B L^2 D\) |
| **Residual & LayerNorm** | \(5 B L D\) | \(5 B L D\) | \(0\) | \(0\) |
| **Total FLOPs** | \(\approx 6 B L D^2 + 15 B L D\) | \(\approx 8 B L D^2 + 4 B L^2 D\) | \(3 B L D^2\) | \(4 B L D^2 + 2 B L^2 D\) |

### 4.2 Memory Footprint Accounting

- **Parameter Count**:
  - `HGRN-2 Mode`: \(3 D^2 + 3 D\) (Recurrence) + \(2 D\) (LayerNorm) \(= 3 D^2 + 5 D\).
  - `Full Attention Mode`: \(4 D^2 + 4 D\) (MHA) + \(2 D\) (LayerNorm) \(= 4 D^2 + 6 D\).

- **Training Activation Footprint**:
  - HGRN-2 Mode: Retaining \(L\) intermediate hidden states \(\mathbf{h}_1 \dots \mathbf{h}_L\) for BPTT backward pass requires \(\mathcal{O}(B \cdot L \cdot D \cdot 2)\) bytes.
  - Full Attention Mode: Storing the full \(B \times H \times L \times L\) attention map requires \(\mathcal{O}(B \cdot H \cdot L^2 \cdot 2)\) bytes.

- **Inference Cache Footprint**:
  - HGRN-2 Mode: \(\mathcal{O}(B \cdot D)\) state memory (constant across time \(L\)).
  - Full Attention Mode: \(\mathcal{O}(2 B \cdot H \cdot L \cdot d_k) = \mathcal{O}(2 B \cdot L \cdot D)\) KV-cache memory.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Python Iterative Loop Bottleneck

In standard PyTorch, evaluating `HGRN2Layer` invokes an unrolled Python `for t in range(L):` loop. At sequence length \(L = 4096\), this dispatches 4,096 distinct PyTorch C++ function calls per forward pass. This creates high CPU thread launch overhead, GPU SM starvation, and low compute kernel occupancy.

### 5.2 Dynamic Range Compression & Gradient Dead Zones

The state clamping operation \(\mathbf{h}_t = \text{clamp}(\hat{\mathbf{h}}_t, -C, C)\) forces all state values exceeding \(C\) or falling below \(-C\) into saturated plateaus. The sub-gradient of the hard clamp function:

\[
\frac{\partial \mathbf{h}_t}{\partial \hat{\mathbf{h}}_t} = \mathbf{I}_{[-C \le \hat{\mathbf{h}}_t \le C]}
\]

When state values saturate against \(\pm C\), the backward gradient evaluates to zero. This creates gradient dead zones, preventing optimization of weights \(\mathbf{W}_f\) and \(\mathbf{W}_i\).

### 5.3 Absence of Channel Cross-Talk in Recurrence

The HGRN-2 state update \(\mathbf{h}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t\) is strictly element-wise across dimension \(D\). Channel mixing occurs exclusively through the linear projections \(\mathbf{W}_f, \mathbf{W}_i, \mathbf{W}_o\). If the output matrix \(\mathbf{W}_o\) becomes ill-conditioned, the layer exhibits channel rank collapse, reducing feature capacity.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Fused Chunked State-Space Recurrence Kernel

To eliminate the sequential Python loop overhead, the recurrence equations are converted into a fused Triton C-like kernel executing in GPU Shared Memory (SRAM):

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _gated_deltanet_fused_kernel(
    X_ptr, Wf_ptr, Wi_ptr, Wo_ptr, Out_ptr,
    stride_b, stride_l, stride_d,
    B, L, D: tl.constexpr,
    CLIP_VAL: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # State register in fast SRAM
    h = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for t in range(L):
        # Compute pointer addresses for current timestep
        ptr_offset = pid_b * stride_b + t * stride_l + offs_d * stride_d
        
        # Load input token x_t
        x_t = tl.load(X_ptr + ptr_offset, mask=mask_d, other=0.0)
        
        # Load pre-computed projections f_raw and i_raw
        f_raw = tl.load(Wf_ptr + ptr_offset, mask=mask_d, other=0.0)
        i_raw = tl.load(Wi_ptr + ptr_offset, mask=mask_d, other=0.0)
        
        # Compute sigmoid activations in SRAM
        f_t = tl.sigmoid(f_raw.to(tl.float32))
        i_t = tl.sigmoid(i_raw.to(tl.float32))
        
        # Recurrent state step
        h_candidate = f_t * h + i_t * x_t
        
        # Bounded clamping
        h = tl.minimum(tl.maximum(h_candidate, -CLIP_VAL), CLIP_VAL)
        
        # Store state output to RAM
        tl.store(Out_ptr + ptr_offset, h.to(X_ptr.dtype.element_ty), mask=mask_d)
```

### 6.2 Parallel Prefetching & Tile Optimization Parameters

- **Kernel Tiling (`BLOCK_D`)**: Selected as 128 or 256 depending on GPU SRAM capacity per SM (e.g., NVIDIA A100 / H100 with 192KB SRAM).
- **Asynchronous Global-to-Shared Memory Copy (`cp.async`)**: Loads projections for timestep \(t+1\) while simultaneously computing state updates for timestep \(t\).
- **Warp Level Primitive Reductions**: Pre-computes LayerNorm statistics (\(\mu, \sigma^2\)) across warps using `tl.sum` in the final fused block stage.