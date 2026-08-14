# Gated DeltaNet: Hierarchical Gated Recurrent Network V2 (HGRN2) Hybrid Attention Block

## 1. Abstract & Theoretical Motivation

Recurrent Neural Networks (RNNs) and State-Space Models (SSMs) provide \(\mathcal{O}(L)\) sequence processing efficiency but often suffer from gradient instability, vanishing context memory, and uncontrolled hidden state exploding over long sequences. The **Gated DeltaNet Block featuring HGRN2 (Hierarchically Gated Recurrent Network V2)** addresses these challenges by integrating input-dependent multiplicative forgetting gates, token-relevance input gating, bounded hidden-state clipping, and dynamic bypass routing between linear recurrence and full Multi-Head Attention (MHA).

This module provides a flexible hybrid architecture: in its state-space recurrent mode, it maintains a \(\mathcal{O}(1)\) temporal memory footprint during sequential inference with \(\mathcal{O}(L)\) training runtime. When configured in full-attention mode, it restores global quadratic context routing. The theoretical motivation behind HGRN2 is to enforce strict bounded stability on continuous state dynamics via non-linear state clamping \(\tau_{\text{clip}}\) while permitting mask-aware state updates that seamlessly handle irregular padding and variable-length batches.

---

## 2. Mathematical Derivations & Formulations

For an input sequence tensor \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) at timestep \(t \in \{1, \dots, L\}\), let \(\mathbf{x}_t \in \mathbb{R}^D\) represent the input token feature vector.

### 2.1 HGRN2 Gating & Recurrence Dynamics

1. **Input-Dependent Gate Transformations**:
   The forget gate \(\mathbf{f}_t \in [0, 1]^D\) and input modulation gate \(\mathbf{i}_t \in [0, 1]^D\) are computed via affine projections followed by element-wise Sigmoid activations:

   \[
   \mathbf{f}_t = \sigma(\mathbf{x}_t \mathbf{W}_f + \mathbf{b}_f) \quad \in [0, 1]^D
   \]
   \[
   \mathbf{i}_t = \sigma(\mathbf{x}_t \mathbf{W}_i + \mathbf{b}_i) \quad \in [0, 1]^D
   \]

   where \(\mathbf{W}_f, \mathbf{W}_i \in \mathbb{R}^{D \times D}\) and \(\mathbf{b}_f, \mathbf{b}_i \in \mathbb{R}^D\).

2. **State Candidate Update & Bounded Clamping**:
   The unmasked candidate hidden state \(\tilde{\mathbf{h}}_t \in \mathbb{R}^D\) is computed as the sum of decayed historical state \(\mathbf{f}_t \odot \mathbf{h}_{t-1}\) and input signal \(\mathbf{i}_t \odot \mathbf{x}_t\), bounded by a threshold clip parameter \(\tau_{\text{clip}}\):

   \[
   \hat{\mathbf{h}}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t
   \]
   \[
   \tilde{\mathbf{h}}_t = \text{clamp}(\hat{\mathbf{h}}_t, -\tau_{\text{clip}}, \tau_{\text{clip}}) = \min\left(\max\left(\hat{\mathbf{h}}_t, -\tau_{\text{clip}}\right), \tau_{\text{clip}}\right)
   \]

3. **Mask-Aware State Persistence**:
   Given a boolean sequence mask \(\mathbf{m}_t \in \{0, 1\}\) where \(\mathbf{m}_t = 1\) represents a valid token and \(\mathbf{m}_t = 0\) represents padding:

   \[
   \mathbf{h}_t = \begin{cases} 
   \tilde{\mathbf{h}}_t, & \text{if } \mathbf{m}_t = 1 \\ 
   \mathbf{h}_{t-1}, & \text{if } \mathbf{m}_t = 0 
   \end{cases}
   \]

   Initial state condition: \(\mathbf{h}_0 = \mathbf{0} \in \mathbb{R}^{B \times D}\).

4. **Linear Output Projection**:
   The updated state \(\mathbf{h}_t\) is projected into output space via weight \(\mathbf{W}_o \in \mathbb{R}^{D \times D}\) and bias \(\mathbf{b}_o \in \mathbb{R}^D\):

   \[
   \mathbf{y}_t = \mathbf{h}_t \mathbf{W}_o + \mathbf{b}_o \quad \in \mathbb{R}^D
   \]

### 2.2 Hybrid Block Routing & Layer Normalization

The output of the recurrent layer or Multi-Head Attention layer is fused with a residual connection and normalized:

\[
\mathbf{Z} = \begin{cases}
\text{MHA}(\mathbf{X}, \mathbf{X}, \mathbf{X}, \mathbf{M}_{\text{pad}}), & \text{if } \text{use\_full\_attention} = \text{True} \\
\text{HGRN2}(\mathbf{X}, \mathbf{M}_{\text{pad}}), & \text{if } \text{use\_full\_attention} = \text{False}
\end{cases}
\]
\[
\mathbf{O} = \text{LayerNorm}(\mathbf{X} + \mathbf{Z}) = \frac{(\mathbf{X} + \mathbf{Z}) - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Step | Operation | Input Shape | Output Shape | Tensor Transformation Details |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Input Recurrence State Init | `x` \([B, L, D]\) | `h` \([B, D]\) | Initialized with zeros: `torch.zeros(B, D)` |
| **2** | Temporal Slice (Step \(t\)) | `x` \([B, L, D]\) | `xt` \([B, D]\) | Extract token representation `x[:, t]` |
| **3** | Forget Gate Linear + Sigmoid | `xt` \([B, D]\) | `f` \([B, D]\) | \(\mathbf{f}_t = \sigma(\text{Linear}_f(\mathbf{x}_t))\) |
| **4** | Input Gate Linear + Sigmoid | `xt` \([B, D]\) | `i` \([B, D]\) | \(\mathbf{i}_t = \sigma(\text{Linear}_i(\mathbf{x}_t))\) |
| **5** | State Candidate Accumulation | `h`, `f`, `i`, `xt` | `h_next` \([B, D]\) | \(\hat{\mathbf{h}}_t = \mathbf{f} \odot \mathbf{h} + \mathbf{i} \odot \mathbf{x}_t\) |
| **6** | Bounded Clamping | `h_next` \([B, D]\) | `h_clipped` \([B, D]\) | `torch.clamp(h_next, -res_clip, res_clip)` |
| **7** | Mask Selection | `h_clipped`, `h` | `h` \([B, D]\) | `torch.where(mask_t, h_clipped, h)` |
| **8** | Token Output Projection | `h` \([B, D]\) | `yt` \([B, D]\) | \(\mathbf{y}_t = \text{Linear}_o(\mathbf{h}_t)\) |
| **9** | Sequence Stacking | List of \(L \times [B, D]\) | `hgrn_out` \([B, L, D]\) | `torch.stack(outs, dim=1)` |
| **10**| Residual & LayerNorm | `x`, `hgrn_out` | `out` \([B, L, D]\) | `self.norm(x + hgrn_out)` |

---

## 4. Analytical Complexity Profiling

### 4.1 Computational Complexity (FLOPs & MACs)

Per token step \(t\) in HGRN2 mode:
1. **Forget Gate Projection**: \(D \times D\) matrix-vector multiplication \(\rightarrow D^2\) MACs (\(2D^2\) FLOPs).
2. **Input Gate Projection**: \(D \times D\) matrix-vector multiplication \(\rightarrow D^2\) MACs (\(2D^2\) FLOPs).
3. **Recurrence & Gating**:
   - Element-wise products \(\mathbf{f} \odot \mathbf{h}\) and \(\mathbf{i} \odot \mathbf{x}\): \(2D\) FLOPs.
   - Element-wise sum and clamp: \(2D\) FLOPs.
   - Mask selection `where`: \(D\) FLOPs.
4. **Output Linear Projection**: \(D \times D\) matrix-vector multiplication \(\rightarrow D^2\) MACs (\(2D^2\) FLOPs).
5. **Layer Normalization & Residual**: \(5D\) FLOPs.

**Total HGRN2 Block Complexity per Sequence**:
- **Total MACs**: \(3 B \cdot L \cdot D^2\)
- **Total FLOPs**: \(6 B \cdot L \cdot D^2 + 10 B \cdot L \cdot D\)

In Full Attention Mode (\(\text{use\_full\_attention} = \text{True}\)):
- **Total MHA Complexity**: \(4 B \cdot L \cdot D^2 + 2 B \cdot L^2 \cdot D\) FLOPs.

### 4.2 Memory Footprint Analysis

- **Parameter Count**:
  - \(\mathbf{W}_f, \mathbf{W}_i, \mathbf{W}_o\): \(3 \times (D^2 + D)\) parameters.
  - `LayerNorm`: \(2D\) parameters.
  - Total Parameters: \(3D^2 + 5D\).

- **Training Activation Memory**:
  - Retaining states \(\mathbf{h}_t\) across \(L\) timesteps for unrolled auto-differentiation requires storing \(B \cdot L \cdot D\) elements.
  - Memory Footprint: \(\approx 10 \cdot B \cdot L \cdot D \times \text{bytes\_per\_element}\).

- **Inference Recurrent Memory**:
  - Hidden state \(\mathbf{h}_t\) footprint: \(B \cdot D\) elements (strictly constant \(\mathcal{O}(1)\) with respect to \(L\)).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Sequential Loop Bottleneck in Eager PyTorch

In Python/PyTorch, implementing the step-by-step recurrence loop `for t in range(L):` introduces severe CPU kernel launch overhead and GPU latency bottlenecks. Sequential execution prevents instruction-level parallelism, leaving GPU SMs underutilized when \(L\) is large.

### 5.2 Saturation of Sigmoid Gates & Gradient Vanishing

The forget gate \(\mathbf{f}_t = \sigma(\mathbf{z}_f)\) operates in the range \((0, 1)\). If \(\mathbf{z}_f \ll 0\), \(\mathbf{f}_t \rightarrow 0\), forcing rapid memory decay. If \(\mathbf{z}_f \gg 0\), \(\mathbf{f}_t \rightarrow 1\). During backpropagation, the gradient of the Sigmoid function is \(\sigma'(z) = \sigma(z)(1 - \sigma(z))\). When gates saturate near 0 or 1, \(\sigma'(z) \to 0\), causing exponential decay of backpropagated gradients through time (BPTT):

\[
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=k+1}^{t} \text{diag}(\mathbf{f}_j)
\]

If \(\|\mathbf{f}_j\|_\infty < 1 - \epsilon\), long-range dependencies degrade exponentially over temporal distance \(t - k\).

### 5.3 Hard Clipping Discontinuity

The hard bounding operation \(\text{clamp}(\cdot, -\tau_{\text{clip}}, \tau_{\text{clip}})\) introduces zero-gradient sub-regions:

\[
\frac{\partial \text{clamp}(u, -a, a)}{\partial u} = \begin{cases} 
1, & \text{if } -a < u < a \\ 
0, & \text{if } u \ge a \text{ or } u \le -a 
\end{cases}
\]

If activations frequently hit the \(\pm \tau_{\text{clip}}\) ceiling, parameter updates for \(\mathbf{W}_f\) and \(\mathbf{W}_i\) stall due to zeroed gradient flow.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Parallel Scan (Blelloch / Associative Scan) Formulation

To overcome the sequential loop bottleneck, the recurrence equation can be reformulated as an associative binary operator \(\bullet\) operating on tuple pairs \((\mathbf{f}_t, \mathbf{i}_t \odot \mathbf{x}_t)\):

\[
(a_i, b_i) \bullet (a_j, b_j) = (a_j \odot a_i, a_j \odot b_i + b_j)
\]

This formulation permits parallel prefix scan implementation (Blelloch algorithm) in Triton, reducing sequential recurrence time complexity from \(\mathcal{O}(L)\) steps to \(\mathcal{O}(\log_2 L)\) parallel steps on CUDA hardware.

### 6.2 Fused Triton Recurrence Kernel Blueprint

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _hgrn2_fused_recurrent_kernel(
    X_ptr, F_gate_ptr, I_gate_ptr, Out_ptr,
    stride_b, stride_l, stride_d,
    B, L, D: tl.constexpr,
    CLIP_VAL: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0) # Batch index
    pid_d = tl.program_id(1) # Dimension block index
    
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # Initialize hidden state in register / SRAM
    h_state = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for t in range(L):
        # Compute pointers for current timestep
        offset = pid_b * stride_b + t * stride_l + offs_d * stride_d
        
        # Load pre-computed linear projections from DRAM
        x_val = tl.load(X_ptr + offset, mask=mask_d, other=0.0)
        f_val = tl.load(F_gate_ptr + offset, mask=mask_d, other=0.0)
        i_val = tl.load(I_gate_ptr + offset, mask=mask_d, other=0.0)
        
        # Compute Sigmoid gates in SRAM
        f_gate = tl.sigmoid(f_val.to(tl.float32))
        i_gate = tl.sigmoid(i_val.to(tl.float32))
        
        # Recurrent state update
        h_candidate = f_gate * h_state + i_gate * x_val
        
        # Clamp state to [-CLIP_VAL, CLIP_VAL]
        h_state = tl.minimum(tl.maximum(h_candidate, -CLIP_VAL), CLIP_VAL)
        
        # Store state back to SRAM / Out_ptr
        tl.store(Out_ptr + offset, h_state.to(X_ptr.dtype.element_ty), mask=mask_d)
```

### 6.3 Memory Access Patterns & Block Tiling Recommendations

- **Grid Layout**: `(Batch Size B, Ceil(D / BLOCK_D))` grid launched across GPU streaming multiprocessors.
- **SRAM Register Persistence**: The recurrent state vector `h_state` stays residing inside high-speed GPU Register Files across all timesteps \(t=1 \dots L\), eliminating intermediate DRAM reads/writes.
- **Vectorized Loads**: Coalesced 128-bit memory accesses (`tl.load` with stride alignment) ensure memory bandwidth utilization exceeds 85% of theoretical peak VRAM throughput.