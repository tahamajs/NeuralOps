# HGRN2-GatedDeltaNet: Hybrid Recurrent-Attention Stack with Dynamic Layer Ratio

## 1. Abstract & Theoretical Motivation

While pure State-Space Models (SSMs) and linear recurrent units achieve linear sequence scaling \(\mathcal{O}(L)\), they occasionally struggle to maintain exact long-range associative recall over long context windows compared to dense Multi-Head Self-Attention (MHA). Conversely, pure MHA networks suffer from quadratic computational and memory scaling \(\mathcal{O}(L^2)\).

The **HGRN2-GatedDeltaNet Hybrid Architecture** reconciles this trade-off by dynamically interleaving linear-time Hierarchical Gated Recurrent Network v2 (HGRN-2) blocks with dense MHA blocks according to a configurable layer ratio \(R:1\). For every sequence of \(R\) linear recurrent blocks, a single dense MHA layer is inserted to periodically re-synchronize global contextual representations. This paper provides a theoretical treatment, mathematical derivations, layer execution shape tracing, FLOP/MAC complexity profiling, architectural failure modes, and GPU Triton acceleration blueprints for HGRN2-GatedDeltaNet.

---

## 2. Mathematical Derivations & Formulations

Let \(\mathbf{X}^{(0)} \in \mathbb{R}^{B \times L \times D}\) be the input hidden state to a network stack of \(N\) layers, where \(B\) is batch size, \(L\) is sequence length, and \(D\) is hidden dimension. Let \(R \in \mathbb{Z}_{\ge 0}\) represent the hybrid ratio parameter.

### 2.1 Cyclic Layer Routing Schedule

For each layer index \(l \in \{0, 1, \dots, N-1\}\), the routing flag \(\text{use\_full}^{(l)} \in \{\text{True}, \text{False}\}\) is computed:

\[
\text{use\_full}^{(l)} = \begin{cases}
\text{True}, & \text{if } l \pmod{R + 1} = R \\
\text{False}, & \text{otherwise}
\end{cases}
\]

The layer transformation \(\mathbf{X}^{(l+1)} = \text{Block}^{(l)}(\mathbf{X}^{(l)})\) is governed by:

\[
\mathbf{X}^{(l+1)} = \begin{cases}
\text{LayerNorm}\left(\mathbf{X}^{(l)} + \text{MHA}(\mathbf{X}^{(l)})\right), & \text{if } l \pmod{R + 1} = R \\
\text{LayerNorm}\left(\mathbf{X}^{(l)} + \text{HGRN2}(\mathbf{X}^{(l)})\right), & \text{otherwise}
\end{cases}
\]

### 2.2 Sub-Layer Formulations

1. **HGRN-2 Linear Recurrence Sub-Layer** (for \(l \pmod{R+1} \neq R\)):
   At timestep \(t \in \{1, \dots, L\}\):

   \[
   \mathbf{f}_t = \sigma(\mathbf{x}_t^{(l)} \mathbf{W}_f^{(l)} + \mathbf{b}_f^{(l)}) \in (0, 1)^D
   \]
   \[
   \mathbf{i}_t = \sigma(\mathbf{x}_t^{(l)} \mathbf{W}_i^{(l)} + \mathbf{b}_i^{(l)}) \in (0, 1)^D
   \]
   \[
   \hat{\mathbf{h}}_t^{(l)} = \mathbf{f}_t \odot \mathbf{h}_{t-1}^{(l)} + \mathbf{i}_t \odot \mathbf{x}_t^{(l)}
   \]
   \[
   \mathbf{h}_t^{(l)} = \text{clamp}(\hat{\mathbf{h}}_t^{(l)}, -1.0, 1.0)
   \]
   \[
   \mathbf{y}_t^{(l)} = \mathbf{h}_t^{(l)} \mathbf{W}_o^{(l)} + \mathbf{b}_o^{(l)}
   \]

2. **Dense Multi-Head Self-Attention Sub-Layer** (for \(l \pmod{R+1} = R\)):

   \[
   \mathbf{Q}^{(l)} = \mathbf{X}^{(l)} \mathbf{W}_Q^{(l)}, \quad \mathbf{K}^{(l)} = \mathbf{X}^{(l)} \mathbf{W}_K^{(l)}, \quad \mathbf{V}^{(l)} = \mathbf{X}^{(l)} \mathbf{W}_V^{(l)}
   \]
   \[
   \mathbf{A}^{(l)} = \text{softmax}\left(\frac{\mathbf{Q}^{(l)} (\mathbf{K}^{(l)})^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right)
   \]
   \[
   \mathbf{Y}^{(l)} = \mathbf{A}^{(l)} \mathbf{V}^{(l)} \mathbf{W}_O^{(l)}
   \]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
                  Input Sequence X^(0) [B, L, D]
                               │
            ┌──────────────────┴──────────────────┐
            │ Layer l = 0, ..., N-1               │
            ▼                                     ▼
   (l mod (R+1) != R)                    (l mod (R+1) == R)
   HGRN2 Recurrence Block                 Dense MHA Block
   - Linear Gating f_t, i_t               - Q, K, V Projections
   - State Update & Clamp [-1, 1]         - Softmax(QK^T / sqrt(d_k))
   - Projection Out                       - Attention Context
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                   Residual: X^(l) + SubLayerOut
                               │
                               ▼
                   LayerNorm: [B, L, D] -> [B, L, D]
                               │
                               ▼
                  Output X^(N) [B, L, D]
```

| Layer Index \(l\) | Block Type | Input Shape | Sub-layer Execution | Residual + LayerNorm | Output Shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| \(l = 0\) | HGRN-2 | \([B, L, D]\) | Gated recurrence pass | `norm(x + hgrn(x))` | \([B, L, D]\) |
| \(\dots\) | HGRN-2 | \([B, L, D]\) | Gated recurrence pass | `norm(x + hgrn(x))` | \([B, L, D]\) |
| \(l = R-1\) | HGRN-2 | \([B, L, D]\) | Gated recurrence pass | `norm(x + hgrn(x))` | \([B, L, D]\) |
| \(l = R\) | Dense MHA | \([B, L, D]\) | Multi-Head Attention pass | `norm(x + mha(x))` | \([B, L, D]\) |
| \(l = R+1\) | HGRN-2 | \([B, L, D]\) | Gated recurrence pass | `norm(x + hgrn(x))` | \([B, L, D]\) |
| \(l = N-1\) | End Layer | \([B, L, D]\) | Final routing block | `norm(x + block(x))`| \([B, L, D]\) |

---

## 4. Analytical Complexity Profiling

### 4.1 Hybrid Analytical FLOP & MAC Profiling

Let \(N\) be the total number of layers, \(R\) the hybrid ratio, \(B\) batch size, \(L\) sequence length, and \(D\) hidden dimension:

- Fraction of HGRN-2 Recurrent Layers: \(\frac{R}{R + 1}\)
- Fraction of Dense MHA Attention Layers: \(\frac{1}{R + 1}\)

1. **Recurrent Layers Complexity (\(N \cdot \frac{R}{R+1}\) layers)**:
   - MACs per layer: \(3 B L D^2\).
   - FLOPs per layer: \(6 B L D^2 + 10 B L D\).
   - Total Recurrent FLOPs: \(N \cdot \frac{R}{R+1} \cdot \left(6 B L D^2 + 10 B L D\right)\).

2. **Attention Layers Complexity (\(N \cdot \frac{1}{R+1}\) layers)**:
   - MACs per layer: \(4 B L D^2 + 2 B L^2 D\).
   - FLOPs per layer: \(8 B L D^2 + 4 B L^2 D + 5 B L D\).
   - Total Attention FLOPs: \(N \cdot \frac{1}{R+1} \cdot \left(8 B L D^2 + 4 B L^2 D + 5 B L D\right)\).

3. **Combined Hybrid Complexity**:
   - **Total FLOPs**:
     \[
     \text{FLOPs}_{\text{total}} = N \cdot B \cdot L \cdot D \left[ \frac{6R + 8}{R + 1} D + \frac{4}{R + 1} L + \frac{10R + 5}{R + 1} \right]
     \]
   - **Scaling Behavior**: When \(R=3\), attention quadratic cost is reduced by \(75\%\) compared to a standard Transformer stack.

### 4.2 Memory Footprint & Activation Accounting

- **Parameter Storage**:
  - HGRN-2 Layer: \(3 D^2 + 5 D\) parameters.
  - MHA Layer: \(4 D^2 + 6 D\) parameters.
  - Total Parameters: \(N \left[ \frac{R(3D^2+5D) + (4D^2+6D)}{R+1} \right]\).

- **Training Activation Memory**:
  - HGRN-2 Layers: \(\mathcal{O}\left(N \cdot \frac{R}{R+1} \cdot B \cdot L \cdot D\right)\) bytes.
  - MHA Layers: \(\mathcal{O}\left(N \cdot \frac{1}{R+1} \cdot B \cdot H \cdot L^2\right)\) bytes.

- **Inference KV-Cache Memory**:
  - Stores KV-cache ONLY for the \(\frac{N}{R+1}\) attention layers, reducing KV-cache RAM footprint by \(\frac{R}{R+1} \times 100\%\).

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Sub-Layer Depth Expansion Overhead

In the current implementation, `inject_hgrn2_gated_deltanet` replaces the single attention module of a host Transformer layer with an entire stack of \(N\) blocks. Consequently, the input is transformed through \(N\) internal residual blocks within a single outer layer, creating potential deep gradient vanishing across un-normalized skip paths if weight initialization scale is uncalibrated.

### 5.2 Context Discrepancy Across Layer Transitions

When transitioning from an HGRN-2 layer (\(\mathcal{O}(1)\) recurrent state memory) into a Dense MHA layer (\(\mathcal{O}(L^2)\) token attention), the MHA layer receives feature maps produced by position-wise linear recurrences. If the HGRN-2 layers suffer from state saturation or rank collapse, the input to MHA exhibits reduced variance, degrading the attention matrix entropy.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Heterogeneous GPU Stream Pipeline

To maximize GPU hardware utilization across mixed MHA and HGRN-2 layers, the stack is executed using a CUDA Stream pipeline that overlaps HGRN-2 recurrent kernel processing with MHA FlashAttention-2 kernels.

```python
import triton
import triton.language as tl
import torch

# Fused Triton HGRN-2 Kernel for Recurrent Layers (Ratio R)
@triton.jit
def _hgrn2_stacked_kernel(
    X_ptr, Out_ptr, Wf_ptr, Wi_ptr,
    stride_b, stride_l, stride_d,
    B, L, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    h = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for t in range(L):
        off = pid_b * stride_b + t * stride_l + offs_d * stride_d
        x = tl.load(X_ptr + off, mask=mask_d, other=0.0)
        f_raw = tl.load(Wf_ptr + off, mask=mask_d, other=0.0)
        i_raw = tl.load(Wi_ptr + off, mask=mask_d, other=0.0)
        
        f = tl.sigmoid(f_raw.to(tl.float32))
        i = tl.sigmoid(i_raw.to(tl.float32))
        
        h = tl.clamp(f * h + i * x, -1.0, 1.0)
        tl.store(Out_ptr + off, h.to(X_ptr.dtype.element_ty), mask=mask_d)
```

### 6.2 Optimization Blueprint & Tile Parameters

- **Dynamic Kernel Dispatch**: For layers where \(l \pmod{R+1} \neq R\), launch `_hgrn2_stacked_kernel` with block size `BLOCK_D = 128`. For layers where \(l \pmod{R+1} = R\), dispatch FlashAttention-2 FP16 kernels.
- **SRAM Reuse**: Keeps intermediate activations in L2 cache across adjacent HGRN-2 layers, avoiding round-trips to high-bandwidth global memory (HBM).