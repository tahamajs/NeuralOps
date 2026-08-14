# HGRN-2 Layer: Hierarchically Gated Recurrent Network v2 Element-Wise Recurrence

## 1. Executive Abstract & Architectural Motivation

Standard Multi-Head Attention scales quadratically $\mathcal{O}(L^2)$ in time and memory. The Hierarchically Gated Recurrent Network v2 (HGRN-2 Layer, Qin et al., 2024) provides an element-wise linear recurrence block that operates in strictly $\mathcal{O}(L)$ sequence time and constant $\mathcal{O}(D)$ state memory.

HGRN-2 replaces 2D matrix recurrences with element-wise vector recurrence updates modulated by dynamic sigmoidal forget gates $f_t = \sigma(W_f x_t)$ and input gates $i_t = \sigma(W_i x_t)$. By incorporating residual state clipping bounds ($\text{Clamp}(h_t, -C, C)$ with $C = 1.0$), HGRN-2 prevents state explosion and vanishing gradient instability, enabling fast and stable long-sequence modeling.

---

## 2. Mathematical Formulations & Derivations

Given input sequence representations $X \in \mathbb{R}^{B \times L \times D}$, gating signals are derived per step $t$:

\[
f_t = \sigma(W_f x_t + b_f) \in (0, 1)^D \quad \text{(Forget Gate)}
\]
\[
i_t = \sigma(W_i x_t + b_i) \in (0, 1)^D \quad \text{(Input Gate)}
\]

### Element-Wise Recurrent State Update & Bounded Clipping:
State vector $h_t \in \mathbb{R}^{B \times D}$ is updated recursively:

\[
\widetilde{h}_t = f_t \odot h_{t-1} + i_t \odot x_t \in \mathbb{R}^{B \times D}
\]
\[
h_t = \text{Clamp}(\widetilde{h}_t, -C, C) \in \mathbb{R}^{B \times D} \quad \text{where } C = 1.0
\]

### Output Gating & Transformation:
Output representation $y_t \in \mathbb{R}^{B \times D}$ is produced via output projection:

\[
y_t = W_O h_t \in \mathbb{R}^{B \times D}
\]

Stacking outputs across steps produces $Y = [y_1, \dots, y_L] \in \mathbb{R}^{B \times L \times D}$.

---

## 3. Step-by-Step Tensor Execution Trace

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, D)[W_f] ──> Sigmoid ──> Forget Gate f: [B, L, D]
  ├──> Linear(D, D)[W_i] ──> Sigmoid ──> Input Gate i: [B, L, D]
  │
  ├──> Recurrent Loop over sequence steps t = 1 ... L:
  │      ├──> Element-wise update: h_tilde_t = f_t * h_{t-1} + i_t * x_t ──> [B, D]
  │      └──> Bounded Clamping: h_t = Clamp(h_tilde_t, -1.0, 1.0) ──> [B, D]
  │
  ├──> Stack states across L ──> H: [B, L, D]
  └──> Linear(D, D)[W_O](H) ──> Output: [B, L, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input | $[B, L, D]$ | Input hidden state sequence |
| 2 | Forget Gate Generation | $[B, L, D]$ | Sigmoidal gate $f_t = \sigma(W_f x_t)$ |
| 3 | Input Gate Generation | $[B, L, D]$ | Sigmoidal gate $i_t = \sigma(W_i x_t)$ |
| 4 | Element-Wise Scan Update | $[B, L, D]$ | Recurrent state updates $\widetilde{h}_t = f_t \odot h_{t-1} + i_t \odot x_t$ |
| 5 | Bounded State Clamping | $[B, L, D]$ | Clamping within $[-1.0, 1.0]$ |
| 6 | Out Projection | $[B, L, D]$ | Final linear transformation $W_O$ |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown ($B$: Batch, $L$: Length, $D$: Model Dim):
- **Gate Projections ($W_f, W_i$)**: $4 B L D^2$ FLOPs
- **Element-wise Recurrence & Clamping**: $4 B L D$ FLOPs
- **Output Projection ($W_O$)**: $2 B L D^2$ FLOPs

**Total Complexity**:
- **Total MACs**: $3 B L D^2$
- **Total FLOPs**: $6 B L D^2$ ($\mathcal{O}(L)$ strictly linear sequence scaling)

### Memory Footprint:
- **Static Weights**: $3 D^2$ parameters.
- **Decoding Memory Footprint**: Bounded by $\mathcal{O}(B \cdot D)$ state memory per step.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

1. **State Saturation under Extreme Inputs**: Constant clamping at $[-1.0, 1.0]$ can clip gradient magnitude if features frequently saturate the boundary.
2. **Channel Independence Bottleneck**: Element-wise recurrence operates independently across hidden channels, relying entirely on linear projections for cross-channel mixing.
3. **Sequential Loop Latency in Unfused Execution**: PyTorch step loops introduce CPU launch latencies.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton kernel for HGRN-2 Layer fuses gate calculation, element-wise recurrence, and state clamping inside GPU SRAM.

```python
# Triton Fused HGRN-2 Scan Blueprint
import triton
import triton.language as tl

@triton.jit
def _hgrn2_layer_fused_kernel(
    X_ptr, W_f_ptr, W_i_ptr, Out_ptr,
    stride_xb, stride_xl, stride_xd,
    L, D,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    
    h_state = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t in range(L):
        x_t = tl.load(X_ptr + pid_b * stride_xb + t * stride_xl + offs_d)
        
        f_t = tl.sigmoid(tl.dot(x_t[None, :], W_f_ptr))
        i_t = tl.sigmoid(tl.dot(x_t[None, :], W_i_ptr))
        
        h_state = f_t * h_state + i_t * x_t
        h_state = tl.clamp(h_state, -1.0, 1.0)
        
        tl.store(Out_ptr + pid_b * stride_xb + t * stride_xl + offs_d, h_state)
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Multi-Head HGRN-2 Layer**: Partition hidden channels into parallel heads with head-specific clamping thresholds $C_h$.
2. **HGRN2-Mamba Hybrid**: Interleave HGRN-2 element-wise layers with Mamba selective scan modules.
3. **FP8 Quantized Scan**: Quantize recurrence states to FP8 format for 2x memory bandwidth acceleration.
