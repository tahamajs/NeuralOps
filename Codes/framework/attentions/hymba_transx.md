# Hymba-TransX: Hybrid Differential Attention and State-Space Model Architecture with Unified RoPE

## 1. Abstract & Theoretical Motivation

Transformers excel at capturing non-local cross-token dependencies via Multi-Head Self-Attention (MHA), but suffer from quadratic compute/memory complexity \(\mathcal{O}(L^2)\) and attention noise. State-Space Models (SSMs), such as Mamba or SimpleSSM, achieve linear computational scaling \(\mathcal{O}(L)\) but can experience local detail loss over long sequence horizons.

The **Hymba-TransX Architecture** fuses both sequence mixing paradigms into a unified hybrid layer. The attention pathway employs **ReLU-Gated Differential Attention** (\(A_{\text{diff}} = \text{ReLU}(A_1 - 0.5 A_2)\)), which subtracts background distraction maps to isolate sharp attention peaks. Simultaneously, a parallel **SimpleSSM** stream processes linear state-space sequence dynamics. A Unified Rotary Position Embedding (RoPE) pre-encodes relative position across both streams, while a learnable Sigmoid gate \(\sigma(\alpha)\) dynamically blends attention and SSM feature flows. This paper details the mathematical formulations, dimensional shape tracing, analytical FLOP/MAC profiling, architectural failure modes, and GPU Triton optimization blueprints.

---

## 2. Mathematical Derivations & Formulations

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) be the input hidden state tensor.

### 2.1 Unified Rotary Position Embedding (Unified RoPE)

Before feature splitting, relative positional information is injected across all \(D\) channels. Frequencies \(\theta_k = b^{-2k/D}\) for \(k \in \{0, \dots, D/2 - 1\}\) construct phase matrix \(\mathbf{\Theta} \in \mathbb{R}^{L \times (D/2)}\):

\[
\mathbf{x}_{\text{rope}, 2k} = \mathbf{x}_{2k} \odot \cos(m \theta_k) - \mathbf{x}_{2k+1} \odot \sin(m \theta_k)
\]
\[
\mathbf{x}_{\text{rope}, 2k+1} = \mathbf{x}_{2k+1} \odot \cos(m \theta_k) + \mathbf{x}_{2k} \odot \sin(m \theta_k)
\]

resulting in position-encoded tensor \(\mathbf{X}_{\text{rope}} \in \mathbb{R}^{B \times L \times D}\).

### 2.2 Differential Attention Pathway

1. **Dual Query & Key Linear Projections**:
   \(\mathbf{X}_{\text{rope}}\) is projected into dual Query-Key pairs and shared Value representations across \(H\) heads with head dimension \(d_k = D / H\):

   \[
   \mathbf{Q}_1 = \mathbf{X}_{\text{rope}} \mathbf{W}_{q1}, \quad \mathbf{K}_1 = \mathbf{X}_{\text{rope}} \mathbf{W}_{k1} \quad \in \mathbb{R}^{B \times H \times L \times d_k}
   \]
   \[
   \mathbf{Q}_2 = \mathbf{X}_{\text{rope}} \mathbf{W}_{q2}, \quad \mathbf{K}_2 = \mathbf{X}_{\text{rope}} \mathbf{W}_{k2} \quad \in \mathbb{R}^{B \times H \times L \times d_k}
   \]
   \[
   \mathbf{V} = \mathbf{X}_{\text{rope}} \mathbf{W}_v \quad \in \mathbb{R}^{B \times H \times L \times d_k}
   \]

2. **Softmax Cancellation & ReLU Gating**:
   Dual Softmax maps \(\mathbf{A}_1, \mathbf{A}_2 \in \mathbb{R}^{B \times H \times L \times L}\) are evaluated:
   \[
   \mathbf{A}_1 = \text{softmax}\left(\frac{\mathbf{Q}_1 \mathbf{K}_1^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right), \quad \mathbf{A}_2 = \text{softmax}\left(\frac{\mathbf{Q}_2 \mathbf{K}_2^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right)
   \]
   The non-linear Differential Attention matrix \(\mathbf{A}_{\text{diff}}\) zero-masks non-salient connections:
   \[
   \mathbf{A}_{\text{diff}} = \text{ReLU}\left(\mathbf{A}_1 - 0.5 \mathbf{A}_2\right) \quad \in \mathbb{R}_{\ge 0}^{B \times H \times L \times L}
   \]

3. **Value Aggregation**:
   \[
   \mathbf{Y}_{\text{attn}} = \text{Reshape}\left(\mathbf{A}_{\text{diff}} \mathbf{V}, [B, L, D]\right) \quad \in \mathbb{R}^{B \times L \times D}
   \]

### 2.3 Parallel SimpleSSM Pathway & Gated Fusion

1. **SSM Sequence Processing**:
   Position-aware features \(\mathbf{X}_{\text{rope}}\) evolve through `SimpleSSM`:
   \[
   \mathbf{Y}_{\text{ssm}} = \text{SimpleSSM}(\mathbf{X}_{\text{rope}}, \mathbf{M}_{\text{pad}}) \quad \in \mathbb{R}^{B \times L \times D}
   \]

2. **Sigmoid Gated Blending**:
   Let \(\theta_\alpha \in \mathbb{R}\) be a learnable scalar (initialized to 0.5). The dynamic blending ratio \(\alpha \in (0, 1)\) is:
   \[
   \alpha = \sigma(\theta_\alpha) = \frac{1}{1 + e^{-\theta_\alpha}}
   \]
   \[
   \mathbf{Y}_{\text{fused}} = \alpha \cdot \mathbf{Y}_{\text{attn}} + (1 - \alpha) \cdot \mathbf{Y}_{\text{ssm}} \quad \in \mathbb{R}^{B \times L \times D}
   \]

3. **Residual Addition & Output Projection**:
   \[
   \mathbf{O} = (\mathbf{X} + \mathbf{Y}_{\text{fused}}) \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \quad \in \mathbb{R}^{B \times L \times D}
   \]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
                     Input Hidden States X [B, L, D]
                                    │
                                    ▼
                     Unified RoPE: X_rope [B, L, D]
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
   Differential Attention Pathway                    SimpleSSM Pathway
   - Q1, K1, Q2, K2, V: [B, H, L, d_k]               - Linear Recurrence Pass
   - A_1, A_2 Softmax: [B, H, L, L]                  - Y_ssm: [B, L, D]
   - A_diff = ReLU(A_1 - 0.5 A_2)                            │
   - Y_attn = A_diff @ V: [B, L, D]                          │
           │                                                 │
           └────────────────────────┬────────────────────────┘
                                    ▼
                   Sigmoid Gated Sum: alpha Y_attn + (1 - alpha) Y_ssm
                                    │
                                    ▼
                   Residual: X + Y_fused [B, L, D]
                                    │
                                    ▼
                     Output GEMM: W_out + b_out [B, L, D]
```

| Execution Step | Operation | Input Tensor Shapes | Output Tensor Shapes | Tensor Dimensional Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **1. Unified RoPE** | `self.rope` | `x` \([B, L, D]\) | `x_rope` | \([B, L, D] \rightarrow [B, L, D]\) |
| **2. Dual Q, K GEMMs** | `W_q1`, `W_k1`, `W_q2`, `W_k2` | `x_rope` | `q1, k1, q2, k2` | \([B, L, D] \rightarrow 4 \times [B, H, L, d_k]\) |
| **3. Value GEMM** | `W_v` | `x_rope` | `v` | \([B, L, D] \rightarrow [B, H, L, d_k]\) |
| **4. Dual Softmax** | Scaled Dot-Product | `q1, k1` and `q2, k2` | `attn1`, `attn2` | \(2 \times ([B, H, L, d_k] \times [B, H, d_k, L] \to [B, H, L, L])\) |
| **5. Differential ReLU**| `F.relu(attn1 - 0.5 * attn2)` | `attn1`, `attn2` | `diff_attn` | \([B, H, L, L] \rightarrow [B, H, L, L]\) |
| **6. Context BMM** | `diff_attn @ v` | `diff_attn`, `v` | `A_out` | \([B, H, L, L] \times [B, H, L, d_k] \rightarrow [B, L, D]\) |
| **7. SimpleSSM Pass** | `self.mamba` | `x_rope` | `M_out` | \([B, L, D] \rightarrow [B, L, D]\) |
| **8. Gated Fusion** | Sigmoid Blend + Residual | `x`, `A_out`, `M_out` | `fused` | \([B, L, D] + [B, L, D] \rightarrow [B, L, D]\) |
| **9. Output GEMM** | `self.out_proj` | `fused` \([B, L, D]\) | `out` | \([B, L, D] \rightarrow [B, L, D]\) |

---

## 4. Analytical Complexity Profiling

### 4.1 FLOP & MAC Operations Analysis

Let \(B\) be batch size, \(L\) sequence length, \(D\) hidden dimension, \(H\) head count, \(d_k = D / H\):

1. **Unified RoPE**:
   - Element-wise sinusoidal multiplications: \(4 B L D\) FLOPs.
2. **Linear Projections (`W_q1`, `W_k1`, `W_q2`, `W_k2`, `W_v`, `W_out`)**:
   - 6 GEMMs of size \((B \cdot L \times D) \times (D \times D)\).
   - MACs: \(6 B \cdot L \cdot D^2\).
   - FLOPs: \(12 B \cdot L \cdot D^2\).
3. **Differential Attention Pathway**:
   - Dual BMMs: \(4 B \cdot H \cdot L^2 \cdot d_k = 4 B L^2 D\) FLOPs.
   - Dual Softmax: \(10 B H L^2\) FLOPs.
   - Map Subtraction + ReLU: \(2 B H L^2\) FLOPs.
   - Value BMM: \(2 B \cdot H \cdot L^2 \cdot d_k = 2 B L^2 D\) FLOPs.
   - Total Attention FLOPs: \(6 B L^2 D + 12 B H L^2\).
4. **SimpleSSM Recurrence Pathway**:
   - State-space recurrence FLOPs: \(6 B \cdot L \cdot D^2 + 10 B L D\).
5. **Total Combined Complexity**:
   - **Total MACs**: \(12 B \cdot L \cdot D^2 + 3 B \cdot L^2 \cdot D\).
   - **Total FLOPs**: \(18 B \cdot L \cdot D^2 + 6 B \cdot L^2 \cdot D + 12 B H L^2 + 14 B L D\).

### 4.2 Memory Footprint Accounting

- **Parameter Count**:
  - `W_q1, W_k1, W_q2, W_k2, W_v`: \(5 D^2\) (bias-free).
  - `SimpleSSM`: \(3 D^2 + 3 D\).
  - `out_proj`: \(D^2 + D\).
  - Gate parameter `alpha`: \(1\).
  - Total Parameters: \(9 D^2 + 4 D + 1\). For \(D=768\): \(\approx 5.31 \times 10^6\) parameters (10.6 MB in FP16).

- **Training Activation Footprint**:
  - Dual Softmax maps \(\mathbf{A}_1, \mathbf{A}_2\): \(2 \times (B \cdot H \cdot L^2 \times 2)\) bytes.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Differential Map Over-Sparsification & Dead Gradient Zones

The operation \(\mathbf{A}_{\text{diff}} = \text{ReLU}(\mathbf{A}_1 - 0.5 \mathbf{A}_2)\) forces all positions where \(\mathbf{A}_{1}(i, j) \le 0.5 \mathbf{A}_{2}(i, j)\) to exact 0. The derivative of ReLU at non-positive values is zero:

\[
\frac{\partial \text{ReLU}(u)}{\partial u} = 0 \quad \forall u \le 0
\]

If \(\mathbf{A}_2\) outputs diffuse high-entropy scores across all tokens, \(\mathbf{A}_1 - 0.5 \mathbf{A}_2\) evaluates negative for over 95% of token pairs, causing widespread gradient dead zones that stall parameter updates for \(\mathbf{W}_{q1}\) and \(\mathbf{W}_{k1}\).

### 5.2 Alpha Gating Saturation

The scalar blending gate \(\alpha = \sigma(\theta_\alpha)\) operates globally across all tokens and dimensions. If \(\theta_\alpha\) saturates to large positive (\(\alpha \to 1\)) or negative (\(\alpha \to 0\)) values during training, the model completely severs either the attention or the SSM pathway, degrading back to a single-stream network.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### 6.1 Fused Differential Softmax & RoPE Kernel

To maximize CUDA Tensor Core occupancy, we integrate Unified RoPE, dual QK GEMMs, and differential Softmax into a single fused Triton kernel.

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _hymba_fused_diff_rope_kernel(
    Q1_ptr, K1_ptr, Q2_ptr, K2_ptr, V_ptr, Out_ptr,
    Cos_ptr, Sin_ptr,
    stride_b, stride_h, stride_l, stride_d,
    B, H, L, D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    
    # 1. Load Cosine/Sine tables and apply inline RoPE to Q1, K1, Q2, K2 in SRAM
    # 2. Compute inline scores: S1 = Q1 @ K1^T / sqrt(d_k), S2 = Q2 @ K2^T / sqrt(d_k)
    # 3. Softmax & Differential ReLU: A_diff = tl.maximum(tl.softmax(S1) - 0.5 * tl.softmax(S2), 0.0)
    # 4. Value matmul: Out_tile = A_diff @ V
    # 5. Store to DRAM
```

### 6.2 Optimization Blueprint Parameters

- **Inline RoPE Tiling**: Performs vector rotations directly inside SM Register Files, eliminating intermediate DRAM writes for position-encoded tensors.
- **SRAM Differential Tiling**: Stores `A_1` and `A_2` tiles in Shared Memory without writing full \(B \times H \times L \times L\) attention matrices back to VRAM, reducing memory footprint by \(\sim 6\times\).