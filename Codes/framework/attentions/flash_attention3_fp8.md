# FlashAttention-3 FP8: Low-Precision 8-Bit Float Transformer Attention for Hopper Architectures

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Self-Attention (MHSA) in Transformer models operating under FP16 or BF16 precision consumes significant High Bandwidth Memory (HBM) activation storage and memory bandwidth. Intermediate attention score materialization requires $\mathcal{O}(L^2)$ space, creating memory bandwidth bottlenecks on modern GPU accelerators.

**FlashAttention-3 FP8** combines exact tiled attention algorithms with hardware-accelerated 8-bit floating point precision (E4M3 format) for NVIDIA Hopper GPUs:
1. **FP8 Quantization Protocol**: Activations $Q, K, V$ are cast to 8-bit floating-point format (`torch.float8_e4m3fn`), halving memory bandwidth consumption and SRAM-HBM data transfer volume.
2. **2x Tensor Core Throughput Acceleration**: Utilizes Hopper WGMMA (Warpgroup Matrix Multiply Accumulate) FP8 instructions, achieving up to $2\times$ higher matrix multiplication TFLOPs over FP16.
3. **IO-Optimal Tiled Memory Reduction**: Integrates online softmax reduction to keep intermediate attention logits inside GPU shared memory (SRAM), maintaining exact attention output without HBM logit allocation.

---

## 2. Mathematical Derivations & Formulations

### 2.1 FP8 E4M3 Precision Quantization
Input hidden activations $X \in \mathbb{R}^{B \times L \times D}$ are projected to query, key, and value tensors $Q, K, V \in \mathbb{R}^{B \times L \times H \times d_h}$ in base precision (FP16/BF16).

Prior to attention core execution, tensors are cast to FP8 E4M3 format (`torch.float8_e4m3fn`), characterized by 1 sign bit, 4 exponent bits, and 3 mantissa bits:

$$Q_{\text{fp8}} = \text{cast}_{\text{fp8\_e4m3}}(Q) \in \mathbb{F}_{p8}^{B \times L \times H \times d_h}$$

$$K_{\text{fp8}} = \text{cast}_{\text{fp8\_e4m3}}(K) \in \mathbb{F}_{p8}^{B \times L \times H \times d_h}$$

$$V_{\text{fp8}} = \text{cast}_{\text{fp8\_e4m3}}(V) \in \mathbb{F}_{p8}^{B \times L \times H \times d_h}$$

### 2.2 Exact Tiled Memory Reduction in FP8
Query, key, and value tensors are partitioned into SRAM blocks $B_r \times d_h$ and $B_c \times d_h$. For query block $Q_{\text{fp8}, i}$ and key-value block $(K_{\text{fp8}, j}, V_{\text{fp8}, j})$, WGMMA FP8 Tensor Core instructions evaluate dot products:

$$S_{i, j} = \frac{Q_{\text{fp8}, i} K_{\text{fp8}, j}^T}{\sqrt{d_h}} \in \mathbb{R}^{B_r \times B_c}$$

Online Softmax statistics $m_i^{(j)}$ (maximum tracking) and $l_i^{(j)}$ (sum tracking) update in SRAM registers:

$$m_i^{(j)} = \max\left( m_i^{(j-1)}, \; \text{rowmax}(S_{i, j}) \right)$$

$$\tilde{P}_{i, j} = \exp\left( S_{i, j} - m_i^{(j)} \right)$$

$$l_i^{(j)} = \exp\left( m_i^{(j-1)} - m_i^{(j)} \right) \odot l_i^{(j-1)} + \text{rowsum}(\tilde{P}_{i, j})$$

The unnormalized context output is updated:

$$O_i^{(j)} = \operatorname{diag}\left( \exp\left( m_i^{(j-1)} - m_i^{(j)} \right) \right) O_i^{(j-1)} + \tilde{P}_{i, j} V_{\text{fp8}, j}$$

Upon sequence loop completion, output $O_i$ is normalized and cast back to base precision:

$$O_i = \text{cast}_{\text{dtype}}\left( \operatorname{diag}\left( l_i^{(\text{final})} \right)^{-1} O_i^{(\text{final})} \right) \in \mathbb{R}^{B_r \times d_h}$$

### 2.3 Output Projection
Assembled context representations $O \in \mathbb{R}^{B \times L \times H \cdot d_h}$ are projected through output linear layer $W_{\text{out}}$:

$$Y = O W_{\text{out}} + b_{\text{out}} \in \mathbb{R}^{B \times L \times D}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input Activation** | $X$ | $[B, L, D]$ | Base activation state |
| **Linear Q/K/V GEMM** | $Q, K, V$ | $[B, L, D] \to [B, L, H, d_h]$ | Linear projection into multi-head view |
| **FP8 Quantization** | $Q_{\text{fp8}}, K_{\text{fp8}}, V_{\text{fp8}}$| $[B, L, H, d_h]$ in FP16/BF16 $\to [B, L, H, d_h]$ in FP8 E4M3 | Cast to `torch.float8_e4m3fn` |
| **TMA Asynchronous Load** | $K_{\text{fp8}, j}, V_{\text{fp8}, j}$| HBM $\to$ SRAM Shared Memory | Hardware TMA load into SRAM tiles |
| **FP8 Tile GEMM $Q K^T$** | $S_{i, j}$ | SRAM: $[B_r, d_h] \times [B_c, d_h]^T \to [B_r, B_c]$ | WGMMA FP8 Tensor Core matmul |
| **Online Softmax Update** | $m_i^{(j)}, l_i^{(j)}$ | SRAM Registers: $[B_r]$ | Online max and sum tracking |
| **FP8 Tile GEMM $\tilde{P} V$** | $O_i^{(j)}$ | SRAM: $[B_r, B_c] \times [B_c, d_h] \to [B_r, d_h]$ | Value contraction into SRAM accumulators |
| **Normalizer Rescaling** | $O_i$ | SRAM Registers: $[B_r, d_h]$ | Division by $l_i^{(\text{final})}$ & cast to base dtype |
| **HBM Output Writeback** | $O$ | SRAM $\to$ HBM: $[B, L, H, d_h]$ | Write context output to global DRAM |
| **Output Projection** | $Y$ | $[B, L, H \cdot d_h] \to [B, L, D]$ | Layer output linear projection |

---

## 4. Analytical Complexity Profiling

### 4.1 Floating Point Operations (FLOPs)
Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H$ head count, and $d_h = D/H$ head dimension.

1. **Linear Projections ($Q, K, V, O$)**:
   $$\text{FLOPs}_{\text{proj}} = 4 \times 2 B L D^2 = 8 B L D^2$$
2. **FlashAttention-3 FP8 Kernel Core**:
   - Query-Key score matmul: $2 B H L^2 d_h = 2 B L^2 D$
   - Softmax online scaling & Value contraction: $2 B H L^2 d_h = 2 B L^2 D$
   $$\text{FLOPs}_{\text{core\_fp8}} = 4 B L^2 D \quad (\text{executed at } 2\times \text{ Tensor Core throughput})$$

**Total FLOP Count**:

$$\text{FLOPs}_{\text{total}} = 8 B L D^2 + 4 B L^2 D$$

### 4.2 Memory Footprint & Bandwidth
- **SRAM-HBM Memory Traffic**: 8-bit precision halves $Q, K, V$ data movement volume from $2 \text{ bytes/element}$ (FP16/BF16) to $1 \text{ byte/element}$ (FP8).
- **HBM Activation IO Complexity**: $\mathcal{O}\left(B L D + \frac{B H L^2 d_h^2}{M_{\text{SRAM}}}\right)$.
- **Parameter Footprint**: $\Theta = 4 D^2$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Dynamic Range Clipping in E4M3 FP8 Format
- **Mechanism**: The E4M3 format supports a maximum finite value of 448. Unscaled attention logits $Q K^T / \sqrt{d_h}$ exceeding 448 overflow to `NaN` during kernel execution.
- **Mitigation**: Apply dynamic pre-scaling factors $S_Q, S_K$ before casting: $Q_{\text{fp8}} = \text{cast}(Q \cdot S_Q)$.

### 5.2 Hardware Requirement Bottleneck
- **Mechanism**: FP8 WGMMA hardware instructions are strictly supported on NVIDIA Hopper GPUs (`sm_90a`). Invoking FP8 kernels on older architectures causes immediate CUDA illegal instruction errors.
- **Mitigation**: Implement automated fallback to PyTorch SDPA (`F.scaled_dot_product_attention`).

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

```
+-----------------------------------------------------------------------------------+
|                        Hopper Hardware Execution Graph                            |
|                                                                                   |
|  Input X [B, L, D] ---> [Linear Q, K, V Projections]                              |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                  |       |
|        v (Producer Warps)                                                 v       |
| [TMA Asynchronous Load Engine]                    [FP8 Quantization Unit]         |
|  - Load Q, K, V Tiles in FP8 E4M3                 - Cast FP16/BF16 -> E4M3 FP8     |
|  - 50% Reduced HBM Traffic                        - Apply Pre-Scaling Factors     |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                          |
|        +---> (Consumer Warps) [FP8 WGMMA Tensor Core Engine]                      |
|               - 2x Throughput FP8 GEMM Q_i K_j^T                                  |
|               - SRAM Online Softmax Rescaling (m_i, l_i)                          |
|               - FP8 Value Reduction \tilde{P}_{i, j} V_j                          |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        |                                                                          |
|        +---> [HBM Async Writeback] ---> Output Y [B, L, D]                        |
+-----------------------------------------------------------------------------------+
```

### 6.1 Triton Strategy 1: Vectorized FP8 Quantization Kernel
- Implement a custom Triton kernel for FP8 quantization: load FP16/BF16 activations into SRAM, evaluate per-channel maximums, scale by $S_Q$, and cast to `torch.float8_e4m3fn` using vectorized 16-byte packed writes.

### 6.2 Triton Strategy 2: Hopper WGMMA FP8 Tensor Core Pipeline
- Issue native FP8 WGMMA matrix instructions on Consumer warps, achieving up to $2\times$ higher TFLOP execution speeds over FP16 Tensor Cores.
- Use TMA asynchronous load engines to prefetch FP8 key-value tiles into shared memory pipelines without warp blocking.
