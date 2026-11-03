---
name: neuralops-memory-bandwidth-saturator
description: Multi-phase engineering workflow for saturating HBM/GDDR DRAM bandwidth on memory-bound elementwise, activation, and reduction kernels.
---

# NeuralOps Memory Bandwidth Saturator Protocol

For memory-bound kernels (elementwise activations like `softplus`, pointwise transformations, fused cross-entropy, fused knowledge distillation, and normalization ops), compute intensity is low ($< 5$ FLOP/byte). Execution speed is dictated almost entirely by memory coalescing, vector load width, and threadblock grid dimensions.

This skill defines an end-to-end 5-phase optimization workflow to achieve $>85\%$ theoretical DRAM bandwidth utilization (e.g., $>790\text{ GB/s}$ of the $936\text{ GB/s}$ peak on RTX 3090).

---

## Phase 1: Theoretical Peak & Arithmetic Intensity Profiling
1. **Identify Memory Bound Status**:
   - Calculate arithmetic intensity $I = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$.
   - If $I < \frac{\text{Peak TFLOPs}}{\text{Peak Bandwidth TB/s}}$ (e.g., $I < 35.6 / 0.936 \approx 38\text{ FLOP/byte}$ on RTX 3090), the kernel is strictly memory-bound.
2. **Establish Empirical Lower Bound**:
   - Profile the PyTorch reference implementation with `torch.cuda.Event`.
   - Calculate effective bandwidth:
     $$\text{BW}_{\text{eff}} = \frac{\text{Bytes Read} + \text{Bytes Written}}{\text{Runtime (seconds)} \times 10^9} \text{ GB/s}$$
   - Compare against theoretical hardware peak.

---

## Phase 2: Vectorized 128-Bit Coalescing & Tile Sizing
1. **Vectorized Load/Store Width**:
   - Ensure consecutive threads load contiguous memory addresses aligned to 16 bytes (128 bits).
   - In Triton, use vectorized pointers matching power-of-two blocks (`BLOCK_SIZE = 2048, 4096, 8192`).
   - In CUDA C++, use `reinterpret_cast<const float4*>` or `reinterpret_cast<const uint4*>`.
2. **Block Sizing for Grid Saturation**:
   - Low block sizes (`BLOCK_SIZE = 1024`) leave memory controllers under-subscribed.
   - Scale block size to $4096$ or $8192$ elements per thread block with `num_warps = 8` or `16`.
   - Ensure the total grid size $G = \lceil N / \text{BLOCK\_SIZE} \rceil$ is at least $4 \times$ the number of streaming multiprocessors (SMs) on the chip to conceal memory latency.

---

## Phase 3: Kernel Fusion & Temporary Allocation Elimination
1. **Intermediate Buffer Elimination**:
   - Prevent round-trips to global HBM memory by fusing operations into single kernels:
     - Fused KD: $\mathcal{L} = \alpha D_{\text{KL}}(P_\tau \parallel Q_\tau) + (1-\alpha) \mathcal{L}_{\text{CE}}(P, y)$. Compute both terms in a single pass over logits.
     - Softplus: $y = \frac{1}{\beta} \ln(1 + e^{\beta x})$. Compute numerically stable log-sum-exp in registers without creating temporary exponential tensors.
2. **Registers Over SRAM**:
   - Keep per-element intermediate variables strictly in registers (`tl.exp`, `tl.log`, `tl.maximum`).
   - Guarantee zero local memory spilling (`st.local`).

---

## Phase 4: Hardware Measurement & Nsight Metrics
1. **Run Roofline Hardware Audit**:
   - Profile using `ncu --metrics dram__throughput.pct_of_peak_sustained_elapsed,gpu__time_duration.sum`.
   - Target metrics:
     - DRAM Bandwidth Utilization: $\ge 85\%$ of sustained peak.
     - L2 Cache Hit Rate: Analyzed for input data locality.
     - Sector Requests per Transaction: Exactly 4 sectors (32 bytes) per transaction (100% coalescing).
2. **Verify Cold-Cache Isolation**:
   - Always run microbenchmarks with L2 cache clearance buffers before measuring execution time.

---

## Phase 5: Verification & Safety Gates
1. **Numerical Invariant Checking**:
   - FP16/BF16 computations must accumulate in FP32 inside the kernel before casting back to output precision.
   - For softplus: Ensure threshold switching ($x \cdot \beta > 20 \implies y = x$) to prevent floating point overflow (`inf`).
2. **Backend Contract Verification**:
   - Verify level 2 (Triton) or level 1 (CUDA) backend is actively registered.
   - Enforce pass on `torch.testing.assert_close(..., atol=1e-5, rtol=1e-5)`.
