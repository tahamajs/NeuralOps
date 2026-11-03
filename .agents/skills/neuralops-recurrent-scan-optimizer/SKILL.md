---
name: neuralops-recurrent-scan-optimizer
description: Multi-phase engineering workflow for parallel prefix scans, chunkwise state recurrences, and associative scan acceleration in linear attention kernels.
---

# NeuralOps Recurrent Scan Optimizer Protocol

Recurrent state updates (such as Gated Linear Attention / `gla_scan`, Hated Gated Recurrent Network / `hgrn2_scan`, Mamba Selective Scan / `selective_scan`, and `fms_scan`) suffer from sequential $O(T)$ latency bottlenecks when implemented naively. High-performance implementations transform sequential recurrences into chunk-parallel associative prefix scans running in $O(T/C)$ parallel steps.

This skill provides a 5-phase engineering protocol to structure, optimize, and verify chunk-parallel associative scans on modern GPU architectures.

---

## Phase 1: Associative Operator Formalization
1. **Define Associative State Transition**:
   - Express the linear recurrence in state-space form:
     $$h_t = a_t \odot h_{t-1} + b_t \odot x_t$$
   - Express the transition tuple as $(a_t, u_t)$ where $u_t = b_t \odot x_t$.
   - Define the binary operator $\oplus$:
     $$(a_j, u_j) \oplus (a_i, u_i) = (a_j \odot a_i, \, a_j \odot u_i + u_j)$$
   - Mathematically verify the associativity property:
     $$((a_3, u_3) \oplus (a_2, u_2)) \oplus (a_1, u_1) = (a_3, u_3) \oplus ((a_2, u_2) \oplus (a_1, u_1))$$
2. **Numerical Stability Formulation**:
   - In decay factors $a_t = \exp(-\Delta_t)$, compute in log-space or clamp $\Delta_t \ge 0$ to guarantee strictly stable non-explosive state dynamics: $|a_t| \le 1.0$.

---

## Phase 2: Chunk-Parallel Hierarchical Decomposition
1. **Three-Tier Chunk Scan Architecture**:
   - Partition sequence length $T$ into chunks of length $C$ (typically $C = 16, 32, \text{ or } 64$):
     - **Intra-Chunk Local Scan**: Each threadblock computes the prefix scan within its chunk of length $C$ using fast shared memory or register shuffling.
     - **Inter-Chunk State Propagation**: Sequentially or hierarchically propagate chunk-boundary hidden states $H_k = h_{k \cdot C}$ across the $T/C$ chunks.
     - **Intra-Chunk Correction**: Broadcast boundary state $H_{k-1}$ into chunk $k$ and add the residual update across all tokens in the chunk.
2. **Memory Footprint**:
   - Avoid writing the full sequence of intermediate recurrent states $(B, H, T, D_k, D_v)$ to global DRAM.
   - Only materialize boundary states $(B, H, T/C, D_k, D_v)$, reducing DRAM writes by a factor of $C$.

---

## Phase 3: Hardware Implementation (Triton / CUDA)
1. **Triton Intra-Chunk Parallelism**:
   - In Triton, represent chunks as tile dimensions `BLOCK_T = 32` or `64`.
   - Use vectorized register arrays for the recurrent state.
   - For linear attention with key-value outer product $u_t = k_t v_t^\top$:
     - Compute intra-chunk attention matrix via local GEMM: $S_{\text{local}} = Q_{\text{chunk}} K_{\text{chunk}}^\top \odot M_{\text{decay}}$.
     - Multiply by $V_{\text{chunk}}$ locally.
2. **CUDA Warp-Shuffle Parallel Scan**:
   - In native CUDA C++, implement intra-warp scan using `__shfl_up_sync()` across 32 threads in the warp.
   - Blelloch or Kogge-Stone scan pattern provides $\log_2(32) = 5$ step latency per warp.

---

## Phase 4: Backward Pass Derivation & Adjoint Scan
1. **Reverse Associative Scan**:
   - Derive adjoint state updates backwards in time:
     $$dh_{t-1} = a_t \odot dh_t, \quad dx_t = b_t \odot dh_t$$
   - Structure the backward pass as a reverse chunk-parallel associative scan.
2. **Recomputation Strategy**:
   - Rather than saving all intermediate states $h_t$ (which exhausts DRAM for long sequences $T \ge 16\text{k}$), recompute intra-chunk states from checkpointed chunk-boundary states $H_k$.

---

## Phase 5: Verification & Parity Certification
1. **Positive Step Enforcements**:
   - Ensure step sizes $\Delta > 0$ to prevent numerical explosion or imaginary roots.
2. **Parity Against PyTorch Reference**:
   - Verify forward difference: $\max |Y_{\text{kernel}} - Y_{\text{ref}}| < 1 \times 10^{-5}$.
   - Verify backward gradients: $\max |dX_{\text{kernel}} - dX_{\text{ref}}| < 1 \times 10^{-5}$.
   - Run `torch.autograd.gradcheck` on CPU/GPU tensors with `atol=1e-3, eps=1e-5`.
