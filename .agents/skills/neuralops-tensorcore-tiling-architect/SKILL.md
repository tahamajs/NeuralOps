---
name: neuralops-tensorcore-tiling-architect
description: Multi-phase 2D SRAM tile scheduling, double buffering, and register budgeting for GEMM, Kronecker, and Nyström matrix kernels on Tensor Cores.
---

# NeuralOps Tensor Core Tiling Architect Protocol

When designing matrix multiplication, projection, Kronecker product, or pseudo-inverse kernels (e.g., `fused_kronecker_gemm`, `fused_nystrom_landmark_pinv`), achieving high compute efficiency requires orchestrating 2D shared memory (SRAM) tiles that saturate Ampere/Ada/Hopper Tensor Cores while avoiding bank conflicts and memory stalls.

This skill outlines a rigorous 5-phase methodology for designing high-performance 2D tiled kernels.

---

## Phase 1: Mathematical Decomposition & Work Partitioning
1. **GEMM Tiling Model**:
   - Decompose $C = A \cdot B$ where $A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}, C \in \mathbb{R}^{M \times N}$.
   - Select macro-tile dimensions $(B_M, B_N, B_K)$ mapped to each Thread Block (or Triton program instance).
   - Partition macro-tiles into warp-level micro-tiles $(W_M, W_N)$ and MMA instruction shapes (e.g., $16 \times 8 \times 16$ for Tensor Cores).
2. **Kronecker Factorization Mapping**:
   - For $Y = (A \otimes B) X$, rewrite as tensor contraction:
     $$Y_{i_1, i_2, j} = \sum_{k_1, k_2} A_{i_1, k_1} B_{i_2, k_2} X_{k_1, k_2, j}$$
   - Restructure into two sequential GEMM stages: $T_{i_2, k_1, j} = B_{i_2, k_2} X_{k_1, k_2, j}$, followed by contraction with $A$. Never materialize the dense $(d_1 d_2 \times d_1 d_2)$ Kronecker matrix.

---

## Phase 2: SRAM Layout & Bank Conflict Elimination
1. **Shared Memory Allocation**:
   - Calculate SRAM requirements per block:
     $$\text{SRAM} = 2 \times (B_M \times B_K + B_K \times B_N) \times \text{sizeof(dtype)}$$
   - Ensure SRAM consumption remains $\le 48\text{ KB}$ per threadblock (or up to $96\text{ KB}$ on Ampere when dynamically opted-in).
2. **Shared Memory Padding (Swizzling)**:
   - To eliminate 32-bank shared memory conflicts during column-wise reads, pad the inner dimension:
     $$\text{stride}_{\text{SRAM}} = B_K + \text{PAD}, \quad \text{PAD} = 8 \text{ elements (16 bytes)}$$
   - In Triton, rely on `tl.dot` layout transformations, maintaining tile dimensions as multiples of 16/32.

---

## Phase 3: Asynchronous Pipeline & Double Buffering
1. **Multi-Stage Software Pipelining**:
   - Overlap global-to-shared memory loads with Tensor Core computation using double buffering (`num_stages=3` or `4`).
   - Stage 0: Issue async load from HBM to SRAM for tile $k=0$.
   - Stage 1: Issue async load for tile $k=1$ while Tensor Cores multiply tile $k=0$.
   - Pipeline advances across the $K$ dimension without CPU-style synchronization barriers.
2. **Register File Budgeting**:
   - Accumulators must reside in FP32 registers ($B_M \times B_N$ per program).
   - Constrain $B_M \times B_N$ such that register usage does not exceed 128 registers per thread, sustaining at least 4 active warps per SM.

---

## Phase 4: Dynamic Autotuning & Shape Specialization
1. **Search Space Grid**:
   - Sweep tile dimensions across:
     - $B_M \in \{32, 64, 128\}$
     - $B_N \in \{32, 64, 128\}$
     - $B_K \in \{32, 64\}$
     - `num_warps` $\in \{4, 8\}$
     - `num_stages` $\in \{2, 3, 4\}$
2. **Heuristic Pruning**:
   - Discard configurations where $B_M \times B_N \times \text{num\_stages} \times 4 > \text{Max Shared Memory}$.
   - Prune configurations where the number of active thread blocks per SM $< 2$.

---

## Phase 5: Parity & Numerical Certification
1. **Accumulation Precision**:
   - Force accumulation in FP32 (`tl.float32`) even when inputs are FP16 or BF16.
   - For Nyström landmark inversion: perform pseudo-inverse via SVD or regularized Cholesky decomposition with jitter $\epsilon \ge 10^{-6}$ added to diagonal elements.
2. **Gradient Checks**:
   - Certify analytical backward pass with `torch.autograd.gradcheck(..., atol=1e-3, eps=1e-5)`.
   - Record and log measured TFLOPs against theoretical Tensor Core peak.
