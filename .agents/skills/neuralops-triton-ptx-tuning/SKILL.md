---
name: neuralops-triton-ptx-tuning
description: Multi-phase Triton PTX lowering, architecture targeting, and JIT compilation tuning across heterogeneous NVIDIA driver and compute capability constraints.
---

# NeuralOps Triton PTX Tuning & Lowering Protocol

When optimizing Triton JIT kernels across diverse cluster hardware and driver baselines (e.g., CUDA 11.3 Driver 465.19 vs CUDA 12.x Driver 535+), PTX forward-compatibility and instruction selection directly govern whether kernels execute natively or trigger silent fallback paths.

This skill provides an exhaustive 5-phase engineering protocol to diagnose, inspect, tune, and lock down Triton compilation pipelines for maximum throughput and compatibility.

---

## Phase 1: Driver Capability & PTX Version Introspection
1. **Query Driver PTX Ceiling**:
   - Inspect NVIDIA driver version and target SM capability:
     ```bash
     nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader
     ```
   - Calculate maximum supported PTX ISA version for the host driver (e.g., Driver 465 supports up to PTX 7.3; CUDA 11.8 / Triton 2.1 defaults to PTX 7.4+ causing `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).
2. **Environment Overrides**:
   - Explicitly configure environment variables prior to Triton invocation:
     ```bash
     export TRITON_PTXAS_PATH=$(which ptxas)
     export TRITON_ALWAYS_COMPILE=1
     export TRITON_PRINT_INTERMEDIATE_ASM=1
     ```

---

## Phase 2: Intermediate Representation (IR) & Assembly Disassembly
1. **Extract Triton IR Pipeline**:
   - Dump AST $\to$ TTIR (Triton Temporal IR) $\to$ TTGIR (Triton GPU IR) $\to$ LLVM-IR $\to$ PTX.
   - Run compilation inspection:
     ```python
     import triton
     # Inspect compiled kernel metadata
     compiled_kernel = kernel[grid](*args)
     print(compiled_kernel.asm["ttir"])
     print(compiled_kernel.asm["ptx"])
     ```
2. **Analyze PTX Register Spilling & Shared Memory Footprint**:
   - Locate register pressure markers in generated PTX:
     - `.reg .b32 %r<N>`: Ensure $N \le 64$ to maintain 100% theoretical occupancy per SM.
     - `.shared .align 16 .b8`: Verify shared memory allocation does not exceed 48KB/96KB SM capacity.
     - Search for local memory spills: `st.local`, `ld.local` (indicates fatal cache thrashing).

---

## Phase 3: Hardware Architecture Micro-Targeting
1. **Compute Capability Tuning**:
   - For Ampere (SM 8.0, SM 8.6, SM 8.9): Enable `num_warps=4` or `num_warps=8`, `num_stages=3` or `4` to overlap global memory pipeline (`cp.async`) with ALU execution.
   - For Turing/Volta (SM 7.0, SM 7.5): Set `num_stages=2` and constrain `num_warps=4` to prevent register exhaustion.
2. **Elimination of Deprecated Constructs**:
   - Guarantee zero `tl.make_block_ptr` / `tl.advance` usage in mainline kernels (strictly banned by repository invariant).
   - Use standard 2D pointer arithmetics: `ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n`.

---

## Phase 4: Dynamic JIT Autotuning Matrix
1. **Configure Triton Autotuner Configurations**:
   - Define multi-stage tuning space across tile sizes and warp counts:
     ```python
     @triton.autotune(
         configs=[
             triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
             triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
             triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
             triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
         ],
         key=['M', 'N', 'K'],
     )
     ```
2. **Cold-Cache Benchmarking**:
   - Flush L2 cache (using a 256MB dummy tensor) between autotuning sweeps to measure pure compute and DRAM throughput rather than warm-cache artifacts.

---

## Phase 5: Verification & Zero-Fallback Enforcement
1. **Dispatch Backend Audit**:
   - Ensure the dispatch manager records `set_last_dispatch_backend("<kernel_name>", "triton")`.
   - Verify that no warning is logged indicating fallback to PyTorch Level 3.
2. **Numerical Parity**:
   - Run forward and backward finite-difference comparisons:
     ```python
     torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-5)
     ```
