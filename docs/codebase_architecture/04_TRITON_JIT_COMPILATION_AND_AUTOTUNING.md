# NeuralOps Codebase Architecture: Part 04
# Triton JIT Kernel Suite: 26 Modules & 62 Unified Operators

**Assigned Subagent**: `Agent 04: Triton JIT Kernel & Autotuning Specialist`
**Target Codebase Paths**:
- `neuralops/core/triton_kernels/`
- `neuralops/kernels/triton/`
- `neuralops/ops/triton/`

---

## 1. Executive Summary & Design Philosophy

While native CUDA C++ delivers unmatched throughput, it requires lengthy ahead-of-time (AOT) compilation cycles and platform-specific tuning. The **Triton JIT Kernel Suite** in `neuralops/core/triton_kernels/` comprises exactly **26 canonical modules housing 62 unified operators** spanning all 4 architectural tiers.

Triton provides compiler-driven optimization at Python execution time:
- Automatic shared memory buffer allocation and synchronization (`tl.syncthreads()` automation).
- Hardware-adaptive autotuning across block sizes (`BLOCK_M`, `BLOCK_N`), warp counts (`num_warps`), and pipeline stages (`num_stages`).
- Universal multi-architecture support across NVIDIA GPUs and Huawei Ascend NPUs (with specialized dialect configurations).

Crucially, in the mainline NeuralOps Triton codebase, **block pointers (`tl.make_block_ptr` / `tl.advance`) are strictly prohibited** to ensure portable JIT stability across all compiler backends (with an explicit exemption only for the Huawei Ascend NPU backend).

---

## 2. Assigned Subagent Responsibilities (Agent 04)

Agent 04 is responsible for:
1. **Module Invariant**: Verifying that `neuralops/core/triton_kernels/` contains exactly 26 canonical modules.
2. **Autotuning Grid Integrity**: Ensuring `@triton.autotune` configurations specify clean power-of-two block sizes without causing compiler timeouts.
3. **Numerical Budget Enforcement**: Mandating that all intermediate accumulations for FP16 and BF16 inputs are performed in FP32 (`tl.float32`) to prevent catastrophic cancellation or overflow.
4. **Ascend & CUDA Portability**: Ensuring code avoids backend-specific intrinsics unless properly guarded by target device conditionals.

---

## 3. Directory & Module Catalog (26 Canonical Modules)

| # | Canonical Module File | Operators Housed | Core Mathematical Operation |
|---|---|---|---|
| 1 | `flash_attention.py` | Forward, Backward, Causal FlashAttn | Tiled online softmax, SRAM KV streaming |
| 2 | `chunk_gla.py` | Chunk Gated Linear Attention | Semi-recurrent chunk state accumulation |
| 3 | `selective_scan.py` | Mamba Selective Scan | Tiled associative prefix scan |
| 4 | `cosformer.py` | Cosformer Linear Attention | Trigonometric distance re-weighting |
| 5 | `fused_linear_cross_entropy.py` | Fused Matmul + Softmax + Loss | Single-pass logits without DRAM allocation |
| 6 | `rotary_embedding.py` | RoPE forward & backward | Complex planar rotation in registers |
| 7 | `rmsnorm.py` | Root Mean Square LayerNorm | Welford variance reduction across hidden dim |
| 8 | `swiglu.py` | Gated Silu Linear Unit | Elementwise activation fusion |
| 9 | `quant_gemm.py` | W4A16, W8A16 Quantized GEMM | Dequantize-on-the-fly matrix multiplication |
| 10 | `block_sparse.py` | Block-Sparse Attention | Bitmap-guided block index filtering |
| 11 | `topk_routing.py` | MoE Top-K Expert Gate | Shared memory sorting & expert assignment |
| 12-26 | `... (other 15 modules)` | Grouped query attention, MLA, etc. | Low-rank projection, sliding window, state decoders |

---

## 4. Architectural Rules & Triton Coding Standards

### 4.1 Banned Idioms: No Block Pointers in Mainline
```python
# BANNED in mainline:
# ptr = tl.make_block_ptr(...)
# ptr = tl.advance(ptr, ...)

# REQUIRED pattern: Standard 1D/2D pointer arithmetic with boundary masking
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tile = tl.load(ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask, other=0.0)
```

### 4.2 Numerical Budget & FP32 Accumulation Standard
```python
# Inputs in FP16/BF16
a = tl.load(...) # fp16
b = tl.load(...) # fp16

# MUST accumulate in FP32
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
acc += tl.dot(a, b) # dot accumulates into acc (fp32)
```

### 4.3 Autotuning Heuristics for SM 8.9 (Ada Lovelace)
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K']
)
```

---

## 5. Audit & Validation Contract

- **Test Path**: `tests/kernels/test_triton_parity.py`
- Verify that every operator passes:
  1. Exact gradient match via PyTorch autograd.
  2. Memory footprint verification (peak SRAM allocation $\le$ available shared memory per CTA).
  3. No NaNs or Infs when executed with extreme input scales ($\pm 10^4$).
