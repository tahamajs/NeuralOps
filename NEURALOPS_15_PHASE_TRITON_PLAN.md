# NeuralOps 15-Phase Triton Kernel Optimization Lifecycle
## Target Architecture: NVIDIA GeForce RTX 3090 (GA102 Ampere, 24GB GDDR6X, SM 8.6)
**Sharif University HPC Cluster (`GPU-3090-1`)**

---

### Executive Objective
Optimize all **26 canonical Triton kernel modules** (housing **62 unified operators** across 4 architectural tiers) for maximum throughput, hardware saturation, and zero numerical degradation on the NVIDIA GeForce RTX 3090 GPU, adhering strictly to the **Fail-Closed Verification Contract** (no fake artifacts, no silent CPU fallbacks, mandatory gradient and tolerance gates).

### Execution Status — 2026-11-03

The first complete remote gate was executed through Torque PBS job `893782.hpc.ce.sharif.edu` on `GPU-3090-1` and completed in `715.70s`:

```text
pytest tests/gpu_exhaustive -v --tb=short
709 passed, 195 failed, 7 skipped
```

This is the frozen remediation baseline. The plan is not complete until the same command reaches zero unexpected failures. The current failure inventory is 115 attention tests, 70 compressor tests, 8 kernel-parity tests, and 2 long-context tests. Triton image compatibility and missing `ninja` are tracked as environment gates; they must be resolved or explicitly classified before performance claims are promoted.

Promotion rule: a phase may be marked complete only with raw command output, a reproducible artifact, and no hidden fallback or undocumented skip. PBS submission is the persistence mechanism for long runs; a second `screen` copy must not be launched for the same workload.

---

### Phase 1: Environment Ingestion & Remote Node Setup
- Connect to `hpc` via Zero-Drop SSH multiplexing (`ControlMaster auto`).
- Access target compute node `GPU-3090-1`.
- Verify NVIDIA Driver (`465.19.01`), CUDA Runtime (`11.3`), PyTorch (`1.12.1+cu113`), and Triton (`3.0.0`).
- Resolve library paths (`LD_LIBRARY_PATH`), GCC 8.2 runtime bindings, and disable offline telemetry hangs.

### Phase 2: Invariant Audit & Static AST Inspection
- Enforce canonical Triton kernel invariants:
  - Exact count of canonical modules: 26 modules (62 unified operators).
  - Absolute ban on `tl.make_block_ptr` / `tl.advance` in mainline Triton (preserves Ampere/Turing stability).
  - Verification of 3-level fallback hierarchy annotations: CUDA C++ $\to$ Triton JIT $\to$ PyTorch reference.

### Phase 3: Hardware-Grounded Rigorous Baseline Benchmarking
- Implement high-precision benchmarking protocol:
  - Strict CUDA event timing (`torch.cuda.Event(enable_timing=True)`).
  - Explicit device synchronization: `torch.cuda.synchronize()` before and after recording.
  - 256MB L2 cache flushing tensor before every timed iteration (`torch.empty(64*1024*1024, dtype=torch.float32).zero_()`).
  - Minimum 10 warmup iterations to trigger JIT compilation and autotuner stabilization.
  - Fail-closed guard: immediate `RuntimeError` if `torch.cuda.is_available()` is False.
- Collect baseline latencies across all 26 modules against PyTorch reference implementations.

### Phase 4: Bottleneck Identification & Classification
- Categorize kernels into 3 performance tiers:
  - **Class A (Accelerated)**: Speedup $\ge 1.5\times$ vs PyTorch reference.
  - **Class B (Parity)**: Speedup between $1.0\times$ and $1.5\times$.
  - **Class C (Degraded / Bottleneck)**: Speedup $< 1.0\times$ (slower than PyTorch reference).
- Identify the bottom 5 bottleneck kernels requiring architectural overhaul.

### Phase 5: Nsight Compute & Roofline Microarchitectural Profiling
- Run CLI profiling on target kernels using NVIDIA Nsight Compute (`ncu`).
- Measure:
  - Memory Bandwidth Saturation (% of 936 GB/s GDDR6X theoretical peak).
  - Compute SM Throughput (% of 35.6 TFLOPS FP32 / 71 TFLOPS FP16 peak).
  - Shared Memory (SRAM) bank conflict rates.
  - Register pressure and spill-to-local-memory (L1/SRAM overflow).

### Phase 6: Block Tiling & Dynamic Grid Autotuning
- Formulate parameter search grids across:
  - `BLOCK_M`: `[32, 64, 128, 256]`
  - `BLOCK_N`: `[32, 64, 128]`
  - `BLOCK_D`: `[32, 64, 128]`
  - `num_warps`: `[2, 4, 8, 16]`
  - `num_stages`: `[2, 3, 4, 5]` (software pipelining on Ampere SM 8.6).
- Autotune each kernel using `triton.autotune` with frozen correctness verification.

### Phase 7: SRAM Tiling & Kernel Fusion
- Fuse elementwise operations directly into attention/scan load-store loops:
  - Fused RMSNorm / LayerNorm + Linear projections.
  - Fused SwiGLU / GELU activations.
  - Fused rotary embedding (RoPE) application during Q/K loading.
- Eliminate intermediate global memory (HBM) round-trips.

### Phase 8: Pointer Arithmetic & 128-Bit Memory Vectorization
- Replace naive strided indexing with aligned 128-bit vector loads (`float4` / `bfloat16x8`).
- Guarantee 16-byte alignment on memory base pointers.
- Structure memory loads to maximize L2 cache line coalescing (128-byte transactions).

### Phase 9: Numerical Budget, Dtype Guards & Autograd Parity
- Accumulate intermediate sums in `torch.float32` for `float16` and `bfloat16` inputs.
- Guard against underflow/overflow with numerical bounds ($[-65504, 65504]$ clamp for FP16).
- Validate autograd gradients against finite-difference references:
  - `torch.autograd.gradcheck(..., atol=1e-3, eps=1e-5)`.

### Phase 10: 3-Level Fallback Dispatch & Zero-Overhead Caching
- Ensure every Triton kernel exports clean dispatch wrappers adhering to the 3-level fallback:
  1. Native CUDA C++ (Level 1).
  2. Triton JIT (Level 2).
  3. PyTorch Vectorized Reference (Level 3).
- Utilize zero-overhead LRU dispatch table with explicit warnings on fallback degradation.

### Phase 11: Long-Context Sequence Scaling (up to 32,768 Tokens)
- Benchmark sequence length scaling: $N \in \{512, 1024, 2048, 4096, 8192, 16384, 32768\}$.
- Verify $O(N)$ linear complexity on associative scans (Cosformer, GLA, GDN, HGRN2).
- Verify chunked linear memory footprint preventing OOM at 32k context on 24GB VRAM.

### Phase 12: Multi-Batch & VRAM Saturation Stress Testing
- Scale batch sizes ($B \in \{1, 2, 4, 8, 16, 32\}$) to saturate Ampere SMs.
- Profile board power draw ($\to 350\text{W}$) and GPU compute utilization ($\to 100\%$).
- Verify stability under sustained thermal load.

### Phase 13: End-to-End Model Integration & Training Validation
- Run in-place attention replacement inside full transformer architectures (`DistilBERT`, `GPT-2`, `RoBERTa`).
- Execute 100 steps of end-to-end forward/backward pretraining with `AutoMaxGPUTrainer`.
- Confirm loss convergence without NaN or Inf anomalies.

### Phase 14: Automated Regression Suite & Continuous CI Gates
- Integrate optimized kernels into unit test suite (`tests/ops/test_triton_kernels.py`).
- Validate compliance with repository invariant scripts:
  - `audit_dead_code.py` $\to 0$ issues.
  - `audit_readme_consistency.py` $\to 0$ errors.

### Phase 15: Final Empirical Ledger Ingestion & Paper Reporting
- Ingest empirical metrics into master SQLite ledger (`results/master_ledger.sqlite`).
- Generate publication-ready LaTeX tables (`results/tables/triton_optimized_leaderboard.tex`).
- Update arXiv monograph hardware section and executive documentation.

## Execution Note — 2026-11-03

The full remote GPU gate remains the frozen baseline for phase promotion: 709 passed,
195 failed, and 7 skipped out of 911 collected tests on RTX 3090 / PyTorch 1.12.1+cu113.
The causal benchmark harness was corrected to read `Registry.get_metadata()` rather than
implementation classes, and its numerical gate remains `max_diff <= 1e-3`.

A targeted PBS rerun using the current default environment was rejected before collection:
PyTorch 2.4.1+cu121 cannot initialize against the node's NVIDIA driver 465.19.01 / CUDA
11.3. This is recorded as an environment blocker, not a passing or failing code result.
The successful exhaustive job used the legacy CUDA 11.3 environment and remains the only
valid GPU evidence until that environment is restored.
