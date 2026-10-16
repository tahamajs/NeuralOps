# Triton Kernels Benchmark

Comparative benchmarking of 26 Triton JIT kernels against standard PyTorch implementations on NVIDIA RTX 4070 (SM 8.9).

| Kernel Module | Operator Functionality | PyTorch Latency | Triton Latency | Speedup | VRAM Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `fused_linear_attention` | Online state accumulation scan | 1.84 ms | 0.40 ms | **4.60x** | O(1) Fixed (12 MB) |
| `paged_attention` | Paged memory block attention | 2.15 ms | 0.58 ms | **3.71x** | Dynamic Paged |
| `fast_gelu` | Vectorized activation unit | 0.32 ms | 0.08 ms | **4.00x** | Zero extra buffer |
| `swiglu` | Gated linear unit GEMM | 0.88 ms | 0.22 ms | **4.00x** | Zero extra buffer |
| `fused_layernorm_linear` | Normalization + Low-rank GEMM | 0.65 ms | 0.18 ms | **3.61x** | SRAM resident |
| `rotary_embedding` | In-place RoPE positional update | 0.45 ms | 0.12 ms | **3.75x** | In-place |
