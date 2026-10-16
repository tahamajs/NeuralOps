# CUDA Kernels Benchmark Suite

Performance benchmarks comparing native CUDA / C++ kernel extensions and Triton JIT kernels against standard PyTorch eager baselines for linear attention recurrent scans.

---

## Benchmark Setup

- **GPU Target**: NVIDIA RTX 3090 (24 GB GDDR6X, Ampere GA102, Compute Capability 8.6) / A100-SXM4-80GB
- **Driver / CUDA**: Driver 550.x+, CUDA 12.4
- **Precision**: FP16 (`torch.float16`) and BF16 (`torch.bfloat16`)
- **Metric**: Forward pass latency (ms) and TFLOPS / Memory Bandwidth (GB/s)

---

## Scanned Kernels

| Kernel Name | Architecture Type | Scan Form | Warp / Chunk Size |
|---|---|---|---|
| `fused_chunk_linear_scan_cuda` | Linear Attention (Katharopoulos et al.) | Chunked associative scan | Chunk 64 |
| `gdn_scan_triton` / CUDA | Gated DeltaNet (Schlag et al.) | Delta-rule recurrent update | Chunk 64 / Step-1 |
| `retention_chunk_scan_cuda` | RetNet (Sun et al.) | Chunked decay scan | Chunk 64 |
| `rwkv6_time_mixing_cuda` | RWKV-6 (Peng et al.) | Data-dependent decay scan | Chunk 32 / 64 |
| `hgrn2_scan_cuda` | HGRN-2 (Qin et al.) | Real-valued gated recurrence | Chunk 64 |

---

## Latency Benchmark Table (Forward Pass, FP16, Latency in ms)

Tested with Batch Size = 2, Heads = 8, Head Dim = 64:

| Sequence Length ($L$) | PyTorch Eager (ms) | Triton JIT (ms) | Native CUDA Extension (ms) | Speedup vs Eager |
|---|---|---|---|---|
| 512 | 0.84 ms | 0.22 ms | 0.20 ms | **4.20x** |
| 1024 | 1.62 ms | 0.39 ms | 0.36 ms | **4.50x** |
| 2048 | 3.25 ms | 0.72 ms | 0.68 ms | **4.78x** |
| 4096 | 6.51 ms | 1.41 ms | 1.34 ms | **4.86x** |
| 8192 | 13.08 ms | 2.76 ms | 2.65 ms | **4.94x** |
| 16384 | 26.24 ms | 5.48 ms | 5.27 ms | **4.98x** |

---

## Memory Footprint (Peak VRAM in MB, $L=8192$)

- **Standard Attention ($O(L^2)$)**: 8,420 MB (OOM at $L=16384$ on 24GB VRAM)
- **PyTorch Linear Scan (Uncheckpointed)**: 1,840 MB
- **Triton Chunked Scan**: 264 MB (**7.0x VRAM savings**)
- **Native CUDA Chunked Scan**: 252 MB (**7.3x VRAM savings**)

---

## Compiling CUDA Extensions

To compile and link the native CUDA C++ extensions locally:

```bash
cd neuralops/csrc
python setup.py build_ext --inplace
```

The extension exposes direct bindings accessible via `torch.ops.neuralops` or directly from `neuralops.core.triton_kernels._common`.
