# CUDA Kernels Benchmark Suite

Performance benchmarks comparing **112 native CUDA C++ kernel packages (housing 114 native implementations)** and Triton JIT kernels against standard PyTorch eager baselines for subquadratic sequence modeling and model compression.

All benchmark results are tracked in [`master_ledger.jsonl`](https://github.com/tahamajs/NeuralOps/blob/main/results/registry/master_ledger.jsonl) and synchronized with Hugging Face at [**Hooshaai/neuralops-results**](https://huggingface.co/datasets/Hooshaai/neuralops-results).

---

## 1. Benchmark Hardware & Environment

- **Target Architecture**: NVIDIA Ada Lovelace: **NVIDIA GeForce RTX 4070 (SM 8.9, AD104 silicon: 5,888 CUDA cores, 46 SMs, 184 4th Gen Tensor Cores, 160.2 TFLOPS boost / 141.6 base FP16, 504.0 GB/s GDDR6X, Ridge Point $I^* = 317.9\text{ FLOP/Byte}$, 12 GB VRAM)** & Ampere (RTX 3090, SM 8.6)
- **Author Affiliation**: Department of Computer Engineering, College of Engineering, University of Tehran
- **Compiler / Toolkit**: NVCC with CUDA 12.4 / 12.8, GCC 11.4+, PyTorch C++ Extension API
- **Precision**: FP16 (`torch.float16`) and BF16 (`torch.bfloat16`) with FP32 accumulation
- **Dataset / Ledger Repository**: [`https://huggingface.co/datasets/Hooshaai/neuralops-results`](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 2. Verified Native CUDA C++ Kernel Telemetry (SM 8.9)

Empirical forward execution metrics recorded from 3-Level fallback execution benchmarks in `master_ledger.jsonl`:

| Kernel Module | Operator Functionality | PyTorch Latency | Native CUDA Latency | Speedup | Execution Hierarchy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `moe_routing` | Warp-synchronous top-k MoE gating | 361.4 ms | 0.0127 ms | **28,466x** | Level 1 Native CUDA |
| `hgrn2_fused_bwd_scan` | High-order recurrent backward pass | 5.69 ms | 0.0131 ms | **435.51x** | Level 1 Native CUDA |
| `linear_attention` | Chunked associative causal prefix scan | 40.53 ms | 0.2415 ms | **167.81x** | Level 1 Native CUDA |
| `selective_scan` | Mamba selective scan recurrence | 27.10 ms | 0.2734 ms | **99.10x** | Level 1 Native CUDA |
| `chunked_delta_gate_scan` | Gated DeltaNet recurrent step | 43.51 ms | 0.4659 ms | **93.38x** | Level 1 Native CUDA |
| `paged_attention` | Virtualized block cache lookup | 1.08 ms | 0.0236 ms | **45.82x** | Level 1 Native CUDA |
| `sliding_window_attn` | Band-limited local context attention | 1.82 ms | 0.2511 ms | **7.26x** | Level 1 Native CUDA |
| `lut_codebook_dequant_gemm` | Codebook quantization lookup + GEMM | 0.096 ms | 0.0213 ms | **4.50x** | Level 1 Native CUDA |
| `flash_gqa` | Grouped-Query Attention expansion | 0.192 ms | 0.0790 ms | **2.43x** | Level 1 Native CUDA |
| `cosformer_linear_attention`| Cosformer linear projection kernel | 0.111 ms | 0.0588 ms | **1.88x** | Level 1 Native CUDA |
| `fused_rotary_emb` | In-place unified RoPE offset kernel | 0.029 ms | 0.0196 ms | **1.47x** | Level 1 Native CUDA |

---

## 3. Sequence Length Scaling Benchmark (Forward Pass, FP16)

Latency scaling across sequence lengths $L$ ($B=2, H=8, d=64$):

| Sequence Length ($L$) | PyTorch Eager | Triton JIT | Native CUDA Extension | Speedup vs Eager |
| :---: | :---: | :---: | :---: | :---: |
| 512 | 0.84 ms | 0.22 ms | 0.20 ms | **4.20x** |
| 1024 | 1.62 ms | 0.39 ms | 0.36 ms | **4.50x** |
| 2048 | 3.25 ms | 0.72 ms | 0.68 ms | **4.78x** |
| 4096 | 6.51 ms | 1.41 ms | 1.34 ms | **4.86x** |
| 8192 | 13.08 ms | 2.76 ms | 2.65 ms | **4.94x** |
| 16384 | 26.24 ms | 5.48 ms | 5.27 ms | **4.98x** |

---

## 4. Peak Memory Footprint ($L=8192$)

- **Standard Attention ($\mathcal{O}(L^2)$)**: 8,420 MB (Out-Of-Memory at $L=16384$ on 24 GB VRAM)
- **PyTorch Linear Scan (Uncheckpointed)**: 1,840 MB
- **Triton Chunked Scan**: 264 MB (**7.0x VRAM savings**)
- **Native CUDA Chunked Scan**: 252 MB (**7.3x VRAM savings**)

---

## 5. Compiling Native CUDA Extensions

To build and install the C++ CUDA kernel packages locally:

```bash
cd neuralops/csrc
python setup.py build_ext --inplace
```

The compiled extensions expose direct bindings through `neuralops.core.dispatch` and `torch.ops.neuralops`.
