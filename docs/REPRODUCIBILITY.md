# Reproducibility Guide & Artifact Verification

## 1. Remote GPU Hardware & Execution Environment

All physical GPU benchmarks, speedup measurements, and autograd verification suites are executed directly on the remote dedicated testbench:

- **Host**: `taha@192.168.1.100`
- **Primary Working Directory**: `/mnt/d/SVD_linear_Attention`
- **Secondary Research Workspace**: `/mnt/d/SVD_Project` (Linked via atomic symlink `lrwxrwxrwx /mnt/d/SVD_Project -> ./SVD_linear_Attention`)
- **Python Virtualenv**: `/mnt/d/venv/bin/python3`
- **GPU Architecture**: NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace SM 8.9, 36 SMs, 8GB GDDR6 VRAM)
- **Host OS & Toolchain**: Linux 6.6.137+ / Ubuntu 22.04 LTS, NVIDIA Driver `595.99.02`, CUDA `13.2` / `12.8`, PyTorch `2.11.0+cu128`, Triton `3.6.0`

## 2. Global Seeds & Invariant Guarantees

All benchmarks use `torch.manual_seed(42)` by default to enforce absolute determinism across all forward and backward passes.
- Architecture counts invariant: **131 Attention Mechanisms**, **72 Compression Engines**, **26 Triton Hardware Kernels**.
- Strict Parameter Invariant: $|\theta(M')| \le |\theta(M_0)|$ enforced across all compression transformations.

## 3. Remote Physical GPU Benchmark Execution Commands

To reproduce the benchmark suites on the remote GPU server:

```bash
# Connect to the remote GPU testbench
ssh gpu  # or ssh taha@192.168.1.100
cd /mnt/d/SVD_linear_Attention

# Set remote Python binary path
export PYTHON_BIN="/mnt/d/venv/bin/python3"
export PYTHONPATH=".:$PYTHONPATH"

# 1. Microbenchmark all 26 Triton JIT Hardware Kernels vs PyTorch Reference
$PYTHON_BIN scripts/benchmarks/benchmark_all_26_kernels.py

# 2. Comprehensive SOTA Gap Closure Benchmark Suite
$PYTHON_BIN scripts/benchmarks/benchmark_sota.py

# 3. 3-Level Execution Hierarchy Benchmark (Native CUDA C++ vs Triton JIT vs PyTorch)
$PYTHON_BIN scripts/benchmarks/benchmark_3level_cuda_triton_pytorch.py

# 4. End-to-end 131 attention mechanisms evaluation
$PYTHON_BIN scripts/benchmarks/benchmark_all_131_attentions.py

# 5. 72 compression engines evaluation
$PYTHON_BIN scripts/benchmarks/benchmark_all_72_compressors.py

# 6. Isolated benchmark sweep across all 108 native CUDA C++ kernels
$PYTHON_BIN scripts/benchmarks/benchmark_all_108_cuda_isolated.py --classify-strict

# 7. Multi-stream concurrency and zero memory drift stress test
$PYTHON_BIN scripts/multi_stream_benchmark.py
$PYTHON_BIN scripts/stress_1000_multistream.py

# 8. Verify canonical architectural invariants (131/72/26)
$PYTHON_BIN scripts/utils/verify_readme_counts.py
```

## 4. Local CLI Benchmarking & Evaluation

For local development or testing with fallback backbones:

```bash
python3 -m neuralops.benchmarks run \
    --module <module_name> \
    --model <model_name> \
    --dataset <dataset> \
    --rank 64 \
    --seed 42
```

