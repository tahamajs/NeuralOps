# NeuralOps Benchmark Suite

This directory contains the production benchmark and stress-testing suite for the **NeuralOps** attention mechanisms, Triton JIT hardware kernels, compilation runtimes, and quantization algorithms.

## Architectural Design: Single Source of Truth

To avoid duplication, all underlying benchmark implementations reside canonically in [`neuralops/benchmarks/`](../neuralops/benchmarks/). The scripts in this directory (`benchmarks/*.py`) serve as clean CLI wrappers that expose both:
- Standalone CLI execution interfaces with standard arguments (`--output`, `--device`, `--dtype`, `--warmup`, `--repeat`, `--quick`).
- Re-exported programmatic APIs (`run_*`) for integration into external pipelines and tests.

---

## Benchmark Catalog & CLI Interfaces

| Script | Target / Core Focus | Default Outputs | Key CLI Arguments |
|---|---|---|---|
| [`bench_llm_bandwidth.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/bench_llm_bandwidth.py) | Memory bandwidth (GB/s), L2 cache residency, DRAM saturation % | `results/benchmark_4_kernels_corrected.csv`, `.json` | `--quick`, `--peak-bw`, `--l2-cache-mb`, `--warmup`, `--repeat` |
| [`benchmark_all_26_comprehensive.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/benchmark_all_26_comprehensive.py) | 26 Triton JIT kernels latency, speedup, launch overhead, parity | `results/triton_26_kernels_comprehensive_benchmark.csv`, `.json`, `docs/reports/TRITON_26_KERNELS_COMPREHENSIVE_REPORT.md` | `--quick`, `--kernels`, `--warmup`, `--repeat`, `--report-md` |
| [`benchmark_all_26_kernels.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/benchmark_all_26_kernels.py) | Direct physical hardware latency, speedup & numerical difference | `results/triton_all_26_kernels_benchmark.csv`, `.json` | `--quick`, `--kernels`, `--warmup`, `--repeat` |
| [`benchmark_torch_compile_131_attentions.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/benchmark_torch_compile_131_attentions.py) | Inductor compilation (`reduce-overhead`, dynamic shapes), zero-recompile | `results/benchmark_torch_compile_131_corrected.csv`, `.json` | `--quick`, `--limit`, `--modules`, `--seq-lens`, `--strict-compile-check` |
| [`benchmark_torch_compile.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/benchmark_torch_compile.py) | Alias to `benchmark_torch_compile_131_attentions.py` | Same as above | Same as above |
| [`benchmark_extrapolation_32k.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/benchmark_extrapolation_32k.py) | Context scaling $L \in [512, 32768]$, peak VRAM, KV memory complexity | `docs/benchmarks/paper_benchmarks_extrapolation_32k.csv`, `.json`, `docs/reports/PAPER_TABLES_EXTRAPOLATION_32K.tex` | `--quick`, `--seq-lens`, `--warmup`, `--repeat`, `--tex-output` |
| [`benchmark_dynamic_range_stress.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/benchmark_dynamic_range_stress.py) | Numerical stability under extreme scales ($10^{-4}$ to $10^{4}$) | `docs/benchmarks/paper_benchmarks_dynamic_range_stress.csv`, `.json`, `docs/reports/PAPER_TABLES_DYNAMIC_RANGE.tex` | `--quick`, `--kernels`, `--warmup`, `--repeat`, `--tex-output` |
| [`measure_multiprecision.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/measure_multiprecision.py) | FP32, FP16, BF16, FP8 (E4M3, E5M2) throughput (TFLOPS) & relative error | `docs/benchmarks/paper_benchmarks_multiprecision_crossval.csv`, `.json`, `docs/reports/PAPER_TABLES_MULTIPRECISION.tex` | `--quick`, `--precisions`, `--kernels`, `--warmup`, `--repeat` |
| [`multi_seed_stress_test_131_72.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/multi_seed_stress_test_131_72.py) | 131 Attentions $\times$ 72 Compressors variance across seeds 42, 1337, 2026 | `results/multi_seed_crossval_stress_test.csv`, `.json` | `--quick`, `--seeds`, `--limit-attentions`, `--limit-compressors` |
| [`stress_test_seeds_100_200_300.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/stress_test_seeds_100_200_300.py) | Attentions & Compressors statistical audit across seeds 100, 200, 300 | `results/multi_seed_stress_test_seeds_100_200_300.csv`, `.json` | `--quick`, `--seeds`, `--limit-attentions`, `--limit-compressors` |
| [`stress_test_seeds_404_777_999.py`](file:///Users/tahamajs/Documents/uni/Project/benchmarks/stress_test_seeds_404_777_999.py) | Backward pass autograd gradient fidelity across seeds 404, 777, 999 | `docs/benchmarks/paper_benchmarks_seeds_404_777_999.csv`, `results/paper_benchmarks_seeds_404_777_999.json` | `--quick`, `--seeds` |

---

## Standard CLI Options

Every benchmark in the suite conforms to a uniform argument schema:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output` | `str` | *Script-dependent* | Path to write output CSV file. |
| `--output-json` | `str` | *Script-dependent* | Path to write structured JSON results. |
| `--device` | `str` | `cuda:0` / `cpu` | Device target for benchmark execution. |
| `--dtype` | `str` | `float16` | Data type precision (`float16`, `float32`, `bfloat16`). |
| `--warmup` | `int` | `2` - `10` | Number of unmeasured warmup iterations before recording events. |
| `--repeat` | `int` | `5` - `50` | Number of measured iterations across which latency is averaged. |
| `--quick` | `flag` | `False` | Fast validation pass with reduced configs/iterations. |

---

## Sample Invocations

### 1. Realistic Memory Bandwidth & Saturation
```bash
python benchmarks/bench_llm_bandwidth.py --quick
python benchmarks/bench_llm_bandwidth.py --output results/bw.csv --warmup 10 --repeat 30
```

### 2. Comprehensive 26 Triton JIT Kernels Sweep
```bash
python benchmarks/benchmark_all_26_comprehensive.py --quick
python benchmarks/benchmark_all_26_comprehensive.py --output results/triton26.csv --report-md docs/reports/TRITON_26.md
```

### 3. Strict Inductor torch.compile Validation
```bash
python benchmarks/benchmark_torch_compile_131_attentions.py --quick
python benchmarks/benchmark_torch_compile_131_attentions.py --seq-lens 128,256,512 --strict-compile-check
```

### 4. Context Length Extrapolation up to 32k Tokens
```bash
python benchmarks/benchmark_extrapolation_32k.py --quick
python benchmarks/benchmark_extrapolation_32k.py --seq-lens 512,1024,2048,4096,8192,16384,32768
```

### 5. Multi-Precision Quantization Cross-Validation
```bash
python benchmarks/measure_multiprecision.py --quick
python benchmarks/measure_multiprecision.py --precisions FP32,FP16,BF16,FP8_E4M3,FP8_E5M2
```

### 6. Multi-Seed Statistical Stress Testing
```bash
python benchmarks/multi_seed_stress_test_131_72.py --quick
python benchmarks/stress_test_seeds_100_200_300.py --quick
python benchmarks/stress_test_seeds_404_777_999.py --quick
```
