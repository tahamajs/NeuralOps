# `neuralops.benchmarks`: Canonical Benchmark & Profiling Engine

This package serves as the **Single Source of Truth** for all NeuralOps benchmarking routines, kernel profiling, compile-time audits, and cross-seed statistical stress tests.

## Benchmark levels

The package covers the four FLA/TritonBench-style levels:

| Level | Entry points | Coverage |
|---|---|---|
| Kernel-level | `ops_registry.py`, `ops_runner.py`, `kernel_benchmark.py` | Correctness-gated CUDA/Triton operators with CUDA-event timings |
| Module-level | `ops_registry.py` | Registered attention/compressor operations plus `causal_conv1d`, `rmsnorm`, `swiglu`, and fused cross-entropy |
| Training throughput | `training_throughput.py` | Forward, backward, optimizer step, tokens/sec, and peak VRAM |
| Generation / evaluation | `generation_speed.py`, `perplexity_eval.py`, `adapters/perplexity.py` | Prefill/decode latency, generation throughput, perplexity, and evaluation latency |

Additional measured categories are available through the same CLI: `roofline`,
`quantization`, `moe`, `serving`, `rope`, `sampling`, `ssm`, `varlen`, and
`sweep`. These category runners report their own finite metrics; they are not
included in the registry-module sweep unless explicitly requested.

The module and kernel paths use a correctness gate before timing.  Lossy
compressors are checked for finite output and positive parameter reduction;
they are not incorrectly compared with `allclose` against the uncompressed
model.  Attention drift above the normal numerical budget is reported as
`PASS_HIGH_DIFF` rather than being hidden.

The extended roofline, quantization, MoE, serving, RoPE, sampling, SSM,
ragged-sequence, and autotuning modules are available as explicit CLI
categories. They are smoke-tested independently and are not silently folded
into the 203-operation registry sweep.

Tracking is disabled by default to keep benchmark latency measurements free of
filesystem and network I/O. Pass `--track` for local tracking, or choose an
explicit backend with `--tracker wandb|mlflow|tensorboard`.

Examples:

```bash
python -m neuralops.benchmarks run --category all --dynamic \
  --output results/benchmark_ops.json
python -m neuralops.benchmarks.generation_speed --model gpt2 --gen-len 32
python -m neuralops.benchmarks.perplexity_eval --model gpt2 --dataset wikitext2
```

## Package Architecture

```
neuralops/benchmarks/
├── __init__.py                                 # Module namespace & PEP 562 lazy exporter
├── api.py / cli.py / config.py / harness.py    # Framework orchestration & metrics
├── bench_llm_bandwidth.py                      # Memory bandwidth & DRAM saturation engine
├── benchmark_all_26_comprehensive.py           # 26 Triton JIT kernels sweep & markdown reporter
├── benchmark_all_26_kernels.py                 # Direct physical hardware JIT benchmarking
├── benchmark_torch_compile_131_attentions.py   # Strict Dynamo/Inductor zero-recompile auditor
├── benchmark_torch_compile.py                  # Canonical alias
├── benchmark_extrapolation_32k.py              # Context length scaling up to 32,768 tokens
├── benchmark_dynamic_range_stress.py           # Dynamic range numerical stability ($10^{-4}$ to $10^{4}$)
├── measure_multiprecision.py                   # Multi-precision (FP32/FP16/BF16/FP8) cross-validation
├── multi_seed_stress_test_131_72.py            # 131 Attentions x 72 Compressors multi-seed audit
├── stress_test_seeds_100_200_300.py            # Statistical variance evaluation (seeds 100, 200, 300)
└── stress_test_seeds_404_777_999.py            # Autograd backward-pass gradient stress test
```

Root wrappers in `benchmarks/` import directly from this package:
```python
from neuralops.benchmarks.bench_llm_bandwidth import run_bandwidth_benchmark, main
```

---

## Python API Usage

All benchmark modules expose clean programmatic functions (`run_*`) that can be called directly from Python code:

### 1. Bandwidth Benchmark
```python
from neuralops.benchmarks.bench_llm_bandwidth import run_bandwidth_benchmark

results = run_bandwidth_benchmark(
    configs=[(4, 512, 2048), (8, 512, 2048)],
    device_str="cuda:0",
    dtype_str="float16",
    warmup=5,
    repeat=20,
    output_csv="results/bandwidth_test.csv",
    output_json="results/bandwidth_test.json",
)
```

### 2. 26 Triton Hardware Kernels Comprehensive Sweep
```python
from neuralops.benchmarks.benchmark_all_26_comprehensive import run_all_26_comprehensive

results = run_all_26_comprehensive(
    device_str="cuda:0",
    dtype_str="float16",
    warmup=10,
    repeat=50,
    output_csv="results/triton_26.csv",
    output_json="results/triton_26.json",
    report_md="docs/reports/TRITON_26.md",
)
```

### 3. Strict torch.compile Inductor Evaluation
```python
from neuralops.benchmarks.benchmark_torch_compile_131_attentions import benchmark_torch_compile

summary = benchmark_torch_compile(
    strict_compile_check=True,
    output_path="results/compile_audit.csv",
    seq_lens=[128, 256, 512],
    repeat=10,
)
```

### 4. Sequence Length Extrapolation
```python
from neuralops.benchmarks.benchmark_extrapolation_32k import run_extrapolation_benchmark

results = run_extrapolation_benchmark(
    seq_lens=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    device_str="cuda:0",
    output_csv="docs/benchmarks/extrapolation.csv",
    output_tex="docs/reports/extrapolation.tex",
)
```

### 5. Multi-Precision Trade-offs
```python
from neuralops.benchmarks.measure_multiprecision import measure_benchmark

results = measure_benchmark(
    device_str="cuda:0",
    warmup=5,
    repeat=20,
    output_csv="docs/benchmarks/multiprecision.csv",
)
```

---

## Design Principles

1. **Zero Top-Level Side Effects**: Importing any module in `neuralops.benchmarks` never initializes CUDA devices, allocates GPU tensors, or forces C++ JIT compilation. All setup is deferred to function calls.
2. **Unified Data Schema**: Outputs are simultaneously exported to structured CSV and JSON with identical column naming and schema.
3. **Relative Path Resolution**: All paths default to relative repository locations (`results/`, `docs/benchmarks/`, `docs/reports/`) and can be overridden via CLI flags.
4. **Resilient Exception Isolation**: Kernel exceptions or numerical instabilities in one test do not crash the remaining sweep. All errors are captured with error messages in the output table.
