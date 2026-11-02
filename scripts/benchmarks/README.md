# NeuralOps Benchmark Engine: Scripts & Tooling Directory

This directory contains executable entry points and helper scripts for running, profiling, and reporting on NeuralOps kernels, attention mechanisms, and model compression engines.

All core benchmarking logic is maintained in the canonical Python package [`neuralops.benchmarks`](../../neuralops/benchmarks/).

---

## 1. Active Tools Overview

| Script | Purpose | Execution Target | Typical Invocation |
| :--- | :--- | :--- | :--- |
| [`benchmark_quickstart.py`](benchmark_quickstart.py) | Human-friendly single-command benchmark runner | CPU / GPU | `python3 scripts/benchmarks/benchmark_quickstart.py --mode all` |
| [`gpu_telemetry.py`](gpu_telemetry.py) | Hardware pre-flight, TF32 enablement & VRAM auto-tuning | CUDA / MPS / CPU | `python3 scripts/benchmarks/gpu_telemetry.py --probe-model` |
| [`gpu_master_suite.py`](gpu_master_suite.py) | Exhaustive sweeps across 131 attentions, 72 compressors, 26 Triton kernels | Remote CUDA | `python3 scripts/benchmarks/gpu_master_suite.py --category all` |
| [`smart_gpu_trainer.py`](smart_gpu_trainer.py) | High-throughput training & 6-pillar knowledge distillation healing | Remote CUDA | `python3 scripts/benchmarks/smart_gpu_trainer.py --steps 100` |
| [`results_exporter.py`](results_exporter.py) | SQLite ledger persistence, JSONL/CSV sync & LaTeX table generator | Agnostic | `python3 scripts/benchmarks/results_exporter.py --demo` |

---

## 2. Legacy Migration Redirectors

In October 2026, legacy benchmark scripts were consolidated into `neuralops.benchmarks`. The following scripts serve as executable migration guides:

- `batch_validate_all_compressors.py` $\rightarrow$ use `python3 -m neuralops.benchmarks.cli run --category compressor`
- `benchmark_all_cuda.py` $\rightarrow$ use `python3 -m neuralops.benchmarks.cli run --category kernel`
- `run_attention_sweep.py` $\rightarrow$ use `python3 -m neuralops.benchmarks.cli run --category attention`
- `run_quality_sweep.py` $\rightarrow$ use `python3 -m neuralops.benchmarks.run_quality_sweep`
- `triton_leaderboard.py` $\rightarrow$ use `python3 neuralops/benchmarks/benchmark_all_26_comprehensive.py`

---

## 3. Canonical Package Entry Points

For scripting and automation, prefer the canonical package CLI:

```bash
# Run all benchmarkable operations
python3 -m neuralops.benchmarks.cli run --backend auto --device cuda:0 --output results/benchmark.csv

# Measure generation latency and decoding speed
python3 -m neuralops.benchmarks.generation_speed --model gpt2 --gen-len 64

# Evaluate language model perplexity
python3 -m neuralops.benchmarks.perplexity_eval --model gpt2 --dataset wikitext2
```
