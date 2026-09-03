# NeuralOps Benchmarks & Evaluation Suite

This directory contains standalone execution benchmarks, orchestrators, and validation suites for the SVD & Linear Attention framework.

## Components

- **`two_phase_benchmark.py`**:
  Orchestrates end-to-end benchmark campaigns in two distinct stages:
  1. **Phase 1 (Latency & Memory Efficiency Sweep)**: High-throughput screening across 230+ attention and compression modules under tight execution budgets.
  2. **Phase 2 (Quality & Statistical Significance Sweep)**: Multi-seed convergence and fine-tuning recovery evaluation on top candidate modules.

## Results Archive

Empirical metrics, per-layer breakdowns, and latency records are persisted under [`Codes/framework/benchmark_results/`](../benchmark_results/):
- **`benchmark_summary.csv`**: Master tabular dataset containing all evaluated runs.
- **`json/`**: Structured per-module execution logs categorized by architectural track (`track1_svd`, `track2_ssm`, `track3_moe`, `track4_kernel_quant`, `track5_pruning`).
- **`BENCHMARK_REPORT.md`**: Executive summary and comparative analysis.
