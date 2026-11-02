# Historical Benchmark Archive (`legacy_pre_purge_2026_10`)

## Provenance and Archival Notice

The files in this directory represent historical benchmark telemetry and evaluation outputs generated prior to commit `710b9b1bd7` (October 19, 2026).

### Context
Prior to October 2026, benchmarks were executed using 87 distinct standalone scripts located under `scripts/benchmarks/` (e.g., `run_attention_sweep.py`, `benchmark_all_cuda.py`, `triton_leaderboard.py`, `batch_validate_all_compressors.py`).

In commit `710b9b1bd7` and `a912a0fbe9`, those legacy scripts were retired and consolidated into the unified CLI suite:
```bash
python3 -m neuralops.benchmarks.cli run [OPTIONS]
```

### Superseded Files
The CSV, JSON, TeX, and Markdown files archived here include:
- `master_108_cuda_modules_benchmark.*`: Legacy runs across 108 CUDA C++ kernels.
- `sweep_131_modules_taha.*` & `sweep_131_modules_tahoora.*`: Independent multi-precision sweeps across 131 attention layers.
- `triton_full_leaderboard_v136.*`: Triton v136 performance benchmarks.
- `MASTER_RESULTS_MATRIX.*`: Pre-consolidation downstream accuracy tables.
- `accuracy_and_ppl_benchmark_*`: Historical perplexity logs on GPT-2 WikiText-2.

### Canonical Modern Workflow
To generate fresh, verified benchmark results compliant with current kernels and tolerances, use the unified CLI interface:
```bash
# Run all benchmark ops
python3 -m neuralops.benchmarks.cli run --category all

# Run specific attention or compressor suites
python3 -m neuralops.benchmarks.cli run --category attention --backend cuda
python3 -m neuralops.benchmarks.cli run --category compressor
```
All modern results are recorded dynamically in the canonical SQLite/JSON ledger via `neuralops.benchmarks`.
