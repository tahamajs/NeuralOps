# 💾 Canonical 5-Format Persistence Engine Specification

Every benchmark execution, matrix sweep, and distillation evaluation in NeuralOps must persist state atomically across 5 standardized representations (`neuralops.benchmarks.results.saver.save_all_formats`).

---

## 📋 The 5 Canonical Formats

1. **JSON (`<run_id>.json`):**
   - Canonical self-contained run document validating against schema `$schema`.
   - Structured path: `results/<category>/<module>/<model>/<dataset>/rank_<rank>/<run_id>.json`.
   - Includes full hardware specifications, driver versions, git provenance (`git_commit`, `git_branch`, `git_dirty`), configuration, metrics, and percentile latency distribution ($p_{10}, p_{50}, p_{90}, p_{99}$).

2. **JSON Lines (`results.jsonl` / `metrics.jsonl`):**
   - Append-only transactional ledger for continuous monitoring across multi-device execution runs.
   - Structured for easy Unix command-line streaming (`jq`, `grep`, `awk`).
   - Located at `results/results.jsonl` (or suite-specific ledger e.g. `results/bert_attention_zoo/metrics.jsonl`).

3. **CSV (`results.csv` / `summary_matrix.csv`):**
   - Flattened tabular matrix designed for spreadsheet visualization, rapid regression filtering, and LaTeX table generation scripts.
   - Located at `results/results.csv`.

4. **Markdown (`<run_id>.md` / `scorecard.md`):**
   - Human-readable scorecard generated alongside the canonical JSON artifact.
   - Suitable for direct inspection, GitHub viewing, and Hugging Face model cards.

5. **Parquet (`results.parquet`):**
   - Strongly-typed columnar storage format designed for analytical SQL queries via DuckDB, PyArrow, or Polars with zero serialization overhead.
   - Located at `results/results.parquet`.

---

## 🔒 Schema Invariant Validation & Reproducibility Fields

All emitted JSON records conform to the authoritative schema and include full reproducibility and hardware telemetry fields:

```json
{
  "$schema": "https://raw.githubusercontent.com/tahamajs/NeuralOps/main/schemas/benchmark_manifest_v1.json",
  "version": "2.1.0",
  "run_id": "bench_20261114_040500",
  "run_key": "flash_attention_2__bert-base-uncased__sst2__rank0__a1b2c3d4e5f6",
  "status": "success",
  "timestamp_start": "2026-11-14T04:05:00.000000+00:00",
  "timestamp_end": "2026-11-14T04:06:12.000000+00:00",
  "run_metadata": {
    "run_id": "bench_20261114_040500",
    "git_commit": "c4795da7ac",
    "git_branch": "main",
    "git_dirty": false,
    "wall_clock_duration_s": 72.15
  },
  "environment": {
    "python_version": "3.11.8",
    "torch_version": "2.4.0",
    "gpu_model": "NVIDIA GeForce RTX 4070",
    "gpu_count": 1,
    "gpu_vram_gb": 12.0,
    "cuda_compute_capability": "8.9",
    "cuda_version": "12.8"
  },
  "hardware_usage": {
    "peak_vram_mb": 4210.5,
    "avg_vram_mb": 3150.2,
    "peak_ram_mb": 2048.0,
    "gpu_utilization_avg": 94.2
  },
  "reproducibility": {
    "random_seed": 42,
    "cublas_workspace_config": ":4096:8",
    "torch_deterministic": true,
    "reproduction_command": "python -m neuralops.benchmarks run --module flash_attention_2 --model bert-base-uncased --dataset sst2"
  },
  "metrics": {
    "accuracy": 0.924,
    "f1": 0.923,
    "loss": 0.215,
    "throughput_tokens_per_s": 14500.0,
    "latency_p50_ms": 1.42,
    "latency_p99_ms": 2.15
  },
  "artifacts": {
    "json_file": "01_unified_runs/flash_attention_2/bert-base-uncased/sst2/rank_0/bench_20261114_040500.json",
    "jsonl_file": "results.jsonl"
  }
}
```

---

## 🛠️ Python API Integration

To persist benchmark results across all 5 formats atomically:

```python
from pathlib import Path
from neuralops.benchmarks.results import BenchmarkResult, save_all_formats

# Create or receive benchmark result
result = BenchmarkResult(
    run_id="bench_20261114_040500",
    module={"name": "flash_attention_2", "type": "attention"},
    model={"name": "bert-base-uncased"},
    dataset={"name": "sst2"},
    metrics={"accuracy": 0.924, "latency_p50_ms": 1.42},
    status="success",
)

# Persist across all 5 canonical formats
saved_paths = save_all_formats(result, output_dir=Path("results"), require_parquet=True)

# saved_paths contains:
# - 'json': Path to results/<category>/<module>/<model>/<dataset>/rank_<rank>/<run_id>.json
# - 'markdown': Path to results/<category>/<module>/<model>/<dataset>/rank_<rank>/<run_id>.md
# - 'jsonl': Path to results/results.jsonl
# - 'csv': Path to results/results.csv
# - 'parquet': Path to results/results.parquet
```

