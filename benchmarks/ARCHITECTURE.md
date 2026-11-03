# Benchmark data flow

The benchmark system has one canonical execution path and several historical
entry points. This document records which pieces are connected.

```text
CLI / root wrapper
        |
        v
BenchmarkConfig (neuralops/benchmarks/config.py)
        |
        v
BenchmarkOrchestrator / BaseRunner
        |
        +--> discovery.py / ops_registry.py --> registered modules and kernels
        +--> adapters/ + core/                 --> models and datasets
        +--> metrics/                           --> correctness and quality metrics
        +--> results/schema.py                  --> typed result contract
        +--> results/saver.py / index.py        --> JSON, JSONL, CSV, Parquet, Markdown
        +--> results/summarizer.py              --> aggregate reports
```

## Canonical path

New benchmark work should use:

```bash
python -m neuralops.benchmarks.cli run \
  --category all --device cuda --output-dir results/<run-name>
```

The root `benchmarks/*.py` files are stable compatibility entry points. The
older `scripts/benchmarks/*.py` files include historical suites and exporters;
they may write legacy flat artifacts such as `results/benchmarks_latest.jsonl`.
Those artifacts are not automatically equivalent to the typed result ledger.

## Configuration sources

There are currently two configuration models:

- `neuralops.benchmarks.config.BenchmarkConfig`: canonical package runner.
- `scripts.benchmarks.benchmark_config.BenchmarkConfig`: legacy script-suite
  configuration with different field names and defaults.

Do not mix these models in a new benchmark. New code must use the package
configuration and `results` serializers. Legacy scripts should be migrated when
they are next modified.

## Result ownership

- Typed runs: `neuralops/benchmarks/results/schema.py` and `results/saver.py`.
- Runner summaries: `neuralops/benchmarks/writers.py`.
- Historical master-suite exports: `scripts/benchmarks/results_exporter.py`.

When comparing runs, use the typed result `run_id`, `run_key`, `status`,
`reproducibility`, and `git_commit` fields. Never merge a flat CSV into the
master ledger without validating its schema first.
