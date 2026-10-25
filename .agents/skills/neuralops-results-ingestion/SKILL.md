---
name: neuralops-results-ingestion
description: "Ingest and validate new empirical benchmark results into the master results ledger."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "ledger", "results", "ingestion", "benchmarks"]
---

# Skill: Results Ingestion Specialist

## When to use
Trigger when recording new empirical benchmark results from kernel sweeps, compression runs, or latency evaluations into the persistent ledger.

## Prerequisites
- SSH access to `taha@192.168.1.100`
- `/mnt/d/venv/bin/python3`

## Steps
1. **Parse Benchmark Output**:
   Extract output metrics from candidate benchmark JSON or CSV logs.
2. **Compute Git Metadata & Timestamp**:
   Obtain exact git commit SHA and current UTC timestamp:
   ```bash
   COMMIT_SHA=$(git rev-parse HEAD)
   TIMESTAMP_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
   ```
3. **Record via Ledger API**:
   Call `log_experiment_result()` from `neuralops.core.results_ledger`:
   ```python
   from neuralops.core.results_ledger import log_experiment_result

   log_experiment_result(
       module_name="selective_ssm",
       category="attention",
       family="recurrent_ssm",
       generation_method={
           "algorithm": "canonical_forward",
           "source_script": "scripts/benchmarks/benchmark_all_108_cuda_isolated.py",
           "source_artifact": "build/lib/selective_ssm.so"
       },
       experimental_setup={
           "hardware": "RTX 4070 (SM 8.9)",
           "dtype": "float16",
           "batch_size": 16,
           "seq_len": 2048
       },
       metrics={
           "latency_ms": 0.42,
           "speedup": 2.85,
           "max_diff": 0.00012,
           "accuracy": 92.4,
           "ppl": 14.1
       },
       git_commit_sha=COMMIT_SHA,
       timestamp_utc=TIMESTAMP_UTC
   )
   ```
4. **Verify Ledger Ingestion**:
   Confirm line count increment in `results/registry/master_ledger.jsonl`:
   ```bash
   wc -l results/registry/master_ledger.jsonl
   ```

## Verification
- Confirm entry exists in `master_ledger.sqlite` and query returns expected latency and metrics.

## Common pitfalls
- Logging dummy metrics without valid hardware execution metadata.
- Forgetting to capture `git_commit_sha` for reproducibility.
