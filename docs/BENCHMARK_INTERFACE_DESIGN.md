# NeuralOps Unified Benchmark Interface Specification

**Date:** October 2026  
**Status:** Design Phase (Step 1)  
**Scope:** Universal benchmarking across all 232 modules (159 Attention + 73 Compressors), 10+ model families, 22 datasets.

---

## 1. CLI Design

The unified CLI provides an intuitive, robust interface with dedicated subcommands: `run`, `resume`, `plan`, `list`, `verify`, and `results`.

### 1.1 Command Specifications

#### `run`
Execute benchmarks across individual modules, category groups, or full multi-dimensional grids.

```bash
# 1. Run a single module
python -m neuralops.benchmarks run \
    --module svd_linear_attention \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64

# 2. Run all attention modules (159 modules)
python -m neuralops.benchmarks run --all-attention --model distilbert-base-uncased --dataset sst2

# 3. Run all compressors (73 compressors)
python -m neuralops.benchmarks run --all-compressors --model distilbert-base-uncased --dataset sst2

# 4. Run all 232 modules
python -m neuralops.benchmarks run --all --model distilbert-base-uncased --dataset sst2

# 5. Run a specific comma-separated subset
python -m neuralops.benchmarks run \
    --modules svd_linear_attention,cosformer,gqa_attention \
    --model distilbert-base-uncased \
    --dataset sst2

# 6. Grid sweep across all models and all datasets
python -m neuralops.benchmarks run \
    --module cosformer \
    --all-models \
    --all-datasets

# 7. Multi-GPU parallel sweep
python -m neuralops.benchmarks run \
    --all \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --parallel 4
```

| Flag | Type | Default | Aliases | Description |
|---|---|---|---|---|
| `--module` | string | `None` | `-m` | Target module name |
| `--modules` | csv/list | `None` | `--module-list` | Multiple comma-separated modules |
| `--all-attention` | flag | `False` | `--attentions` | Select all 159 attention modules |
| `--all-compressors`| flag | `False` | `--compressors` | Select all 73 compression engines |
| `--all` | flag | `False` | `--all-modules` | Select all 232 modules |
| `--model` | string | `distilbert-base-uncased` | `--model-name` | Model name or HuggingFace path |
| `--all-models` | flag | `False` | - | Grid across all canonical model families |
| `--dataset` | string | `sst2` | `--task` | Dataset name |
| `--all-datasets` | flag | `False` | - | Grid across all GLUE datasets |
| `--rank` | int | `64` | `-r` | Compression rank |
| `--ranks` | csv/list | `None` | `--rank-list` | Comma-separated ranks (e.g. `16,32,64,128`) |
| `--epochs` | int | `3` | - | Fine-tuning epochs |
| `--recovery-steps`| int | `50` | `--steps` | Recovery gradient steps |
| `--batch-size` | int | `32` | `-b` | Evaluation and training batch size |
| `--seq-len` | int | `128` | `--max-length` | Sequence length |
| `--lr` | float | `5e-5` | `--learning-rate`| Optimizer learning rate |
| `--parallel` | int | `1` | `-j`, `--gpus` | Number of worker processes / GPUs |
| `--output-dir` | Path | `results/run_<timestamp>` | `-o` | Output directory root |
| `--timeout` | int | `600` | - | Per-module timeout in seconds |

#### `resume`
Resume an interrupted sweep from its persistent `run_state.json`.
```bash
python -m neuralops.benchmarks resume --run-id 90ab0eb7-fcfd-4cd9-a696-b0a466ecec5e
python -m neuralops.benchmarks resume --output-dir results/full_sweep_20261008
```

#### `plan` (Dry-Run Preview)
Preview the exact combinatorial grid of runs, estimated durations, and GPU memory requirements without running.
```bash
python -m neuralops.benchmarks plan --all --model distilbert-base-uncased --dataset sst2
```

#### `list`
Filter and list modules by capabilities.
```bash
python -m neuralops.benchmarks list --type attention --supports causal
python -m neuralops.benchmarks list --type compressor
```

#### `verify`
Verify environment, PyTorch installation, GPU VRAM, and HuggingFace connectivity.
```bash
python -m neuralops.benchmarks verify
```

#### `results`
Inspect and format results from prior runs.
```bash
python -m neuralops.benchmarks results --latest
python -m neuralops.benchmarks results --run-id <uuid> --summary
```

---

## 2. Python API

A clean, non-blocking Python API designed for both interactive scripting and automated workflows:

```python
from neuralops.benchmarks import benchmark, Runner

# 1. Single Execution
result = benchmark(
    module="svd_linear_attention",
    model="distilbert-base-uncased",
    dataset="sst2",
    rank=64,
    device="cuda",
)
print("Status:", result.status)
print("Accuracy:", result.metrics["accuracy"])
print("Peak VRAM:", result.resources["peak_vram_mb"])

# 2. Grid Orchestration
runner = Runner(
    modules="all-attention",       # or "all", "all-compressors", or list
    models=["distilbert-base-uncased", "roberta-base"],
    datasets=["sst2", "mrpc"],
    ranks=[32, 64, 128],
    parallel=4,                    # 4 parallel worker processes across GPUs
    output_dir="results/full_sweep",
    resume=True,
)

# Execution modes:
results = runner.run()             # Synchronous (blocking)
runner.run_async()                 # Non-blocking background execution
status_info = runner.status()      # Query live progress: {completed: 45, total: 232, failed: 0}
df = runner.summary()              # Returns aggregated pandas DataFrame

# Streaming generator
for r in runner.stream():
    print(f"[{r.status}] {r.module.name} on {r.dataset.name}: Acc={r.metrics.accuracy}")
```

### 2.1 API Classes & Signatures

```python
class Runner:
    def __init__(
        self,
        modules: str | list[str] = "all",
        models: str | list[str] = "distilbert-base-uncased",
        datasets: str | list[str] = "sst2",
        ranks: int | list[int] = 64,
        parallel: int = 1,
        output_dir: Path | str = "results",
        resume: bool = True,
        timeout_s: int = 600,
        verbose: bool = False,
    ) -> None: ...

    def run(self) -> list[BenchmarkResult]: ...
    def run_async(self) -> None: ...
    def status(self) -> dict[str, Any]: ...
    def stream(self) -> Generator[BenchmarkResult, None, None]: ...
    def summary(self) -> pd.DataFrame: ...
    def cancel(self) -> None: ...
```

---

## 3. Unified JSON Schema

Every run produces an identical, strictly validated JSON document:

```json
{
  "run_id": "90ab0eb7-fcfd-4cd9-a696-b0a466ecec5e",
  "run_name": "full_attention_sweep_20261008_120000",
  "timestamp_start": "2026-10-08T12:00:00Z",
  "timestamp_end": "2026-10-08T12:01:23Z",
  "git_commit": "7354d98e81",

  "module": {
    "name": "svd_linear_attention",
    "type": "attention",
    "family": "svd",
    "supports_causal": true,
    "supports_gqa": true,
    "supports_rope": true,
    "requires_cuda": false
  },

  "model": {
    "name": "distilbert-base-uncased",
    "family": "distilbert",
    "num_params": 66955010,
    "hidden_size": 768,
    "num_heads": 12,
    "num_layers": 6
  },

  "dataset": {
    "name": "sst2",
    "task": "classification",
    "num_labels": 2,
    "num_train": 67349,
    "num_eval": 872
  },

  "config": {
    "rank": 64,
    "lora_rank": 16,
    "epochs": 3,
    "lr": 0.00005,
    "batch_size": 32,
    "seq_len": 128,
    "seed": 42,
    "calibration_samples": 128
  },

  "metrics": {
    "accuracy": 0.8647,
    "baseline_accuracy": 0.912,
    "accuracy_drop": 0.0473,
    "f1": 0.864,
    "precision": 0.865,
    "recall": 0.8647,
    "loss": 0.3241,
    "perplexity": null
  },

  "resources": {
    "peak_vram_mb": 620.5,
    "avg_vram_mb": 480.2,
    "latency_ms_per_token": 0.0234,
    "throughput_tok_per_sec": 42500.0,
    "duration_s": 24.3,
    "num_params_before": 66955010,
    "num_params_after": 57840233,
    "compression_ratio": 1.157
  },

  "status": "success",
  "error": null,
  "traceback": null,
  "warnings": []
}
```

For failed runs:
```json
{
  "run_id": "...",
  "status": "failed",
  "error": "RuntimeError: CUDA out of memory at step 42",
  "traceback": "Traceback (most recent call last): ...",
  "metrics": null,
  "resources": null
}
```

---

## 4. File Structure

```text
neuralops/benchmarks/
├── __init__.py              # Exports: benchmark, Runner, BenchmarkResult
├── __main__.py              # python -m neuralops.benchmarks entrypoint
├── api.py                   # High-level Python API: benchmark(), Runner
├── cli.py                   # Subcommands: run, resume, plan, list, verify, results
├── result.py                # BenchmarkResult, ModuleMetadata, ModelMetadata dataclasses
├── config.py                # BenchmarkConfig, GridConfig dataclasses
├── registry.py              # Canonical discovery and registry indexing
├── discovery.py             # Automatic capability inspection (causal, GQA, RoPE, CUDA)
├── executor.py              # Subprocess-isolated single-run worker
├── orchestrator.py          # Parallel queue execution + state persistence + resume
├── writers.py               # JSON, CSV, Markdown, leaderboard writers
├── core/
│   ├── __init__.py
│   ├── arguments.py         # Shared argparse definitions
│   ├── model_loader.py      # HuggingFace model loading and adapter injection
│   ├── data_loader.py       # GLUE, Wikitext, dataloaders
│   ├── compression.py       # Compression engine unified wrapper
│   ├── training.py          # Post-compression recovery fine-tuning
│   ├── evaluation.py        # GLUE evaluation and perplexity scoring
│   ├── stats.py             # Welch t-test, bootstrap CI, FDR correction
│   └── kd.py                # Knowledge distillation utilities
└── adapters/
    ├── __init__.py
    ├── classification.py    # GLUE benchmark task adapter
    ├── perplexity.py        # Wikitext language modeling adapter
    ├── zeroshot.py          # Zero-shot evaluation adapter
    └── ruler.py             # Long-context RULER NIAH adapter
```

---

## 5. Auto-Discovery & Metadata Protocol

The discovery engine inspects modules dynamically without requiring code modifications:

1. **Protocol Attributes:**
   ```python
   supports_causal: bool = False
   supports_gqa: bool = False
   supports_rope: bool = False
   requires_cuda: bool = False
   supported_models: list[str] = ["all"]
   benchmark_category: str = "attention"
   ```
2. **Current Registry State:**
   - **`supports_causal`**: Explicitly declared on 83 modules; remaining 76 attention modules inferred from constructor parameter signature (`is_causal` / `causal`).
   - **`supports_gqa`**: Explicitly declared on 82 modules; remaining inferred from `num_kv_heads`.
   - **`supports_rope`**: Explicitly declared on 38 modules; remaining inferred from rotary/position embedding parameters.
   - **`requires_cuda`**: 0 modules require CUDA exclusively; all 232 provide pure PyTorch CPU fallbacks.
3. **Fallback Defaults:** If a module does not declare an attribute, `discovery.py` uses heuristic signature reflection with safe defaults (`supports_causal=False`, `requires_cuda=False`).

---

## 6. Resume & Multi-GPU Parallel Execution

### 6.1 Resumability
- Every execution session maintains `run_state.json` inside its output directory:
  ```json
  {
    "run_id": "90ab0eb7-fcfd-4cd9-a696-b0a466ecec5e",
    "completed": ["svd_linear_attention:distilbert:sst2:64"],
    "failed": [],
    "pending": ["cosformer:distilbert:sst2:64"]
  }
  ```
- Re-running with `--resume` checks `run_state.json` and skips completed tasks immediately. Runs are completely idempotent.

### 6.2 Parallel Multi-GPU Workers
- The grid is partitioned across `N` available GPUs.
- Each worker runs in an isolated subprocess with `CUDA_VISIBLE_DEVICES=<gpu_id>`.
- Failures in one module (e.g. OOM) are isolated to that subprocess; the worker catches the error, writes a `status="failed"` result file, resets VRAM, and proceeds to the next candidate.
- All results are aggregated into `summary.csv`, `summary.md`, and `leaderboard.csv` upon completion.

---

## 7. Summary Reports Generated

At the conclusion of a sweep, the following files are produced in the output directory:
1. `runs/<module>_<model>_<dataset>_<rank>.json`: Raw JSON documents for each individual run.
2. `summary.csv`: Flat tabular CSV of all runs.
3. `summary.md`: Publication-ready GitHub Markdown table.
4. `summary.json`: Hierarchical JSON grouped by module type and family.
5. `leaderboard.csv`: Top-ranked modules sorted by compression ratio vs accuracy drop.
6. `failures.log`: Human-readable log detailing any failed runs with stack traces.

---

## 8. Example: Running All 232 Modules

### Command:
```bash
python -m neuralops.benchmarks run \
    --all \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 32 \
    --parallel 4 \
    --output-dir results/full_sweep_20261008
```

### Execution Flow:
1. Discovers 159 attention modules + 73 compressor modules = 232 total.
2. Launches 4 concurrent workers (across GPUs 0, 1, 2, 3).
3. Evaluates 232 candidates in ~1.5 hours (parallelized).
4. Emits `results/full_sweep_20261008/summary.md` and `leaderboard.csv`.

---

## 9. Backward Compatibility Plan

All legacy runner script paths are retained as thin deprecation shims:

| Old Script | New Invocation | Shim Action |
|---|---|---|
| `run_quality_sweep.py` | `python -m neuralops.benchmarks run ...` | Translates legacy CLI flags and calls `Runner` |
| `run_gpt2_sweep.py` | `python -m neuralops.benchmarks run --model gpt2 --dataset wikitext2 ...` | Wraps `Runner(category="lm")` |
| `kd_quick_test.py` | `python -m neuralops.benchmarks run --kd ...` | Delegates to KD adapter |
| `run_rank_sweeper.py` | `python -m neuralops.benchmarks run --ranks 16,32,64,128 ...` | Delegates to rank sweep grid |
| `multi_mode_benchmark.py` | `python -m neuralops.benchmarks run --modes ...` | Delegates to multi-mode adapter |
| `run_multiseed_rigor.py` | `python -m neuralops.benchmarks run --seeds 42,1337,2024 ...` | Delegates to multi-seed statistical runner |
| `run_seqlen_scaling_bench.py` | `python -m neuralops.benchmarks run --seq-lens ...` | Delegates to sequence scaling runner |
| `run_ruler_benchmark.py` | `python -m neuralops.benchmarks run --adapter ruler ...` | Delegates to RULER adapter |
| `run_all_benchmarks.py` | `python -m neuralops.benchmarks run --all ...` | Unified orchestrator |
| `run_one_module_isolated.py` | `python -m neuralops.benchmarks.executor ...` | Internal subprocess worker entrypoint |


---

## 10. Corrections Applied (C.1 — C.5)

### C.1 Grid Size Estimation in `plan`
- When `--all-models` or `--all-datasets` is provided, `plan` outputs total combinations: `N_modules * N_models * N_datasets * N_ranks`.
- Estimated runtime calculated using a baseline heuristic (e.g. 20-30s per standard GLUE SST-2 run on modern GPUs).
- If total runs > 500, user prompt requires explicit `--yes` flag to proceed with execution.

### C.2 Configurable Timeouts
- Default `timeout_s` is 1800s (30 minutes).
- Overridable globally with `--timeout <seconds>` or `--no-timeout` for unbounded execution.
- Overridable per module via `--timeout-per-module <module=secs,module=secs>`.
- In output schema:
  ```json
  "timeout_s": 1800,
  "timed_out": false
  ```

### C.3 Manual Capability Overrides (`discovery_overrides.json`)
- `neuralops/benchmarks/discovery_overrides.json` specifies manual adjustments:
  ```json
  {
    "sparsegpt": {"requires_cuda": false, "timeout_s": 3600},
    "triton_svd_linear_attention": {"requires_cuda": false},
    "flash_attention3": {"requires_cuda": true, "supports_rope": true}
  }
  ```
- Lookup resolution hierarchy:
  1. `discovery_overrides.json`
  2. Class attributes (`supports_causal`, `supports_gqa`, `supports_rope`, etc.)
  3. Signature reflection (`inspect.signature`)

### C.4 Legacy Runner Shims
- Legacy runners are relocated to `neuralops/benchmarks/_legacy/run_X.py`.
- Deprecation shims at original paths emit `DeprecationWarning` and re-export `main` and public symbols.

### C.5 Automated Quality & Safety Warnings in Schema
Populate `warnings: list[str]` in the output JSON schema when:
- CPU fallback occurred instead of requested GPU/CUDA execution.
- Execution duration took > 5x estimated average.
- Post-compression accuracy drop exceeds 20.0% (suspicious degeneration).
- Total parameter count before and after compression is identical for a compressor (inactive compressor).
