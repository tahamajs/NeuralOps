# NeuralOps Benchmarks Architecture Audit & Unified Design

**Scope:** `neuralops/benchmarks/` (22 files, ~15,500 lines of code)  
**Date:** October 2026  
**Status:** Audit & Design Complete (Phases 1 & 2)

---

# PHASE 1 — FULL AUDIT REPORT

## 1.1 File Inventory

| File | Lines | Classes | Functions | Primary Purpose | Dead Code? | Duplications? |
|---|---|---|---|---|---|---|
| `run_quality_sweep.py` | 5,740 | 1 | 39 | Primary multi-model accuracy & quality benchmark | YES (imports, redefined funcs) | HIGH (parse_args, stats, kd, eval) |
| `kd_quick_test.py` | 1,872 | 2 | 37 | Knowledge distillation validation test | YES (unused variables) | HIGH (PPL eval, cache logic) |
| `run_gpt2_sweep.py` | 1,459 | 1 | 18 | GPT-2 language model perplexity benchmark | YES (unused import) | HIGH (CSV writing, KD recovery) |
| `multi_mode_benchmark.py` | 1,459 | 7 | 35 | Multi-mode transformer injection benchmarking | YES (7 unused imports) | MED (eval loop, model loader) |
| `run_all_modules_scheduler.py` | 810 | 0 | 13 | Ad-hoc parallel scheduler for all modules | NO | HIGH (CLI arg wrapper, worker pools) |
| `run_rank_sweeper.py` | 583 | 0 | 17 | SVD rank vs accuracy sweep runner | NO | MED (CSV/Markdown plotting, cmd building) |
| `run_long_context_sweep.py` | 494 | 0 | 6 | Long context sequence length sweep | YES (undefined name `Any`) | HIGH (identical to seqlen scaling) |
| `run_ruler_benchmark.py` | 485 | 0 | 5 | RULER / Needle In A Haystack benchmark | NO | MED (Synthetic context generation) |
| `run_seqlen_scaling_bench.py` | 485 | 0 | 4 | Sequence length scaling evaluation | NO | HIGH (duplicate of seqlen_scaling) |
| `run_multiseed_rigor.py` | 460 | 0 | 5 | Multi-seed statistical significance runner | NO | HIGH (duplicate of run_multiseed) |
| `run_all_benchmarks.py` | 419 | 0 | 5 | Orchestrator for all benchmark runs | NO | MED (subprocess command runner) |
| `run_multiseed.py` | 318 | 0 | 3 | Legacy multi-seed evaluator | NO | HIGH (duplicate of multiseed_rigor) |
| `run_seqlen_scaling.py` | 282 | 0 | 3 | Minimal sequence length scaling runner | NO | HIGH (duplicate of seqlen_scaling_bench) |
| `gpu_live_probe.py` | 255 | 0 | 4 | Real-time GPU & process monitor utility | NO | LOW |
| `run_one_module_isolated.py` | 219 | 0 | 1 | Subprocess worker for isolated execution | YES (redefined func) | HIGH (shim to run_quality_sweep) |
| `run_whitened_svd.py` | 146 | 0 | 6 | Whitened SVD calibration runner | NO | LOW |
| `launch_parallel_sweep.py` | 103 | 0 | 1 | Multi-worker parallel runner wrapper | NO | LOW |
| `harvest_gpu_benchmark_data.py` | 38 | 0 | 1 | JSON data harvester utility | NO | LOW |
| `core/arguments.py` | 687 | 0 | 1 | Monolithic shared argument parser | YES (undefined names) | CRITICAL (copy-paste of quality_sweep) |
| `core/stats.py` | 81 | 0 | 4 | Statistical significance calculations | YES (redefined imports) | HIGH (duplicate of quality_sweep) |
| `core/__init__.py` | 26 | 0 | 0 | Core package init | NO | NO |
| `__init__.py` | 51 | 0 | 0 | Benchmarks package exports | NO | NO |

---

## 1.2 CLI Analysis & Inconsistencies

There are **11 different custom `parse_args` implementations** and **15 different `main` functions**, leading to serious naming divergences across runners:

| Argument Concept | `run_quality_sweep.py` | `run_gpt2_sweep.py` | `run_rank_sweeper.py` | `kd_quick_test.py` | `multi_mode_benchmark.py` |
|---|---|---|---|---|---|
| **Module under test** | `--modules`, `--module` | `--modules`, `--category` | `--modules` | `--module`, `--modules` | `--modes` |
| **Model target** | `--model` | `--model_path` | `--model`, `--model_family` | `--model_name` | `--model-name` |
| **Dataset target** | `--dataset`, `--dataset_dir` | `--dataset_name`, `--split` | `--dataset` | Hardcoded Wikitext-2 | `--dataset` |
| **Sequence Length** | `--seq_len`, `--max_seq_len` | `--seq_len`, `--max_length` | Hardcoded / passed | `--seq_len`, `--max_length` | `--seq-len` |
| **Output Directory** | `--output_dir` | `--output_dir` | `--output_base`, `--output_dir` | `--run_dir`, `--output_dir` | `--output-dir` |
| **Rank** | `--rank`, `--lora_rank` | `--rank` | `--ranks` (comma-sep) | `--rank` | `--rank` |
| **Flag format** | `snake_case` (`--batch_size`) | `snake_case` (`--model_path`) | `snake_case` (`--recovery_steps`) | `snake_case` (`--save_weights`) | `kebab-case` (`--batch-size`) |

**Key Findings:**
1. `core/arguments.py` was originally an exact copy-paste snapshot of `run_quality_sweep.py:parse_args`, but drifted and has missing imports (`Path`, `logging`, `LOG`).
2. Runners don't share arg definitions; they manually declare ad-hoc flags or shell out via CLI strings.

---

## 1.3 Output Format Analysis

Outputs are written to fragmented locations with incompatible schema structures:

1. **Target Directories:**
   - `results/`
   - `results/01_unified_runs/`
   - `results/02_rank_sweeps/`
   - `results/03_zeroshot_evals/`
   - `results/05_glue_finetuning/multiseed/`
   - `results/06_adhoc_experiments/perplexity_seqlen/`
   - `results/06_adhoc_experiments/gpt2_sweep/`
   - `logs/isolated_results/`
   - `./benchmark_results/`
2. **Schema Inconsistencies:**
   - Metric keys diverge: `acc` vs `accuracy_pct` vs `metrics.accuracy` vs `eval_accuracy`.
   - Perplexity format: `perplexity` vs `ppl` vs `test_ppl` vs `baseline_ppl`.
   - Timing: `elapsed_s` vs `latency_ms` vs `duration_sec` vs `total_time`.
   - Metadata: Missing Git commit hash, hardware information, or PyTorch environment tags in half the runners.

---

## 1.4 Code Duplication Patterns

| Duplicated Component | Sharing Files | Duplicated Lines | Canonical Reference |
|---|---|---|---|
| **Model & Tokenizer Loading** | `run_quality_sweep`, `run_gpt2_sweep`, `kd_quick_test`, `multi_mode_benchmark`, `run_whitened_svd` | ~450 lines | `run_quality_sweep.py` |
| **Wikitext / GLUE Data Loading** | `run_quality_sweep`, `run_gpt2_sweep`, `kd_quick_test`, `run_long_context_sweep` | ~380 lines | `run_quality_sweep.py` |
| **Statistical Metrics (t-test, CI, bootstrap)** | `run_quality_sweep.py`, `core/stats.py` | ~120 lines | `core/stats.py` |
| **CLI Argument Parsing** | `run_quality_sweep.py`, `core/arguments.py` | ~680 lines | Consolidated unified parser |
| **Sequence Length Sweep Logic** | `run_seqlen_scaling.py`, `run_seqlen_scaling_bench.py`, `run_long_context_sweep.py` | ~650 lines | Single `seqlen_scaling/runner.py` |
| **Multi-seed Rigor Runner** | `run_multiseed.py`, `run_multiseed_rigor.py` | ~400 lines | Single `multiseed/runner.py` |
| **KD (Knowledge Distillation) Loop** | `run_quality_sweep.py`, `kd_quick_test.py`, `run_gpt2_sweep.py` | ~500 lines | Shared KD training utility |

---

## 1.5 Dead Code & Syntax Errors

1. **Undefined Names / Syntax Issues:**
   - `run_long_context_sweep.py`: line 44 references undefined `Any`.
   - `core/arguments.py`: lines 547, 560, 563, 602, 640 reference undefined `Path`, `logging`, `LOG`, `os`, `_auto_parallel_and_batch`.
2. **Redefined / Dead Functions:**
   - `run_quality_sweep.py:4158`: redefinition of unused `save_lora_weights`.
   - `run_one_module_isolated.py:75`: redefinition of `run_single_module_main`.
   - `core/stats.py`: repeated `import numpy as np` / `import scipy.stats` at module & function level.
3. **Unused Imports:**
   - `multi_mode_benchmark.py`: `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, `DistilBertConfig`, `PreTrainedModel`, `detect_architecture`, `get_transformer_layers`, `get_gpu_memory_stats`.
   - `run_gpt2_sweep.py`: unused `neuralops.compression_engine`.

---

## 1.6 Top Long Functions (> 100 lines)

1. `run_quality_sweep.py:3519` — `run_single_module_main()` (1,825 lines) 🚨 **CRITICAL**
2. `run_quality_sweep.py:2325` — `_run_candidate_pipeline()` (1,191 lines) 🚨 **CRITICAL**
3. `run_quality_sweep.py:1629` — `parse_args()` (664 lines)
4. `core/arguments.py:7` — `parse_args()` (649 lines)
5. `run_gpt2_sweep.py:871` — `main()` (584 lines)
6. `kd_quick_test.py:1148` — `run_kd_method()` (478 lines)
7. `multi_mode_benchmark.py:869` — `run_single_mode()` (244 lines)
8. `run_quality_sweep.py:5510` — `main()` (226 lines)
9. `run_quality_sweep.py:1356` — `finetune_lora()` (222 lines)
10. `kd_quick_test.py:1659` — `main()` (210 lines)
11. `run_all_benchmarks.py:213` — `main()` (202 lines)
12. `run_one_module_isolated.py:25` — `main()` (190 lines)
13. `run_multiseed.py:129` — `main()` (185 lines)
14. `run_quality_sweep.py:1024` — `evaluate_comprehensive()` (180 lines)
15. `run_quality_sweep.py:521` — `compute_compression_metrics()` (177 lines)
16. `run_long_context_sweep.py:314` — `main()` (176 lines)
17. `run_all_modules_scheduler.py:647` — `main()` (159 lines)
18. `kd_quick_test.py:247` — `save_run_details()` (154 lines)
19. `run_ruler_benchmark.py:331` — `main()` (150 lines)
20. `run_quality_sweep.py:756` — `collect_and_export_full_metrics()` (145 lines)

---

## 1.7 Circular Imports & Dependencies

- `run_all_benchmarks.py` imports `run_gpt2_sweep`.
- `run_multiseed.py` imports `run_quality_sweep`.
- `run_one_module_isolated.py` imports `run_quality_sweep`.
- No circular package-level dependencies detected.

---

# PHASE 2 — UNIFIED BENCHMARKS ARCHITECTURE DESIGN

## 2.1 Target Directory Structure

All files bounded under **< 800 lines**:

```
neuralops/benchmarks/
├── __init__.py
├── cli.py                    # Unified CLI dispatcher (subcommand router)
├── runner.py                 # Abstract BaseRunner class
├── config.py                 # BenchmarkRunConfig dataclasses
├── writers.py                # Standardized JSON / CSV / Markdown writers
├── core/
│   ├── __init__.py
│   ├── arguments.py          # Clean shared argparse builder
│   ├── stats.py              # Statistical testing (t-test, bootstrap, FDR)
│   ├── model_loader.py       # Robust HuggingFace model/tokenizer loading
│   ├── data_loader.py        # Tokenized batching, GLUE & Wikitext data loading
│   └── kd.py                 # Shared Knowledge Distillation loss & cache
├── quality_sweep/            # Split from run_quality_sweep.py (5,740 lines)
│   ├── __init__.py
│   ├── runner.py             # Main pipeline orchestration (~500 lines)
│   ├── candidates.py         # Candidate generation & module filtering (~350 lines)
│   ├── pipeline.py           # Per-candidate execution pipeline (~550 lines)
│   ├── training.py           # Fine-tuning & LoRA loop (~450 lines)
│   ├── evaluation.py         # Comprehensive eval & PPL (~400 lines)
│   └── reports.py            # Result aggregation & export (~350 lines)
├── rank_sweep/
│   ├── __init__.py
│   └── runner.py             # Rank curve sweeper (< 500 lines)
├── multi_mode/
│   ├── __init__.py
│   ├── runner.py             # Multi-mode benchmark runner (< 500 lines)
│   ├── modes.py              # Mode configuration (< 300 lines)
│   └── transforms.py         # Transform engine (< 400 lines)
├── multiseed/
│   ├── __init__.py
│   └── runner.py             # Unified multi-seed runner (< 500 lines)
├── seqlen_scaling/
│   ├── __init__.py
│   └── runner.py             # Unified sequence length scaling (< 500 lines)
├── ruler/
│   ├── __init__.py
│   └── runner.py             # RULER NIAH benchmark (< 450 lines)
├── gpt2_sweep/
│   ├── __init__.py
│   ├── runner.py             # GPT-2 perplexity runner (< 500 lines)
│   └── evaluation.py         # Calibration & recovery (< 450 lines)
└── kd_test/
    ├── __init__.py
    ├── runner.py             # Quick KD runner (< 500 lines)
    └── methods.py            # KD methods & cache (< 500 lines)
```

## 2.2 Unified CLI Interface

CLI command syntax:
```bash
python -m neuralops.benchmarks <subcommand> [options]
```

Subcommands:
- `quality-sweep`: Run full quality / accuracy sweep.
- `rank-sweep`: Run rank vs quality scaling sweep.
- `multi-mode`: Run multi-mode structural benchmarking.
- `multiseed`: Run statistical multi-seed verification.
- `seqlen-scaling`: Run sequence length scaling throughput & latency benchmark.
- `ruler`: Run long-context RULER benchmark.
- `gpt2-sweep`: Run GPT-2 decoder module sweep.
- `kd-test`: Run quick knowledge distillation validation test.
- `list`: List all registered benchmark tasks and modules.
- `verify`: Verify model, dataset, and GPU readiness.

Each runner extends `BaseRunner`:
```python
class BaseRunner(ABC):
    def __init__(self, config: BenchmarkRunConfig): ...
    @abstractmethod
    def run(self) -> BenchmarkResult: ...
    def write_results(self, result: BenchmarkResult) -> Path: ...
```

## 2.3 Unified Output Schema (`BenchmarkResult`)

Canonical schema stored at `results/<benchmark>/<dataset>/<model>/<module>/result.json`:

```json
{
  "run_id": "90ab0eb7-fcfd-4cd9-a696-b0a466ecec5e",
  "timestamp": "2026-10-08T13:50:00Z",
  "benchmark": "quality_sweep",
  "module": "svd_linear_attention",
  "model": "distilbert-base-uncased",
  "dataset": "sst2",
  "config": {
    "rank": 64,
    "batch_size": 32,
    "seq_len": 128,
    "recovery_steps": 50,
    "lr": 5e-5,
    "seed": 42
  },
  "metrics": {
    "accuracy_pct": 86.47,
    "baseline_accuracy_pct": 91.20,
    "accuracy_drop_pct": 4.73,
    "perplexity": null,
    "eval_loss": 0.3241
  },
  "resources": {
    "peak_vram_mb": 620.5,
    "latency_ms": 12.4,
    "throughput_tok_per_sec": 42500.0
  },
  "status": "PASS",
  "duration_s": 24.3,
  "git_commit": "7354d98e81"
}
```

## 2.4 Migration Mapping

| Old File Path | Target File Path(s) | Action |
|---|---|---|
| `run_quality_sweep.py` | `quality_sweep/` (`runner.py`, `candidates.py`, `pipeline.py`, `training.py`, `evaluation.py`, `reports.py`) | Split 5,740 lines into 6 clean submodules + add shim |
| `kd_quick_test.py` | `kd_test/runner.py`, `kd_test/methods.py` | Split 1,872 lines into 2 submodules + add shim |
| `run_gpt2_sweep.py` | `gpt2_sweep/runner.py`, `gpt2_sweep/evaluation.py` | Split 1,459 lines into 2 submodules + add shim |
| `multi_mode_benchmark.py` | `multi_mode/runner.py`, `multi_mode/modes.py`, `multi_mode/transforms.py` | Split 1,459 lines into 3 submodules + add shim |
| `run_rank_sweeper.py` | `rank_sweep/runner.py` | Refactor to inherit BaseRunner (< 500 lines) + add shim |
| `run_multiseed_rigor.py` & `run_multiseed.py` | `multiseed/runner.py` | Merge redundant runners into unified multiseed runner + add shims |
| `run_seqlen_scaling.py`, `run_seqlen_scaling_bench.py`, `run_long_context_sweep.py` | `seqlen_scaling/runner.py` | Merge 3 redundant files into unified seqlen scaling runner + add shims |
| `run_ruler_benchmark.py` | `ruler/runner.py` | Refactor into runner (< 450 lines) + add shim |
| `run_all_benchmarks.py` & `run_all_modules_scheduler.py` | Integrated into `cli.py` & `launch_parallel_sweep.py` | Move to top-level CLI orchestrator + add shims |
| `run_one_module_isolated.py` | Integrated into `quality_sweep/pipeline.py` | Retain as thin subprocess worker shim (< 100 lines) |
| `run_whitened_svd.py` | Merged into `quality_sweep/candidates.py` / `core/` | Retain as shim |
| `core/arguments.py` | Cleaned and deduplicated | Fix undefined names, unify shared flags |
| `core/stats.py` | Cleaned and deduplicated | Remove redundant imports, fix docstrings |
