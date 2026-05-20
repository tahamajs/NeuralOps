

# Experiment Runners

Two production‑grade scripts to run **any single experiment** or **batch‑test every component** of the Efficient Transformer Toolkit.

## Quick Start

```bash
# Single experiment
python run.py --model distilbert-base-uncased --task sst2 --attn_type galerkin --rank 64 --method lora

# Full batch sweep (all attentions, all compressors, all GLUE tasks)
python batch_runner.py --epochs 1 --max_samples 500

# Resume a previous batch run
python batch_runner.py --resume

# Batch with 4 parallel workers
python batch_runner.py --parallel 4
```

---

## `run.py` – Single Experiment Runner

A universal script that executes **one** experiment end‑to‑end. It uses every module of the framework:

- **Unified configuration** – merges a YAML file with CLI arguments into a `Config` dataclass.
- **Model loading** – `load_model()` handles attention injection, weight copying, and optional `UnifiedTransformerModel` wrapping.
- **Data pipeline** – `DataConfig`, `TaskType`, `load_dataset()`, `create_dataloader()`. Supports all GLUE tasks and custom datasets.
- **Training** – `HuggingFaceTrainer` with `ResultsManager`, `ExperimentTracker`, and callback support.
- **Results persistence** – `ResultsSaver` saves the model, generates plots, exports ONNX, and pushes to cloud storage.
- **Offline‑first** – environment variables are set before any HuggingFace import.

### Usage

```bash
python run.py [--config config.yaml] [options]
```

### Key Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config file | None |
| `--model` | HuggingFace model name | `distilbert-base-uncased` |
| `--task` | GLUE task name | `sst2` |
| `--attn_type` | Attention registry key | `full` |
| `--compressor` | Compressor registry key | `none` |
| `--method` | Training method (LoRA, Mixout, …) | `baseline` |
| `--rank` | Compression rank | `32` |
| `--epochs` | Number of epochs | `5` |
| `--batch_size` | Batch size per GPU | `16` |
| `--lr` | Learning rate | `2e-5` |
| `--seed` | Random seed | `42` |
| `--output_dir` | Base output directory | `./experiments` |
| `--exp_name` | Experiment name (auto‑generated if not provided) | None |
| `--fp16` | Enable mixed precision | `False` |
| `--max_train_samples` | Limit training samples (`-1` = all) | `-1` |
| `--save_visuals` | Save attention heatmaps | `False` |

All other `CompressionConfig` fields can be set via CLI or YAML.

### Example

```bash
python run.py --model bert-base-uncased --task mrpc --attn_type nova --rank 64 --method lora --epochs 3 --fp16
```

Output is saved to `./experiments/<exp_name>/` containing the model, logs, plots, and `final_metrics.json`.

---

## `batch_runner.py` – Batch Orchestrator

Systematically tests **every attention variant and compressor** across all specified models, datasets, methods, and seeds. It launches `run.py` for each combination, collects results into a master CSV, and generates cumulative plots.

### Features

- **Automatic discovery** – reads `ATTENTION_REGISTRY`, `COMPRESSOR_REGISTRY`, and `available_methods()`.
- **Resumable** – with `--resume`, it skips experiments already present in the master CSV.
- **Parallel execution** – `--parallel N` runs N experiments concurrently (uses `ProcessPoolExecutor`).
- **Live monitoring** – each sub‑process streams its console output to a `run.log` file; the master CSV is updated after every experiment.
- **Cumulative plots** – after each experiment (or periodically), a heatmap of accuracies is saved, giving a real‑time overview.
- **Final pivot table** – at completion, a `pivot_summary.csv` is created for quick analysis.

### Usage

```bash
python batch_runner.py [--epochs EPOCHS] [--max_samples N] [--resume] [--parallel N]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--epochs` | Number of epochs per experiment | `1` |
| `--max_samples` | Max training samples per experiment | `1000` |
| `--resume` | Skip already‑completed runs | `False` |
| `--parallel` | Number of parallel workers | `1` |

All other parameters are hard‑coded at the top of the script (model list, dataset list, seeds, skip list). You can edit them directly to customise the sweep.

### Example

```bash
python batch_runner.py --epochs 2 --max_samples 2000 --parallel 4
```

After completion, you’ll find:

- `experiments_batch/all_results.csv` – every experiment result
- `experiments_batch/pivot_summary.csv` – mean accuracy per attention/compressor × task
- `experiments_batch/final_cumulative.png` – heatmap of all attention performances

Each individual experiment folder (`experiments_batch/<dataset>/<exp_name>/`) contains the same artifacts as a single run.

---

## Offline Mode

Both scripts automatically set HuggingFace to offline mode. Ensure your models and datasets are cached locally in the `./models/` and `./data/glue/` directories (configured via `framework.utils.paths`).

---

## Extending

- **Add a new attention** – simply register it in `ATTENTION_REGISTRY`; it will be picked up automatically by `batch_runner.py`.
- **Add a new compressor** – register in `COMPRESSOR_REGISTRY`.
- **Change sweep scope** – edit the `BASE_MODELS`, `DATASETS`, `SEEDS` lists in `batch_runner.py`.
- **Skip broken modules** – add their keys to the `SKIP` set.

