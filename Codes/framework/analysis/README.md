# 📊 Analysis Package

Comprehensive tools for analysing, visualising, and comparing transformer compression experiments – especially SVD‑based attention compression.  
Provides publication‑ready plots, metrics tracking, layer‑wise analysis, model comparison, and automatic report generation.

## 📁 Structure

```
framework/analysis/
├── __init__.py
├── plotting.py            # All plot functions (comprehensive grid, tradeoff, Pareto, rank allocation)
├── metrics.py             # MetricsTracker, compute_metrics, aggregate_results
├── layerwise_analyzer.py  # Layer‑wise hidden state analysis (cosine similarity, effective rank, intrinsic dimension)
├── model_comparison.py    # Compare multiple runs (load, summarise, generate plots)
├── report_generator.py    # HTML, CSV, LaTeX reports
├── callbacks.py           # Trainer callback for logging and saving training curves
├── utils.py               # Filtering, best configs, compressibility detection
└── README.md
```

## 🚀 Quick Start

### 1. Basic analysis of a single experiment

```python
import pandas as pd
from framework.analysis import create_comprehensive_plots, plot_tradeoff

df = pd.read_csv("results/run1/performance.csv")
create_comprehensive_plots(df, "plots/run1")
plot_tradeoff(df, "plots/run1")
```

### 2. Track metrics during training

```python
from framework.analysis import MetricsTracker, LoggingCallback

tracker = MetricsTracker("results/exp1/metrics.json")
trainer.add_callback(LoggingCallback(tracker, logger, plots_dir="results/exp1/plots"))
```

### 3. Compare multiple runs

```python
from framework.analysis import quick_compare

# Directories containing performance.csv files
dirs = ["results/bert_base", "results/bert_large", "results/distilbert"]
quick_compare(dirs, output_dir="comparison")
```

### 4. Layer‑wise compressibility analysis

```python
from framework.analysis import LayerwiseAnalyzer, detect_least_compressible_layers

analyzer = LayerwiseAnalyzer(model, tokenizer)
result = analyzer.analyze(["Example sentence 1", "Example sentence 2"])
print(result['cosine_similarities'])
critical = detect_least_compressible_layers(model, tokenizer, texts, k=3)
print(f"Keep layers {critical} at full rank")
```

### 5. Generate a full report

```python
from framework.analysis import generate_full_report

df = pd.read_csv("results/summary.csv")
generate_full_report(df, "reports/experiment", experiment_name="my_experiment")
```

## 🧩 Detailed Module Documentation

### `plotting.py`

| Function | Description |
|----------|-------------|
| `create_comprehensive_plots(df, output_dir)` | 3×3 grid: accuracy vs rank (model/task), speedup, param reduction, memory reduction, accuracy drop distribution, heatmap, best config per dataset. |
| `plot_tradeoff(df, output_dir, metric)` | Scatter plot of speedup vs accuracy (or another metric) with rank annotations. |
| `plot_pareto_front(df, output_dir, x_metric, y_metric)` | Highlights non‑dominated (Pareto‑optimal) configurations. |
| `plot_rank_allocation(recommended_ranks, output_dir)` | Bar chart of recommended ranks per layer. |
| `plot_training_history(history, save_path)` | Line plot of training/evaluation loss. |

All plots use `seaborn-v0_8-darkgrid` style, configurable.

### `metrics.py`

- **`MetricsTracker(filepath)`** – logs metrics as JSON lines; methods: `.log()`, `.save()`, `.to_dataframe()`.
- **`compute_metrics(predictions, labels, task)`** – returns accuracy, F1, Matthews (classification) or Pearson/Spearman (regression).
- **`aggregate_results(results_list)`** – computes mean, std, and list of values per key.

### `layerwise_analyzer.py`

- **`LayerwiseAnalyzer(model, tokenizer)`** – register hooks, forward pass, compute cosine similarity between consecutive layers.
- **`estimate_intrinsic_dimension(hidden_states, k)`** – nearest‑neighbour based intrinsic dimension estimate.

### `model_comparison.py`

- **`ModelComparison(results_dirs, output_dir)`** – loads all `performance.csv`, concatenates, generates plots.
  - `load_all_results() -> DataFrame`
  - `generate_all_plots()` – comprehensive grid, tradeoff, Pareto.
  - `best_accuracy_table() -> DataFrame`
  - `summary_table() -> DataFrame`
- **`quick_compare(dirs, output_dir)`** – one‑call convenience function.

### `report_generator.py`

- **`generate_html_report(df, output_dir, title)`** – creates an HTML page with summary table.
- **`generate_latex_table(df, output_path, caption)`** – exports a LaTeX `table` environment.
- **`generate_full_report(df, output_dir, experiment_name)`** – produces HTML, CSV, LaTeX, and all plots.

### `callbacks.py`

- **`LoggingCallback(tracker, logger, plots_dir, save_plots)`** – Trainer callback that logs metrics, saves training history plots, and flushes tracker.

### `utils.py`

- **`filter_results(df, models, tasks, ranks, min_accuracy)`** – filters DataFrame.
- **`get_best_configurations(df, metric)`** – returns best config per (task, model).
- **`detect_least_compressible_layers(model, tokenizer, calibration_texts, k)`** – identifies layers that should **not** be heavily compressed (based on large changes in cosine similarity).

## 🔧 Integration with the Toolkit

- The `analysis` package expects DataFrames with at least columns `model`, `task`, `rank`, `accuracy`.
- Optional columns: `speedup_factor`, `param_reduction`, `memory_reduction`, `task_type`, `accuracy_drop`.
- The `LoggingCallback` works with any Hugging Face `Trainer`.
- The layer‑wise analyzer requires a model that can be hooked (any HF transformer).

## 📜 Dependencies

- `matplotlib`, `seaborn`, `pandas`, `numpy`, `scipy`, `scikit‑learn`
- `torch`, `transformers` (for `LayerwiseAnalyzer`)

## 📄 License

Part of the **Efficient Transformer Toolkit** – MIT License.
