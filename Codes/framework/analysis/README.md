# 📊 Analysis & Visualization Package

A complete toolkit for logging, tracking, plotting, and comparing your compression experiments.  
All outputs are automatically saved in a structured `results/` folder.

## 📁 Package Structure

```
framework/analysis/
├── __init__.py
├── callbacks.py           # LoggingCallback for HuggingFace Trainer
├── metrics.py             # Metrics computation + MetricsTracker (JSONL)
├── plotting.py            # Comprehensive plotting functions
├── model_comparison.py    # Compare multiple runs & save comparison plots
├── layerwise_analyzer.py  # Inspect hidden representations & attention
├── report_generator.py    # HTML, CSV, LaTeX reports
└── README.md              # This file
```

## 🚀 Quick Start

```python
from framework.analysis import (
    LoggingCallback, MetricsTracker, ModelComparison
)
import logging

# 1. Set up metrics tracking (auto‑creates directories)
tracker = MetricsTracker("results/my_experiment/metrics.json")

# 2. Create logger
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

# 3. Add callback to HuggingFace Trainer
trainer.add_callback(
    LoggingCallback(
        tracker,
        logger,
        plots_dir="results/my_experiment/plots",
        save_plots=True
    )
)

# 4. Train as usual
trainer.train()
```

After training, your `results/` folder will contain:

```
results/my_experiment/
├── metrics.json                # Step‑wise metrics (JSON lines)
└── plots/
    └── training_history.png     # Loss curves from training
```

If you run multiple experiments and want to compare them:

```python
from framework.analysis import quick_compare

quick_compare(
    results_dirs=["results/run1", "results/run2", "results/run3"],
    output_dir="results/comparison"
)
```

This will create `results/comparison/plots/` with comprehensive 3×3 grid plots and tradeoff charts.

---

## 📦 Modules

### 1. `callbacks.py` – LoggingCallback

A `TrainerCallback` that automatically:
- Captures `loss`, `eval_loss`, and any other logged metrics.
- Records them via `MetricsTracker`.
- Logs them to the console using the provided logger.
- Generates and saves a `training_history.png` plot at the end of training.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tracker` | `MetricsTracker` | Instance to store metrics |
| `logger` | `logging.Logger` | Logger for console output |
| `plots_dir` | `str` (optional) | Directory to save plots (default: `None`) |
| `save_plots` | `bool` | Whether to save plots (default: `True`) |

### 2. `metrics.py` – Metrics Computation & Tracking

- **`compute_metrics(preds, labels, task)`** – computes accuracy, F1, Matthews, Pearson/Spearman based on task type.
- **`aggregate_results(list_of_dicts)`** – aggregates multiple runs (mean, std).
- **`MetricsTracker`** – simple JSON‑lines logger for step‑wise metrics.

**MetricsTracker methods:**

| Method | Description |
|--------|-------------|
| `log(metrics, step, flush=False)` | Record a dict (optionally with a step number) |
| `save()` / `_flush()` | Write in‑memory records to the JSON file |
| `to_dataframe()` | Load all records into a pandas DataFrame |

### 3. `plotting.py` – Publication‑Ready Plots

Contains functions that generate and save plots directly to disk.

| Function | Output |
|----------|--------|
| `create_comprehensive_plots(df, plots_dir)` | 3×3 grid: accuracy vs rank, speedup, param reduction, best config, etc. |
| `plot_training_history(history, save_path)` | Loss curves (train + eval) |
| `plot_tradeoff(df, plots_dir)` | Accuracy vs speedup scatter plot |

All plots use `seaborn-v0_8-darkgrid` style, 300 DPI, and are saved as PNG files.

### 4. `model_comparison.py` – Compare Multiple Experiments

- **`ModelComparison(results_dirs, output_dir)`** – loads CSV/JSON results from several run directories.
- **`quick_compare(results_dirs, output_dir)`** – one‑liner to load and generate all comparison plots.

The comparison collects all `performance.csv` and `metrics.json` files and creates:

```
results/comparison/plots/
├── comprehensive_analysis.png
└── tradeoff.png
```

### 5. `layerwise_analyzer.py` – Inspect Model Internals

- **`LayerwiseAnalyzer(model, tokenizer)`** – registers hooks and captures hidden states.
- **`analyze(text)`** – returns cosine similarities, effective ranks, intrinsic dimensions, attention entropy.
- **`save_results(results, path)`** – saves analysis to JSON.

### 6. `report_generator.py` – HTML, CSV, LaTeX Reports

- **`generate_html_report(df, output_dir, title)`** – creates a styled HTML report with summary tables and best configurations.
- **`generate_csv_summary(df, output_path)`** – saves a grouped CSV summary.
- **`generate_latex_table(df, output_path)`** – produces a LaTeX table for papers.

---

## 📂 Output Directory Convention

All output is organized under a user‑specified `results/` directory.  
A typical workflow produces:

```
results/
├── experiment1/
│   ├── metrics.json
│   └── plots/
│       └── training_history.png
├── experiment2/
│   ├── metrics.json
│   └── plots/
│       └── training_history.png
└── comparison/
    └── plots/
        ├── comprehensive_analysis.png
        └── tradeoff.png
```

You can customise the root folder by changing the `plots_dir` parameter or the `output_dir` in `quick_compare`.

---

## 🔄 Integration with the CLI

The analysis package is fully integrated with the CLI’s logging system.  
When you run:

```bash
python cli.py train --preset fast --task mrpc
```

The CLI automatically:
- Creates a timestamped run directory under `logs/`
- Sets up logging and metrics tracking
- Passes the `LoggingCallback` to the trainer
- Saves training plots in `logs/<run>/plots/`

After training, you can directly call:

```bash
python cli.py train --task mrpc --attention aurora
```

And find the plots in `logs/mrpc_aurora_<timestamp>/plots/`.

---

## 🛠️ Customization

- **Add your own plots:** Import `plotting.py` functions or add new ones; they just need a DataFrame and a save path.
- **Use without Trainer:** Call `MetricsTracker` directly and then pass the DataFrame to `create_comprehensive_plots`.
- **Layer analysis:** Use `LayerwiseAnalyzer` on any HuggingFace model to understand which layers can be pruned more aggressively.

---

## 📜 License

Part of the **Efficient Transformer Toolkit** – MIT License.
